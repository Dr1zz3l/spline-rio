"""
Joint offline calibration of radar extrinsics and time offsets.

Jointly optimises:
  - Rotation perturbation δR ∈ so(3) around nominal R_bs  (1–3 DOF)
  - Radar time offset dt_radar                             (1 DOF)
  - Translation T_bs  (optional, --optimize-translation)  (0–3 DOF)

Also calibrates IMU-to-MoCap timing via gyro cross-correlation (independent).

Usage:
    python calibrate_extrinsics.py slow_racing_best_velocity [bag2 ...] \\
        [--optimize-translation] [--full-rotation]
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares
from scipy.signal import correlate

from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (
    rotation_matrix_from_euler,
    predict_doppler_velocity,
    unwrap_doppler,
)
from config_loader import load_config


# ==================== SO(3) helpers ====================

def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map so(3) → SO(3)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        return np.eye(3) + np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0],
        ])
    return Rotation.from_rotvec(omega).as_matrix()


def rotation_to_euler_deg(R: np.ndarray):
    """Convert rotation matrix to [roll, pitch, yaw] degrees (ZYX convention)."""
    r = Rotation.from_matrix(R)
    # as_euler('ZYX') returns [yaw, pitch, roll]
    euler_zyx = r.as_euler('ZYX', degrees=True)
    return np.array([euler_zyx[2], euler_zyx[1], euler_zyx[0]])  # [roll, pitch, yaw]


# ==================== Data loading ====================

def load_bag_data(bag_key: str, bags_cfg: dict, cfg: dict) -> dict:
    """Load a single bag and extract MoCap + radar + IMU arrays."""
    bags = bags_cfg.get('bags', {})
    flipped_bags = set(bags_cfg.get('flipped', []))
    timing = bags_cfg.get('timing', {})

    if bag_key in bags:
        bag_path = bags[bag_key]
    else:
        bag_path = bag_key

    start_offset = 5.0
    duration = 120.0
    if bag_key in timing:
        start_offset, duration = timing[bag_key]

    print(f"  Loading {bag_key} → {bag_path}  (offset={start_offset}s, dur={duration}s)...")
    bag_data = load_bag_topics(bag_path, verbose=False)

    t_start = bag_data.start_time + start_offset
    t_end = t_start + duration

    agiros_states = [s for s in bag_data.agiros_state if t_start <= s.timestamp <= t_end]
    radar_frames  = [f for f in bag_data.radar_velocity if t_start <= f.timestamp <= t_end]
    imu_data      = [d for d in bag_data.imu_data if t_start <= d.timestamp <= t_end]

    print(f"    MoCap: {len(agiros_states)}, Radar frames: {len(radar_frames)}, IMU: {len(imu_data)}")

    # V_max per bag type
    radar_cfg = bags_cfg.get('radar_config', {})
    if 'best_velocity' in bag_key:
        _rc = radar_cfg.get('best_velocity', {})
    else:
        _rc = radar_cfg.get('default', {})
    v_max = _rc.get('v_max', 4.99)

    # Yaw-flip flag
    is_flipped = bag_key in flipped_bags

    return {
        'bag_key': bag_key,
        'agiros_states': agiros_states,
        'radar_frames': radar_frames,
        'imu_data': imu_data,
        'v_max': v_max,
        'is_flipped': is_flipped,
    }


# ==================== Interpolators ====================

def build_interpolators(agiros_states: list) -> dict:
    """Build velocity/omega/SLERP interpolators from MoCap states."""
    times      = np.array([s.timestamp for s in agiros_states])
    velocities = np.array([s.velocity for s in agiros_states])
    omegas     = np.array([s.angular_velocity for s in agiros_states])
    quats      = np.array([s.orientation for s in agiros_states])  # [qx,qy,qz,qw]

    vel_interp   = interp1d(times, velocities, axis=0, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    omega_interp = interp1d(times, omegas, axis=0, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    slerp = Slerp(times, Rotation.from_quat(quats))

    return {
        'vel_interp':   vel_interp,
        'omega_interp': omega_interp,
        'slerp':        slerp,
        't_min':        times[0],
        't_max':        times[-1],
    }


# ==================== Radar point collection ====================

def collect_radar_points(radar_frames: list, t_min: float, t_max: float,
                         min_range: float) -> dict:
    """Extract flat arrays of (timestamp, position_sensor, v_meas) from radar frames."""
    timestamps = []
    positions  = []
    v_meas_arr = []

    for frame in radar_frames:
        t = frame.timestamp
        if t < t_min or t > t_max:
            continue
        for i in range(frame.num_points()):
            p_s = frame.positions[i]
            r = frame.ranges[i] if frame.ranges is not None else np.linalg.norm(p_s)
            if r < min_range:
                continue
            timestamps.append(t)
            positions.append(p_s)
            v_meas_arr.append(frame.velocities[i])

    return {
        'timestamps': np.array(timestamps),
        'positions':  np.array(positions),
        'v_meas':     np.array(v_meas_arr),
    }


# ==================== Forward model ====================

def compute_predicted_dopplers(points: dict, interps: dict,
                                R_bs: np.ndarray, T_bs: np.ndarray,
                                dt_radar: float) -> np.ndarray:
    """Vectorized Doppler prediction for all radar points."""
    timestamps = points['timestamps']
    positions  = points['positions']
    t_min = interps['t_min']
    t_max = interps['t_max']

    v_preds = np.empty(len(timestamps))
    for idx in range(len(timestamps)):
        t = np.clip(timestamps[idx] + dt_radar, t_min, t_max)
        v_world    = interps['vel_interp'](t)
        omega_body = interps['omega_interp'](t)
        R_wb       = interps['slerp'](t).as_matrix()
        v_preds[idx] = predict_doppler_velocity(
            v_world, omega_body, R_wb,
            positions[idx:idx+1], T_bs, R_bs
        )[0]

    return v_preds


def compute_predicted_dopplers_batch(points: dict, interps: dict,
                                     R_bs: np.ndarray, T_bs: np.ndarray,
                                     dt_radar: float) -> np.ndarray:
    """Batch version: interpolate all timestamps at once then vectorise inner loop."""
    timestamps = points['timestamps']
    positions  = points['positions']
    t_min = interps['t_min']
    t_max = interps['t_max']
    t_eval = np.clip(timestamps + dt_radar, t_min, t_max)

    v_world_all    = interps['vel_interp'](t_eval)    # (N,3)
    omega_body_all = interps['omega_interp'](t_eval)  # (N,3)
    R_wb_all       = interps['slerp'](t_eval).as_matrix()  # (N,3,3)

    N = len(timestamps)
    v_preds = np.empty(N)
    for i in range(N):
        v_preds[i] = predict_doppler_velocity(
            v_world_all[i], omega_body_all[i], R_wb_all[i],
            positions[i:i+1], T_bs, R_bs
        )[0]
    return v_preds


# ==================== Residual function ====================

def make_residual_fn(bag_data_list: list, k_arrays: list,
                     R_nominal: np.ndarray, T_nominal: np.ndarray,
                     optimize_translation: bool,
                     prior_weights: dict):
    """Return a closure for scipy.optimize.least_squares.

    Parameter vector x:
      [δR[0], δR[1], δR[2], dt_radar, (T[0], T[1], T[2])]
    """
    def residual_fn(x):
        delta_r   = x[0:3]
        dt_radar  = x[3]
        if optimize_translation:
            T_bs = x[4:7]
        else:
            T_bs = T_nominal

        R_bs = R_nominal @ so3_exp(delta_r)

        all_residuals = []
        for bd, k_arr in zip(bag_data_list, k_arrays):
            interps = bd['interps']
            points  = bd['points']

            v_pred = compute_predicted_dopplers_batch(points, interps, R_bs, T_bs, dt_radar)
            v_meas_unwrapped = points['v_meas'] + k_arr * 2.0 * bd['v_max']
            r = v_meas_unwrapped - v_pred
            all_residuals.append(r)

        # Rotation priors (as soft constraints in the residual vector)
        lam_roll = prior_weights.get('roll', 100.0)
        lam_yaw  = prior_weights.get('yaw',  100.0)
        prior_residuals = [
            np.sqrt(lam_roll) * delta_r[0],
            np.sqrt(lam_yaw)  * delta_r[2],
        ]

        return np.concatenate(all_residuals + [prior_residuals])

    return residual_fn


# ==================== IMU timing calibration ====================

def calibrate_imu_timing(bag_data_list: list, config_offset: float) -> dict:
    """Gyro cross-correlation to estimate IMU-to-MoCap time offset."""
    print("\n--- IMU Timing (gyro cross-correlation) ---")
    all_offsets = []

    per_bag = {}
    for bd in bag_data_list:
        bag_key     = bd['bag_key']
        agiros_states = bd['agiros_states']
        imu_data      = bd['imu_data']

        if not imu_data:
            print(f"  {bag_key}: no IMU data, skipping")
            continue

        # Build MoCap omega interpolator
        mocap_times  = np.array([s.timestamp for s in agiros_states])
        mocap_omegas = np.array([s.angular_velocity for s in agiros_states])
        omega_interp = interp1d(mocap_times, mocap_omegas, axis=0, kind='linear',
                                bounds_error=False, fill_value='extrapolate')

        gyro_times = np.array([d.timestamp for d in imu_data])
        gyro_meas  = np.array([d.angular_velocity for d in imu_data])

        # Only use samples inside MoCap range
        mask = (gyro_times >= mocap_times[0] + 0.01) & (gyro_times <= mocap_times[-1] - 0.01)
        gyro_times = gyro_times[mask]
        gyro_meas  = gyro_meas[mask]

        if len(gyro_times) < 50:
            print(f"  {bag_key}: insufficient overlapping IMU samples, skipping")
            continue

        gyro_pred = omega_interp(gyro_times)
        gyro_times_rel = gyro_times - gyro_times[0]

        dt_sync = 0.005  # 200 Hz grid
        t_grid = np.arange(gyro_times_rel[0], gyro_times_rel[-1], dt_sync)
        gyro_meas_reg = np.zeros((len(t_grid), 3))
        gyro_pred_reg = np.zeros((len(t_grid), 3))
        for ax_i in range(3):
            gyro_meas_reg[:, ax_i] = np.interp(t_grid, gyro_times_rel, gyro_meas[:, ax_i])
            gyro_pred_reg[:, ax_i] = np.interp(t_grid, gyro_times_rel, gyro_pred[:, ax_i])

        max_shift_samples = int(0.5 / dt_sync)
        best_shifts = []
        for ax_i in range(3):
            sig_m = gyro_meas_reg[:, ax_i] - gyro_meas_reg[:, ax_i].mean()
            sig_p = gyro_pred_reg[:, ax_i] - gyro_pred_reg[:, ax_i].mean()
            cc = correlate(sig_m, sig_p, mode='full')
            mid = len(sig_p) - 1
            cc_window = cc[mid - max_shift_samples:mid + max_shift_samples + 1]
            lags_window = np.arange(-max_shift_samples, max_shift_samples + 1) * dt_sync
            best_lag = lags_window[np.argmax(cc_window)]
            best_shifts.append(best_lag)

        median_offset = float(np.median(best_shifts))
        per_bag[bag_key] = {
            'X': best_shifts[0], 'Y': best_shifts[1], 'Z': best_shifts[2],
            'median': median_offset,
        }
        all_offsets.append(median_offset)
        print(f"  {bag_key:40s}: {median_offset*1000:+6.1f} ms  "
              f"(X:{best_shifts[0]*1000:+.0f}  Y:{best_shifts[1]*1000:+.0f}  Z:{best_shifts[2]*1000:+.0f})")

    overall = float(np.median(all_offsets)) if all_offsets else config_offset
    print(f"  Median overall:  {overall*1000:+.1f} ms  (config: {config_offset*1000:+.1f} ms)")
    return {'per_bag': per_bag, 'overall_median': overall}


# ==================== Main calibration ====================

def run_calibration():
    parser = argparse.ArgumentParser(
        description='Joint calibration of radar extrinsics and time offsets')
    parser.add_argument('bags', nargs='+', help='Bag name(s) or paths')
    parser.add_argument('--optimize-translation', action='store_true',
                        help='Also optimise translation T_bs (3 DOF)')
    parser.add_argument('--full-rotation', action='store_true',
                        help='Free all 3 rotation DOF (default: pitch only, roll/yaw priors)')
    args = parser.parse_args()

    print("=" * 70)
    print("RADAR EXTRINSIC CALIBRATION")
    print("=" * 70)

    # --- Config ---
    cfg = load_config()
    bags_cfg  = cfg['bags']
    ext_cfg   = cfg['extrinsics']
    solver_cfg = cfg['solver']

    MIN_RANGE = solver_cfg['min_range']

    ROTATION_EULER = np.array(ext_cfg['rotation_euler_deg'], dtype=float)
    TRANSLATION    = np.array(ext_cfg['translation_body_m'], dtype=float)
    imu_mocap_offset_cfg  = float(ext_cfg.get('imu_mocap_offset_sec', 0.020))
    radar_imu_offset_cfg  = float(ext_cfg.get('radar_imu_offset_sec', 0.140))
    # radar_total_offset = imu_mocap_offset - radar_imu_offset
    dt_radar_init = imu_mocap_offset_cfg - radar_imu_offset_cfg  # typically -0.120

    R_nominal = rotation_matrix_from_euler(
        np.radians(ROTATION_EULER[0]),
        np.radians(ROTATION_EULER[1]),
        np.radians(ROTATION_EULER[2]),
    )

    # --- Load bags ---
    print("\nLoading bags...")
    bag_data_list = []
    for bag_key in args.bags:
        bd = load_bag_data(bag_key, bags_cfg, cfg)
        if not bd['agiros_states'] or not bd['radar_frames']:
            print(f"  WARNING: {bag_key} has insufficient data, skipping")
            continue
        # Build interpolators
        bd['interps'] = build_interpolators(bd['agiros_states'])
        # Collect radar points
        interps = bd['interps']
        # Apply yaw flip to nominal rotation if bag is flipped
        R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)
        R_nom_bag = R_yaw_flip @ R_nominal if bd['is_flipped'] else R_nominal
        T_nom_bag = R_yaw_flip @ TRANSLATION if bd['is_flipped'] else TRANSLATION.copy()
        bd['R_nominal_bag'] = R_nom_bag
        bd['T_nominal_bag'] = T_nom_bag

        pts = collect_radar_points(
            bd['radar_frames'],
            interps['t_min'], interps['t_max'],
            MIN_RANGE
        )
        bd['points'] = pts
        print(f"  {bag_key}: {len(pts['timestamps'])} radar points collected")
        bag_data_list.append(bd)

    if not bag_data_list:
        print("ERROR: No valid bags loaded.")
        sys.exit(1)

    # --- IMU timing calibration ---
    imu_result = calibrate_imu_timing(bag_data_list, imu_mocap_offset_cfg)
    imu_mocap_offset_cal = imu_result['overall_median']

    # --- Radar extrinsics optimisation ---
    print("\n--- Radar Extrinsic Optimisation ---")

    # Use first bag's nominal (for single-bag; multi-bag assumes same nominal).
    # For multi-bag with mixed flip, we need per-bag transforms; the residual
    # function applies R_nominal_bag from each bag's own entry.
    # We parameterise δR around the non-flipped R_nominal and compose per-bag.
    T_nominal = TRANSLATION.copy()

    # Priors: strong on roll/yaw unless --full-rotation
    if args.full_rotation:
        prior_weights = {'roll': 0.0, 'yaw': 0.0}
        print("  Full rotation: all 3 DOF free (no priors)")
    else:
        prior_weights = {'roll': 100.0, 'yaw': 100.0}
        print("  Partial rotation: pitch free, roll/yaw with priors λ=100")

    # Build residual function that handles per-bag flip internally
    def residual_fn_multibag(x):
        delta_r  = x[0:3]
        dt_radar = x[3]
        T_bs = x[4:7] if args.optimize_translation else T_nominal

        all_residuals = []
        for bd, k_arr in zip(bag_data_list, k_arrays):
            interps = bd['interps']
            points  = bd['points']
            R_bs = bd['R_nominal_bag'] @ so3_exp(delta_r)
            if args.optimize_translation and bd['is_flipped']:
                R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)
                T_bs_bag = R_yaw_flip @ T_bs
            else:
                T_bs_bag = T_bs

            v_pred = compute_predicted_dopplers_batch(points, interps, R_bs, T_bs_bag, dt_radar)
            v_meas_unwrapped = points['v_meas'] + k_arr * 2.0 * bd['v_max']
            all_residuals.append(v_meas_unwrapped - v_pred)

        # Rotation priors
        lam_roll = prior_weights.get('roll', 100.0)
        lam_yaw  = prior_weights.get('yaw', 100.0)
        priors = []
        if lam_roll > 0:
            priors.append(np.sqrt(lam_roll) * delta_r[0])
        if lam_yaw > 0:
            priors.append(np.sqrt(lam_yaw) * delta_r[2])

        return np.concatenate(all_residuals + [np.array(priors)])

    # Initial parameter vector
    n_params = 4 + (3 if args.optimize_translation else 0)
    x0 = np.zeros(n_params)
    x0[3] = dt_radar_init
    if args.optimize_translation:
        x0[4:7] = T_nominal

    # Bounds
    rot_bound  = np.radians(20.0)
    dt_lo, dt_hi = -0.300, 0.050
    lb = [-rot_bound, -rot_bound, -rot_bound, dt_lo]
    ub = [ rot_bound,  rot_bound,  rot_bound, dt_hi]
    if args.optimize_translation:
        lb += [-0.3, -0.3, -0.3]
        ub += [ 0.3,  0.3,  0.3]

    # --- Outer loop with unwrapping ---
    MAX_OUTER = 3
    k_arrays = [np.zeros(len(bd['points']['timestamps']), dtype=int) for bd in bag_data_list]

    # Compute initial k_i
    for i, bd in enumerate(bag_data_list):
        v_pred_init = compute_predicted_dopplers_batch(
            bd['points'], bd['interps'],
            bd['R_nominal_bag'], T_nominal, dt_radar_init
        )
        k_arrays[i] = np.round(
            (v_pred_init - bd['points']['v_meas']) / (2.0 * bd['v_max'])
        ).astype(int)
        n_unwrapped = np.sum(k_arrays[i] != 0)
        print(f"  {bd['bag_key']}: initial unwrap: {n_unwrapped}/{len(k_arrays[i])} points")

    result = None
    for outer_iter in range(MAX_OUTER):
        print(f"\n  Outer iteration {outer_iter + 1}/{MAX_OUTER}...")
        result = least_squares(
            residual_fn_multibag, x0,
            method='trf', loss='huber', f_scale=1.0,
            bounds=(lb, ub),
            x_scale='jac',
            max_nfev=2000,
            verbose=0,
        )
        x_opt = result.x
        print(f"    cost={result.cost:.4f}, nfev={result.nfev}, status={result.status}")

        # Recompute k_i at solution
        changed = False
        for i, bd in enumerate(bag_data_list):
            delta_r  = x_opt[0:3]
            dt_radar = x_opt[3]
            T_bs     = x_opt[4:7] if args.optimize_translation else T_nominal
            R_bs     = bd['R_nominal_bag'] @ so3_exp(delta_r)
            if args.optimize_translation and bd['is_flipped']:
                R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)
                T_bs = R_yaw_flip @ T_bs

            v_pred_new = compute_predicted_dopplers_batch(
                bd['points'], bd['interps'], R_bs, T_bs, dt_radar
            )
            k_new = np.round(
                (v_pred_new - bd['points']['v_meas']) / (2.0 * bd['v_max'])
            ).astype(int)
            n_changed = np.sum(k_new != k_arrays[i])
            if n_changed > 0:
                changed = True
                print(f"    {bd['bag_key']}: {n_changed} unwrap changes")
            k_arrays[i] = k_new

        x0 = x_opt
        if not changed:
            print(f"  Converged (no unwrap changes).")
            break

    # --- Extract results ---
    x_opt     = result.x
    delta_r   = x_opt[0:3]
    dt_radar  = x_opt[3]
    T_opt     = x_opt[4:7] if args.optimize_translation else T_nominal

    # Calibrated rotation (compose with non-flipped nominal)
    R_cal = R_nominal @ so3_exp(delta_r)
    euler_cal  = rotation_to_euler_deg(R_cal)
    euler_init = ROTATION_EULER.copy()

    # Timing
    # dt_radar = imu_mocap_offset - radar_imu_offset  ← from definition
    # We want to update radar_imu_offset:
    #   radar_imu_offset_cal = imu_mocap_offset_cal - dt_radar
    radar_imu_offset_cal = imu_mocap_offset_cal - dt_radar

    # 1-sigma uncertainties from Jacobian.
    # result.jac from TRF with loss='huber' is the sqrt-weight-scaled Jacobian,
    # so J^T J ≈ Fisher information already scaled.  We compute s² from the
    # *raw* (unscaled) Doppler residuals to get physically meaningful units.
    J = result.jac  # shape (M, n_params) — scaled by sqrt(huber weights)
    sigma_dr_deg = np.full(3, float('nan'))
    sigma_dt_ms  = float('nan')
    sigma_T      = np.full(3, float('nan'))
    try:
        JtJ = J.T @ J
        # Raw residuals (Doppler only, prior rows excluded) for s²
        n_raw = sum(len(bd['points']['timestamps']) for bd in bag_data_list)
        raw_res = residual_fn_multibag(x_opt)[:n_raw]
        n_dof = max(n_raw - n_params, 1)
        s2 = np.sum(raw_res**2) / n_dof
        cov = np.linalg.pinv(JtJ) * s2
        sigma = np.sqrt(np.abs(np.diag(cov)))
        sigma_dr_deg = np.degrees(sigma[0:3])
        sigma_dt_ms  = sigma[3] * 1000
        if args.optimize_translation:
            sigma_T = sigma[4:7] * 1000  # mm
    except (np.linalg.LinAlgError, Exception):
        pass

    # Before/after residuals
    def compute_rmse_corr(x_vec, k_arrs):
        residuals_all = []
        preds_all = []
        meas_all  = []
        for bd, k_arr in zip(bag_data_list, k_arrs):
            dR   = x_vec[0:3]
            dt   = x_vec[3]
            T_bs = x_vec[4:7] if args.optimize_translation else T_nominal
            R_bs = bd['R_nominal_bag'] @ so3_exp(dR)
            if args.optimize_translation and bd['is_flipped']:
                R_yaw_flip_loc = rotation_matrix_from_euler(0.0, 0.0, np.pi)
                T_bs = R_yaw_flip_loc @ T_bs
            vp = compute_predicted_dopplers_batch(bd['points'], bd['interps'], R_bs, T_bs, dt)
            vm = bd['points']['v_meas'] + k_arr * 2.0 * bd['v_max']
            preds_all.append(vp)
            meas_all.append(vm)
            residuals_all.append(vm - vp)
        all_r = np.concatenate(residuals_all)
        all_p = np.concatenate(preds_all)
        all_m = np.concatenate(meas_all)
        rmse = np.sqrt(np.mean(all_r**2))
        corr = np.corrcoef(all_m, all_p)[0, 1]
        return rmse, corr

    x_before = np.zeros(n_params)
    x_before[3] = dt_radar_init
    if args.optimize_translation:
        x_before[4:7] = T_nominal
    k_init = [np.zeros(len(bd['points']['timestamps']), dtype=int) for bd in bag_data_list]
    rmse_before, corr_before = compute_rmse_corr(x_before, k_init)
    rmse_after,  corr_after  = compute_rmse_corr(x_opt, k_arrays)

    total_pts = sum(len(bd['points']['timestamps']) for bd in bag_data_list)
    total_unwrapped = sum(np.sum(k != 0) for k in k_arrays)

    # --- Report ---
    print("\n")
    print("=" * 70)
    print("=== RADAR EXTRINSIC CALIBRATION RESULTS ===")
    bag_summaries = ", ".join(
        f"{bd['bag_key']} (N={len(bd['points']['timestamps'])})"
        for bd in bag_data_list
    )
    print(f"Bags: {bag_summaries}")

    print(f"\n--- IMU Timing (gyro cross-correlation) ---")
    for bk, v in imu_result['per_bag'].items():
        print(f"  {bk:40s}: {v['median']*1000:+6.1f} ms  "
              f"(X:{v['X']*1000:+.0f}  Y:{v['Y']*1000:+.0f}  Z:{v['Z']*1000:+.0f})")
    print(f"  Calibrated median:  {imu_mocap_offset_cal*1000:+.1f} ms  "
          f"(config: {imu_mocap_offset_cfg*1000:+.1f} ms)")

    print(f"\n--- Calibration Result ({outer_iter + 1} outer iterations) ---")
    header = f"  {'Parameter':<18}  {'Initial':>12}  {'Calibrated':>12}  {'Δ':>10}  {'1-σ':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def fmt_rot(name, idx, is_prior):
        init_v = euler_init[idx]
        cal_v  = euler_cal[idx]
        delta  = cal_v - init_v
        unc    = "[prior]" if is_prior else f"±{sigma_dr_deg[idx]:.2f}°"
        print(f"  {name:<18}  {init_v:>11.2f}°  {cal_v:>11.2f}°  {delta:>+9.2f}°  {unc:>10}")

    fmt_rot("roll",  0, is_prior=(prior_weights.get('roll', 0) > 0 and not args.full_rotation))
    fmt_rot("pitch", 1, is_prior=False)
    fmt_rot("yaw",   2, is_prior=(prior_weights.get('yaw', 0) > 0 and not args.full_rotation))

    init_dt_ms = dt_radar_init * 1000
    cal_dt_ms  = dt_radar * 1000
    print(f"  {'dt_radar':<18}  {init_dt_ms:>11.1f}ms  {cal_dt_ms:>11.1f}ms  "
          f"{cal_dt_ms - init_dt_ms:>+9.1f}ms  ±{sigma_dt_ms:.1f}ms")

    if args.optimize_translation:
        print(f"\n  Translation (mm):")
        axis_names = ['Tx', 'Ty', 'Tz']
        for i, name in enumerate(axis_names):
            init_v = T_nominal[i] * 1000
            cal_v  = T_opt[i] * 1000
            print(f"  {name:<18}  {init_v:>11.1f}mm  {cal_v:>11.1f}mm  "
                  f"{cal_v - init_v:>+9.1f}mm  ±{sigma_T[i]:.1f}mm")

    pct_change = (rmse_after - rmse_before) / rmse_before * 100
    print(f"\n  {'Metric':<18}  {'Before':>12}  {'After':>12}  {'Δ':>10}")
    print("  " + "-" * 54)
    print(f"  {'RMSE':<18}  {rmse_before:>11.3f}   {rmse_after:>11.3f}   {pct_change:>+9.1f}%")
    print(f"  {'Correlation':<18}  {corr_before:>12.4f}  {corr_after:>12.4f}")
    pct_unwrap = total_unwrapped / total_pts * 100 if total_pts > 0 else 0
    print(f"\n  Unwrap changes at solution: {total_unwrapped}/{total_pts} pts ({pct_unwrap:.1f}%)")

    # --- Suggested YAML ---
    print(f"\n--- Suggested config/extrinsics.yaml ---")
    T_out = T_opt if args.optimize_translation else TRANSLATION
    print(f"rotation_euler_deg: [{euler_cal[0]:.2f}, {euler_cal[1]:.2f}, {euler_cal[2]:.2f}]")
    print(f"translation_body_m: [{T_out[0]:.4f}, {T_out[1]:.4f}, {T_out[2]:.4f}]")
    print(f"imu_mocap_offset_sec: {imu_mocap_offset_cal:.4f}")
    print(f"radar_imu_offset_sec: {radar_imu_offset_cal:.4f}")
    print(f"  # (dt_radar = imu_mocap_offset - radar_imu_offset = {dt_radar*1000:.1f} ms)")


if __name__ == '__main__':
    run_calibration()
