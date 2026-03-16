"""
PHYSICS UNIT TEST: MoCap vs Sensors  (v2 — uses Agiros acceleration field)

Feed Ground Truth (MoCap) into the Forward Model and compare predictions
against actual sensor readings.  No optimisation — pure physics check.

Diagnoses:
- Radar Doppler sign errors  (scatter plot, correlation)
- Gravity sign / IMU convention errors  (accel Z mean)
- Extrinsic calibration issues  (radar correlation)
- Time synchronisation problems  (radar corr vs time-offset sweep)
- Angular-velocity frame mismatch  (body-frame vs world-frame comparison)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import butter, filtfilt

from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (
    rotation_matrix_from_euler,
    predict_doppler_velocity,
    unwrap_doppler,
)
from config_loader import load_config


def zero_phase_lowpass(x, dt, cutoff_hz=12.0, order=4):
    """Apply zero-phase Butterworth low-pass filter along axis 0."""
    if x is None or len(x) < 8 or dt <= 0:
        return x
    fs = 1.0 / dt
    nyq = 0.5 * fs
    if cutoff_hz >= 0.95 * nyq:
        return x

    b, a = butter(order, cutoff_hz / nyq, btype='low')
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(x) <= padlen:
        return x
    return filtfilt(b, a, x, axis=0)


def run_physics_diagnostics():
    print("=" * 80)
    print("PHYSICS UNIT TEST  v2: MoCap vs Sensors")
    print("=" * 80)

    # ==================== Configuration ====================
    cfg = load_config()
    bags_cfg = cfg['bags']
    ext = cfg['extrinsics']

    bags = bags_cfg.get('bags', {})
    flipped_bags = set(bags_cfg.get('flipped', []))
    timing = bags_cfg.get('timing', {})

    positional_args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    bag_key = positional_args[0] if positional_args else "original"

    if bag_key in bags:
        BAG_PATH = bags[bag_key]
    else:
        BAG_PATH = bag_key  # Allow direct path

    START_OFFSET = 5.0
    DURATION = 120.0
    if bag_key in timing:
        START_OFFSET, DURATION = timing[bag_key]

    print(f"Bag: {bag_key} → {BAG_PATH}")

    # --- Extrinsic calibration ---
    ROTATION_EULER = np.array(ext['rotation_euler_deg'], dtype=float)  # [roll, pitch, yaw] deg
    TRANSLATION = np.array(ext['translation_body_m'], dtype=float)

    FLIP_BODY_FRAME = bag_key in flipped_bags
    if "--flip" in sys.argv:
        FLIP_BODY_FRAME = True
    if "--no-flip" in sys.argv:
        FLIP_BODY_FRAME = False

    ROTATION_EULER_RAD = np.radians(ROTATION_EULER)
    R_base = rotation_matrix_from_euler(
        ROTATION_EULER_RAD[0], ROTATION_EULER_RAD[1], ROTATION_EULER_RAD[2],
    )

    if FLIP_BODY_FRAME:
        R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)  # R_z(180°)
        sensor_rotation = R_yaw_flip @ R_base
        sensor_translation = R_yaw_flip @ TRANSLATION
        print(f"  ★ Body frame FLIPPED (R_z(180°) applied) — bag '{bag_key}' uses rotated agiros frame")
    else:
        sensor_rotation = R_base
        sensor_translation = TRANSLATION.copy()

    solver_cfg = cfg['solver']
    MIN_RANGE = solver_cfg['min_range']
    # Per-bag v_max: bags with "best_velocity" in their name use 3.84 m/s config
    radar_cfg = cfg['bags'].get('radar_config', {})
    if 'best_velocity' in bag_key:
        _rc = radar_cfg.get('best_velocity', {})
    else:
        _rc = radar_cfg.get('default', {})
    V_MAX_UNAMBIGUOUS = _rc.get('v_max', 4.99)
    g_world = np.array([0, 0, -9.81])

    # ==================== Load Data ====================
    print(f"\nLoading {BAG_PATH}...")
    bag_data = load_bag_topics(BAG_PATH, verbose=True)

    t_start = bag_data.start_time + START_OFFSET
    t_end = t_start + DURATION

    agiros_states = [s for s in bag_data.agiros_state if t_start <= s.timestamp <= t_end]
    radar_frames  = [f for f in bag_data.radar_velocity if t_start <= f.timestamp <= t_end]
    imu_data      = [d for d in bag_data.imu_data if t_start <= d.timestamp <= t_end]

    print(f"\nTime window: {START_OFFSET:.1f}s + {DURATION:.1f}s")
    print(f"  MoCap states: {len(agiros_states)}")
    print(f"  Radar frames: {len(radar_frames)}")
    print(f"  IMU samples:  {len(imu_data)}")

    # Check whether Agiros acceleration field is populated
    has_accel = agiros_states[0].acceleration is not None
    print(f"  Agiros has acceleration field: {has_accel}")

    if not agiros_states or not radar_frames:
        print("ERROR: Insufficient data!")
        return

    # ==================== Build MoCap Interpolators ====================
    mocap_times      = np.array([s.timestamp for s in agiros_states])
    mocap_positions  = np.array([s.position for s in agiros_states])
    mocap_velocities = np.array([s.velocity for s in agiros_states])
    mocap_quats      = np.array([s.orientation for s in agiros_states])  # [qx,qy,qz,qw]
    mocap_omegas     = np.array([s.angular_velocity for s in agiros_states])

    # Check MoCap data quality
    dt_mocap = np.diff(mocap_times)
    vel_jumps = np.linalg.norm(np.diff(mocap_velocities, axis=0), axis=1) / dt_mocap
    pos_jumps = np.linalg.norm(np.diff(mocap_positions, axis=0), axis=1) / dt_mocap
    print(f"  MoCap dt: mean={dt_mocap.mean()*1000:.1f}ms, max={dt_mocap.max()*1000:.1f}ms, min={dt_mocap.min()*1000:.1f}ms")
    print(f"  MoCap vel rate: p50={np.percentile(vel_jumps, 50):.1f}, p99={np.percentile(vel_jumps, 99):.1f}, max={vel_jumps.max():.1f} m/s²")
    print(f"  MoCap pos rate: p50={np.percentile(pos_jumps, 50):.1f}, p99={np.percentile(pos_jumps, 99):.1f}, max={pos_jumps.max():.1f} m/s")
    n_spikes = np.sum(vel_jumps > 500)
    if n_spikes > 0:
        print(f"  ⚠ {n_spikes} velocity spikes > 500 m/s² detected — MoCap tracking issues!")

    vel_interp   = interp1d(mocap_times, mocap_velocities, axis=0, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    omega_interp = interp1d(mocap_times, mocap_omegas, axis=0, kind='linear',
                            bounds_error=False, fill_value='extrapolate')

    # SLERP for rotation
    scipy_rots  = Rotation.from_quat(mocap_quats)
    mocap_slerp = Slerp(mocap_times, scipy_rots)

    # Acceleration interpolator — build from velocity differentiation
    # First: remove near-duplicate timestamps (MoCap has duplicates with µs jitter)
    MIN_DT = 0.001  # 1ms minimum between samples
    dt_raw = np.diff(mocap_times)
    good_mask = dt_raw > MIN_DT
    good_idx = np.where(good_mask)[0]
    n_removed = len(dt_raw) - np.sum(good_mask)
    print(f"  Removed {n_removed} near-duplicate MoCap timestamps (dt < {MIN_DT*1000:.0f}ms)")

    # Build clean velocity array for differentiation
    clean_times = mocap_times[np.concatenate([[0], good_idx + 1])]
    clean_vels  = mocap_velocities[np.concatenate([[0], good_idx + 1])]
    dt_clean = np.diff(clean_times)
    dv_clean = np.diff(clean_vels, axis=0)
    accel_num = dv_clean / dt_clean[:, None]
    accel_num_times = 0.5 * (clean_times[:-1] + clean_times[1:])

    # Savitzky-Golay smoothing on the clean data
    from scipy.signal import savgol_filter
    sg_window = 15  # ~50ms at 300Hz
    if sg_window % 2 == 0:
        sg_window += 1
    if len(clean_vels) > sg_window:
        vel_smooth = np.zeros_like(clean_vels)
        for ax_i in range(3):
            vel_smooth[:, ax_i] = savgol_filter(clean_vels[:, ax_i], sg_window, 3)
        dv_smooth = np.diff(vel_smooth, axis=0)
        accel_smooth = dv_smooth / dt_clean[:, None]
        accel_veldiff_interp = interp1d(accel_num_times, accel_smooth, axis=0, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
        print(f"  Using SavGol smoothed velocity diff (window={sg_window}, {len(clean_times)} clean samples)")
    else:
        accel_veldiff_interp = interp1d(accel_num_times, accel_num, axis=0, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
        print("  Using raw velocity differentiation")

    if has_accel:
        mocap_accels = np.array([s.acceleration for s in agiros_states])
        accel_agiros_interp = interp1d(mocap_times, mocap_accels, axis=0, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
    else:
        accel_agiros_interp = None

    # We'll test both sources in the accel check section

    # ==================== RADAR DOPPLER CHECK (with time-offset sweep) ==========
    print("\n--- Radar Doppler Check ---")

    def radar_correlation(dt_offset, wrap_alias=False, unwrap=False):
        """Compute correlation between measured and predicted Doppler with a time offset.

        Modes:
          wrap_alias=False, unwrap=False: raw v_pred vs raw v_meas
          wrap_alias=True:                wrap v_pred to [-V_MAX,+V_MAX] vs v_meas
          unwrap=True:                    unwrap v_meas toward raw v_pred vs raw v_pred
        """
        preds, meas = [], []
        for frame in radar_frames:
            t = frame.timestamp + dt_offset
            if t < mocap_times[0] or t > mocap_times[-1]:
                continue
            v_world    = vel_interp(t)
            omega_body = omega_interp(t)
            R_wb       = mocap_slerp(np.clip(t, mocap_times[0], mocap_times[-1])).as_matrix()

            for i in range(frame.num_points()):
                p_s = frame.positions[i]
                r = frame.ranges[i] if frame.ranges is not None else np.linalg.norm(p_s)
                if r < MIN_RANGE:
                    continue
                v_pred = predict_doppler_velocity(
                    v_world, omega_body, R_wb,
                    p_s.reshape(1, 3), sensor_translation, sensor_rotation
                )[0]
                v_meas_i = frame.velocities[i]
                if wrap_alias:
                    v_pred = ((v_pred + V_MAX_UNAMBIGUOUS) % (2 * V_MAX_UNAMBIGUOUS)) - V_MAX_UNAMBIGUOUS
                preds.append(v_pred)
                meas.append(v_meas_i)
        preds = np.array(preds)
        meas  = np.array(meas)
        if len(preds) < 10:
            return 0.0, preds, meas, meas - preds
        if unwrap:
            meas = unwrap_doppler(meas, preds, V_MAX_UNAMBIGUOUS)
        corr = np.corrcoef(meas, preds)[0, 1]
        return corr, preds, meas, meas - preds

    # Sweep time offsets to find best alignment — try raw, alias-wrapped, and unwrapped
    offsets = np.linspace(-0.4, 0.4, 61)
    corrs_raw = []
    corrs_alias = []
    corrs_unwrap = []
    for dt in offsets:
        c_raw,    _, _, _ = radar_correlation(dt, wrap_alias=False, unwrap=False)
        c_alias,  _, _, _ = radar_correlation(dt, wrap_alias=True,  unwrap=False)
        c_unwrap, _, _, _ = radar_correlation(dt, wrap_alias=False, unwrap=True)
        corrs_raw.append(c_raw)
        corrs_alias.append(c_alias)
        corrs_unwrap.append(c_unwrap)
    corrs_raw   = np.array(corrs_raw)
    corrs_alias = np.array(corrs_alias)
    corrs_unwrap = np.array(corrs_unwrap)

    best_idx_raw   = np.argmax(corrs_raw)
    best_idx_alias = np.argmax(corrs_alias)
    best_idx_unwrap = np.argmax(corrs_unwrap)
    print(f"  V_MAX_UNAMBIGUOUS = {V_MAX_UNAMBIGUOUS:.2f} m/s (from radar config, bag_key='{bag_key}')")
    print(f"  Time-offset sweep (raw):      best offset = {offsets[best_idx_raw]*1000:.1f} ms  (corr = {corrs_raw[best_idx_raw]:.4f})")
    print(f"  Time-offset sweep (aliased):  best offset = {offsets[best_idx_alias]*1000:.1f} ms  (corr = {corrs_alias[best_idx_alias]:.4f})")
    print(f"  Time-offset sweep (unwrapped):best offset = {offsets[best_idx_unwrap]*1000:.1f} ms  (corr = {corrs_unwrap[best_idx_unwrap]:.4f})")

    # Pick whichever method gives better correlation (unwrap takes priority over alias if tied)
    best_corrs = {
        'raw':    corrs_raw[best_idx_raw],
        'alias':  corrs_alias[best_idx_alias],
        'unwrap': corrs_unwrap[best_idx_unwrap],
    }
    best_mode = max(best_corrs, key=best_corrs.get)
    if best_mode == 'alias':
        best_offset = offsets[best_idx_alias]
        use_alias = True
        use_unwrap = False
        print(f"  → Using ALIASED predictions (best corr = {best_corrs['alias']:.4f})")
    elif best_mode == 'unwrap':
        best_offset = offsets[best_idx_unwrap]
        use_alias = False
        use_unwrap = True
        print(f"  → Using UNWRAPPED measurements (best corr = {best_corrs['unwrap']:.4f})")
    else:
        best_offset = offsets[best_idx_raw]
        use_alias = False
        use_unwrap = False
        print(f"  → Using RAW predictions (best corr = {best_corrs['raw']:.4f})")

    # Re-evaluate at best offset
    corr, pred_dopplers, meas_dopplers, residuals_radar = radar_correlation(
        best_offset, wrap_alias=use_alias, unwrap=use_unwrap)
    radar_times_rel = []
    for frame in radar_frames:
        t = frame.timestamp + best_offset
        if t < mocap_times[0] or t > mocap_times[-1]:
            continue
        for i in range(frame.num_points()):
            r = frame.ranges[i] if frame.ranges is not None else np.linalg.norm(frame.positions[i])
            if r < MIN_RANGE:
                continue
            radar_times_rel.append(frame.timestamp - t_start)
    radar_times_rel = np.array(radar_times_rel)

    print(f"  Points: {len(pred_dopplers)}")
    print(f"  Pred range: [{pred_dopplers.min():.3f}, {pred_dopplers.max():.3f}] m/s")
    print(f"  Meas range: [{meas_dopplers.min():.3f}, {meas_dopplers.max():.3f}] m/s")
    print(f"  Residual mean: {residuals_radar.mean():.4f} m/s")
    print(f"  Residual std:  {residuals_radar.std():.4f} m/s")
    print(f"  Residual RMSE: {np.sqrt(np.mean(residuals_radar**2)):.4f} m/s")
    print(f"  Correlation: {corr:.4f}  (1.0=perfect, -1.0=sign error)")

    # Quantization diagnostic
    unique_meas = np.unique(np.round(meas_dopplers, 3))
    if len(unique_meas) > 2:
        diffs = np.diff(np.sort(unique_meas))
        bin_width = np.median(diffs)
        print(f"  Doppler quantization: {len(unique_meas)} unique values, bin width ≈ {bin_width:.3f} m/s")

    # Also try negated Doppler (sign-error test)
    corr_neg = np.corrcoef(-meas_dopplers, pred_dopplers)[0, 1]
    print(f"  Correlation (negated meas): {corr_neg:.4f}  (high → sign error in measurement)")

    # Check subset of points that should NOT be aliased (|v_pred_raw| < 0.8 * V_MAX)
    corr_raw_full, pred_raw, _, _ = radar_correlation(best_offset, wrap_alias=False)
    safe_mask = np.abs(pred_raw) < 0.8 * V_MAX_UNAMBIGUOUS
    n_safe = np.sum(safe_mask)
    print(f"\n  Aliasing analysis:")
    print(f"    Points with |v_pred| < {0.8*V_MAX_UNAMBIGUOUS:.1f}: {n_safe}/{len(pred_raw)} ({100*n_safe/len(pred_raw):.0f}%)")
    if n_safe > 10:
        corr_safe = np.corrcoef(meas_dopplers[safe_mask], pred_raw[safe_mask])[0, 1]
        rmse_safe = np.sqrt(np.mean((meas_dopplers[safe_mask] - pred_raw[safe_mask])**2))
        print(f"    Correlation (safe subset): {corr_safe:.4f}")
        print(f"    RMSE (safe subset): {rmse_safe:.4f} m/s")
    n_aliased = np.sum(np.abs(pred_raw) > V_MAX_UNAMBIGUOUS)
    print(f"    Points with |v_pred| > {V_MAX_UNAMBIGUOUS:.1f}: {n_aliased}/{len(pred_raw)} ({100*n_aliased/len(pred_raw):.0f}%)")

    # Unwrap diagnostic: compare all three modes at best_offset
    print(f"\n  Unwrap diagnostic (at best_offset={best_offset*1000:.0f}ms):")
    _c_raw,    _p_raw,    _m_raw,    _ = radar_correlation(best_offset, wrap_alias=False, unwrap=False)
    _c_alias,  _p_alias,  _m_alias,  _ = radar_correlation(best_offset, wrap_alias=True,  unwrap=False)
    _c_unwrap, _p_unwrap, _m_unwrap, _ = radar_correlation(best_offset, wrap_alias=False, unwrap=True)
    _rmse_raw    = np.sqrt(np.mean((_m_raw    - _p_raw)**2))
    _rmse_alias  = np.sqrt(np.mean((_m_alias  - _p_alias)**2))
    _rmse_unwrap = np.sqrt(np.mean((_m_unwrap - _p_unwrap)**2))
    print(f"    raw:      corr={_c_raw:.4f}  RMSE={_rmse_raw:.4f} m/s")
    print(f"    aliased:  corr={_c_alias:.4f}  RMSE={_rmse_alias:.4f} m/s")
    print(f"    unwrapped:corr={_c_unwrap:.4f}  RMSE={_rmse_unwrap:.4f} m/s")
    # Count unwrapped points (k != 0) and show k distribution
    _two_vmax = 2.0 * V_MAX_UNAMBIGUOUS
    _k = np.round((_p_raw - _m_raw) / _two_vmax).astype(int)
    _k_nonzero = np.sum(_k != 0)
    if _k_nonzero > 0:
        _k_vals, _k_counts = np.unique(_k[_k != 0], return_counts=True)
        _k_str = ', '.join(f'k={v}: {c}' for v, c in zip(_k_vals, _k_counts))
        print(f"    Unwrapped {_k_nonzero}/{len(_k)} points ({100*_k_nonzero/len(_k):.0f}%): {_k_str}")
    else:
        print(f"    No aliased points detected (all k=0)")

    # ==================== ACCELEROMETER CHECK ====================
    print("\n--- Accelerometer Check ---")

    # First: validate Agiros acceleration field against velocity differentiation
    if has_accel:
        agiros_accels_raw = np.array([s.acceleration for s in agiros_states])
        
        n_check = min(1000, len(accel_num_times))
        idx_check = np.linspace(0, len(accel_num_times)-1, n_check, dtype=int)
        agiros_at_check = np.array([accel_agiros_interp(accel_num_times[i]) for i in idx_check])
        veldiff_at_check = accel_num[idx_check]
        
        for ax_i, name in enumerate(['X', 'Y', 'Z']):
            c = np.corrcoef(agiros_at_check[:, ax_i], veldiff_at_check[:, ax_i])[0, 1]
            offset = (agiros_at_check[:, ax_i] - veldiff_at_check[:, ax_i]).mean()
            print(f"  Agiros accel vs vel_diff {name}: corr={c:.4f}, offset={offset:.3f} m/s²")
        
        agiros_z_mean = agiros_accels_raw[:, 2].mean()
        veldiff_z_mean = accel_num[:, 2].mean()
        print(f"  Agiros accel Z mean: {agiros_z_mean:.3f} (vel_diff Z mean: {veldiff_z_mean:.3f})")
        print(f"  → If ~0: probably NOT coordinate acceleration. If close to vel_diff: good.")
        print()

    # Try multiple forward model variants x acceleration sources
    accel_times = []
    accel_meas  = []
    # Build predictions for each (source, model) combo
    combos = {}
    label_vd = 'veldiff'
    label_ag = 'agiros'

    for d in imu_data[::4]:
        t = d.timestamp
        if t < mocap_times[0] + 0.02 or t > mocap_times[-1] - 0.02:
            continue
        if t < accel_num_times[0] or t > accel_num_times[-1]:
            continue

        z_imu   = d.linear_acceleration
        a_vd    = accel_veldiff_interp(t)
        R_wb    = mocap_slerp(t).as_matrix()
        R_bw    = R_wb.T

        accel_times.append(t - t_start)
        accel_meas.append(z_imu)

        # Velocity-diff source (includes untransformed/world-frame variants for debugging)
        key = f'{label_vd}: world a (raw)'
        combos.setdefault(key, []).append(a_vd)
        key = f'{label_vd}: world (a-g)'
        combos.setdefault(key, []).append(a_vd - g_world)
        key = f'{label_vd}: body R_bw*a'
        combos.setdefault(key, []).append(R_bw @ a_vd)
        key = f'{label_vd}: body R_bw*(a-g)'
        combos.setdefault(key, []).append(R_bw @ (a_vd - g_world))
        key = f'{label_vd}: body R_bw*a+[0,0,g] (legacy)'
        combos.setdefault(key, []).append(R_bw @ a_vd + np.array([0, 0, 9.81]))

        # Agiros source (if available)
        if accel_agiros_interp is not None:
            a_ag = accel_agiros_interp(t)
            key = f'{label_ag}: world a (raw)'
            combos.setdefault(key, []).append(a_ag)
            key = f'{label_ag}: world (a-g)'
            combos.setdefault(key, []).append(a_ag - g_world)
            key = f'{label_ag}: body R_bw*a'
            combos.setdefault(key, []).append(R_bw @ a_ag)
            key = f'{label_ag}: body R_bw*(a-g)'
            combos.setdefault(key, []).append(R_bw @ (a_ag - g_world))
            key = f'{label_ag}: body R_bw*a+[0,0,g] (legacy)'
            combos.setdefault(key, []).append(R_bw @ a_ag + np.array([0, 0, 9.81]))

    accel_times = np.array(accel_times)
    accel_meas  = np.array(accel_meas)
    for k in combos:
        combos[k] = np.array(combos[k])

    # Low-pass filter both IMU accel and model predictions for cleaner comparison.
    if len(accel_times) > 12:
        dt_acc = float(np.median(np.diff(accel_times)))
        accel_cutoff_hz = 10.0
        accel_filter_order = 4
        accel_meas = zero_phase_lowpass(
            accel_meas,
            dt_acc,
            cutoff_hz=accel_cutoff_hz,
            order=accel_filter_order,
        )
        for k in combos:
            combos[k] = zero_phase_lowpass(
                combos[k],
                dt_acc,
                cutoff_hz=accel_cutoff_hz,
                order=accel_filter_order,
            )
        print(
            f"  Applied zero-phase low-pass filter to accel signals "
            f"(Butterworth order={accel_filter_order}, cutoff={accel_cutoff_hz:.1f} Hz, dt≈{dt_acc*1000:.1f} ms)"
        )

    print(f"  Samples: {len(accel_times)}")
    if not combos:
        print("ERROR: No acceleration models could be evaluated!")
        return

    print(f"\n  Forward model variant comparison (source: model):")
    best_model = None
    best_total_corr = -999
    for model_name, pred in combos.items():
        corrs_ax = []
        for ax_i in range(3):
            c = np.corrcoef(accel_meas[:, ax_i], pred[:, ax_i])[0, 1]
            corrs_ax.append(c)
        total = sum(corrs_ax)
        marker = " ← BEST" if total > best_total_corr else ""
        if total > best_total_corr:
            best_total_corr = total
            best_model = model_name
        rmse = np.sqrt(np.mean((accel_meas - pred)**2))
        print(f"    {model_name:38s}: X={corrs_ax[0]:+.4f}  Y={corrs_ax[1]:+.4f}  Z={corrs_ax[2]:+.4f}  "
              f"Σ={total:+.4f}  RMSE={rmse:.2f}{marker}")

    # Use the best variant for remaining analysis
    accel_pred = combos[best_model]
    accel_residual = accel_meas - accel_pred
    print(f"\n  Using model: {best_model}")
    for ax_i, name in enumerate(['X', 'Y', 'Z']):
        print(f"  {name}: meas_mean={accel_meas[:, ax_i].mean():.3f}  "
              f"pred_mean={accel_pred[:, ax_i].mean():.3f}  "
              f"res_mean={accel_residual[:, ax_i].mean():.3f}  "
              f"res_std={accel_residual[:, ax_i].std():.3f}")

    accel_times    = np.array(accel_times)
    accel_meas     = np.array(accel_meas)
    accel_pred     = np.array(accel_pred)
    accel_residual = accel_meas - accel_pred

    print(f"  Samples: {len(accel_times)}")
    for ax_i, name in enumerate(['X', 'Y', 'Z']):
        print(f"  {name}: meas_mean={accel_meas[:, ax_i].mean():.3f}  "
              f"pred_mean={accel_pred[:, ax_i].mean():.3f}  "
              f"res_mean={accel_residual[:, ax_i].mean():.3f}  "
              f"res_std={accel_residual[:, ax_i].std():.3f}")

    # ==================== GYROSCOPE CHECK (body vs world frame test) ===========
    print("\n--- Gyroscope Check ---")

    gyro_times = []
    gyro_meas  = []
    gyro_pred_body  = []  # omega directly (assumed body frame)
    gyro_pred_world = []  # R_bw @ omega  (would be correct if omega is in world frame)

    for d in imu_data[::4]:
        t = d.timestamp
        if t < mocap_times[0] + 0.01 or t > mocap_times[-1] - 0.01:
            continue

        omega_mocap = omega_interp(t)
        R_wb = mocap_slerp(t).as_matrix()

        gyro_times.append(t - t_start)
        gyro_meas.append(d.angular_velocity)
        gyro_pred_body.append(omega_mocap)                # if omega is already body frame
        gyro_pred_world.append(R_wb.T @ omega_mocap)      # if omega is world frame → transform

    gyro_times      = np.array(gyro_times)
    gyro_meas       = np.array(gyro_meas)
    gyro_pred_body  = np.array(gyro_pred_body)
    gyro_pred_world = np.array(gyro_pred_world)

    # Determine which frame fits better
    res_body  = gyro_meas - gyro_pred_body
    res_world = gyro_meas - gyro_pred_world
    rmse_body  = np.sqrt(np.mean(res_body**2))
    rmse_world = np.sqrt(np.mean(res_world**2))
    print(f"  Samples: {len(gyro_times)}")
    print(f"  RMSE (omega = body frame):  {rmse_body:.4f} rad/s")
    print(f"  RMSE (omega = world frame): {rmse_world:.4f} rad/s")
    if rmse_body < rmse_world:
        print("  → MoCap omega is in BODY frame (as expected)")
        gyro_pred = gyro_pred_body
        gyro_residual = res_body
    else:
        print("  → MoCap omega is in WORLD frame — transforming to body frame")
        gyro_pred = gyro_pred_world
        gyro_residual = res_world

    for ax_i, name in enumerate(['X', 'Y', 'Z']):
        print(f"  {name}: meas_mean={gyro_meas[:, ax_i].mean():.3f}  "
              f"pred_mean={gyro_pred[:, ax_i].mean():.3f}  "
              f"res_mean={gyro_residual[:, ax_i].mean():.3f}  "
              f"res_std={gyro_residual[:, ax_i].std():.3f}")

    # ==================== GYRO CROSS-CORRELATION TIME SYNC ====================
    print("\n--- Gyro Cross-Correlation Time Sync ---")
    # Resample both signals onto a regular grid at ~200 Hz
    dt_sync = 0.005
    t_grid = np.arange(gyro_times[0], gyro_times[-1], dt_sync)
    gyro_meas_reg = np.zeros((len(t_grid), 3))
    gyro_pred_reg = np.zeros((len(t_grid), 3))
    for ax_i in range(3):
        gyro_meas_reg[:, ax_i] = np.interp(t_grid, gyro_times, gyro_meas[:, ax_i])
        gyro_pred_reg[:, ax_i] = np.interp(t_grid, gyro_times, gyro_pred[:, ax_i])

    # Cross-correlate each axis, pick best shift
    from scipy.signal import correlate
    max_shift_samples = int(0.5 / dt_sync)  # search ±500ms
    best_shifts = []
    for ax_i, name in enumerate(['X', 'Y', 'Z']):
        sig_m = gyro_meas_reg[:, ax_i] - gyro_meas_reg[:, ax_i].mean()
        sig_p = gyro_pred_reg[:, ax_i] - gyro_pred_reg[:, ax_i].mean()
        cc = correlate(sig_m, sig_p, mode='full')
        mid = len(sig_p) - 1
        cc_window = cc[mid - max_shift_samples:mid + max_shift_samples + 1]
        lags_window = np.arange(-max_shift_samples, max_shift_samples + 1) * dt_sync
        best_lag_idx = np.argmax(cc_window)
        best_lag = lags_window[best_lag_idx]
        best_shifts.append(best_lag)
        print(f"  {name}: best lag = {best_lag*1000:+.1f} ms")

    imu_mocap_offset = np.median(best_shifts)
    print(f"  → Median IMU-MoCap offset: {imu_mocap_offset*1000:+.1f} ms")
    print(f"    (positive = IMU timestamps are AHEAD of MoCap)")

    # Re-evaluate gyro with time correction
    gyro_pred_shifted = np.zeros_like(gyro_pred)
    for ax_i in range(3):
        gyro_pred_shifted[:, ax_i] = np.interp(
            gyro_times, gyro_times + imu_mocap_offset, gyro_pred[:, ax_i])
    res_shifted = gyro_meas - gyro_pred_shifted
    rmse_shifted = np.sqrt(np.mean(res_shifted**2))
    print(f"  Gyro RMSE after shift: {rmse_shifted:.4f} rad/s (was {rmse_body:.4f})")

    # Re-evaluate accel with time correction
    accel_pred_shifted = np.zeros_like(accel_pred)
    for ax_i in range(3):
        accel_pred_shifted[:, ax_i] = np.interp(
            accel_times, accel_times + imu_mocap_offset, accel_pred[:, ax_i])
    accel_res_shifted = accel_meas - accel_pred_shifted
    print(f"\n  Accel after time correction:")
    for ax_i, name in enumerate(['X', 'Y', 'Z']):
        corr_shifted = np.corrcoef(accel_meas[:, ax_i], accel_pred_shifted[:, ax_i])[0, 1]
        corr_orig = np.corrcoef(accel_meas[:, ax_i], accel_pred[:, ax_i])[0, 1]
        print(f"    {name}: corr {corr_orig:.4f} → {corr_shifted:.4f}  "
              f"res_std {accel_residual[:, ax_i].std():.3f} → {accel_res_shifted[:, ax_i].std():.3f}")

    # Re-evaluate radar with combined time correction
    print(f"\n  Radar after time correction (radar offset = {best_offset*1000:.0f}ms + IMU corr = {imu_mocap_offset*1000:.0f}ms):")
    corr_radar_corrected, _, _, res_radar_corrected = radar_correlation(
        best_offset + imu_mocap_offset, wrap_alias=use_alias, unwrap=use_unwrap)
    print(f"    Correlation: {corr_radar_corrected:.4f} (was {corr:.4f})")
    print(f"    RMSE: {np.sqrt(np.mean(res_radar_corrected**2)):.4f} m/s")
    print("\n--- Generating Plots ---")

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle(f'Physics Unit Test v2: {bag_key}', fontsize=14, fontweight='bold')

    # --- Plot 1: Radar Doppler Scatter ---
    ax = axes[0, 0]
    ax.scatter(meas_dopplers, pred_dopplers, alpha=0.15, s=3, c='steelblue')
    lims = [min(meas_dopplers.min(), pred_dopplers.min()) - 0.5,
            max(meas_dopplers.max(), pred_dopplers.max()) + 0.5]
    ax.plot(lims, lims, 'r--', linewidth=2, label='y=x (ideal)')
    ax.plot(lims, [-x for x in lims], 'g--', linewidth=1, alpha=0.5, label='y=-x (sign error)')
    ax.set_xlabel('Measured Doppler (m/s)')
    ax.set_ylabel('Predicted Doppler (MoCap) (m/s)')
    ax.set_title(f'Radar Sign Check (corr={corr:.3f}, offset={best_offset*1000:.0f}ms)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(lims); ax.set_ylim(lims)

    # --- Plot 2: Radar Residual over Time ---
    ax = axes[0, 1]
    ax.scatter(radar_times_rel, residuals_radar, alpha=0.15, s=3, c='steelblue')
    ax.axhline(0, color='r', linestyle='--', linewidth=1)
    ax.axhline(residuals_radar.mean(), color='orange', linestyle='-', linewidth=2,
               label=f'Mean: {residuals_radar.mean():.3f} m/s')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Residual (meas - pred) (m/s)')
    ax.set_title('Radar Residual vs Time'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Plot 3: Time-Offset Sweep ---
    ax = axes[0, 2]
    ax.plot(offsets * 1000, corrs_raw,    'b-',  linewidth=2, label='Raw')
    ax.plot(offsets * 1000, corrs_alias,  'r-',  linewidth=2, label='Alias-wrapped')
    ax.plot(offsets * 1000, corrs_unwrap, 'g--', linewidth=2, label='Unwrapped meas')
    ax.axvline(best_offset * 1000, color='k', linestyle='--', label=f'Best: {best_offset*1000:.1f}ms')
    ax.set_xlabel('Time Offset (ms)'); ax.set_ylabel('Correlation')
    ax.set_title('Radar Correlation vs Time Offset'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Plot 4: Accel Z (Gravity Check) ---
    ax = axes[1, 0]
    ax.plot(accel_times, accel_meas[:, 2], 'b-', alpha=0.5, linewidth=0.5, label='IMU Z (measured)')
    ax.plot(accel_times, accel_pred[:, 2], 'r-', alpha=0.5, linewidth=0.5, label='Predicted Z (MoCap)')
    ax.axhline(9.81, color='gray', linestyle=':', alpha=0.5, label='g=9.81')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Accel Z (m/s²)')
    ax.set_title('Accel Z: Gravity Check'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Plot 5: Accel X ---
    ax = axes[1, 1]
    ax.plot(accel_times, accel_meas[:, 0], 'b-', alpha=0.5, linewidth=0.5, label='IMU X (measured)')
    ax.plot(accel_times, accel_pred[:, 0], 'r-', alpha=0.5, linewidth=0.5, label='Predicted X (MoCap)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Accel X (m/s²)')
    ax.set_title('Accel X: Lateral Check'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Plot 6: Accel Y ---
    ax = axes[1, 2]
    ax.plot(accel_times, accel_meas[:, 1], 'b-', alpha=0.5, linewidth=0.5, label='IMU Y (measured)')
    ax.plot(accel_times, accel_pred[:, 1], 'r-', alpha=0.5, linewidth=0.5, label='Predicted Y (MoCap)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Accel Y (m/s²)')
    ax.set_title('Accel Y: Lateral Check'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Plot 7: Gyroscope X ---
    ax = axes[2, 0]
    ax.plot(gyro_times, gyro_meas[:, 0], 'b-', alpha=0.5, linewidth=0.5, label='IMU X')
    ax.plot(gyro_times, gyro_pred[:, 0], 'r--', alpha=0.8, linewidth=1, label='MoCap X')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('ω_x (rad/s)')
    ax.set_title('Gyro X'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Plot 8: Gyroscope Y ---
    ax = axes[2, 1]
    ax.plot(gyro_times, gyro_meas[:, 1], 'b-', alpha=0.5, linewidth=0.5, label='IMU Y')
    ax.plot(gyro_times, gyro_pred[:, 1], 'r--', alpha=0.8, linewidth=1, label='MoCap Y')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('ω_y (rad/s)')
    ax.set_title('Gyro Y'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Plot 9: Gyroscope Z ---
    ax = axes[2, 2]
    ax.plot(gyro_times, gyro_meas[:, 2], 'b-', alpha=0.5, linewidth=0.5, label='IMU Z')
    ax.plot(gyro_times, gyro_pred[:, 2], 'r--', alpha=0.8, linewidth=1, label='MoCap Z')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('ω_z (rad/s)')
    ax.set_title('Gyro Z'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outname = f'physics_check_{bag_key}.png'
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    print(f"Saved: {outname}")

    # ==================== EXTRA PLOT: ALL ACCEL MODELS ====================
    combo_names = list(combos.keys())
    n_models = len(combo_names)
    fig2, axes2 = plt.subplots(
        3,
        n_models,
        figsize=(max(18, 4.0 * n_models), 10),
        sharex=True,
        squeeze=False,
    )
    fig2.suptitle(f'Accel model comparison: {bag_key}', fontsize=13, fontweight='bold')

    axis_names = ['X', 'Y', 'Z']
    for col, model_name in enumerate(combo_names):
        pred = combos[model_name]
        for row in range(3):
            ax2 = axes2[row, col]
            ax2.plot(accel_times, accel_meas[:, row], 'k-', alpha=0.45, linewidth=0.6, label='IMU')
            ax2.plot(accel_times, pred[:, row], 'tab:blue', alpha=0.8, linewidth=0.7, label='Model')
            if row == 0:
                ax2.set_title(model_name, fontsize=8)
            if col == 0:
                ax2.set_ylabel(f'Accel {axis_names[row]} (m/s²)')
            if row == 2:
                ax2.set_xlabel('Time (s)')

            corr_ax = np.corrcoef(accel_meas[:, row], pred[:, row])[0, 1]
            ax2.text(
                0.02,
                0.92,
                f'corr={corr_ax:+.2f}',
                transform=ax2.transAxes,
                fontsize=7,
                va='top',
                ha='left',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6, linewidth=0.0),
            )
            ax2.grid(True, alpha=0.25)

            if row == 0 and col == 0:
                ax2.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outname_models = f'physics_check_{bag_key}_accel_models.png'
    plt.savefig(outname_models, dpi=160, bbox_inches='tight')
    print(f"Saved: {outname_models}")

    # ==================== VERDICT ====================
    print(f"\n{'VERDICT':#^80}")
    if abs(corr) > 0.8 and corr > 0:
        print(f"[OK] Radar: Positive correlation ({corr:.3f}) — signs correct (offset={best_offset*1000:.0f}ms)")
    elif abs(corr) > 0.8 and corr < 0:
        print(f"[!!] Radar: NEGATIVE correlation ({corr:.3f}) — SIGN ERROR!")
    elif abs(corr_neg) > 0.8:
        print(f"[!!] Radar: Negated correlation ({corr_neg:.3f}) — measurement sign is flipped!")
    else:
        print(f"[??] Radar: Low correlation ({corr:.3f}) — extrinsics or time sync issue")

    accel_z_offset = accel_residual[:, 2].mean()
    accel_z_std = accel_residual[:, 2].std()
    if abs(accel_z_offset) < 2.0 and accel_z_std < 5.0:
        print(f"[OK] Accel Z: Offset {accel_z_offset:.3f}±{accel_z_std:.3f} m/s² — gravity correct")
    elif abs(accel_z_offset) > 15:
        print(f"[!!] Accel Z: Offset {accel_z_offset:.3f} m/s² — GRAVITY SIGN ERROR (~2g)")
    else:
        print(f"[??] Accel Z: Offset {accel_z_offset:.3f}±{accel_z_std:.3f} m/s² — suspicious")

    for ax_i, name in enumerate(['X', 'Y', 'Z']):
        accel_corr = np.corrcoef(accel_meas[:, ax_i], accel_pred[:, ax_i])[0, 1]
        print(f"     Accel {name} correlation: {accel_corr:.4f}")

    gyro_offset_norm = np.linalg.norm(gyro_residual.mean(axis=0))
    print(f"     Gyro frame: {'body' if rmse_body < rmse_world else 'WORLD (transformed)'}")
    if gyro_offset_norm < 0.1:
        print(f"[OK] Gyro: Mean offset {gyro_offset_norm:.4f} rad/s — reasonable bias")
    else:
        print(f"[??] Gyro: Mean offset {gyro_offset_norm:.4f} rad/s — large bias or frame mismatch")


if __name__ == "__main__":
    run_physics_diagnostics()