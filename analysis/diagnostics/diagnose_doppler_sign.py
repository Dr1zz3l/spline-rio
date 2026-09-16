#!/usr/bin/env python3
"""
Doppler Sign Convention Diagnostic (Tests 2 & 5)

Compares the two independent doppler forward models at MoCap ground truth:
  1. predict_doppler_velocity()      — used by validate_physics.py
  2. radar_residual_with_jacobians() — used by validate_nonlinear_solver.py (SymForce)

For each radar point, prints:
  v_meas | dot(u,v) | predict_doppler | symforce_vpred | which matches v_meas?

Also tests with yaw_flip ON and OFF.

Usage:
    python diagnostics/diagnose_doppler_sign.py [bag_key]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (
    rotation_matrix_from_euler,
    predict_doppler_velocity,
    quat_to_rotation_matrix,
)
from codegen.generated_jacobians import (
    radar_residual_with_jacobians,
    Rot3,
)
from config_loader import load_config


def run_sign_diagnostic(bag_key="circle", flip=None):
    cfg = load_config()
    bags_cfg = cfg['bags']
    ext = cfg['extrinsics']

    bags = bags_cfg.get('bags', {})
    flipped_bags = set(bags_cfg.get('flipped', []))
    timing = bags_cfg.get('timing', {})

    BAG_PATH = bags.get(bag_key, bag_key)
    START_OFFSET, DURATION = timing.get(bag_key, [5.0, 120.0])

    ROTATION_EULER = np.array(ext['rotation_euler_deg'], dtype=float)
    _t_base = np.array(ext['translation_body_m'], dtype=float)

    FLIP_BODY_FRAME = bag_key in flipped_bags
    if flip is not None:
        FLIP_BODY_FRAME = flip

    R_base = rotation_matrix_from_euler(
        np.radians(ROTATION_EULER[0]),
        np.radians(ROTATION_EULER[1]),
        np.radians(ROTATION_EULER[2]),
    )
    R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)

    if FLIP_BODY_FRAME:
        sensor_rotation = R_yaw_flip @ R_base
        sensor_translation = R_yaw_flip @ _t_base
    else:
        sensor_rotation = R_base
        sensor_translation = _t_base.copy()

    print(f"\n{'=' * 80}")
    print(f"Doppler Sign Diagnostic: bag={bag_key}, flip={FLIP_BODY_FRAME}")
    print(f"{'=' * 80}")

    # Load data
    bag_data = load_bag_topics(BAG_PATH, verbose=False)
    t_start = bag_data.start_time + START_OFFSET
    t_end = t_start + DURATION

    agiros_states = [s for s in bag_data.agiros_state if t_start <= s.timestamp <= t_end]
    radar_frames = [f for f in bag_data.radar_velocity if t_start <= f.timestamp <= t_end]

    if not agiros_states or not radar_frames:
        print("ERROR: Insufficient data!")
        return

    # Build interpolators
    mocap_times = np.array([s.timestamp for s in agiros_states])
    mocap_velocities = np.array([s.velocity for s in agiros_states])
    mocap_omegas = np.array([s.angular_velocity for s in agiros_states])
    mocap_quats = np.array([s.orientation for s in agiros_states])

    vel_interp = interp1d(mocap_times, mocap_velocities, axis=0, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
    omega_interp = interp1d(mocap_times, mocap_omegas, axis=0, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    scipy_rots = Rotation.from_quat(mocap_quats)
    mocap_slerp = Slerp(mocap_times, scipy_rots)

    # Collect point-wise comparisons
    MIN_RANGE = 0.2
    results = []

    # Use subset of frames for table output
    frame_indices = np.linspace(0, len(radar_frames) - 1, min(20, len(radar_frames)), dtype=int)

    for fi in range(len(radar_frames)):
        frame = radar_frames[fi]
        t = frame.timestamp
        if t < mocap_times[0] or t > mocap_times[-1]:
            continue

        v_world = vel_interp(t)
        omega_body = omega_interp(t)
        R_wb = mocap_slerp(np.clip(t, mocap_times[0], mocap_times[-1])).as_matrix()

        for i in range(frame.num_points()):
            p_s = frame.positions[i]
            r = frame.ranges[i] if frame.ranges is not None else np.linalg.norm(p_s)
            if r < MIN_RANGE:
                continue

            v_meas = frame.velocities[i]
            u_sensor = p_s / np.linalg.norm(p_s)

            # --- Path 1: predict_doppler_velocity (validate_physics.py) ---
            v_predict = predict_doppler_velocity(
                v_world, omega_body, R_wb,
                p_s.reshape(1, 3), sensor_translation, sensor_rotation
            )[0]

            # --- Manual: raw dot(u_body, v_ant) without negation ---
            R_bw = R_wb.T
            v_body = R_bw @ v_world
            v_lever = np.cross(omega_body, sensor_translation)
            v_ant = v_body + v_lever
            u_body = sensor_rotation @ u_sensor
            dot_uv = np.dot(u_body, v_ant)

            # --- Path 2: SymForce radar_residual_with_jacobians ---
            R_nom_quat = Rot3.from_rotation_matrix(R_wb)
            R_bs_quat = Rot3.from_rotation_matrix(sensor_rotation)
            res, _, _, _, _ = radar_residual_with_jacobians(
                v_world, R_nom_quat,
                np.zeros(3),  # delta = 0 (at ground truth)
                omega_body,
                u_sensor, sensor_translation, R_bs_quat,
                v_meas, 1e-10
            )
            # res = v_meas - v_pred_symforce
            v_pred_symforce = v_meas - res[0]

            results.append({
                'v_meas': v_meas,
                'dot_uv': dot_uv,
                'predict_doppler': v_predict,  # = -dot_uv
                'symforce_vpred': v_pred_symforce,  # = dot_uv (from SymForce)
                'residual_symforce': res[0],
            })

    results = np.array([(r['v_meas'], r['dot_uv'], r['predict_doppler'],
                         r['symforce_vpred'], r['residual_symforce']) for r in results])

    v_meas_all = results[:, 0]
    dot_uv_all = results[:, 1]
    predict_all = results[:, 2]
    symforce_all = results[:, 3]
    res_sf_all = results[:, 4]

    # --- Print summary table (subset) ---
    print(f"\nTotal radar points evaluated: {len(results)}")
    print(f"\n{'Point-wise comparison (subset of ~50 points)':=^80}")
    print(f"{'v_meas':>8}  {'dot(u,v)':>8}  {'predict':>8}  {'sf_vpred':>8}  {'sf_resid':>8}")
    print("-" * 50)
    indices = np.linspace(0, len(results)-1, min(50, len(results)), dtype=int)
    for idx in indices:
        print(f"{v_meas_all[idx]:8.3f}  {dot_uv_all[idx]:8.3f}  {predict_all[idx]:8.3f}  "
              f"{symforce_all[idx]:8.3f}  {res_sf_all[idx]:8.3f}")

    # --- Correlations ---
    print(f"\n{'Correlation Analysis':=^80}")
    corr_meas_dot = np.corrcoef(v_meas_all, dot_uv_all)[0, 1]
    corr_meas_neg_dot = np.corrcoef(v_meas_all, -dot_uv_all)[0, 1]
    corr_meas_predict = np.corrcoef(v_meas_all, predict_all)[0, 1]
    corr_meas_symforce = np.corrcoef(v_meas_all, symforce_all)[0, 1]

    print(f"  corr(v_meas, +dot(u,v))        = {corr_meas_dot:+.4f}")
    print(f"  corr(v_meas, -dot(u,v))        = {corr_meas_neg_dot:+.4f}")
    print(f"  corr(v_meas, predict_doppler)  = {corr_meas_predict:+.4f}  [= -dot(u,v)]")
    print(f"  corr(v_meas, symforce_vpred)   = {corr_meas_symforce:+.4f}  [= +dot(u,v)]")

    # --- Residual statistics ---
    print(f"\n{'Residual Statistics':=^80}")
    # If sign is correct: residual = v_meas - v_pred should have mean ~0
    res_positive = v_meas_all - dot_uv_all       # SymForce convention: v_meas - dot(u,v)
    res_negative = v_meas_all - (-dot_uv_all)    # Negated convention: v_meas - (-dot(u,v))

    print(f"  Residual v_meas - dot(u,v)   [SymForce]:  mean={res_positive.mean():+.4f}  std={res_positive.std():.4f}  |mean|/std={abs(res_positive.mean())/res_positive.std():.2f}")
    print(f"  Residual v_meas - (-dot(u,v)) [negated]:  mean={res_negative.mean():+.4f}  std={res_negative.std():.4f}  |mean|/std={abs(res_negative.mean())/res_negative.std():.2f}")
    print(f"  SymForce actual residuals:                 mean={res_sf_all.mean():+.4f}  std={res_sf_all.std():.4f}")

    # --- RMSE ---
    rmse_positive = np.sqrt(np.mean(res_positive**2))
    rmse_negative = np.sqrt(np.mean(res_negative**2))
    print(f"\n  RMSE (v_meas - dot(u,v))   = {rmse_positive:.4f} m/s")
    print(f"  RMSE (v_meas - (-dot(u,v))) = {rmse_negative:.4f} m/s")

    # --- Huber suppression ---
    HUBER_DELTA = 1.0
    frac_huber_sf = np.mean(np.abs(res_sf_all) > HUBER_DELTA)
    frac_huber_pos = np.mean(np.abs(res_positive) > HUBER_DELTA)
    frac_huber_neg = np.mean(np.abs(res_negative) > HUBER_DELTA)
    print(f"\n  Fraction |residual| > {HUBER_DELTA} m/s (Huber-downweighted):")
    print(f"    SymForce (v_meas - dot):     {frac_huber_pos:.1%}")
    print(f"    Negated  (v_meas + dot):     {frac_huber_neg:.1%}")

    # --- Code path consistency check (Test 5) ---
    print(f"\n{'Code Path Consistency (Test 5)':=^80}")
    diff = predict_all - (-symforce_all)  # predict = -dot, symforce = +dot, so predict = -symforce
    print(f"  predict_doppler vs -symforce_vpred:")
    print(f"    max |diff| = {np.abs(diff).max():.2e}")
    print(f"    mean |diff| = {np.abs(diff).mean():.2e}")
    consistent = np.abs(diff).max() < 1e-6
    print(f"    Consistent: {consistent}")
    if consistent:
        print(f"    -> predict_doppler = -symforce_vpred (sign conventions are exactly opposite)")

    # --- Verdict ---
    print(f"\n{'VERDICT':=^80}")
    # Use RMSE as the definitive test (correlation sign is ambiguous with flip)
    if rmse_negative < rmse_positive:
        print(f"  Best convention: v_pred = -dot(u,v) (RMSE {rmse_negative:.4f} < {rmse_positive:.4f})")
        print(f"  -> TI radar: positive Doppler = receding target")
        print(f"  -> predict_doppler (-dot) is CORRECT for this flip setting")
        print(f"  -> SymForce residual (v_meas - dot) needs sign fix")
    else:
        print(f"  Best convention: v_pred = +dot(u,v) (RMSE {rmse_positive:.4f} < {rmse_negative:.4f})")
        print(f"  -> SymForce residual (v_meas - dot) is CORRECT for this flip setting")
        print(f"  -> predict_doppler (-dot) has wrong sign for this flip setting")

    return results


if __name__ == "__main__":
    positional_args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    bag_key = positional_args[0] if positional_args else "circle"

    # Run with default flip setting
    print("\n" + "#" * 80)
    print("# TEST WITH DEFAULT FLIP SETTING")
    print("#" * 80)
    run_sign_diagnostic(bag_key, flip=None)

    # Run with flip forced ON
    print("\n" + "#" * 80)
    print("# TEST WITH FLIP = ON")
    print("#" * 80)
    run_sign_diagnostic(bag_key, flip=True)

    # Run with flip forced OFF
    print("\n" + "#" * 80)
    print("# TEST WITH FLIP = OFF")
    print("#" * 80)
    run_sign_diagnostic(bag_key, flip=False)
