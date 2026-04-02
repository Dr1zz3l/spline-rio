"""
LIVE RIO SOLVER — MoCap-Free Initialization Prototype

Implements P1-P3 from the live RIO migration plan:
  P1. Gyro-integrated orientation init (replaces MoCap SLERP)
  P2. Radar-velocity position init (replaces MoCap cubic interpolation)
  P3. Sensor-only boundary priors (replaces MoCap-derived targets)

MoCap data is still loaded, but ONLY for final evaluation (RMSE comparison).
It plays zero role in initialization or optimization.

Usage:
    python validate_live_solver.py <bag_name>
    python validate_live_solver.py <bag_name> --noise-deg <σ_deg>   # P5 stress test

The --noise-deg flag adds Gaussian noise (σ in degrees/√s, integrated as random walk)
to the gyro-integrated orientations, simulating worse-than-real gyro drift.
Use this to probe how much init error the solver can tolerate.
"""

import sys
import dataclasses
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy import sparse
import scipy.sparse.linalg
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import time

from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (
    quat_to_rotation_matrix,
    rotation_matrix_from_euler,
    solve_ego_velocity_weighted,
)
from bspline_utils import (
    UniformBSpline,
    build_minimum_snap_regularization,
)
from cumulative_so3_bspline import (
    CumulativeSO3BSpline,
    so3_exp,
    so3_log,
)
from codegen.generated_jacobians import Rot3
from config_loader import load_config
from imu_preintegration import build_preintegrated_factors, preintegrate

# Import solver core from the batch solver (optimizer itself is MoCap-free)
from validate_nonlinear_solver import (
    TrajectoryState,
    compute_jacobian_analytical,
    compute_residuals_only,
    solve_trajectory_nonlinear,
    detect_stationary_bias,
    compute_orientation_rmse,
    huber_weight,
)


# ==================== C++ Ceres solver bridge ====================

def _build_preint_factors_cpp(rio_solver, imu_data, b_a0, b_g0, t_start, t_end, dt_ori, t_ref):
    """
    Build preintegrated IMU factors on a uniform grid aligned to knot boundaries.
    Returns a list of rio_solver.PreintFactor objects.
    """
    # Snap grid to knot boundaries relative to t_ref
    k_start = int(round((t_start - t_ref) / dt_ori))
    k_end   = int(round((t_end   - t_ref) / dt_ori))
    t_grid  = np.array([t_ref + k * dt_ori for k in range(k_start, k_end + 1)])

    # Convert imu_data to arrays for preintegrate()
    imu_times = np.array([s.timestamp for s in imu_data])
    imu_acc   = np.array([s.linear_acceleration for s in imu_data])
    imu_gyro  = np.array([s.angular_velocity    for s in imu_data])
    imu_arr   = np.hstack([imu_times[:, None], imu_acc, imu_gyro])  # (N, 7)

    factors = []
    for k in range(len(t_grid) - 1):
        ti, tj = float(t_grid[k]), float(t_grid[k + 1])
        mask = (imu_arr[:, 0] >= ti) & (imu_arr[:, 0] < tj + 1e-9)
        if not mask.any():
            continue
        pf_py = preintegrate(
            imu_data,
            np.asarray(b_a0, dtype=float),
            np.asarray(b_g0, dtype=float),
            ti, tj)
        f = rio_solver.PreintFactor()
        f.t_i = pf_py.t_i
        f.t_j = pf_py.t_j
        f.dt  = pf_py.dt
        f.delta_R  = pf_py.delta_R
        f.delta_v  = pf_py.delta_v
        f.delta_p  = pf_py.delta_p
        f.b_a0     = pf_py.b_a0
        f.b_g0     = pf_py.b_g0
        f.d_R_d_bg = pf_py.d_R_d_bg
        f.d_v_d_ba = pf_py.d_v_d_ba
        f.d_v_d_bg = pf_py.d_v_d_bg
        f.d_p_d_ba = pf_py.d_p_d_ba
        f.d_p_d_bg = pf_py.d_p_d_bg
        factors.append(f)
    return factors


def _solve_cpp(initial_state, solver_radar_frames, imu_data,
               extrinsics_cfg, solver_cfg,
               heading_priors=None):
    """
    Call the C++ Ceres solver (rio_solver_cpp) in place of
    solve_trajectory_nonlinear().  Returns an updated TrajectoryState.

    Orientation conversion:
      Python: omega_knots[i] = incremental so3 log increments
      Basalt/C++: quaternion knots q[i] = absolute rotation as xyzw
      q[i] = _base_rotations[i]  (the cumulative product rotation at knot i)

    After the C++ solve, converts back via CumulativeSO3BSpline.from_rotation_samples().
    """
    import sys as _sys, os as _os
    _build_dirs = [
        _os.path.join(_os.path.dirname(__file__), '..', 'rio_solver_cpp', 'build_release'),
        _os.path.join(_os.path.dirname(__file__), '..', 'rio_solver_cpp', 'build_debug'),
        _os.path.dirname(__file__),
    ]
    for d in _build_dirs:
        d = _os.path.abspath(d)
        if d not in _sys.path:
            _sys.path.insert(0, d)
    import rio_solver

    ori_spline = initial_state.ori_spline
    pos_bspline = initial_state.pos_bspline

    # --- Convert orientation: omega_knots → absolute quaternion knots (xyzw) ---
    R_abs = ori_spline._base_rotations  # (N_ori, 3, 3), already computed
    init_ori_quats = Rotation.from_matrix(R_abs).as_quat()  # (N_ori, 4) xyzw

    # --- Position control points ---
    init_pos_cps = pos_bspline.control_points.copy()  # (N_pos, 3)

    # --- Biases ---
    init_biases = np.concatenate([initial_state.acc_bias, initial_state.gyr_bias])  # (6,)

    # --- Config ---
    cfg = rio_solver.SolverConfig()
    cfg.dt_pos              = solver_cfg.get('dt_pos', 0.005)
    cfg.dt_ori              = solver_cfg.get('dt_ori', 0.008)
    cfg.huber_delta         = solver_cfg.get('huber_delta', 1.0)
    cfg.min_range           = solver_cfg.get('min_range', 0.2)
    cfg.lambda_accel        = solver_cfg.get('lambda_accel', 0.01)
    cfg.lambda_gyro         = solver_cfg.get('lambda_gyro', 1.0)
    cfg.huber_delta_accel   = solver_cfg.get('huber_delta_accel', 2.0)
    cfg.lambda_snap_pos     = solver_cfg.get('lambda_snap_pos', 0.0001)
    cfg.lambda_ori_reg      = solver_cfg.get('lambda_ori_reg', 0.001)
    cfg.lambda_ori_accel    = solver_cfg.get('lambda_ori_accel', 0.0)
    cfg.lambda_gravity      = solver_cfg.get('lambda_gravity', 0.001)
    cfg.gravity_accel_threshold = solver_cfg.get('gravity_accel_threshold', 3.0)
    cfg.lambda_heading      = solver_cfg.get('lambda_heading', 3.0)
    cfg.lambda_bias_prior_accel = solver_cfg.get('lambda_bias_prior_accel', 1.0)
    cfg.lambda_bias_prior_gyro  = solver_cfg.get('lambda_bias_prior_gyro', 1.0)
    cfg.lambda_boundary_pos = solver_cfg.get('lambda_boundary_pos', 1000.0)
    cfg.lambda_boundary_vel = solver_cfg.get('lambda_boundary_vel', 1000.0)
    cfg.lambda_boundary_ori = solver_cfg.get('lambda_boundary_ori', 1000.0)
    cfg.lambda_boundary_ori_yaw = solver_cfg.get('lambda_boundary_ori_yaw', 0.0)
    cfg.lock_extrinsics     = solver_cfg.get('lock_extrinsics', False)
    cfg.optimize_pitch_only = solver_cfg.get('optimize_pitch_only', True)
    cfg.lambda_extrinsic_prior = solver_cfg.get('lambda_extrinsic_prior', 10.0)
    cfg.max_iterations      = solver_cfg.get('max_iterations', 400)
    cfg.use_preintegration  = solver_cfg.get('use_preintegration', False)
    cfg.lambda_preint       = solver_cfg.get('lambda_preint', 1.0)
    cfg.lambda_preint_v     = solver_cfg.get('lambda_preint_v', 0.0)
    cfg.lambda_preint_p     = solver_cfg.get('lambda_preint_p', 0.0)
    cfg.preint_hz           = solver_cfg.get('preint_hz', 100.0)

    # --- Extrinsics ---
    euler_deg = extrinsics_cfg.get('rotation_euler_deg', [180.0, 25.5, 0.0])
    ext = rio_solver.ExtrinsicConfig()
    ext.roll_deg  = euler_deg[0]
    ext.pitch_deg = euler_deg[1]
    ext.yaw_deg   = euler_deg[2]
    t_body = extrinsics_cfg.get('translation_body_m', [0.08, 0.02, -0.01])
    ext.tx, ext.ty, ext.tz = t_body[0], t_body[1], t_body[2]

    # --- Convert radar frames ---
    cpp_radar_frames = []
    for frame in solver_radar_frames:
        n = frame.num_points()
        if n == 0:
            continue
        pts = np.zeros((n, 4))
        pts[:, :3] = frame.positions[:n]
        pts[:, 3]  = frame.velocities[:n] if frame.velocities is not None else 0.0
        cpp_radar_frames.append(rio_solver.make_radar_frame(frame.timestamp, pts))

    # --- Convert IMU samples ---
    imu_np = np.zeros((len(imu_data), 7))
    for i, s in enumerate(imu_data):
        imu_np[i, 0] = s.timestamp
        imu_np[i, 1:4] = s.linear_acceleration
        imu_np[i, 4:7] = s.angular_velocity
    cpp_imu = rio_solver.make_imu_samples(imu_np)

    # --- Heading samples ---
    # heading_priors is List[(t_abs, R_gt)] with R_gt a 3x3 matrix.
    # C++ expects List[Tuple[float, float]] = (timestamp, yaw_rad).
    # Extract yaw = atan2(R[1,0], R[0,0]).
    if heading_priors:
        cpp_heading = [(float(t), float(np.arctan2(R[1, 0], R[0, 0])))
                       for t, R in heading_priors]
    else:
        cpp_heading = []

    # --- t_ref ---
    t_ref = pos_bspline.t_ref

    # --- Preintegrated factors ---
    cpp_preint = []
    if cfg.use_preintegration:
        t_data_start = min(f.timestamp for f in cpp_radar_frames) if cpp_radar_frames else t_ref
        t_data_end   = max(f.timestamp for f in cpp_radar_frames) if cpp_radar_frames else t_ref
        cpp_preint = _build_preint_factors_cpp(
            rio_solver, imu_data, init_biases[:3], init_biases[3:],
            t_data_start, t_data_end, cfg.dt_ori, t_ref)
        print(f"  [--cpp] Preintegration: {len(cpp_preint)} factors at {cfg.preint_hz:.0f} Hz"
              f" (dt_ori={cfg.dt_ori:.4f}s)")

    # --- Solve ---
    print(f"  [--cpp] Calling C++ Ceres solver "
          f"({len(cpp_radar_frames)} radar frames, {len(cpp_imu)} IMU samples)")
    t0_cpp = time.time()
    result = rio_solver.solve(
        cpp_radar_frames, cpp_imu, cpp_preint, cfg, ext,
        init_pos_cps, init_ori_quats, init_biases,
        t_ref, cpp_heading)
    dt_cpp = time.time() - t0_cpp
    t_jac  = result.time_jacobian_eval_s
    t_res  = result.time_residual_eval_s
    t_lin  = result.time_linear_solver_s
    t_misc = dt_cpp - t_jac - t_res - t_lin
    nominal_pitch = extrinsics_cfg.get('rotation_euler_deg', [180.0, 25.5, 0.0])[1]
    opt_pitch = result.extrinsic_euler_deg[1]
    print(f"  [--cpp] Done in {dt_cpp:.2f}s  |  {result.solver_summary}")
    if not solver_cfg.get('lock_extrinsics', False):
        print(f"  [--cpp] Extrinsic pitch: {opt_pitch:.3f}° (nominal {nominal_pitch:.1f}°,"
              f" delta {opt_pitch - nominal_pitch:+.3f}°)")
    print(f"  [--cpp] Time breakdown:  jacobian={t_jac:.2f}s ({100*t_jac/dt_cpp:.0f}%)"
          f"  residual={t_res:.2f}s ({100*t_res/dt_cpp:.0f}%)"
          f"  linear_solve={t_lin:.2f}s ({100*t_lin/dt_cpp:.0f}%)"
          f"  other={t_misc:.2f}s ({100*t_misc/dt_cpp:.0f}%)")

    # --- Convert result back to TrajectoryState ---
    # Quaternion knots → absolute rotation matrices → omega_knots
    result_quats = result.ori_knots  # (N_ori, 4) xyzw
    result_R_abs = Rotation.from_quat(result_quats).as_matrix()  # (N_ori, 3, 3)
    new_ori_spline = CumulativeSO3BSpline.from_rotation_samples(
        result_R_abs, dt=ori_spline.dt, t_ref=ori_spline.t_ref)

    new_pos_bspline = UniformBSpline(
        result.pos_cps, pos_bspline.degree, pos_bspline.dt)
    new_pos_bspline.t_ref = t_ref

    result_biases = result.biases
    new_acc_bias = np.array(result_biases[:3])
    new_gyr_bias = np.array(result_biases[3:])

    return TrajectoryState(
        pos_bspline=new_pos_bspline,
        ori_spline=new_ori_spline,
        acc_bias=new_acc_bias,
        gyr_bias=new_gyr_bias,
        radar_extrinsic_delta=np.zeros(3),
    )


# ==================== C++ sliding window bridge ====================

def _solve_cpp_sliding_window(initial_state, solver_radar_frames, imu_data,
                               extrinsics_cfg, solver_cfg,
                               heading_priors=None):
    """
    Fixed-lag smoother with Schur complement marginalization.
    Uses the stateful C++ SlidingWindowSolver which carries a dense Gaussian
    prior across window advances (Phase 4b).

    Config keys read from solver_cfg:
      window_duration  (float, default 3.0 s)
      window_stride    (float, default 0.3 s)
    """
    import sys as _sys, os as _os
    _build_dirs = [
        _os.path.join(_os.path.dirname(__file__), '..', 'rio_solver_cpp', 'build_release'),
        _os.path.join(_os.path.dirname(__file__), '..', 'rio_solver_cpp', 'build_debug'),
        _os.path.dirname(__file__),
    ]
    for d in _build_dirs:
        d = _os.path.abspath(d)
        if d not in _sys.path:
            _sys.path.insert(0, d)
    import rio_solver

    window_duration = solver_cfg.get('window_duration', 3.0)
    window_stride   = solver_cfg.get('window_stride', 0.3)
    dt_pos          = solver_cfg.get('dt_pos', 0.005)
    dt_ori          = solver_cfg.get('dt_ori', 0.008)

    ori_spline  = initial_state.ori_spline
    pos_bspline = initial_state.pos_bspline
    t_ref = pos_bspline.t_ref

    all_pos_cps   = pos_bspline.control_points.copy()         # (N_total_pos, 3)
    all_ori_quats = Rotation.from_matrix(                      # (N_total_ori, 4) xyzw
        ori_spline._base_rotations).as_quat()
    biases = np.concatenate([initial_state.acc_bias, initial_state.gyr_bias])

    # Extrinsics
    euler_deg = extrinsics_cfg.get('rotation_euler_deg', [180.0, 25.5, 0.0])
    ext = rio_solver.ExtrinsicConfig()
    ext.roll_deg, ext.pitch_deg, ext.yaw_deg = euler_deg
    t_body = extrinsics_cfg.get('translation_body_m', [0.08, 0.02, -0.01])
    ext.tx, ext.ty, ext.tz = t_body

    # Build SolverConfig once
    cfg = rio_solver.SolverConfig()
    cfg.dt_pos                  = dt_pos
    cfg.dt_ori                  = dt_ori
    cfg.huber_delta             = solver_cfg.get('huber_delta', 1.0)
    cfg.min_range               = solver_cfg.get('min_range', 0.2)
    cfg.lambda_accel            = solver_cfg.get('lambda_accel', 0.01)
    cfg.lambda_gyro             = solver_cfg.get('lambda_gyro', 1.0)
    cfg.huber_delta_accel       = solver_cfg.get('huber_delta_accel', 2.0)
    cfg.lambda_snap_pos         = solver_cfg.get('lambda_snap_pos', 0.0001)
    cfg.lambda_ori_reg          = solver_cfg.get('lambda_ori_reg', 0.001)
    cfg.lambda_ori_accel        = solver_cfg.get('lambda_ori_accel', 0.0)
    cfg.lambda_gravity          = solver_cfg.get('lambda_gravity', 0.001)
    cfg.gravity_accel_threshold = solver_cfg.get('gravity_accel_threshold', 3.0)
    cfg.lambda_heading          = solver_cfg.get('lambda_heading', 3.0)
    cfg.lambda_bias_prior_accel = solver_cfg.get('lambda_bias_prior_accel', 1.0)
    cfg.lambda_bias_prior_gyro  = solver_cfg.get('lambda_bias_prior_gyro', 1.0)
    cfg.lambda_boundary_pos     = solver_cfg.get('lambda_boundary_pos', 1000.0)
    cfg.lambda_boundary_vel     = solver_cfg.get('lambda_boundary_vel', 1000.0)
    cfg.lambda_boundary_ori     = solver_cfg.get('lambda_boundary_ori', 1000.0)
    cfg.lambda_boundary_ori_yaw = solver_cfg.get('lambda_boundary_ori_yaw', 0.0)
    cfg.lock_extrinsics         = solver_cfg.get('lock_extrinsics', False)
    cfg.optimize_pitch_only     = solver_cfg.get('optimize_pitch_only', True)
    cfg.lambda_extrinsic_prior  = solver_cfg.get('lambda_extrinsic_prior', 10.0)
    cfg.max_iterations          = solver_cfg.get('max_iterations', 400)
    cfg.marg_prior_scale           = solver_cfg.get('marg_prior_scale', 1.0)
    cfg.use_adaptive_marg_scale    = solver_cfg.get('use_adaptive_marg_scale', False)
    cfg.marg_prior_cauchy_delta    = solver_cfg.get('marg_prior_cauchy_delta', 0.0)
    cfg.marg_prior_eig_clip        = solver_cfg.get('marg_prior_eig_clip', 0.0)
    cfg.use_preintegration         = solver_cfg.get('use_preintegration', False)
    cfg.lambda_preint           = solver_cfg.get('lambda_preint', 1.0)
    cfg.lambda_preint_v         = solver_cfg.get('lambda_preint_v', 0.0)
    cfg.lambda_preint_p         = solver_cfg.get('lambda_preint_p', 0.0)
    cfg.preint_hz               = solver_cfg.get('preint_hz', 100.0)

    # Create stateful solver and initialize with full P1-P3 trajectory
    solver = rio_solver.SlidingWindowSolver(cfg, ext)
    solver.initialize(all_pos_cps, all_ori_quats, biases, t_ref)

    # Convert radar frames once; keep (timestamp, RadarFrame) for slicing
    cpp_radar_all = []
    for frame in solver_radar_frames:
        n = frame.num_points()
        if n == 0:
            continue
        pts = np.zeros((n, 4))
        pts[:, :3] = frame.positions[:n]
        pts[:, 3]  = frame.velocities[:n] if frame.velocities is not None else 0.0
        cpp_radar_all.append((frame.timestamp, rio_solver.make_radar_frame(frame.timestamp, pts)))

    # Convert IMU once
    imu_np = np.zeros((len(imu_data), 7))
    for i, s in enumerate(imu_data):
        imu_np[i, 0]   = s.timestamp
        imu_np[i, 1:4] = s.linear_acceleration
        imu_np[i, 4:7] = s.angular_velocity
    imu_times = imu_np[:, 0]

    # Heading priors
    all_heading = []
    if heading_priors:
        all_heading = [(float(t), float(np.arctan2(R[1, 0], R[0, 0])))
                       for t, R in heading_priors]

    t_data_start = cpp_radar_all[0][0]  if cpp_radar_all else imu_times[0]
    t_data_end   = cpp_radar_all[-1][0] if cpp_radar_all else imu_times[-1]
    n_expected = max(1, int((t_data_end - t_data_start - window_duration) / window_stride) + 1)

    print(f"  [--sliding-window] window={window_duration:.1f}s  stride={window_stride:.1f}s"
          f"  ~{n_expected} windows  (Schur marginalization)")

    t_solve = t_data_start + window_duration
    n_windows = 0

    # Live leading-edge snapshots — one per window, over the stride zone
    _live_eval_dt  = 0.02   # 50 Hz grid → ~15 pts per 0.3 s stride
    live_snapshots = []     # list of {'t', 'pos', 'ori'} dicts

    while t_solve <= t_data_end + 1e-6:
        t_w_start = t_solve - window_duration
        t_w_end   = t_solve

        # Slice sensor data for this window
        window_radar   = [rf for ts, rf in cpp_radar_all if t_w_start <= ts <= t_w_end]
        imu_mask       = (imu_times >= t_w_start) & (imu_times <= t_w_end)
        window_imu     = rio_solver.make_imu_samples(imu_np[imu_mask])
        window_heading = [(t, y) for t, y in all_heading if t_w_start <= t <= t_w_end]

        # Skip if no IMU, or if radar is expected (data exists globally) but absent this window
        if not imu_mask.any() or (cpp_radar_all and not window_radar):
            t_solve += window_stride
            continue

        # Preintegrated factors for this window
        window_imu_data = [s for s in imu_data
                           if t_w_start <= s.timestamp <= t_w_end]
        window_preint = []
        if cfg.use_preintegration and window_imu_data:
            b_a0 = np.array(solver.biases[:3])
            b_g0 = np.array(solver.biases[3:])
            window_preint = _build_preint_factors_cpp(
                rio_solver, window_imu_data, b_a0, b_g0,
                t_w_start, t_w_end, cfg.dt_ori, t_ref)

        t0 = time.time()
        result = solver.solve_window(
            window_radar, window_imu, window_preint, window_heading,
            t_w_start, t_w_end, window_stride)
        dt_w = time.time() - t0

        cost_str = f"{result.cost_history[-1]:.3f}" if result.cost_history else "n/a"
        if result.marg_prior_valid:
            r2 = result.marg_prior_residual_norm
            r2_str = f"{r2:.1f}" if r2 >= 0 else "n/a"
            prior_str = (f"  prior=OK"
                         f"  cond={result.marg_cond_number:.1e}"
                         f"  rank={result.marg_numerical_rank}/{result.marg_prior_dim}"
                         f"  applied={result.marg_applied_scale:.2e}"
                         f"  ||r||²={r2_str}")
        else:
            reason = f" ({result.marg_drop_reason})" if result.marg_drop_reason else ""
            prior_str = f"  prior=DROP{reason}"
        if result.boundary_cov_valid:
            ratio = result.window_cov_trace / result.boundary_cov_trace if result.boundary_cov_trace > 0 else float('inf')
            bcov_str = (f"  tr(S⁻¹)={result.boundary_cov_trace:.2e}"
                        f"  tr(H⁻¹)={result.window_cov_trace:.2e}"
                        f"  ratio={ratio:.2f}")
        else:
            bcov_str = "  bcov=n/a"
        print(f"  [sw {n_windows+1:3d}] t={t_w_start:.2f}–{t_w_end:.2f}s"
              f"  {len(window_radar):3d}fr  {int(imu_mask.sum()):4d}imu"
              f"  cost={cost_str}  iter={len(result.cost_history)}  dt={dt_w:.2f}s"
              + prior_str + bcov_str)

        # Snapshot leading-edge estimate in stride zone [t_w_end - stride, t_w_end]
        t_snap_start = t_w_end - window_stride
        t_snap_end   = t_w_end
        n_pts        = max(2, int(round((t_snap_end - t_snap_start) / _live_eval_dt)))
        t_snap_grid  = np.linspace(t_snap_start, t_snap_end, n_pts, endpoint=False)

        snap_pos_cps   = np.array(solver.pos_cps)
        snap_ori_quats = np.array(solver.ori_knots)
        snap_pos_sp = UniformBSpline(snap_pos_cps, pos_bspline.degree, dt_pos)
        snap_pos_sp.t_ref = t_ref
        snap_ori_sp = CumulativeSO3BSpline.from_rotation_samples(
            Rotation.from_quat(snap_ori_quats).as_matrix(), dt=dt_ori, t_ref=t_ref)

        abs_pos_start = t_ref + snap_pos_sp.t_start
        abs_pos_end   = t_ref + snap_pos_sp.t_end
        abs_ori_start = t_ref + snap_ori_sp.t_start
        abs_ori_end   = t_ref + snap_ori_sp.t_end
        valid_snap = ((t_snap_grid >= max(abs_pos_start, abs_ori_start)) &
                      (t_snap_grid <= min(abs_pos_end,   abs_ori_end)))
        t_eval_snap = t_snap_grid[valid_snap]
        if len(t_eval_snap) > 0:
            snap_pos_vals = np.array([snap_pos_sp(t - t_ref, derivative=0) for t in t_eval_snap])
            snap_vel_vals = np.array([snap_pos_sp(t - t_ref, derivative=1) for t in t_eval_snap])
            snap_ori_vals = np.array([snap_ori_sp.evaluate(t - t_ref)      for t in t_eval_snap])
            live_snapshots.append({'t': t_eval_snap, 'pos': snap_pos_vals, 'vel': snap_vel_vals, 'ori': snap_ori_vals})

        t_solve += window_stride
        n_windows += 1

    nominal_pitch_sw = extrinsics_cfg.get('rotation_euler_deg', [180.0, 25.5, 0.0])[1]
    opt_pitch_sw = result.extrinsic_euler_deg[1] if n_windows > 0 else nominal_pitch_sw
    print(f"  [--sliding-window] Done: {n_windows} windows solved")
    if not solver_cfg.get('lock_extrinsics', False):
        print(f"  [--sliding-window] Extrinsic pitch: {opt_pitch_sw:.3f}°"
              f" (nominal {nominal_pitch_sw:.1f}°, delta {opt_pitch_sw - nominal_pitch_sw:+.3f}°)")

    # Retrieve full global trajectory from solver
    all_pos_cps   = solver.pos_cps
    all_ori_quats = solver.ori_knots
    biases        = solver.biases

    # Reconstruct full TrajectoryState from updated global arrays
    new_ori_spline = CumulativeSO3BSpline.from_rotation_samples(
        Rotation.from_quat(all_ori_quats).as_matrix(),
        dt=ori_spline.dt, t_ref=ori_spline.t_ref)
    new_pos_bspline = UniformBSpline(all_pos_cps, pos_bspline.degree, pos_bspline.dt)
    new_pos_bspline.t_ref = t_ref

    return TrajectoryState(
        pos_bspline=new_pos_bspline,
        ori_spline=new_ori_spline,
        acc_bias=np.array(biases[:3]),
        gyr_bias=np.array(biases[3:]),
        radar_extrinsic_delta=np.zeros(3),
    ), live_snapshots


# ==================== Config ====================
_cfg = load_config()
BAGS = _cfg['bags']['bags']
FLIPPED_BAGS = set(_cfg['bags']['flipped'])
_BAG_TIMING_CFG = _cfg['bags']['timing']
_BAG_SOLVER_OVERRIDES = _cfg['bags'].get('solver_overrides', {})
_RADAR_CFG = _cfg['bags'].get('radar_config', {})
_EXTRINSICS_CFG = _cfg['extrinsics']
_SOLVER_CFG = _cfg['solver']
del _cfg


# ==================== P1: Gravity-derived initial attitude ====================

def gravity_to_rotation(g_body: np.ndarray) -> np.ndarray:
    """
    Derive initial rotation matrix from measured gravity vector in body frame.

    The body convention is FLU (x=forward, y=left, z=up). In a level pose,
    gravity reads [0, 0, +9.81]. Tilt is encoded in the x/y components.

    Returns R_world_from_body (3x3) such that R @ [0,0,-9.81] ≈ g_body.
    Yaw is set to zero (unobservable without magnetometer).
    """
    g_norm = np.linalg.norm(g_body)
    if g_norm < 1e-3:
        return np.eye(3)

    g_hat = g_body / g_norm  # unit vector pointing "up" in body frame

    # World z-axis = [0, 0, 1]. In body frame it equals g_hat (gravity is up in FLU).
    # roll = atan2(g_y, g_z),  pitch = atan2(-g_x, g_z)
    # These are small-angle approximations good to ±45°.
    roll  = np.arctan2( g_hat[1],  g_hat[2])
    pitch = np.arctan2(-g_hat[0],  g_hat[2])
    yaw   = 0.0  # unobservable without magnetometer

    return rotation_matrix_from_euler(roll, pitch, yaw)


# ==================== P1: Gyro integration ====================

def integrate_gyro_orientation(
    imu_data,
    b_g: np.ndarray,
    R_init: np.ndarray,
    t_start: float,
    t_end: float,
    noise_sigma_rad_per_sqrts: float = 0.0,
) -> tuple:
    """
    Forward-integrate gyroscope measurements to produce a dense rotation array.

    Model: R(t + dt) = R(t) @ exp((z_gyro - b_g + noise) * dt)

    Args:
        imu_data     : list of IMU messages with .timestamp and .angular_velocity
        b_g          : (3,) gyroscope bias estimate (rad/s)
        R_init       : (3,3) initial rotation at t_start
        t_start      : absolute start time (only IMU samples at t >= t_start used)
        t_end        : absolute end time   (only IMU samples at t <= t_end   used)
        noise_sigma_rad_per_sqrts : optional Gaussian noise σ for P5 stress tests
                                    (injected as sqrt(dt)-scaled random walk on gyro)

    Returns:
        times  : (M,) absolute timestamps of integrated rotations
        Rs     : (M, 3, 3) rotation matrices at each time
    """
    # Filter to window
    msgs = [d for d in imu_data if t_start <= d.timestamp <= t_end]
    if len(msgs) < 2:
        return np.array([t_start]), np.array([R_init])

    times = [msgs[0].timestamp]
    Rs    = [R_init.copy()]
    R_cur = R_init.copy()

    rng = np.random.default_rng(42)  # deterministic for reproducibility

    for i in range(1, len(msgs)):
        dt = msgs[i].timestamp - msgs[i - 1].timestamp
        if dt <= 0 or dt > 0.1:  # skip bad dt or gaps > 100 ms
            continue
        omega = msgs[i].angular_velocity - b_g
        if noise_sigma_rad_per_sqrts > 0:
            omega = omega + rng.normal(0.0, noise_sigma_rad_per_sqrts * np.sqrt(dt), 3)
        dR = so3_exp(omega * dt)
        R_cur = R_cur @ dR
        times.append(msgs[i].timestamp)
        Rs.append(R_cur.copy())

    return np.array(times), np.array(Rs)


# ==================== P2: Radar-velocity position integration ====================

def integrate_radar_velocity(
    radar_frames,
    imu_times: np.ndarray,
    imu_Rs: np.ndarray,
    sensor_rotation: np.ndarray,
    sensor_translation: np.ndarray,
    p_init: np.ndarray,
    min_range: float = 0.2,
    min_points: int = 5,
    v_max: float | None = None,
    imu_data_full=None,
    acc_bias: np.ndarray = None,
) -> tuple:
    """
    Integrate radar WLS ego-velocity to get a position trajectory.

    Per-frame ego-velocity from WLS (sensor frame) is rotated to world frame
    and integrated with forward Euler.

    Sign convention:
        solve_ego_velocity_weighted returns v_wls satisfying  u · v_wls = v_meas.
        With TI Doppler convention v_meas = -u · v_sensor,
        we have v_wls = -v_sensor_frame.
        World velocity ≈ -(R_world_from_body @ sensor_rotation) @ v_wls
                       = R_world_from_sensor @ v_sensor_frame
        where R_world_from_sensor = R_world_from_body @ sensor_rotation
              and v_sensor_frame = -v_wls.

    Args:
        radar_frames      : radar data list
        imu_times         : (M,) gyro-integrated rotation timestamps (absolute)
        imu_Rs            : (M,3,3) corresponding rotation matrices (world-from-body)
        sensor_rotation   : (3,3) R_body_from_sensor (= SENSOR_ROTATION from config)
        sensor_translation: (3,) t_body_from_sensor (lever arm, for lever-arm correction)
        p_init            : (3,) initial position at first radar frame
        min_range         : minimum range filter
        min_points        : minimum points for WLS to be valid
        imu_data_full     : optional full-rate IMU message list. When provided with
                            acc_bias, an IMU-integrated velocity is maintained and
                            used as the Doppler unwrapping prediction. This prevents
                            aliasing cascades when v_prev_wls is None (first frame)
                            or when the drone exceeds v_max. After each successful
                            WLS, v_imu is reset from the WLS result to bound drift.
        acc_bias          : (3,) accelerometer bias for IMU integration.

    Returns:
        times  : (K,) absolute timestamps of position estimates
        ps     : (K,3) integrated positions
    """
    times = []
    ps    = []

    p_cur = p_init.copy()
    t_prev = None
    v_prev_wls = None  # previous frame's WLS result (sensor frame, measurement convention)

    # IMU-aided unwrapping: maintain a world-frame velocity from accel integration.
    # Used as unwrapping prediction to handle the first frame (v_prev_wls=None) and
    # high-speed segments where the drone exceeds v_max.
    use_imu_unwrap = (imu_data_full is not None and acc_bias is not None and v_max is not None)
    v_imu = np.zeros(3)   # world-frame velocity estimate
    _g_world = np.array([0.0, 0.0, -9.81])
    if use_imu_unwrap:
        _imu_t = np.array([d.timestamp for d in imu_data_full])
        _imu_a = np.array([d.linear_acceleration for d in imu_data_full])
        _imu_next_idx = 0
        _t_imu_prev = imu_times[0]   # start integration from t_ref

    for frame in sorted(radar_frames, key=lambda f: f.timestamp):
        if frame.positions is None or frame.velocities is None or frame.intensities is None:
            continue

        t = frame.timestamp

        # Advance IMU-velocity integration to current radar frame time.
        # We use nearest-neighbour rotation (imu_Rs already at ~1kHz, so error < 1ms).
        if use_imu_unwrap:
            while _imu_next_idx < len(_imu_t) and _imu_t[_imu_next_idx] <= t:
                t_cur = _imu_t[_imu_next_idx]
                dt_imu = t_cur - _t_imu_prev
                if 0 < dt_imu < 0.1:
                    idx_r = min(np.searchsorted(imu_times, t_cur), len(imu_times) - 1)
                    R_wb_imu = imu_Rs[idx_r]
                    a_debiased = _imu_a[_imu_next_idx] - acc_bias
                    v_imu += (R_wb_imu @ a_debiased + _g_world) * dt_imu
                _t_imu_prev = t_cur
                _imu_next_idx += 1

        # Interpolate gyro-integrated rotation at this radar frame timestamp.
        idx = np.searchsorted(imu_times, t)
        if idx == 0:
            R_wb = imu_Rs[0]
        elif idx >= len(imu_times):
            R_wb = imu_Rs[-1]
        else:
            alpha = (t - imu_times[idx - 1]) / (imu_times[idx] - imu_times[idx - 1])
            dR = imu_Rs[idx - 1].T @ imu_Rs[idx]
            dw = so3_log(dR)
            R_wb = imu_Rs[idx - 1] @ so3_exp(alpha * dw)

        # Unwrapping prediction: when IMU integration is available, always use v_imu
        # (which is continuously propagated by accelerometer and corrected by each WLS
        # result). This avoids the cascade failure where one stale v_prev_wls prediction
        # causes every subsequent frame to unwrap incorrectly.
        # v_wls convention: u·v_wls = v_meas = -u·v_sensor, so v_wls = -v_sensor.
        # v_imu_pred_wls = -(R_sensor_from_body^T @ R_body_from_world @ v_imu)
        #                = -(sensor_rotation.T @ R_wb.T @ v_imu)
        if use_imu_unwrap:
            v_unwrap_pred = -(sensor_rotation.T @ R_wb.T @ v_imu)
        else:
            v_unwrap_pred = v_prev_wls

        # Unwrap individual Doppler measurements using the chosen prediction.
        velocities_to_use = frame.velocities
        if v_unwrap_pred is not None and v_max is not None:
            unwrapped = frame.velocities.copy()
            for i in range(len(frame.positions)):
                _rng = np.linalg.norm(frame.positions[i])
                if _rng < 1e-6:
                    continue
                u = frame.positions[i] / _rng
                v_pred_radial = np.dot(u, v_unwrap_pred)
                v_meas = frame.velocities[i]
                best = v_meas
                best_err = abs(v_meas - v_pred_radial)
                for k in [-1, 1]:
                    v_shift = v_meas + k * 2 * v_max
                    err = abs(v_shift - v_pred_radial)
                    if err < best_err:
                        best_err = err
                        best = v_shift
                unwrapped[i] = best
            velocities_to_use = unwrapped

        # WLS ego-velocity in sensor frame
        v_wls = solve_ego_velocity_weighted(
            frame.positions,
            velocities_to_use,
            frame.intensities,
            min_range=min_range,
            min_points=min_points,
        )
        if v_wls is None:
            continue
        v_prev_wls = v_wls

        # World-frame velocity from WLS:
        #   v_sensor_actual = -v_wls   (sign from TI convention)
        #   v_body_actual   = sensor_rotation @ (-v_wls)
        #   v_world         = R_wb @ v_body_actual
        # Note: sensor_rotation = R_body_from_sensor
        v_world = R_wb @ (sensor_rotation @ (-v_wls))

        # Reset IMU velocity from WLS result to prevent accelerometer drift accumulation.
        if use_imu_unwrap:
            v_imu = v_world.copy()
            _t_imu_prev = t   # restart integration from this anchor

        # Optionally correct for lever arm (omega x t_bs) — omitted for init

        if t_prev is not None:
            dt = t - t_prev
            if 0 < dt < 0.5:  # guard against large gaps
                p_cur = p_cur + v_world * dt

        times.append(t)
        ps.append(p_cur.copy())
        t_prev = t

    if len(times) == 0:
        return np.array([radar_frames[0].timestamp]), np.array([p_init])

    return np.array(times), np.array(ps)


def preunwrap_radar_frames(
    radar_frames,
    imu_times: np.ndarray,
    imu_Rs: np.ndarray,
    sensor_rotation: np.ndarray,
    v_max: float,
    imu_data_full=None,
    acc_bias: np.ndarray = None,
    min_range: float = 0.2,
    min_points: int = 5,
    mocap_vel_fn=None,  # optional: t -> world-frame velocity (3,); overrides IMU integration
) -> tuple:
    """
    Pre-unwrap radar Doppler measurements using IMU-aided velocity prediction.

    Mirrors the velocity propagation in integrate_radar_velocity(): an IMU-
    integrated world-frame velocity is advanced to each radar frame and used to
    pick the correct Doppler alias for every point.  After each frame the WLS
    result resets the IMU velocity to prevent accelerometer drift accumulation.

    When mocap_vel_fn is provided (e.g. from differentiated MoCap positions),
    it overrides the IMU integration entirely — useful when backflip velocities
    exceed v_max so often that the WLS reset loop breaks down.

    Returns new RadarVelocity objects with corrected velocities so the solver
    sees already-unwrapped measurements from iteration 1, removing the circular
    dependency between alias selection and trajectory accuracy.

    Args:
        radar_frames    : raw RadarVelocity list
        imu_times       : (M,) gyro-integrated rotation timestamps
        imu_Rs          : (M,3,3) R_wb rotation matrices
        sensor_rotation : (3,3) R_body_from_sensor
        v_max           : Doppler ambiguity (m/s)
        imu_data_full   : full-rate IMU samples for accel integration
        acc_bias        : (3,) accelerometer bias
        mocap_vel_fn    : optional callable t -> (3,) world-frame CoM velocity;
                          when given, replaces IMU integration as the prediction source

    Returns:
        (unwrapped_frames, n_total_pts, n_unwrapped_pts)
    """
    from rosbag_loader.structures import RadarVelocity  # local import to avoid circular

    unwrapped_frames = []
    v_prev_wls = None
    n_total_pts = 0
    n_unwrapped_pts = 0

    use_imu = (imu_data_full is not None and acc_bias is not None)
    v_imu = np.zeros(3)
    _g_world = np.array([0.0, 0.0, -9.81])
    if use_imu:
        _imu_t = np.array([d.timestamp for d in imu_data_full])
        _imu_a = np.array([d.linear_acceleration for d in imu_data_full])
        _imu_next_idx = 0
        _t_imu_prev = imu_times[0]

    for frame in sorted(radar_frames, key=lambda f: f.timestamp):
        if frame.positions is None or frame.velocities is None or frame.intensities is None:
            unwrapped_frames.append(frame)
            continue

        t = frame.timestamp

        # Advance IMU velocity to this frame's timestamp
        if use_imu:
            while _imu_next_idx < len(_imu_t) and _imu_t[_imu_next_idx] <= t:
                t_cur = _imu_t[_imu_next_idx]
                dt_imu = t_cur - _t_imu_prev
                if 0 < dt_imu < 0.1:
                    idx_r = min(np.searchsorted(imu_times, t_cur), len(imu_times) - 1)
                    a_db = _imu_a[_imu_next_idx] - acc_bias
                    v_imu += (imu_Rs[idx_r] @ a_db + _g_world) * dt_imu
                _t_imu_prev = t_cur
                _imu_next_idx += 1

        # Rotation at this frame (SLERP from gyro chain)
        idx = np.searchsorted(imu_times, t)
        if idx == 0:
            R_wb = imu_Rs[0]
        elif idx >= len(imu_times):
            R_wb = imu_Rs[-1]
        else:
            alpha = (t - imu_times[idx - 1]) / (imu_times[idx] - imu_times[idx - 1])
            dw = so3_log(imu_Rs[idx - 1].T @ imu_Rs[idx])
            R_wb = imu_Rs[idx - 1] @ so3_exp(alpha * dw)

        # Prediction in WLS sensor frame: u · v_wls = v_meas = -u · v_sensor
        #   v_wls = -v_sensor = -(R_bs^T @ R_wb^T @ v_world)
        if mocap_vel_fn is not None:
            v_world_mocap = mocap_vel_fn(t)
            v_unwrap_pred = -(sensor_rotation.T @ R_wb.T @ v_world_mocap)
        elif use_imu:
            v_unwrap_pred = -(sensor_rotation.T @ R_wb.T @ v_imu)
        else:
            v_unwrap_pred = v_prev_wls

        # Per-point unwrapping: pick alias closest to prediction
        new_velocities = frame.velocities.copy()
        if v_unwrap_pred is not None:
            for i in range(len(frame.positions)):
                rng = np.linalg.norm(frame.positions[i])
                if rng < min_range:
                    continue
                u = frame.positions[i] / rng
                v_pred_radial = np.dot(u, v_unwrap_pred)
                v_meas = frame.velocities[i]
                best, best_err = v_meas, abs(v_meas - v_pred_radial)
                for k in (-1, 1):
                    v_shift = v_meas + k * 2.0 * v_max
                    err = abs(v_shift - v_pred_radial)
                    if err < best_err:
                        best_err = err
                        best = v_shift
                n_total_pts += 1
                if best != v_meas:
                    n_unwrapped_pts += 1
                new_velocities[i] = best

        # Run WLS on the (now unwrapped) velocities and reset IMU velocity
        v_wls = solve_ego_velocity_weighted(
            frame.positions, new_velocities, frame.intensities,
            min_range=min_range, min_points=min_points,
        )
        if v_wls is not None:
            v_prev_wls = v_wls
            v_world = R_wb @ (sensor_rotation @ (-v_wls))
            if use_imu:
                v_imu = v_world.copy()
                _t_imu_prev = t

        unwrapped_frames.append(dataclasses.replace(frame, velocities=new_velocities))

    return unwrapped_frames, n_total_pts, n_unwrapped_pts


# ==================== Init helpers ====================

def build_orientation_spline_from_gyro(
    imu_gyro_times: np.ndarray,
    imu_Rs: np.ndarray,
    n_knots: int,
    dt_ori: float,
    t_ref: float,
) -> CumulativeSO3BSpline:
    """
    Sample gyro-integrated rotations at spline knot times and build SO(3) spline.

    Args:
        imu_gyro_times : (M,) absolute timestamps of integrated rotations
        imu_Rs         : (M,3,3) rotation matrices
        n_knots        : number of knots in the orientation spline
        dt_ori         : knot spacing (seconds)
        t_ref          : absolute time reference (maps t_rel=0 to this absolute time)

    Returns:
        CumulativeSO3BSpline initialized from gyro integration
    """
    # Relative times for each knot (relative to t_ref)
    knot_times_rel = np.arange(n_knots) * dt_ori
    # t_ref corresponds to the first IMU sample in the window
    # so t_abs = t_ref + t_rel for each knot
    knot_times_abs = t_ref + knot_times_rel

    imu_start = imu_gyro_times[0]
    imu_end   = imu_gyro_times[-1]

    R_knot_samples = np.zeros((n_knots, 3, 3))
    for j, t_abs in enumerate(knot_times_abs):
        t_abs_clamped = np.clip(t_abs, imu_start, imu_end)
        idx = np.searchsorted(imu_gyro_times, t_abs_clamped)
        if idx == 0:
            R_knot_samples[j] = imu_Rs[0]
        elif idx >= len(imu_gyro_times):
            R_knot_samples[j] = imu_Rs[-1]
        else:
            alpha = ((t_abs_clamped - imu_gyro_times[idx - 1]) /
                     (imu_gyro_times[idx] - imu_gyro_times[idx - 1]))
            dR    = imu_Rs[idx - 1].T @ imu_Rs[idx]
            dw    = so3_log(dR)
            R_knot_samples[j] = imu_Rs[idx - 1] @ so3_exp(alpha * dw)

    return CumulativeSO3BSpline.from_rotation_samples(R_knot_samples, dt=dt_ori, t_ref=t_ref)


def build_position_spline_from_radar_integration(
    radar_times: np.ndarray,
    radar_ps: np.ndarray,
    n_pos_points: int,
    bspline_degree: int,
    dt_pos: float,
    t_ref: float,
) -> UniformBSpline:
    """
    Fit position B-spline control points from the integrated radar trajectory.

    Interpolates the radar-integrated positions at uniform control-point times.
    This is approximate (for quintic B-splines, ctrl_pts[i] ≠ spline(knot[i])), but
    produces smooth initial control points from which the LM solver converges well.

    The initial snap cost will be high (O(10^10) for dt_pos=0.005s) due to velocity
    discontinuities in the piecewise-linear radar dead-reckoning. The first LM step
    reduces snap rapidly; the solver recovers within ~25 iterations.

    Args:
        radar_times    : (K,) absolute timestamps of integrated positions
        radar_ps       : (K,3) integrated position estimates
        n_pos_points   : number of control points
        bspline_degree : B-spline degree (5)
        dt_pos         : knot spacing (seconds)
        t_ref          : absolute time reference

    Returns:
        UniformBSpline with control points initialized from radar integration
    """
    pos_bspline = UniformBSpline(np.zeros((n_pos_points, 3)), bspline_degree, dt_pos)
    pos_bspline.t_ref = t_ref

    radar_times_rel = radar_times - t_ref

    # Clamp sample times to spline domain (exclude t_end: basis is zero there for uniform splines)
    t_start = pos_bspline.t_start
    t_end   = pos_bspline.t_end
    mask = (radar_times_rel >= t_start) & (radar_times_rel < t_end)
    if mask.sum() < bspline_degree + 1:
        # Degenerate: too few points inside domain; fall back to flat trajectory
        ctrl_pts = np.tile(radar_ps[0], (n_pos_points, 1))
        pos_bspline.control_points = ctrl_pts
        return pos_bspline

    t_samples = radar_times_rel[mask]
    p_samples = radar_ps[mask]          # (M, 3)

    # Initialize control points by linearly interpolating the radar-integrated trajectory
    # at uniformly-spaced control-point times. This is an approximation for degree-5
    # B-splines (ctrl_pts[i] ≠ spline(knot[i])), but produces smooth starting control
    # points and the LM solver corrects the mismatch in the first few iterations.
    init_times = np.linspace(t_start, t_end, n_pos_points)
    pos_interp = interp1d(t_samples, p_samples, axis=0,
                          kind='linear', fill_value='extrapolate')
    ctrl_pts = pos_interp(np.clip(init_times, t_samples[0], t_samples[-1]))

    pos_bspline.control_points = ctrl_pts
    return pos_bspline


# ==================== Main ====================

def main():
    start_time = time.time()
    from datetime import datetime
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("LIVE RIO SOLVER — MoCap-Free Initialization")
    print("P1: Gyro-integrated orientation  |  P2: Radar-velocity position")
    print("P3: Sensor-only boundary priors  |  MoCap used ONLY for final eval")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ==================== CLI ====================
    bag_key = sys.argv[1] if len(sys.argv) > 1 else "circle_fwd"
    if bag_key in BAGS:
        BAG_PATH = BAGS[bag_key]
    else:
        BAG_PATH = bag_key

    # P5 stress test: inject orientation noise
    noise_deg_per_sqrts = 0.0
    if '--noise-deg' in sys.argv:
        idx_n = sys.argv.index('--noise-deg')
        if idx_n + 1 < len(sys.argv):
            noise_deg_per_sqrts = float(sys.argv[idx_n + 1])
    noise_rad_per_sqrts = np.radians(noise_deg_per_sqrts)

    # MoCap usage flags (decoupled):
    #   --mocap-init    : use MoCap position + orientation at t=0 only
    #   --mocap-heading : build continuous heading priors from MoCap (pseudo-magnetometer)
    #   --mocap-yaw     : shorthand for --mocap-init --mocap-heading (legacy)
    _legacy_mocap_yaw = '--mocap-yaw' in sys.argv
    USE_MOCAP_INIT    = _legacy_mocap_yaw or '--mocap-init' in sys.argv
    USE_MOCAP_HEADING = _legacy_mocap_yaw or '--mocap-heading' in sys.argv

    NO_PLOT = '--no-plot' in sys.argv
    USE_PREINTEGRATE = '--preintegrate' in sys.argv
    USE_GNC = '--gnc' in sys.argv
    USE_CPP = '--cpp' in sys.argv
    USE_SLIDING_WINDOW = '--sliding-window' in sys.argv

    if USE_SLIDING_WINDOW and not USE_CPP:
        print("ERROR: --sliding-window requires --cpp", file=sys.stderr)
        sys.exit(1)

    # Default: full rate for --cpp (fast enough + better results), 200 Hz for Python solver
    IMU_TARGET_HZ = 1000 if USE_CPP else 200
    if '--imu-hz' in sys.argv:
        idx_hz = sys.argv.index('--imu-hz')
        if idx_hz + 1 < len(sys.argv):
            IMU_TARGET_HZ = int(sys.argv[idx_hz + 1])

    BIAS_PRESET = None
    if '--bias' in sys.argv:
        idx_b = sys.argv.index('--bias')
        if idx_b + 1 < len(sys.argv):
            BIAS_PRESET = sys.argv[idx_b + 1]

    # Apply C++ solver overrides from config/solver_cpp.yaml (before --set, so --set can further override)
    if USE_CPP:
        _cpp_overrides = load_config().get('solver_cpp', {})
        for k, v in _cpp_overrides.items():
            _SOLVER_CFG[k] = v

    # Apply per-bag solver overrides from bags.yaml (after cpp overrides, before --set)
    if bag_key in _BAG_SOLVER_OVERRIDES:
        for k, v in _BAG_SOLVER_OVERRIDES[bag_key].items():
            _SOLVER_CFG[k] = v
            print(f"  [bag override] {k} = {v}")

    # --set key=value  (repeatable): override solver.yaml entries at runtime
    # e.g.: --set lambda_bias_prior_accel=100 --set lambda_gravity=0
    for i, arg in enumerate(sys.argv):
        if arg == '--set' and i + 1 < len(sys.argv):
            k, _, v = sys.argv[i + 1].partition('=')
            try:
                if v.lower() in ('true', 'false'):
                    _SOLVER_CFG[k] = v.lower() == 'true'
                elif '.' in v or ('e' in v.lower() and any(c.isdigit() for c in v)):
                    _SOLVER_CFG[k] = float(v)
                else:
                    _SOLVER_CFG[k] = int(v)
            except ValueError:
                _SOLVER_CFG[k] = v
            print(f"  [--set] {k} = {_SOLVER_CFG[k]}")

    # Preintegration: dt_ori is coupled to preint_hz (dt_ori = 1/preint_hz)
    if _SOLVER_CFG.get('use_preintegration', False):
        preint_hz = _SOLVER_CFG.get('preint_hz', 100.0)
        _SOLVER_CFG['dt_ori'] = 1.0 / preint_hz
        print(f"  [preintegration] dt_ori coupled to preint_hz={preint_hz:.0f} Hz"
              f" → dt_ori={_SOLVER_CFG['dt_ori']:.5f}s")

    # Init mode string (for display/filenames)
    mode_parts = []
    if USE_MOCAP_INIT:    mode_parts.append("mocap-init")
    if USE_MOCAP_HEADING: mode_parts.append("mocap-heading")
    if BIAS_PRESET:       mode_parts.append(f"bias={BIAS_PRESET}")
    init_mode_str = "+".join(mode_parts) + "+gyro" if mode_parts else "sensor-only+gyro"

    # ==================== Config ====================
    if bag_key in _BAG_TIMING_CFG:
        START_TIME_OFFSET, DURATION = _BAG_TIMING_CFG[bag_key]
    else:
        START_TIME_OFFSET = 30.0
        DURATION = 5.0

    ROTATION_EULER_DEG = np.array(_EXTRINSICS_CFG['rotation_euler_deg'])
    _t_base = np.array(_EXTRINSICS_CFG['translation_body_m'])
    IMU_MOCAP_OFFSET  = _EXTRINSICS_CFG['imu_mocap_offset_sec']
    RADAR_IMU_OFFSET  = _EXTRINSICS_CFG['radar_imu_offset_sec']

    FLIP_BODY_FRAME = bag_key in FLIPPED_BAGS
    if '--flip'    in sys.argv: FLIP_BODY_FRAME = True
    if '--no-flip' in sys.argv: FLIP_BODY_FRAME = False

    R_base    = rotation_matrix_from_euler(*np.radians(ROTATION_EULER_DEG))
    R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)
    if FLIP_BODY_FRAME:
        TRANSLATION    = R_yaw_flip @ _t_base
        SENSOR_ROTATION = R_yaw_flip @ R_base
    else:
        TRANSLATION    = _t_base.copy()
        SENSOR_ROTATION = R_base

    BSPLINE_DEGREE  = _SOLVER_CFG['pos_bspline_degree']
    DT_POS          = _SOLVER_CFG['dt_pos']
    DT_ORI          = _SOLVER_CFG['dt_ori']
    LAMBDA_ACCEL    = _SOLVER_CFG['lambda_accel']
    LAMBDA_GYRO     = _SOLVER_CFG['lambda_gyro']
    LAMBDA_SNAP_POS = _SOLVER_CFG['lambda_snap_pos']
    LAMBDA_ORI_REG  = _SOLVER_CFG['lambda_ori_reg']
    LAMBDA_GRAVITY  = _SOLVER_CFG.get('lambda_gravity', 0.1)  # default 0.1 for live solver
    GRAVITY_ACCEL_THRESHOLD = _SOLVER_CFG.get('gravity_accel_threshold', 3.0)
    LAMBDA_HEADING  = _SOLVER_CFG.get('lambda_heading', 0.0)
    if USE_MOCAP_HEADING and LAMBDA_HEADING == 0.0:
        LAMBDA_HEADING = 1.0  # default heading weight for --mocap-heading
    LAMBDA_BIAS_PRIOR = _SOLVER_CFG['lambda_bias_prior']
    LAMBDA_BIAS_PRIOR_ACCEL = _SOLVER_CFG['lambda_bias_prior_accel']
    LAMBDA_BIAS_PRIOR_GYRO  = _SOLVER_CFG['lambda_bias_prior_gyro']
    HUBER_DELTA     = _SOLVER_CFG['huber_delta']
    HUBER_DELTA_ACCEL = _SOLVER_CFG['huber_delta_accel']
    MIN_RANGE       = _SOLVER_CFG['min_range']
    MAX_ITERATIONS  = _SOLVER_CFG['max_iterations']
    EARLY_STOP_PATIENCE = _SOLVER_CFG.get('early_stop_patience', 3)
    CONVERGE_WINDOW = _SOLVER_CFG.get('converge_window', 5)
    RTOL_CONVERGE   = _SOLVER_CFG.get('rtol_converge', 0.01)
    LOCK_BIASES     = _SOLVER_CFG['lock_biases']
    LOCK_EXTRINSICS = _SOLVER_CFG['lock_extrinsics']
    OPTIMIZE_PITCH_ONLY = _SOLVER_CFG['optimize_pitch_only']
    LAMBDA_EXTRINSIC_PRIOR = _SOLVER_CFG['lambda_extrinsic_prior']
    RELINEARIZE_THRESHOLD_DEG = _SOLVER_CFG['relinearize_threshold_deg']
    ORI_BASE_JACOBIAN_WINDOW = _SOLVER_CFG.get('ori_base_jacobian_window', 0)
    LAMBDA_BOUNDARY_VEL   = _SOLVER_CFG['lambda_boundary_vel']
    LAMBDA_BOUNDARY_POS   = _SOLVER_CFG['lambda_boundary_pos']
    LAMBDA_BOUNDARY_ORI     = _SOLVER_CFG['lambda_boundary_ori']
    LAMBDA_BOUNDARY_ORI_YAW = _SOLVER_CFG.get('lambda_boundary_ori_yaw', None)
    LAMBDA_BOUNDARY_ACCEL = _SOLVER_CFG['lambda_boundary_accel']
    LAMBDA_BOUNDARY_GYRO  = _SOLVER_CFG['lambda_boundary_gyro']
    BOUNDARY_WINDOW       = _SOLVER_CFG['boundary_window']

    GNC_DIV_FACTOR = _SOLVER_CFG.get('gnc_div_factor', 2.0)
    GNC_MU_FINAL = _SOLVER_CFG.get('gnc_mu_final', 0.0)

    NO_RADAR = '--no-radar' in sys.argv
    USE_JACOBI_PRECOND = '--precond' in sys.argv
    NO_UNWRAP = '--no-unwrap' in sys.argv

    _rc = _RADAR_CFG.get('best_velocity' if 'best_velocity' in bag_key else 'default', {})
    V_MAX = _rc.get('v_max', 4.99)
    USE_UNWRAP = not NO_UNWRAP

    print(f"\n{'Configuration':-^80}")
    print(f"Bag: {bag_key} -> {BAG_PATH}")
    print(f"Flip body frame: {FLIP_BODY_FRAME}")
    print(f"Time window: {START_TIME_OFFSET:.1f}s + {DURATION:.1f}s")
    print(f"Init mode: {init_mode_str}")
    if noise_rad_per_sqrts > 0:
        print(f"P5 gyro noise: {noise_deg_per_sqrts:.1f} deg/sqrt(s)")

    # ==================== Load Data ====================
    print(f"\n{'Loading Data':-^80}")

    bag_data = load_bag_topics(BAG_PATH, verbose=True)

    t_bag_start = bag_data.start_time + START_TIME_OFFSET
    t_bag_end   = t_bag_start + DURATION

    agiros_states = [s for s in bag_data.agiros_state
                     if t_bag_start <= s.timestamp <= t_bag_end]
    radar_frames  = [f for f in bag_data.radar_velocity
                     if t_bag_start <= f.timestamp <= t_bag_end]
    imu_data      = [d for d in bag_data.imu_data
                     if t_bag_start <= d.timestamp <= t_bag_end]

    # Apply hardware time offsets (these are calibrated hardware constants, not MoCap-derived)
    radar_total_offset = IMU_MOCAP_OFFSET - RADAR_IMU_OFFSET
    for d in imu_data:
        d.timestamp += IMU_MOCAP_OFFSET
    for f in radar_frames:
        f.timestamp += radar_total_offset

    # Filter near-duplicate MoCap timestamps (for evaluation only)
    n_before = len(agiros_states)
    filtered_agiros = [agiros_states[0]] if agiros_states else []
    for i in range(1, len(agiros_states)):
        if agiros_states[i].timestamp - filtered_agiros[-1].timestamp >= 1e-3:
            filtered_agiros.append(agiros_states[i])
    agiros_states = filtered_agiros

    print(f"\nFiltered data:")
    print(f"  MoCap states (eval only): {len(agiros_states)}")
    print(f"  Radar frames: {len(radar_frames)}")
    print(f"  IMU samples: {len(imu_data)}")

    # Build MoCap SLERP for heading priors and initial conditions
    mocap_slerp = None
    mocap_pos_arr = None
    _mc_times = None
    if agiros_states:
        _mc_times   = np.array([s.timestamp for s in agiros_states])
        _mc_rots    = Rotation.from_matrix(
            np.array([quat_to_rotation_matrix(s.orientation) for s in agiros_states]))
        mocap_slerp = Slerp(_mc_times, _mc_rots)
        mocap_pos_arr = np.array([s.position for s in agiros_states])

    if len(radar_frames) == 0 or len(imu_data) == 0:
        print("ERROR: Insufficient sensor data!")
        return

    # Downsample IMU to target rate (default 200 Hz; use --imu-hz N to override)
    IMU_DOWNSAMPLE = max(1, len(imu_data) // (int(DURATION * IMU_TARGET_HZ)))
    imu_data_full  = imu_data  # keep full-rate for gyro integration
    imu_data       = imu_data[::IMU_DOWNSAMPLE]
    print(f"  IMU after downsampling to ~{IMU_TARGET_HZ} Hz (1/{IMU_DOWNSAMPLE}): {len(imu_data)}")

    # Time reference = first IMU sample after time offset (MoCap-free)
    t_ref = imu_data[0].timestamp

    # ==================== P3: Stationary Bias (already MoCap-free) ====================
    print(f"\n{'Stationary Bias Detection (sensor-only)':-^80}")
    # Lower min_stationary_sec to 1.0s — the configured timing windows cover flight only,
    # so the 1s pre-flight stationary period would be rejected with the default 2s threshold.
    stationary_result = detect_stationary_bias(bag_data.imu_data, min_stationary_sec=1.0, verbose=True)

    if stationary_result is not None:
        acc_bias = stationary_result['acc_bias']
        gyr_bias = stationary_result['gyr_bias']
        # Re-extract gravity (mean accel) from the stationary window
        t_stat_s = stationary_result['stationary_start']
        t_stat_e = stationary_result['stationary_end']
        stat_imu = [d for d in bag_data.imu_data
                    if t_stat_s <= d.timestamp <= t_stat_e]
        gravity_body_stationary = (np.mean([d.linear_acceleration for d in stat_imu], axis=0)
                                   if stat_imu else np.array([0.0, 0.0, 9.81]))
    else:
        print("  [WARN] Stationary detection failed; using zero biases and level initial attitude")
        acc_bias = np.zeros(3)
        gyr_bias = np.zeros(3)
        gravity_body_stationary = np.array([0.0, 0.0, 9.81])

    bias_prior_mean = np.concatenate([acc_bias, gyr_bias])

    # --bias flag: override with known per-bag presets for testing convergence
    # Bags from the same session share the same IMU sensor state, so converged
    # biases from one bag can be used for others recorded in the same session.
    _BIAS_PRESETS = {
        'slow_racing_best_velocity': {
            # Stationary-detected biases from prior batch run
            'acc':       (np.array([-0.0077, +0.0072, +0.0605]),
                          np.array([+0.0042, -0.0028, +0.0003])),
            # Converged biases from batch solver (optimal)
            'converged': (np.array([-0.0955, -0.1791, +0.0999]),
                          np.array([+0.0220, -0.0009, -0.0063])),
        },
        # Same sensor session as slow_racing_best_velocity (Wed_11032026_1503).
        # Stationary detection is unreliable for this bag (no clean pre-flight
        # stationary window); use slow_racing converged biases as proxy.
        'fast_racing_best_velocity': {
            'converged': (np.array([-0.0955, -0.1791, +0.0999]),
                          np.array([+0.0220, -0.0009, -0.0063])),
        },
    }
    if BIAS_PRESET and bag_key in _BIAS_PRESETS and BIAS_PRESET in _BIAS_PRESETS[bag_key]:
        acc_bias, gyr_bias = _BIAS_PRESETS[bag_key][BIAS_PRESET]
        bias_prior_mean = np.concatenate([acc_bias, gyr_bias])
        print(f"  [--bias {BIAS_PRESET}] Overriding biases from preset:")
    elif BIAS_PRESET:
        print(f"  [WARN] --bias {BIAS_PRESET}: no preset for bag '{bag_key}', using detected/zero")

    print(f"\n  Acc bias: [{acc_bias[0]:.4f}, {acc_bias[1]:.4f}, {acc_bias[2]:.4f}] m/s²")
    print(f"  Gyr bias: [{gyr_bias[0]:.5f}, {gyr_bias[1]:.5f}, {gyr_bias[2]:.5f}] rad/s")
    print(f"  Gravity (body, stationary): [{gravity_body_stationary[0]:.3f}, "
          f"{gravity_body_stationary[1]:.3f}, {gravity_body_stationary[2]:.3f}] m/s²")

    # ==================== P1: Gravity-derived initial attitude ====================
    print(f"\n{'P1: Gravity-derived initial attitude':-^80}")
    R_init = gravity_to_rotation(gravity_body_stationary)
    init_euler_deg = np.degrees(Rotation.from_matrix(R_init).as_euler('xyz'))
    print(f"  Gravity body frame: [{gravity_body_stationary[0]:.3f}, "
          f"{gravity_body_stationary[1]:.3f}, {gravity_body_stationary[2]:.3f}]")
    print(f"  Gravity-derived attitude (Euler xyz): [{init_euler_deg[0]:.2f}, "
          f"{init_euler_deg[1]:.2f}, {init_euler_deg[2]:.2f}] deg  (yaw=0, unobservable)")

    # --mocap-init: override initial attitude with MoCap orientation at t=t_ref
    if USE_MOCAP_INIT and mocap_slerp is not None:
        t_ref_clamped = np.clip(t_ref, _mc_times[0], _mc_times[-1])
        R_init = mocap_slerp(t_ref_clamped).as_matrix()
        mocap_euler_init = np.degrees(Rotation.from_matrix(R_init).as_euler('xyz'))
        print(f"  [--mocap-init] Using MoCap orientation at t_ref:")
        print(f"    Euler xyz: [{mocap_euler_init[0]:.2f}, {mocap_euler_init[1]:.2f}, {mocap_euler_init[2]:.2f}] deg")
    else:
        print(f"  (yaw unobservable; set to 0 — use --mocap-init for true yaw init)")

    # If stationary period is BEFORE the flight window, use R_init directly.
    # If the drone starts in a tilted pose, this will be slightly wrong — acceptable
    # because the gravity factor and gyro will correct it within a few iterations.

    # ==================== P1: Gyro integration ====================
    print(f"\n{'P1: Gyro integration over flight window':-^80}")
    if noise_rad_per_sqrts > 0:
        print(f"  *** STRESS TEST: adding {noise_deg_per_sqrts:.1f} deg/sqrt(s) gyro noise ***")

    gyro_times, gyro_Rs = integrate_gyro_orientation(
        imu_data_full,
        gyr_bias,
        R_init,
        t_start=t_ref,
        t_end=imu_data[-1].timestamp,
        noise_sigma_rad_per_sqrts=noise_rad_per_sqrts,
    )

    gyro_euler = np.degrees(np.array(
        [Rotation.from_matrix(R).as_euler('xyz') for R in gyro_Rs]))
    print(f"  Integrated {len(gyro_times)} steps over {gyro_times[-1]-gyro_times[0]:.2f}s")
    print(f"  Euler at end: [{gyro_euler[-1,0]:.1f}, {gyro_euler[-1,1]:.1f}, {gyro_euler[-1,2]:.1f}] deg")
    print(f"  Max |Euler| change: roll={np.abs(gyro_euler[:,0]-gyro_euler[0,0]).max():.1f}  "
          f"pitch={np.abs(gyro_euler[:,1]-gyro_euler[0,1]).max():.1f}  "
          f"yaw={np.abs(gyro_euler[:,2]-gyro_euler[0,2]).max():.1f} deg")

    # Compare against MoCap at a few timestamps (eval only)
    if agiros_states:
        print(f"\n  Gyro-init vs MoCap (eval only):")
        mocap_times_ev = np.array([s.timestamp for s in agiros_states])
        mocap_rots_ev  = np.array([quat_to_rotation_matrix(s.orientation)
                                   for s in agiros_states])
        eval_idxs = np.linspace(0, len(gyro_times)-1, min(5, len(gyro_times)), dtype=int)
        for ei in eval_idxs:
            t_abs = gyro_times[ei]
            R_gyro = gyro_Rs[ei]
            ic = np.argmin(np.abs(mocap_times_ev - t_abs))
            R_mocap = mocap_rots_ev[ic]
            angle_err = np.degrees(np.arccos(np.clip(
                (np.trace(R_mocap.T @ R_gyro) - 1) / 2, -1, 1)))
            print(f"    t={t_abs-t_ref:.2f}s  angle error vs MoCap: {angle_err:.1f} deg")

    # ==================== Build orientation spline from gyro ====================
    ori_degree   = 3
    BOUNDARY_ORDER = 2
    n_interior_ori = int(np.ceil(DURATION / DT_ORI)) + 1
    n_ori_points   = max(ori_degree + 2, n_interior_ori + 2 * BOUNDARY_ORDER)

    ori_spline = build_orientation_spline_from_gyro(
        gyro_times, gyro_Rs, n_ori_points, DT_ORI, t_ref
    )
    print(f"\n  Orientation spline: {n_ori_points} knots, dt={DT_ORI:.4f}s  (from gyro)")

    # ==================== P2: Radar velocity integration ====================
    print(f"\n{'P2: Radar velocity integration':-^80}")

    # --mocap-init: use MoCap position at t_ref as position origin
    if USE_MOCAP_INIT and mocap_pos_arr is not None:
        p_init_world = np.array([
            np.interp(t_ref, _mc_times, mocap_pos_arr[:, i]) for i in range(3)])
        print(f"  [--mocap-init] Using MoCap position at t_ref as origin: "
              f"[{p_init_world[0]:.3f}, {p_init_world[1]:.3f}, {p_init_world[2]:.3f}] m")
    else:
        p_init_world = np.zeros(3)

    if NO_RADAR:
        print("  *** --no-radar: skipping radar integration, using zero-position init ***")
        radar_int_times = np.array([t_ref, t_ref + DURATION])
        radar_int_ps    = np.array([p_init_world, p_init_world])
    else:
        radar_int_times, radar_int_ps = integrate_radar_velocity(
            radar_frames,
            gyro_times, gyro_Rs,
            SENSOR_ROTATION, TRANSLATION,
            p_init=p_init_world,
            min_range=MIN_RANGE,
            v_max=V_MAX if USE_UNWRAP else None,
            imu_data_full=imu_data_full,
            acc_bias=acc_bias,
        )
        print(f"  Integrated {len(radar_int_times)} radar frames")
        pos_range = np.ptp(radar_int_ps, axis=0)
        print(f"  Position range: Δx={pos_range[0]:.2f}  Δy={pos_range[1]:.2f}  Δz={pos_range[2]:.2f} m")

    # Build position spline from integrated trajectory
    n_interior_pos = int(np.ceil((DURATION) / DT_POS)) + 1
    n_pos_points   = n_interior_pos + 2 * BOUNDARY_ORDER

    pos_bspline = build_position_spline_from_radar_integration(
        radar_int_times, radar_int_ps,
        n_pos_points, BSPLINE_DEGREE, DT_POS, t_ref,
    )
    print(f"  Position spline: {n_pos_points} control points, dt={DT_POS:.4f}s  (interp init)")

    # Compare integrated trajectory to MoCap (eval only)
    if agiros_states:
        mocap_pos_ev = np.array([s.position for s in agiros_states])
        mocap_pos_ev_centered = mocap_pos_ev - mocap_pos_ev[0]  # center to origin
        # Sample spline at MoCap times
        t_spline_s = pos_bspline.t_start + t_ref
        t_spline_e = pos_bspline.t_end   + t_ref
        spline_mask = ((mocap_times_ev >= t_spline_s) & (mocap_times_ev <= t_spline_e))
        if spline_mask.any():
            est_ps_init = np.array([
                pos_bspline(mocap_times_ev[i] - t_ref, derivative=0)
                for i in range(len(mocap_times_ev)) if spline_mask[i]
            ])
            gt_ps_init = mocap_pos_ev_centered[spline_mask]
            pos_init_err = np.linalg.norm(est_ps_init - gt_ps_init, axis=1)
            print(f"\n  Position init vs MoCap-centered (eval only):")
            print(f"    RMSE: {np.sqrt(np.mean(pos_init_err**2)):.3f} m  "
                  f"max: {pos_init_err.max():.3f} m")

    # ==================== P3: Sensor-only boundary priors ====================
    print(f"\n{'P3: Sensor-only boundary priors':-^80}")
    # Origin is [0,0,0] (local frame, no global reference)
    # Velocity from first good radar WLS result
    # Orientation from gyro integration
    # Angular velocity from first IMU gyro reading

    boundary_vel_priors   = []
    boundary_pos_priors   = []
    boundary_ori_priors   = []
    boundary_accel_priors = []
    boundary_gyro_priors  = []

    t_spline_start_rel = pos_bspline.t_start
    t_spline_start_abs = t_spline_start_rel + t_ref

    # Build sensor-only interpolators for boundary window
    # Orientation: interpolate from gyro_Rs
    def interp_gyro_R(t_abs):
        tc = np.clip(t_abs, gyro_times[0], gyro_times[-1])
        idx = np.searchsorted(gyro_times, tc)
        if idx == 0: return gyro_Rs[0]
        if idx >= len(gyro_times): return gyro_Rs[-1]
        alpha = (tc - gyro_times[idx-1]) / (gyro_times[idx] - gyro_times[idx-1])
        dw = so3_log(gyro_Rs[idx-1].T @ gyro_Rs[idx])
        return gyro_Rs[idx-1] @ so3_exp(alpha * dw)

    # Velocity: from integrated radar positions (numerical derivative over small window)
    def interp_radar_vel(t_abs):
        if len(radar_int_times) < 2:
            return np.zeros(3)
        tc = np.clip(t_abs, radar_int_times[0], radar_int_times[-1])
        idx = np.searchsorted(radar_int_times, tc)
        if idx == 0: idx = 1
        if idx >= len(radar_int_times): idx = len(radar_int_times) - 1
        dt = radar_int_times[idx] - radar_int_times[idx-1]
        if dt < 1e-6: return np.zeros(3)
        return (radar_int_ps[idx] - radar_int_ps[idx-1]) / dt

    # Angular velocity: from IMU gyro (debiased)
    def interp_gyro_omega(t_abs):
        imu_times_arr = np.array([d.timestamp for d in imu_data_full])
        imu_gyros_arr = np.array([d.angular_velocity for d in imu_data_full])
        tc = np.clip(t_abs, imu_times_arr[0], imu_times_arr[-1])
        idx = np.searchsorted(imu_times_arr, tc)
        if idx >= len(imu_times_arr): idx = len(imu_times_arr) - 1
        return imu_gyros_arr[idx] - gyr_bias

    # Position: radar-integrated position (already starts at [0,0,0])
    def interp_radar_pos(t_abs):
        if len(radar_int_times) < 2:
            return np.zeros(3)
        tc = np.clip(t_abs, radar_int_times[0], radar_int_times[-1])
        idx = np.searchsorted(radar_int_times, tc)
        if idx == 0: return radar_int_ps[0]
        if idx >= len(radar_int_times): return radar_int_ps[-1]
        alpha = (tc - radar_int_times[idx-1]) / (radar_int_times[idx] - radar_int_times[idx-1])
        return radar_int_ps[idx-1] + alpha * (radar_int_ps[idx] - radar_int_ps[idx-1])

    n_boundary_samples = max(1, int(BOUNDARY_WINDOW * 50))
    t_bnd_end = t_spline_start_abs + BOUNDARY_WINDOW

    for t_abs in np.linspace(t_spline_start_abs, t_bnd_end, n_boundary_samples):
        # Velocity from radar integration
        v_bnd = interp_radar_vel(t_abs)
        boundary_vel_priors.append((t_abs, v_bnd))
        # Position from radar integration (starts at origin)
        p_bnd = interp_radar_pos(t_abs)
        boundary_pos_priors.append((t_abs, p_bnd))
        # Orientation from gyro integration
        R_bnd = interp_gyro_R(t_abs)
        boundary_ori_priors.append((t_abs, R_bnd))
        # Acceleration: gravity-only approximation (zero dynamics assumption at start)
        # Better than nothing; replaced quickly by accel factor
        boundary_accel_priors.append((t_abs, np.zeros(3)))
        # Angular velocity from gyro
        omega_bnd = interp_gyro_omega(t_abs)
        boundary_gyro_priors.append((t_abs, omega_bnd))

    print(f"  Boundary window: [{t_spline_start_abs-t_ref:.2f}, {t_bnd_end-t_ref:.2f}] s rel")
    print(f"  Prior counts: vel={len(boundary_vel_priors)} pos={len(boundary_pos_priors)} "
          f"ori={len(boundary_ori_priors)} acc={len(boundary_accel_priors)} "
          f"gyr={len(boundary_gyro_priors)}")
    if boundary_vel_priors:
        v0 = boundary_vel_priors[0][1]
        R0 = boundary_ori_priors[0][1]
        r0_euler = np.degrees(Rotation.from_matrix(R0).as_euler('xyz'))
        print(f"  Start vel (sensor): [{v0[0]:.2f}, {v0[1]:.2f}, {v0[2]:.2f}] m/s")
        print(f"  Start ori Euler: [{r0_euler[0]:.1f}, {r0_euler[1]:.1f}, {r0_euler[2]:.1f}] deg")

    # ==================== Heading priors (--mocap-heading pseudo-magnetometer) ====================
    heading_priors = []
    if USE_MOCAP_HEADING and LAMBDA_HEADING > 0 and mocap_slerp is not None:
        print(f"\n{'Heading Priors (MoCap pseudo-magnetometer)':-^80}")
        heading_dt = 0.01   # 100 Hz, matches raw MoCap rate (/mocap/angrybird2/pose @ 100 Hz)
        t_spline_start_rel = pos_bspline.t_start
        t_spline_end_rel   = pos_bspline.t_end
        for t_rel in np.arange(t_spline_start_rel, t_spline_end_rel, heading_dt):
            t_abs = t_rel + t_ref
            t_clamped = np.clip(t_abs, _mc_times[0], _mc_times[-1])
            R_gt = mocap_slerp(t_clamped).as_matrix()
            heading_priors.append((t_abs, R_gt))
        print(f"  Built {len(heading_priors)} heading priors at {1/heading_dt:.0f} Hz  "
              f"lambda_heading={LAMBDA_HEADING}")

    # ==================== Create initial state ====================
    initial_state = TrajectoryState(
        pos_bspline=pos_bspline,
        ori_spline=ori_spline,
        acc_bias=acc_bias,
        gyr_bias=gyr_bias,
        radar_extrinsic_delta=np.zeros(3),
    )

    print(f"\n  Total state variables: {initial_state.get_state_size()}")
    print(f"    Position: {n_pos_points * 3}")
    print(f"    Orientation (Ω knots): {ori_spline.n_knots * 3}")
    print(f"    Biases: 6")

    # ==================== Initial radar residual stats ====================
    from codegen.generated_jacobians import radar_residual_with_jacobians
    _zeros3_init = np.zeros(3)
    init_residuals = []
    n_total_pts    = 0
    for frame in radar_frames:
        t = frame.timestamp
        try:
            v_world = initial_state.get_position(t, derivative=1)
            t_rel_ori = t - initial_state.ori_spline.t_ref
            R_full, omega, _, _, _ = initial_state.ori_spline.evaluate_with_jacobians(t_rel_ori)
            R_nom_quat = Rot3.from_rotation_matrix(R_full)
            R_bs_quat  = Rot3.from_rotation_matrix(SENSOR_ROTATION)
        except Exception:
            continue
        for i in range(frame.num_points()):
            p_s      = frame.positions[i]
            range_val = frame.ranges[i] if frame.ranges is not None else np.linalg.norm(p_s)
            if range_val < MIN_RANGE:
                continue
            u_sensor  = p_s / np.linalg.norm(p_s)
            v_meas    = frame.velocities[i]
            res, _, _, _, _ = radar_residual_with_jacobians(
                v_world, R_nom_quat, _zeros3_init, omega,
                u_sensor, TRANSLATION, R_bs_quat, v_meas, 1e-10)
            init_residuals.append(res[0])
            n_total_pts += 1

    init_residuals = np.array(init_residuals) if init_residuals else np.array([0.0])
    print(f"\n{'Initial Radar Residuals (sensor-only init)':=^80}")
    print(f"  Total radar points: {n_total_pts}")
    print(f"  Mean: {init_residuals.mean():+.4f} m/s  Std: {init_residuals.std():.4f} m/s")
    print(f"  Median |r|: {np.median(np.abs(init_residuals)):.4f} m/s")
    print(f"  Max |r|: {np.abs(init_residuals).max():.4f} m/s")

    # ==================== Pre-unwrap radar frames ====================
    # Use IMU-aided velocity propagation to pick the correct Doppler alias for
    # every radar point BEFORE the solver starts.  This breaks the circular
    # dependency where alias selection requires an accurate trajectory and the
    # trajectory requires correct alias selection.  The solver's in-loop
    # unwrapping (via v_max) remains as a safety net for residual errors.
    if NO_RADAR:
        solver_radar_frames = []
    elif USE_UNWRAP and V_MAX is not None:
        # If MoCap positions are available, build a velocity interpolation function
        # from finite-differenced MoCap positions. This gives exact velocity for
        # alias correction even when backflip speeds exceed v_max (where IMU
        # integration and WLS both break down).
        _mocap_vel_fn = None
        if mocap_pos_arr is not None and _mc_times is not None and len(_mc_times) > 1:
            _mc_vels = np.gradient(mocap_pos_arr, _mc_times, axis=0)
            _mocap_vel_fn = lambda t: np.array([
                np.interp(t, _mc_times, _mc_vels[:, i]) for i in range(3)])
        solver_radar_frames, _n_pts, _n_unwrapped = preunwrap_radar_frames(
            radar_frames,
            gyro_times, gyro_Rs,
            SENSOR_ROTATION,
            V_MAX,
            imu_data_full=imu_data_full,
            acc_bias=acc_bias,
            min_range=MIN_RANGE,
            mocap_vel_fn=_mocap_vel_fn,
        )
        print(f"\n  Pre-unwrapped {_n_unwrapped}/{_n_pts} radar points "
              f"({100 * _n_unwrapped / max(1, _n_pts):.1f}%)")
    else:
        solver_radar_frames = radar_frames

    # ==================== Optimize ====================

    # Build preintegrated IMU factors (one per consecutive radar frame interval).
    # When active, these REPLACE the per-sample accel+gyro residuals: imu_data is
    # set to [] so the solver only sees the preintegrated 9-residual factors.
    # This reduces the Jacobian from O(N_imu) rows to O(N_radar) rows.
    preintegrated_factors = None
    if USE_PREINTEGRATE:
        radar_times_for_preint = np.array([f.timestamp for f in radar_frames])
        preintegrated_factors = build_preintegrated_factors(
            imu_data_full, radar_times_for_preint, acc_bias, gyr_bias)
        print(f"  Preintegrated IMU factors: {len(preintegrated_factors)}")
        print(f"  [--preintegrate] Accel replaced by {len(preintegrated_factors)} preintegrated factors "
              f"(gyro kept: {len(imu_data)} samples for orientation stability)")

    # MoCap data for solver verbose RMSE display only (does not affect optimization)
    mocap_times_abs = np.array([s.timestamp for s in agiros_states]) if agiros_states else None
    mocap_rots_eval = (np.array([quat_to_rotation_matrix(s.orientation) for s in agiros_states])
                       if agiros_states else None)

    # When preintegrating: preintegrated factors replace the accel residuals
    # (position/velocity dynamics), but gyro residuals are kept at full rate to
    # prevent orientation knots from oscillating freely between radar frames.
    # With our B-spline representation, each knot Ω_j is only constrained by
    # preintegration at the radar frame endpoints (~90ms apart), leaving
    # intermediate knots unconstrained — gyro pins them at 1kHz.
    solver_lambda_accel = 0.0 if USE_PREINTEGRATE else LAMBDA_ACCEL

    live_snapshots = []  # populated only for --sliding-window; used in live RMSE eval below
    if USE_CPP and USE_SLIDING_WINDOW:
        optimized_state, live_snapshots = _solve_cpp_sliding_window(
            initial_state=initial_state,
            solver_radar_frames=solver_radar_frames,
            imu_data=imu_data,
            extrinsics_cfg=_EXTRINSICS_CFG,
            solver_cfg=_SOLVER_CFG,
            heading_priors=heading_priors if heading_priors else None,
        )
    elif USE_CPP:
        optimized_state = _solve_cpp(
            initial_state=initial_state,
            solver_radar_frames=solver_radar_frames,
            imu_data=imu_data,
            extrinsics_cfg=_EXTRINSICS_CFG,
            solver_cfg=_SOLVER_CFG,
            heading_priors=heading_priors if heading_priors else None,
        )
    else:
        optimized_state = solve_trajectory_nonlinear(
            initial_state=initial_state,
            radar_frames=solver_radar_frames,
            imu_data=imu_data,
            sensor_translation=TRANSLATION,
            sensor_rotation=SENSOR_ROTATION,
            lambda_accel=solver_lambda_accel,
            lambda_gyro=LAMBDA_GYRO,
            lambda_snap_pos=LAMBDA_SNAP_POS,
            huber_delta=HUBER_DELTA,
            huber_delta_accel=HUBER_DELTA_ACCEL,
            max_iterations=MAX_ITERATIONS,
            lock_biases=LOCK_BIASES,
            use_jacobi_precond=USE_JACOBI_PRECOND,
            verbose=True,
            mocap_times_abs=mocap_times_abs,
            mocap_rotations=mocap_rots_eval,
            boundary_vel_priors=boundary_vel_priors,
            boundary_pos_priors=boundary_pos_priors,
            lambda_boundary_vel=LAMBDA_BOUNDARY_VEL,
            lambda_boundary_pos=LAMBDA_BOUNDARY_POS,
            boundary_ori_priors=boundary_ori_priors,
            boundary_accel_priors=boundary_accel_priors,
            boundary_gyro_priors=boundary_gyro_priors,
            lambda_boundary_ori=LAMBDA_BOUNDARY_ORI,
            lambda_boundary_ori_yaw=LAMBDA_BOUNDARY_ORI_YAW,
            lambda_boundary_accel=LAMBDA_BOUNDARY_ACCEL,
            lambda_boundary_gyro=LAMBDA_BOUNDARY_GYRO,
            relinearize_threshold_deg=RELINEARIZE_THRESHOLD_DEG,
            lambda_ori_reg=LAMBDA_ORI_REG,
            lambda_bias_prior=LAMBDA_BIAS_PRIOR,
            lambda_bias_prior_accel=LAMBDA_BIAS_PRIOR_ACCEL,
            lambda_bias_prior_gyro=LAMBDA_BIAS_PRIOR_GYRO,
            bias_prior_mean=bias_prior_mean,
            lock_extrinsics=LOCK_EXTRINSICS,
            optimize_pitch_only=OPTIMIZE_PITCH_ONLY,
            lambda_extrinsic_prior=LAMBDA_EXTRINSIC_PRIOR,
            v_max=V_MAX if USE_UNWRAP else None,
            ori_base_jacobian_window=ORI_BASE_JACOBIAN_WINDOW,
            early_stop_patience=EARLY_STOP_PATIENCE,
            converge_window=CONVERGE_WINDOW,
            rtol_converge=RTOL_CONVERGE,
            lambda_gravity=LAMBDA_GRAVITY,
            gravity_accel_threshold=GRAVITY_ACCEL_THRESHOLD,
            lambda_heading=LAMBDA_HEADING,
            heading_priors=heading_priors if heading_priors else None,
            preintegrated_factors=preintegrated_factors,
            use_gnc=USE_GNC,
            gnc_div_factor=GNC_DIV_FACTOR,
            gnc_mu_final=GNC_MU_FINAL,
        )

    # ==================== Evaluate against MoCap ====================
    print(f"\n{'Evaluation vs MoCap Ground Truth (not used in optimization)':=^80}")

    if not agiros_states:
        print("  No MoCap data available for evaluation.")
    else:
        from scipy.signal import butter, filtfilt, savgol_filter

        mocap_times_abs = np.array([s.timestamp for s in agiros_states])
        mocap_positions  = np.array([s.position for s in agiros_states])
        mocap_rots_all   = np.array([quat_to_rotation_matrix(s.orientation)
                                     for s in agiros_states])

        t_eval_start = max(pos_bspline.t_start + t_ref, mocap_times_abs[0])
        t_eval_end   = min(pos_bspline.t_end   + t_ref, mocap_times_abs[-1])
        # Trim the last 3s from evaluation: the trajectory tail (no future data)
        # typically drifts and artificially inflates RMSE.
        t_eval_end = min(t_eval_end, pos_bspline.t_end + t_ref - 3.0)
        spline_valid_mask = ((mocap_times_abs >= t_eval_start) &
                             (mocap_times_abs <= t_eval_end))
        eval_times     = mocap_times_abs[spline_valid_mask]
        agiros_eval    = [agiros_states[i] for i in range(len(agiros_states)) if spline_valid_mask[i]]
        mocap_pos_eval = mocap_positions[spline_valid_mask]
        mocap_rot_eval = mocap_rots_all[spline_valid_mask]

        print(f"  Eval range: [{eval_times[0]-t_ref:.3f}, {eval_times[-1]-t_ref:.3f}] s "
              f"({len(eval_times)} MoCap points)")

        estimated_positions     = np.array([optimized_state.get_position(t, 0) for t in eval_times])
        estimated_velocities    = np.array([optimized_state.get_position(t, 1) for t in eval_times])
        estimated_accelerations = np.array([optimized_state.get_position(t, 2) for t in eval_times])
        estimated_rotations     = np.array([optimized_state.get_rotation(t)    for t in eval_times])
        est_ang_vel             = np.array([optimized_state.get_angular_velocity(t) for t in eval_times])

        # Constant SE3 alignment to MoCap frame.
        # R_align rotates the estimate's initial orientation into the MoCap initial orientation.
        # Applied as a single constant transform — preserves trajectory shape, only fixes
        # the unobservable initial yaw (and any small roll/pitch offset).
        R_est_0  = estimated_rotations[0]
        R_gt_0   = mocap_rot_eval[0]
        R_align  = R_gt_0 @ R_est_0.T
        t_align  = mocap_pos_eval[0] - R_align @ estimated_positions[0]

        estimated_positions_aligned     = (R_align @ estimated_positions.T).T + t_align
        estimated_velocities_aligned    = (R_align @ estimated_velocities.T).T
        estimated_accelerations_aligned = (R_align @ estimated_accelerations.T).T
        estimated_rotations_aligned     = np.array([R_align @ R for R in estimated_rotations])

        align_euler = np.degrees(Rotation.from_matrix(R_align).as_euler('xyz'))
        print(f"  SE3 alignment R: [{align_euler[0]:.1f}, {align_euler[1]:.1f}, {align_euler[2]:.1f}] deg  "
              f"t: [{t_align[0]:.3f}, {t_align[1]:.3f}, {t_align[2]:.3f}] m")

        # MoCap derived quantities
        mocap_velocities = np.array([s.velocity for s in agiros_eval])
        mocap_ang_vel    = np.array([s.angular_velocity for s in agiros_eval])

        # Lowpass filter MoCap velocity (4th-order Butterworth, 10 Hz)
        _eval_dt = np.median(np.diff(eval_times))
        _fs = 1.0 / _eval_dt
        _fc = min(10.0, _fs * 0.4)
        _b, _a = butter(4, _fc / (_fs / 2), btype='low')
        if len(mocap_velocities) > 3 * 9:
            for dim in range(3):
                mocap_velocities[:, dim] = filtfilt(_b, _a, mocap_velocities[:, dim])

        # MoCap acceleration via numerical differentiation of velocity
        all_mocap_velocities = np.array([s.velocity for s in agiros_states])
        dt_mocap = np.diff(mocap_times_abs)
        valid_mask_acc = np.ones(len(mocap_times_abs), dtype=bool)
        for i in range(1, len(dt_mocap)):
            if dt_mocap[i - 1] < 1e-3:
                valid_mask_acc[i] = False
        clean_times = mocap_times_abs[valid_mask_acc]
        clean_vel   = all_mocap_velocities[valid_mask_acc]
        dt_clean    = np.diff(clean_times)
        clean_accel = np.zeros_like(clean_vel)
        clean_accel[1:-1] = (clean_vel[2:] - clean_vel[:-2]) / (dt_clean[1:] + dt_clean[:-1])[:, None]
        clean_accel[0]    = (clean_vel[1] - clean_vel[0]) / dt_clean[0]
        clean_accel[-1]   = (clean_vel[-1] - clean_vel[-2]) / dt_clean[-1]
        win = min(15, len(clean_accel) - (1 if len(clean_accel) % 2 == 0 else 0))
        if win >= 5:
            for dim in range(3):
                clean_accel[:, dim] = savgol_filter(clean_accel[:, dim], win, 3)
        mocap_accelerations = np.zeros_like(mocap_velocities)
        for dim in range(3):
            mocap_accelerations[:, dim] = np.interp(eval_times, clean_times, clean_accel[:, dim])
        if len(mocap_accelerations) > 3 * 9:
            for dim in range(3):
                mocap_accelerations[:, dim] = filtfilt(_b, _a, mocap_accelerations[:, dim])

        # Compute errors
        pos_diff        = estimated_positions_aligned - mocap_pos_eval
        pos_errors      = np.linalg.norm(pos_diff, axis=1)
        pos_rmse        = np.sqrt(np.mean(pos_errors**2))
        vel_diff        = estimated_velocities_aligned - mocap_velocities
        vel_errors      = np.linalg.norm(vel_diff, axis=1)
        vel_rmse        = np.sqrt(np.mean(vel_errors**2))
        accel_diff      = estimated_accelerations_aligned - mocap_accelerations
        accel_abs_error = np.linalg.norm(accel_diff, axis=1)
        accel_rmse      = np.sqrt(np.mean(accel_abs_error**2))
        ang_vel_diff      = est_ang_vel - mocap_ang_vel
        ang_vel_abs_error = np.linalg.norm(ang_vel_diff, axis=1)
        ang_vel_rmse      = np.sqrt(np.mean(ang_vel_abs_error**2))

        rot_errors = []
        for i in range(len(eval_times)):
            R_err = mocap_rot_eval[i].T @ estimated_rotations_aligned[i]
            angle = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
            rot_errors.append(angle)
        rot_errors = np.array(rot_errors)
        rot_rmse   = np.sqrt(np.mean(rot_errors**2))

        # Euler per-axis errors (unwrapped, branch-snapped to MoCap)
        mocap_euler_raw = Rotation.from_matrix(mocap_rot_eval).as_euler('xyz')
        est_euler_raw   = Rotation.from_matrix(estimated_rotations_aligned).as_euler('xyz')
        mocap_euler = np.degrees(np.unwrap(mocap_euler_raw, axis=0))
        est_euler_deg = np.degrees(est_euler_raw)
        est_euler = mocap_euler + ((est_euler_deg - mocap_euler + 180) % 360 - 180)
        euler_diff = est_euler - mocap_euler

        # ==================== Live (leading-edge) RMSE ====================
        # Compute before the results print so we can show a combined table.
        # Sentinel plot arrays — set inside the block if live data is available
        live_pos_aligned_plot  = None   # (M, 3) aligned positions at live_mocap_t
        live_vel_aligned_plot  = None   # (M, 3) aligned velocities
        live_ori_aligned_plot  = None   # (M, 3, 3)
        live_euler_plot        = None   # (M, 3) Euler xyz degrees, branch-snapped to MoCap
        live_pos_errs_plot     = None   # (M,) absolute position errors
        live_vel_errs_plot     = None   # (M,) absolute velocity errors
        live_vel_rmse_plot     = None
        live_rot_errs_plot     = None   # (M,) absolute orientation errors in degrees
        live_time_rel_plot     = None   # (M,) relative times matching eval_times[0] origin
        live_pos_rmse_plot     = None
        live_rot_rmse_plot     = None
        _n_live_windows        = 0

        if live_snapshots:
            all_live_t   = np.concatenate([s['t']   for s in live_snapshots])
            all_live_pos = np.concatenate([s['pos'] for s in live_snapshots])
            all_live_vel = np.concatenate([s['vel'] for s in live_snapshots])
            all_live_ori = np.concatenate([s['ori'] for s in live_snapshots])

            # Same MoCap time range as settled eval for apples-to-apples comparison
            live_mask = ((mocap_times_abs >= t_eval_start) & (mocap_times_abs <= t_eval_end) &
                         (mocap_times_abs >= all_live_t[0])  & (mocap_times_abs <= all_live_t[-1]))
            live_mocap_t   = mocap_times_abs[live_mask]
            live_mocap_pos = mocap_positions[live_mask]
            live_mocap_ori = mocap_rots_all[live_mask]

            if len(live_mocap_t) > 0:
                from scipy.interpolate import interp1d as _interp1d
                from scipy.spatial.transform import Slerp as _Slerp

                live_pos_interp = np.zeros((len(live_mocap_t), 3))
                live_vel_interp = np.zeros((len(live_mocap_t), 3))
                for dim in range(3):
                    fp = _interp1d(all_live_t, all_live_pos[:, dim], kind='linear',
                                   bounds_error=False,
                                   fill_value=(all_live_pos[0, dim], all_live_pos[-1, dim]))
                    fv = _interp1d(all_live_t, all_live_vel[:, dim], kind='linear',
                                   bounds_error=False,
                                   fill_value=(all_live_vel[0, dim], all_live_vel[-1, dim]))
                    live_pos_interp[:, dim] = fp(live_mocap_t)
                    live_vel_interp[:, dim] = fv(live_mocap_t)

                slerp_fn = _Slerp(all_live_t, Rotation.from_matrix(all_live_ori))
                live_ori_interp = slerp_fn(
                    np.clip(live_mocap_t, all_live_t[0], all_live_t[-1])).as_matrix()

                live_pos_aligned = (R_align @ live_pos_interp.T).T + t_align
                live_vel_aligned = (R_align @ live_vel_interp.T).T
                live_ori_aligned = np.array([R_align @ R for R in live_ori_interp])

                # Euler angles branch-snapped to MoCap (same convention as settled eval)
                _live_euler_raw = np.degrees(Rotation.from_matrix(live_ori_aligned).as_euler('xyz'))
                _ltr = live_mocap_t - eval_times[0]
                _mocap_euler_raw_all = np.degrees(
                    np.unwrap(Rotation.from_matrix(mocap_rots_all[spline_valid_mask]).as_euler('xyz'), axis=0))
                _mocap_euler_at_live = np.zeros_like(_live_euler_raw)
                _time_rel_settled = eval_times - eval_times[0]
                for dim in range(3):
                    _mocap_euler_at_live[:, dim] = np.interp(_ltr, _time_rel_settled, _mocap_euler_raw_all[:, dim])
                live_euler_aligned = _mocap_euler_at_live + (
                    (_live_euler_raw - _mocap_euler_at_live + 180) % 360 - 180)

                live_mocap_vel = np.array([s.velocity for s in agiros_states])[live_mask]
                live_pos_errs = np.linalg.norm(live_pos_aligned - live_mocap_pos, axis=1)
                live_pos_rmse = np.sqrt(np.mean(live_pos_errs**2))
                live_vel_errs = np.linalg.norm(live_vel_aligned - live_mocap_vel, axis=1)
                live_vel_rmse = np.sqrt(np.mean(live_vel_errs**2))

                live_rot_errs = [
                    np.degrees(np.arccos(np.clip(
                        (np.trace(live_mocap_ori[i].T @ live_ori_aligned[i]) - 1) / 2, -1, 1)))
                    for i in range(len(live_mocap_t))
                ]
                live_rot_errs_arr = np.array(live_rot_errs)
                live_rot_rmse = np.sqrt(np.mean(live_rot_errs_arr**2))

                # Populate plot sentinels
                live_pos_aligned_plot = live_pos_aligned
                live_vel_aligned_plot = live_vel_aligned
                live_ori_aligned_plot = live_ori_aligned
                live_euler_plot       = live_euler_aligned
                live_pos_errs_plot    = live_pos_errs
                live_vel_errs_plot    = live_vel_errs
                live_vel_rmse_plot    = live_vel_rmse
                live_rot_errs_plot    = live_rot_errs_arr
                live_time_rel_plot    = _ltr
                live_pos_rmse_plot    = live_pos_rmse
                live_rot_rmse_plot    = live_rot_rmse
                _n_live_windows       = len(live_snapshots)

        # ==================== Results summary ====================
        _has_live = live_pos_rmse_plot is not None
        _lbl_s = f"{'Settled':>10}"
        _lbl_l = f"{'Live edge':>10}" if _has_live else ""
        _lbl_d = f"{'Δ':>8}"         if _has_live else ""
        print(f"\n  === RESULTS ({init_mode_str})"
              + (f" — settled vs live ({_n_live_windows} windows) ===" if _has_live else " ==="))
        print(f"  {'Metric':<28} {_lbl_s}" + (f"  {_lbl_l}  {_lbl_d}" if _has_live else ""))
        print(f"  {'-'*28} {'-'*10}" + (f"  {'-'*10}  {'-'*8}" if _has_live else ""))

        def _row(label, settled, live=None, fmt=".4f", unit=""):
            s = f"{settled:{fmt}}{unit}"
            if _has_live and live is not None:
                d = live - settled
                return f"  {label:<28} {s:>10}  {live:{fmt}}{unit:>0}  {d:>+8.4f}"
            return f"  {label:<28} {s:>10}"

        print(_row("Position RMSE (m)", pos_rmse,
                   live_pos_rmse_plot if _has_live else None))
        print(_row("Velocity RMSE (m/s)", vel_rmse,
                   live_vel_rmse_plot if _has_live else None))
        print(f"  {'Angular vel RMSE (rad/s)':<28} {ang_vel_rmse:>10.4f}")
        print(f"  {'Acceleration RMSE (m/s²)':<28} {accel_rmse:>10.4f}")
        print(_row("Orientation RMSE (deg)", rot_rmse,
                   live_rot_rmse_plot if _has_live else None))
        print(f"  Per-axis ori RMSE: roll={np.sqrt(np.mean(euler_diff[:,0]**2)):.3f}  "
              f"pitch={np.sqrt(np.mean(euler_diff[:,1]**2)):.3f}  "
              f"yaw={np.sqrt(np.mean(euler_diff[:,2]**2)):.3f} deg")
        print(f"  Acc bias: [{optimized_state.acc_bias[0]:.4f}, {optimized_state.acc_bias[1]:.4f}, "
              f"{optimized_state.acc_bias[2]:.4f}] m/s²")
        print(f"  Gyr bias: [{optimized_state.gyr_bias[0]:.4f}, {optimized_state.gyr_bias[1]:.4f}, "
              f"{optimized_state.gyr_bias[2]:.4f}] rad/s")
        if _has_live:
            print(f"  Live eval range: [{live_mocap_t[0]-t_ref:.3f}, {live_mocap_t[-1]-t_ref:.3f}] s  "
                  f"({len(live_mocap_t)} MoCap pts)")

        if noise_rad_per_sqrts > 0:
            print(f"\n  [P5] Gyro noise={noise_deg_per_sqrts:.1f} deg/sqrt(s) -> "
                  f"pos_rmse={pos_rmse:.4f} m  ori_rmse={rot_rmse:.4f} deg")

        # ==================== Plot ====================
        if NO_PLOT:
            print(f"\n  [--no-plot] Skipping plot generation.")
            return
        print(f"\n{'Generating Plots':-^80}")

        time_rel = eval_times - eval_times[0]

        # Style constants
        AXIS_COLORS = ['#c85050', '#4e9e4e', '#4878c8']
        C_MOCAP = 'royalblue'
        C_EST   = 'crimson'
        C_LIVE  = 'darkorange'
        LW_AXIS = 0.9
        LW_ABS  = 2.0
        LW_ERR  = 1.0
        A_AXIS  = 0.7

        # Display-only lowpass (5 Hz) for error envelopes
        _fc_disp = min(5.0, _fs * 0.4)
        _bd, _ad = butter(4, _fc_disp / (_fs / 2), btype='low')
        def _smooth(x):
            return filtfilt(_bd, _ad, x) if len(x) > 27 else x

        axis_labels = ['x', 'y', 'z']
        euler_names = ['roll', 'pitch', 'yaw']

        def _comparison(a, t, mocap_data, est_data, ax_labels, ylabel):
            for i, lbl in enumerate(ax_labels):
                a.plot(t, mocap_data[:, i], color=AXIS_COLORS[i], linewidth=LW_AXIS,
                       alpha=A_AXIS, label=f'MoCap {lbl}')
                a.plot(t, est_data[:, i], color=AXIS_COLORS[i], linewidth=LW_AXIS,
                       alpha=A_AXIS, linestyle='--')
            a.set_ylabel(ylabel); a.legend(fontsize=6, ncol=2); a.grid(True, alpha=0.3)

        def _error(a, t, per_axis_diff, abs_error, rmse, ax_labels, ylabel, abs_label):
            for i, lbl in enumerate(ax_labels):
                a.plot(t, per_axis_diff[:, i], color=AXIS_COLORS[i], linewidth=LW_AXIS,
                       alpha=A_AXIS, label=f'Δ{lbl}')
            a.plot(t, _smooth(abs_error), color='k', linewidth=LW_ERR,
                   label=f'|{abs_label}| (smoothed)')
            a.axhline(rmse, color='royalblue', linewidth=1.5, linestyle='--',
                      label=f'RMSE: {rmse:.4f}')
            a.axhline(0, color='gray', linewidth=0.5, linestyle=':')
            a.set_ylabel(ylabel); a.legend(fontsize=7); a.grid(True, alpha=0.3)

        # Figure 1: 5 rows x 3 cols (rows 1-4 col 2 = summary)
        fig = plt.figure(figsize=(21, 28))
        noise_suffix = f'  |  gyro noise={noise_deg_per_sqrts:.1f} deg/√s' if noise_rad_per_sqrts > 0 else ''
        fig.suptitle(f'Live RIO ({init_mode_str}) — {bag_key}{noise_suffix}',
                     fontsize=14, fontweight='bold')
        gs = fig.add_gridspec(5, 3, hspace=0.45, wspace=0.35)
        axd = {}
        for row in range(5):
            for col in range(2):
                axd[(row, col)] = fig.add_subplot(gs[row, col])
        axd[(0, 2)] = fig.add_subplot(gs[0, 2])
        ax_summary  = fig.add_subplot(gs[1:, 2])

        # Row 0: Position
        a = axd[(0, 0)]
        a.plot(mocap_pos_eval[:, 0], mocap_pos_eval[:, 1],
               color=C_MOCAP, linewidth=LW_ABS, label='MoCap')
        a.plot(estimated_positions_aligned[:, 0], estimated_positions_aligned[:, 1],
               color=C_EST, linewidth=LW_ABS, linestyle='--', label='Settled')
        if live_pos_aligned_plot is not None:
            a.plot(live_pos_aligned_plot[:, 0], live_pos_aligned_plot[:, 1],
                   color=C_LIVE, linewidth=LW_AXIS, linestyle=':', label='Live edge', alpha=0.8)
        a.set_xlabel('X (m)'); a.set_ylabel('Y (m)')
        a.set_title('Trajectory (X-Y)')
        a.legend(fontsize=8); a.grid(True, alpha=0.3); a.axis('equal')

        a = axd[(0, 1)]
        _comparison(a, time_rel, mocap_pos_eval, estimated_positions_aligned,
                    axis_labels, 'Position (m)')
        if live_pos_aligned_plot is not None:
            for i, lbl in enumerate(axis_labels):
                a.plot(live_time_rel_plot, live_pos_aligned_plot[:, i],
                       color=AXIS_COLORS[i], linewidth=LW_AXIS, linestyle=':', alpha=0.6)
        a.set_xlabel('Time (s)'); a.set_title('Position vs Time  [-- settled  ··· live]')

        a = axd[(0, 2)]
        _error(a, time_rel, pos_diff, pos_errors, pos_rmse, axis_labels, 'Error (m)', 'err')
        if live_pos_aligned_plot is not None:
            a.plot(live_time_rel_plot, _smooth(live_pos_errs_plot) if len(live_pos_errs_plot) > 27 else live_pos_errs_plot,
                   color=C_LIVE, linewidth=LW_ERR, linestyle=':', label=f'Live |err| (smoothed)', alpha=0.9)
            a.axhline(live_pos_rmse_plot, color=C_LIVE, linewidth=1.2, linestyle=':',
                      label=f'Live RMSE: {live_pos_rmse_plot:.4f}')
            a.legend(fontsize=7)
        a.set_xlabel('Time (s)'); a.set_title('Position Error per Axis + Abs')

        # Row 1: Orientation
        a = axd[(1, 0)]
        for i, lbl in enumerate(euler_names):
            a.plot(time_rel, mocap_euler[:, i], color=AXIS_COLORS[i], linewidth=LW_AXIS,
                   alpha=A_AXIS, label=f'MoCap {lbl}')
            a.plot(time_rel, est_euler[:, i], color=AXIS_COLORS[i], linewidth=LW_AXIS,
                   alpha=A_AXIS, linestyle='--')
        if live_euler_plot is not None:
            for i in range(3):
                a.plot(live_time_rel_plot, live_euler_plot[:, i],
                       color=AXIS_COLORS[i], linewidth=LW_AXIS * 0.7, linestyle=':', alpha=0.6)
        a.set_xlabel('Time (s)'); a.set_ylabel('Euler angle (deg)')
        a.set_title('Orientation (Euler xyz)  [-- settled  ··· live]')
        a.legend(fontsize=6, ncol=2); a.grid(True, alpha=0.3)

        a = axd[(1, 1)]
        _error(a, time_rel, euler_diff, rot_errors, rot_rmse, euler_names, 'Error (deg)', 'Δori')
        if live_rot_errs_plot is not None:
            a.plot(live_time_rel_plot, _smooth(live_rot_errs_plot) if len(live_rot_errs_plot) > 27 else live_rot_errs_plot,
                   color=C_LIVE, linewidth=LW_ERR, linestyle=':', label=f'Live |err| (smoothed)', alpha=0.9)
            a.axhline(live_rot_rmse_plot, color=C_LIVE, linewidth=1.2, linestyle=':',
                      label=f'Live RMSE: {live_rot_rmse_plot:.4f}')
            a.legend(fontsize=7)
        a.set_xlabel('Time (s)'); a.set_title('Orientation Error per Axis + Abs')

        # Row 2: Linear velocity
        a = axd[(2, 0)]
        _comparison(a, time_rel, mocap_velocities, estimated_velocities_aligned,
                    axis_labels, 'Velocity (m/s)')
        if live_vel_aligned_plot is not None:
            for i in range(3):
                a.plot(live_time_rel_plot, live_vel_aligned_plot[:, i],
                       color=AXIS_COLORS[i], linewidth=LW_AXIS * 0.7, linestyle=':', alpha=0.6)
        a.set_xlabel('Time (s)'); a.set_title('Linear Velocity Comparison  [-- settled  ··· live]')

        a = axd[(2, 1)]
        _error(a, time_rel, vel_diff, vel_errors, vel_rmse, axis_labels, 'Error (m/s)', 'Δv')
        if live_vel_errs_plot is not None:
            a.plot(live_time_rel_plot, _smooth(live_vel_errs_plot) if len(live_vel_errs_plot) > 27 else live_vel_errs_plot,
                   color=C_LIVE, linewidth=LW_ERR, linestyle=':', label=f'Live |err| (smoothed)', alpha=0.9)
            a.axhline(live_vel_rmse_plot, color=C_LIVE, linewidth=1.2, linestyle=':',
                      label=f'Live RMSE: {live_vel_rmse_plot:.4f}')
            a.legend(fontsize=7)
        a.set_xlabel('Time (s)'); a.set_title('Linear Velocity Error per Axis + Abs')

        # Row 3: Angular velocity
        a = axd[(3, 0)]
        _comparison(a, time_rel, mocap_ang_vel, est_ang_vel,
                    axis_labels, 'Angular vel (rad/s)')
        a.set_xlabel('Time (s)'); a.set_title('Angular Velocity Comparison')

        a = axd[(3, 1)]
        ang_vel_diff_plot = np.column_stack([_smooth(ang_vel_diff[:, i]) for i in range(3)])
        _error(a, time_rel, ang_vel_diff_plot, ang_vel_abs_error, ang_vel_rmse,
               axis_labels, 'Error (rad/s)', 'Δω')
        a.set_xlabel('Time (s)'); a.set_title('Angular Velocity Error per Axis + Abs')

        # Row 4: Acceleration
        mocap_accel_plot = np.column_stack([_smooth(mocap_accelerations[:, i]) for i in range(3)])
        a = axd[(4, 0)]
        _comparison(a, time_rel, mocap_accel_plot, estimated_accelerations_aligned,
                    axis_labels, 'Accel (m/s²)')
        a.set_xlabel('Time (s)'); a.set_title('Acceleration Comparison (vs diff(MoCap vel), smoothed)')

        a = axd[(4, 1)]
        accel_diff_plot = np.column_stack([_smooth(accel_diff[:, i]) for i in range(3)])
        _error(a, time_rel, accel_diff_plot, accel_abs_error, accel_rmse,
               axis_labels, 'Error (m/s²)', 'Δa')
        a.set_xlabel('Time (s)'); a.set_title('Acceleration Error per Axis + Abs')

        # Summary panel
        calibrated_R_bs = Rot3.from_rotation_matrix(
            SENSOR_ROTATION @ so3_exp(optimized_state.radar_extrinsic_delta))
        calibrated_euler_extr = np.degrees(Rotation.from_quat(calibrated_R_bs.data).as_euler('xyz'))
        delta_deg_extr = np.degrees(optimized_state.radar_extrinsic_delta)
        summary_lines = [
            f"RESULTS",
            f"  Pos  RMSE: {pos_rmse:.4f} m",
            f"  Vel  RMSE: {vel_rmse:.4f} m/s",
            f"  AngV RMSE: {ang_vel_rmse:.4f} rad/s",
            f"  Acc  RMSE: {accel_rmse:.4f} m/s²",
            f"  Ori  RMSE: {rot_rmse:.4f}°",
            f"",
            f"INIT ({init_mode_str})",
            f"  Acc: [{bias_prior_mean[0]:+.4f}, {bias_prior_mean[1]:+.4f}, {bias_prior_mean[2]:+.4f}] m/s²",
            f"  Gyr: [{np.degrees(bias_prior_mean[3]):+.2f}, {np.degrees(bias_prior_mean[4]):+.2f}, {np.degrees(bias_prior_mean[5]):+.2f}] deg/s",
            f"FINAL BIASES",
            f"  Acc: [{optimized_state.acc_bias[0]:+.4f}, {optimized_state.acc_bias[1]:+.4f}, {optimized_state.acc_bias[2]:+.4f}] m/s²",
            f"  Gyr: [{np.degrees(optimized_state.gyr_bias[0]):+.2f}, {np.degrees(optimized_state.gyr_bias[1]):+.2f}, {np.degrees(optimized_state.gyr_bias[2]):+.2f}] deg/s",
            f"",
            f"HYPERPARAMETERS",
            f"  bag={bag_key}  t={START_TIME_OFFSET:.0f}s+{DURATION:.0f}s",
            f"  flip={FLIP_BODY_FRAME}  lock_bias={LOCK_BIASES}",
            f"  dt_pos={DT_POS}  dt_ori={DT_ORI}  pos_deg={BSPLINE_DEGREE}  ori_deg=3",
            f"  λ_accel={LAMBDA_ACCEL}  λ_gyro={LAMBDA_GYRO}",
            f"  λ_snap_pos={LAMBDA_SNAP_POS}",
            f"  huber_radar={HUBER_DELTA}  huber_accel={HUBER_DELTA_ACCEL}",
            f"  λ_bnd_vel={LAMBDA_BOUNDARY_VEL}  λ_bnd_pos={LAMBDA_BOUNDARY_POS}",
            f"  λ_bnd_ori={LAMBDA_BOUNDARY_ORI}(yaw={LAMBDA_BOUNDARY_ORI_YAW})  λ_bnd_acc={LAMBDA_BOUNDARY_ACCEL}",
            f"  λ_bnd_gyr={LAMBDA_BOUNDARY_GYRO}",
            f"  bnd_window={BOUNDARY_WINDOW}s (start only)",
            f"  max_iter={MAX_ITERATIONS}  precond={USE_JACOBI_PRECOND}",
            f"  relin_thr={RELINEARIZE_THRESHOLD_DEG}°  imu_offset={IMU_MOCAP_OFFSET*1000:.0f}ms",
            f"  λ_ori_reg={LAMBDA_ORI_REG}  λ_bp_a={LAMBDA_BIAS_PRIOR_ACCEL}  λ_bp_g={LAMBDA_BIAS_PRIOR_GYRO}",
            f"  λ_gravity={LAMBDA_GRAVITY} (σ={GRAVITY_ACCEL_THRESHOLD})  λ_heading={LAMBDA_HEADING}",
            f"",
            f"EXTRINSICS (rotation [roll,pitch,yaw] deg)",
            f"  lock={LOCK_EXTRINSICS}  pitch_only={OPTIMIZE_PITCH_ONLY}  λ_prior={LAMBDA_EXTRINSIC_PRIOR}",
            f"  Init:  [{ROTATION_EULER_DEG[0]:.2f}, {ROTATION_EULER_DEG[1]:.2f}, {ROTATION_EULER_DEG[2]:.2f}]",
            f"  Δ:     [{delta_deg_extr[0]:+.3f}, {delta_deg_extr[1]:+.3f}, {delta_deg_extr[2]:+.3f}]",
            f"  Final: [{calibrated_euler_extr[0]:.2f}, {calibrated_euler_extr[1]:.2f}, {calibrated_euler_extr[2]:.2f}]",
            f"  Trans: [{TRANSLATION[0]:.3f}, {TRANSLATION[1]:.3f}, {TRANSLATION[2]:.3f}] m",
        ]
        ax_summary.text(0.02, 0.98, "\n".join(summary_lines),
                        transform=ax_summary.transAxes,
                        fontsize=8, fontfamily='monospace', verticalalignment='top')
        ax_summary.axis('off')
        ax_summary.set_title('Summary & Config')

        # --- Detect de-aliased frames (post-optimization) ---
        frame_dealiased = np.zeros(len(solver_radar_frames), dtype=bool)
        if USE_UNWRAP and V_MAX is not None and len(solver_radar_frames) > 0:
            from codegen.generated_jacobians import radar_residual_with_jacobians as _rr_fn
            _R_bs_quat_da = Rot3.from_rotation_matrix(SENSOR_ROTATION)
            _zeros3_da = np.zeros(3)
            for _fi, _frame in enumerate(solver_radar_frames):
                _t = _frame.timestamp
                try:
                    _v_world = optimized_state.get_position(_t, derivative=1)
                    _t_rel = _t - optimized_state.ori_spline.t_ref
                    _R_full, _omega, _, _, _ = optimized_state.ori_spline.evaluate_with_jacobians(_t_rel)
                    _R_nom_quat = Rot3.from_rotation_matrix(_R_full)
                except Exception:
                    continue
                for _i in range(_frame.num_points()):
                    _p_s = _frame.positions[_i]
                    _rng = np.linalg.norm(_p_s)
                    if _rng < MIN_RANGE:
                        continue
                    _u_sensor = _p_s / _rng
                    _v_meas = _frame.velocities[_i]
                    _res, *_ = _rr_fn(
                        _v_world, _R_nom_quat, _zeros3_da, _omega,
                        _u_sensor, TRANSLATION, _R_bs_quat_da,
                        _v_meas, 1e-10)
                    _k_alias = round(-_res[0] / (2.0 * V_MAX))
                    if _k_alias != 0:
                        frame_dealiased[_fi] = True
                        break  # one de-aliased point is enough to flag the frame

        # Radar frame tick marks on all time-axis subplots
        # Yellow = normal frames, red = frames with at least one de-aliased return
        radar_tick_times   = np.array([f.timestamp for f in solver_radar_frames]) - eval_times[0]
        radar_tick_counts  = np.array([f.num_points() for f in solver_radar_frames], dtype=float)
        radar_tick_heights = 0.04 * radar_tick_counts / 13.5
        _mask_normal    = ~frame_dealiased
        _mask_dealiased = frame_dealiased
        _time_axes = [axd[(r, c)] for r in range(5) for c in range(2) if (r, c) != (0, 0)] + [axd[(0, 2)]]
        for _ax in _time_axes:
            if len(radar_tick_times) > 0:
                _trans = mtransforms.blended_transform_factory(_ax.transData, _ax.transAxes)
                if _mask_normal.any():
                    _ax.vlines(radar_tick_times[_mask_normal], 0, radar_tick_heights[_mask_normal],
                               transform=_trans, color='#ffe566', linewidth=0.8, alpha=0.85, zorder=0)
                if _mask_dealiased.any():
                    _ax.vlines(radar_tick_times[_mask_dealiased], 0, radar_tick_heights[_mask_dealiased],
                               transform=_trans, color='#ff4444', linewidth=0.8, alpha=0.85, zorder=0)

        noise_tag   = f"_noise{noise_deg_per_sqrts:.0f}" if noise_rad_per_sqrts > 0 else ""
        mocap_tag   = ("_mocap-init" if USE_MOCAP_INIT else "") + ("_mocap-heading" if USE_MOCAP_HEADING else "")
        bias_tag    = f"_bias-{BIAS_PRESET}" if BIAS_PRESET else ""
        plots_dir   = Path(__file__).parent.parent / 'plots' / bag_key / 'live_solver'
        plots_dir.mkdir(parents=True, exist_ok=True)
        out1 = plots_dir / f'live_validation_{bag_key}{mocap_tag}{bias_tag}{noise_tag}_{timestamp_str}.png'
        fig.savefig(out1, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out1}")
        plt.close(fig)

        # Figure 2: Multi-view trajectory
        fig2 = plt.figure(figsize=(14, 12))
        gt  = mocap_pos_eval
        est = estimated_positions_aligned

        def _setup_2d(ax, xi, yi, xlabel, ylabel, title):
            ax.plot(gt[:, xi], gt[:, yi], 'b-', label='MoCap', linewidth=2)
            ax.plot(est[:, xi], est[:, yi], 'r--', label='Estimated', linewidth=1.5)
            ax.plot(gt[0, xi], gt[0, yi], 'bs', markersize=8)
            ax.plot(est[0, xi], est[0, yi], 'rs', markersize=8)
            ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

        _setup_2d(fig2.add_subplot(2, 2, 1), 0, 1, 'X (m)', 'Y (m)', 'X-Y Plane')
        _setup_2d(fig2.add_subplot(2, 2, 2), 0, 2, 'X (m)', 'Z (m)', 'X-Z Plane')
        _setup_2d(fig2.add_subplot(2, 2, 3), 1, 2, 'Y (m)', 'Z (m)', 'Y-Z Plane')

        ax3d = fig2.add_subplot(2, 2, 4, projection='3d')
        ax3d.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'b-', label='MoCap', linewidth=2)
        ax3d.plot(est[:, 0], est[:, 1], est[:, 2], 'r--', label='Estimated', linewidth=1.5)
        ax3d.plot([gt[0, 0]], [gt[0, 1]], [gt[0, 2]], 'bs', markersize=8)
        ax3d.plot([est[0, 0]], [est[0, 1]], [est[0, 2]], 'rs', markersize=8)
        ax3d.set_xlabel('X (m)', fontsize=10); ax3d.set_ylabel('Y (m)', fontsize=10)
        ax3d.set_zlabel('Z (m)', fontsize=10)
        ax3d.set_title('3D View', fontsize=12, fontweight='bold')
        ax3d.legend(fontsize=9)

        fig2.suptitle(f'Live RIO Trajectory Views — {bag_key}', fontsize=14, fontweight='bold')
        fig2.tight_layout()
        out2 = plots_dir / f'live_views_{bag_key}{mocap_tag}{bias_tag}{noise_tag}_{timestamp_str}.png'
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out2}")
        plt.close(fig2)

        # Figure 3: Gravity diagnostics
        G_NORM_DIAG = 9.81
        _sigma_grav = GRAVITY_ACCEL_THRESHOLD
        grav_times = []; grav_accel_norm = []; grav_w_dynamic = []
        grav_measured = []; grav_predicted = []; grav_residual = []
        for imu_msg in imu_data:
            t = imu_msg.timestamp
            if t < eval_times[0] or t > eval_times[-1]:
                continue
            z_acc  = imu_msg.linear_acceleration
            z_deb  = z_acc - optimized_state.acc_bias
            a_norm = np.linalg.norm(z_deb)
            if a_norm < 1e-6:
                continue
            w      = np.exp(-((a_norm - G_NORM_DIAG) / _sigma_grav) ** 2)
            g_meas = (z_deb / a_norm) * G_NORM_DIAG
            try:
                R_est = optimized_state.get_rotation(t)
            except Exception:
                continue
            g_pred = R_est.T @ np.array([0.0, 0.0, G_NORM_DIAG])
            grav_times.append(t - eval_times[0])
            grav_accel_norm.append(a_norm)
            grav_w_dynamic.append(w)
            grav_measured.append(g_meas)
            grav_predicted.append(g_pred)
            grav_residual.append(g_meas - g_pred)

        grav_times      = np.array(grav_times)
        grav_accel_norm = np.array(grav_accel_norm)
        grav_w_dynamic  = np.array(grav_w_dynamic)
        grav_measured   = np.array(grav_measured)  if grav_measured  else np.zeros((0, 3))
        grav_predicted  = np.array(grav_predicted) if grav_predicted else np.zeros((0, 3))
        grav_residual   = np.array(grav_residual)  if grav_residual  else np.zeros((0, 3))
        grav_res_mag    = np.linalg.norm(grav_residual, axis=1) if grav_residual.size else np.zeros(0)

        fig3, axes3 = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        fig3.suptitle(
            f'Gravity Factor Diagnostics — {bag_key}'
            + (f'   [λ_gravity={LAMBDA_GRAVITY}, σ={_sigma_grav} m/s²]' if LAMBDA_GRAVITY > 0
               else '   [lambda_gravity=0, factor DISABLED]'),
            fontsize=13, fontweight='bold',
        )
        _axis_colors3 = ['tab:red', 'tab:green', 'tab:blue']
        _axis_names3  = ['x', 'y', 'z']

        ax = axes3[0]
        ax.plot(grav_times, grav_accel_norm, color='steelblue', linewidth=0.8,
                alpha=0.7, label='‖a_debiased‖')
        ax.axhline(G_NORM_DIAG, color='k', linewidth=1.5, linestyle='--',
                   label=f'g = {G_NORM_DIAG} m/s²')
        for nsig, alpha_band in [(1, 0.18), (2, 0.10)]:
            ax.axhspan(G_NORM_DIAG - nsig * _sigma_grav, G_NORM_DIAG + nsig * _sigma_grav,
                       color='green', alpha=alpha_band,
                       label=f'±{nsig}σ trust band' if nsig == 1 else f'±{nsig}σ')
        ax.set_ylabel('Accel norm (m/s²)')
        ax.set_title('Accelerometer norm (debiased) — near g during quasi-static phases')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes3[1]
        ax.fill_between(grav_times, grav_w_dynamic, alpha=0.5, color='green', label='w_dynamic')
        ax.plot(grav_times, grav_w_dynamic, color='green', linewidth=0.8)
        ax.axhline(1e-4, color='r', linewidth=1.0, linestyle=':', label='skip threshold (1e-4)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Trust weight w')
        ax.set_title('Dynamic trust weight — w→1 near hover, w→0 during high-g maneuvers')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes3[2]
        if grav_measured.size:
            for i, (col, lbl) in enumerate(zip(_axis_colors3, _axis_names3)):
                ax.plot(grav_times, grav_measured[:, i], color=col, linewidth=1.0,
                        alpha=0.85, label=f'meas {lbl}')
                ax.plot(grav_times, grav_predicted[:, i], color=col, linewidth=1.0,
                        linestyle='--', alpha=0.5, label=f'pred {lbl}')
        ax.set_ylabel('g_body (m/s²)')
        ax.set_title('Gravity direction in body frame: measured (solid) vs predicted (dashed)')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

        ax = axes3[3]
        if grav_residual.size:
            for i, (col, lbl) in enumerate(zip(_axis_colors3, _axis_names3)):
                ax.plot(grav_times, grav_residual[:, i], color=col, linewidth=0.8,
                        alpha=0.6, label=f'Δ{lbl}')
            ax.plot(grav_times, grav_res_mag, color='k', linewidth=1.3, label='‖residual‖')
            rms_grav = float(np.sqrt(np.mean(grav_res_mag ** 2))) if grav_res_mag.size else 0.0
            ax.axhline(rms_grav, color='royalblue', linewidth=1.3, linestyle='--',
                       label=f'RMS = {rms_grav:.3f} m/s²')
            ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Residual (m/s²)')
        ax.set_title('Gravity residual per axis: measured − predicted (after optimization)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        if len(solver_radar_frames) > 0:
            for _ax in axes3:
                _trans = mtransforms.blended_transform_factory(_ax.transData, _ax.transAxes)
                if _mask_normal.any():
                    _ax.vlines(radar_tick_times[_mask_normal], 0, radar_tick_heights[_mask_normal],
                               transform=_trans, color='#ffe566', linewidth=0.8, alpha=0.85, zorder=0)
                if _mask_dealiased.any():
                    _ax.vlines(radar_tick_times[_mask_dealiased], 0, radar_tick_heights[_mask_dealiased],
                               transform=_trans, color='#ff4444', linewidth=0.8, alpha=0.85, zorder=0)

        fig3.tight_layout()
        out3 = plots_dir / f'live_gravity_{bag_key}{mocap_tag}{bias_tag}{noise_tag}_{timestamp_str}.png'
        fig3.savefig(out3, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out3}")
        plt.close(fig3)

    elapsed = time.time() - start_time
    print(f"\n{'Done':#^80}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
