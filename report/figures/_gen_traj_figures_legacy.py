#!/usr/bin/env python3
"""
Generate paper-quality trajectory comparison figures for slow_racing and fast_racing.
Produces single-panel XY-plane plots with large fonts, settled + live-edge overlay.

Usage (from repo root):
    cd analysis
    ../.venv/bin/python3 ../report/figures/gen_traj_figures.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../analysis/lib'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp as _Slerp
from scipy.interpolate import interp1d as _interp1d

# ── Matplotlib paper settings ──────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'lines.linewidth': 2.0,
    'figure.dpi': 200,
})

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import analysis modules ────────────────────────────────────────────────
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from validate_live_solver import (
    gravity_to_rotation, integrate_gyro_orientation,
    build_orientation_spline_from_gyro,
    integrate_radar_velocity, preunwrap_radar_frames,
    build_position_spline_from_radar_integration,
    _solve_cpp_sliding_window, _solve_cpp,
    detect_stationary_bias,
)

# ── Config ─────────────────────────────────────────────────────────────────
cfg      = load_config()
bags_cfg = cfg['bags']
base_dir = Path(os.path.expanduser('~/MyData/radar-iwr6843-driver'))

BAGS = [
    ('slow_racing_best_velocity', 'Slow racing'),
    ('fast_racing_best_velocity', 'Fast racing'),
]


def se3_align(estimated_positions, mocap_pos_eval):
    """Constant rotation+translation alignment (yaw only)."""
    R_gt_0  = np.eye(3)
    R_est_0 = np.eye(3)
    R_align = R_gt_0 @ R_est_0.T
    t_align = mocap_pos_eval[0] - R_align @ estimated_positions[0]
    return (R_align @ estimated_positions.T).T + t_align, R_align, t_align


def run_and_plot(bag_key, label):
    print(f'\n{"="*60}')
    print(f'Processing: {bag_key}')

    # ── Load bag ──────────────────────────────────────────────
    rel_path  = bags_cfg['bags'][bag_key]
    bag_path  = str(base_dir / rel_path)
    start_off, dur = bags_cfg['timing'][bag_key]

    data = load_bag_topics(bag_path, verbose=False)
    t0_bag = data.start_time
    t_ref  = t0_bag + start_off        # flight window start
    t_end  = t_ref + dur

    # IMU
    imu_raw = data.imu_data
    imu_t   = np.array([s.timestamp for s in imu_raw])
    imu_acc = np.array([s.linear_acceleration for s in imu_raw])
    imu_gyr = np.array([s.angular_velocity for s in imu_raw])
    mask_imu = (imu_t >= t_ref) & (imu_t <= t_end)
    imu_data = np.column_stack([imu_t[mask_imu],
                                imu_acc[mask_imu],
                                imu_gyr[mask_imu]])   # (N,7)

    # MoCap (ground truth)
    st_all = data.agiros_state
    t_st   = np.array([s.timestamp for s in st_all])
    mask_st = (t_st >= t_ref) & (t_st <= t_end)
    st_w   = [s for s, m in zip(st_all, mask_st) if m]
    mocap_times_abs = np.array([s.timestamp for s in st_w])
    mocap_positions = np.array([s.position   for s in st_w])
    mocap_rots_all  = Rotation.from_quat(
        np.array([[*s.orientation[:3], s.orientation[3]] for s in st_w])
    ).as_matrix()

    # Radar
    rv_all = data.radar_velocity
    t_rv   = np.array([r.timestamp for r in rv_all])
    radar_frames = [r for r, t in zip(rv_all, t_rv) if t_ref <= t <= t_end]

    print(f'  IMU: {len(imu_data)} samples, Radar: {len(radar_frames)} frames, '
          f'MoCap: {len(st_w)} pts')

    # ── Run solver (re-use cached run output if same numbers match) ──
    # We import and call _solve_cpp_sliding_window directly.
    # For simplicity, we just re-run a quick batch solve here to get trajectory.
    # This mirrors what validate_live_solver.py does.

    # Bias detection
    imu_full = imu_data
    b_a, b_g = detect_stationary_bias(imu_full, t_ref - t0_bag)

    # P1: orientation spline
    ori_spline = build_orientation_spline_from_gyro(imu_full, b_g, t_ref)

    # P2: position spline
    rframes_unwrapped = preunwrap_radar_frames(radar_frames, imu_full, b_a, b_g, t_ref,
                                               ori_spline, cfg['solver'])
    pos_spline, _ = build_position_spline_from_radar_integration(
        rframes_unwrapped, imu_full, b_a, b_g, t_ref, ori_spline, cfg['solver']
    )

    from validate_live_solver import TrajectoryState
    initial_state = TrajectoryState(
        pos_spline=pos_spline,
        ori_spline=ori_spline,
        b_a=b_a, b_g=b_g,
    )

    # Run C++ SW solver
    solver_cfg = {**cfg['solver'], **cfg.get('solver_cpp', {})}
    # Apply per-bag overrides
    overrides = bags_cfg.get('solver_overrides', {}).get(bag_key, {})
    solver_cfg.update(overrides)

    heading_priors = []
    for s in st_w:
        R = Rotation.from_quat([*s.orientation[:3], s.orientation[3]]).as_matrix()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        heading_priors.append((s.timestamp, yaw))

    optimized_state, live_snapshots = _solve_cpp_sliding_window(
        initial_state, rframes_unwrapped, imu_full,
        heading_priors, t_ref, t_end, solver_cfg,
        use_mocap_init=True, mocap_pos0=mocap_positions[0],
        mocap_ori0=mocap_rots_all[0],
    )

    # ── Evaluate trajectory ─────────────────────────────────────
    pos_sp  = optimized_state.pos_spline
    ori_sp  = optimized_state.ori_spline

    t_eval_start = max(pos_sp.t_start + t_ref, mocap_times_abs[0])
    t_eval_end   = min(pos_sp.t_end   + t_ref, mocap_times_abs[-1]) - 3.0
    mask_eval    = ((mocap_times_abs >= t_eval_start) &
                    (mocap_times_abs <= t_eval_end))
    mocap_pos_eval = mocap_positions[mask_eval]

    t_rel  = mocap_times_abs[mask_eval] - t_ref
    est_pos = np.array([pos_sp(ti, derivative=0) for ti in t_rel])

    # SE3 align
    est_aligned, R_align, t_align = se3_align(est_pos, mocap_pos_eval)

    # Live edge alignment
    live_pos_aligned_plot = None
    live_mocap_pos_plot   = None
    if live_snapshots:
        all_live_t   = np.concatenate([s['t']   for s in live_snapshots])
        all_live_pos = np.concatenate([s['pos'] for s in live_snapshots])
        live_mask    = ((mocap_times_abs >= t_eval_start) &
                        (mocap_times_abs <= t_eval_end)   &
                        (mocap_times_abs >= all_live_t[0]) &
                        (mocap_times_abs <= all_live_t[-1]))
        if live_mask.sum() > 0:
            live_mocap_t   = mocap_times_abs[live_mask]
            live_mocap_pos = mocap_positions[live_mask]
            live_pos_interp = np.column_stack([
                _interp1d(all_live_t, all_live_pos[:, d], kind='linear',
                           bounds_error=False,
                           fill_value=(all_live_pos[0, d], all_live_pos[-1, d])
                           )(live_mocap_t)
                for d in range(3)
            ])
            live_pos_aligned_plot = (R_align @ live_pos_interp.T).T + t_align
            live_mocap_pos_plot   = live_mocap_pos

    # ── Plot: single XY panel ────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))

    ax.plot(mocap_pos_eval[:, 0], mocap_pos_eval[:, 1],
            color='royalblue', lw=2.2, label='Ground truth (MoCap)', zorder=3)
    ax.plot(est_aligned[:, 0], est_aligned[:, 1],
            color='tomato', lw=1.8, ls='--', label='Settled estimate', zorder=2)
    if live_pos_aligned_plot is not None:
        ax.plot(live_pos_aligned_plot[:, 0], live_pos_aligned_plot[:, 1],
                color='darkorange', lw=1.4, ls=':', label='Live edge', zorder=4)

    # Start markers
    ax.plot(mocap_pos_eval[0, 0], mocap_pos_eval[0, 1],
            's', color='royalblue', ms=8, zorder=5)
    ax.plot(est_aligned[0, 0], est_aligned[0, 1],
            's', color='tomato', ms=8, zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    fig.tight_layout()

    outfile = OUT_DIR / f'traj_{bag_key}.pdf'
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved: {outfile}')
    plt.close(fig)
    return outfile


if __name__ == '__main__':
    for bag_key, label in BAGS:
        try:
            run_and_plot(bag_key, label)
        except Exception as e:
            import traceback
            print(f'ERROR on {bag_key}: {e}')
            traceback.print_exc()
    print('\nDone.')
