"""
Generate error-over-time figure (Fig 5).

Requires extended .npz arrays produced by:
  cd analysis/
  python validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp --save-arrays --no-plot
  python validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp --save-arrays --no-plot
  python validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp --sliding-window --save-arrays --no-plot
  python validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp --sliding-window --save-arrays --no-plot

Two-row layout: slow racing (top), fast racing (bottom).
Each row has three subplots: position error, velocity error, and orientation error vs time.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'analysis'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType, not Type 3 (IEEE PDF eXpress)
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from pathlib import Path

BAGS = [
    ('slow_racing_best_velocity', 'Slow racing'),
    ('fast_racing_best_velocity', 'Fast racing'),
]

PLOTS_ROOT = Path(__file__).parent.parent.parent / 'plots'
OUT_DIR    = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAG = 'mocap-init_mocap-heading'

matplotlib.rcParams.update({
    # authored ~13in, displayed full text width ~7.16in (figure*, ~0.55x);
    # fonts sized for ~10pt on page
    'font.size':       18,
    'axes.labelsize':  18,
    'axes.titlesize':  17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'lines.linewidth': 1.8,
    'figure.dpi':      150,
})

C_BATCH = '#d62728'
C_LIVE  = 'darkorange'

fig, axes = plt.subplots(len(BAGS), 3, figsize=(13, 4.5 * len(BAGS)))
if len(BAGS) == 1:
    axes = axes[np.newaxis, :]

for row, (bag_key, bag_label) in enumerate(BAGS):
    bag_dir   = PLOTS_ROOT / bag_key / 'live_solver'
    batch_npz = bag_dir / f'traj_arrays_{bag_key}_{TAG}_batch.npz'
    sw_npz    = bag_dir / f'traj_arrays_{bag_key}_{TAG}_sw.npz'

    if not batch_npz.exists():
        print(f"Missing (re-run with --save-arrays): {batch_npz}")
        continue
    if not sw_npz.exists():
        print(f"Missing (re-run with --save-arrays): {sw_npz}")
        continue

    b = np.load(batch_npz)
    s = np.load(sw_npz)

    # Check for extended arrays
    if 't_rel' not in b:
        print(f"WARNING: {batch_npz} lacks extended arrays. "
              f"Re-run validate_live_solver.py with --save-arrays.")
        continue

    t_batch      = b['t_rel']
    pos_err_b    = b['pos_errors']
    vel_err_b    = b['vel_errors']
    rot_err_b    = b['rot_errors']    # deg

    has_live = 'live_t_rel' in s and s['live_t_rel'] is not None
    if has_live:
        t_live    = s['live_t_rel']
        pos_err_l = s['live_pos_errs']
        vel_err_l = s['live_vel_errs']
        rot_err_l = s['live_rot_errs']   # deg

    ax_pos = axes[row, 0]
    ax_vel = axes[row, 1]
    ax_ori = axes[row, 2]

    # Position error
    ax_pos.plot(t_batch, pos_err_b, color=C_BATCH, linestyle='--',
                label='Batch estimate', zorder=2)
    if has_live:
        ax_pos.plot(t_live, pos_err_l, color=C_LIVE, linestyle=':',
                    alpha=0.9, label='SW live edge', zorder=3)
    ax_pos.set_xlabel('Time (s)')
    ax_pos.set_ylabel('Position error (m)')
    ax_pos.set_title(f'{bag_label} — position')
    ax_pos.legend(loc='upper left')
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_ylim(bottom=0)

    # Velocity error
    ax_vel.plot(t_batch, vel_err_b, color=C_BATCH, linestyle='--',
                label='Batch estimate', zorder=2)
    if has_live:
        ax_vel.plot(t_live, vel_err_l, color=C_LIVE, linestyle=':',
                    alpha=0.9, label='SW live edge', zorder=3)
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity error (m/s)')
    ax_vel.set_title(f'{bag_label} — velocity')
    ax_vel.grid(True, alpha=0.3)
    ax_vel.set_ylim(bottom=0)

    # Orientation error
    ax_ori.plot(t_batch, rot_err_b, color=C_BATCH, linestyle='--',
                label='Batch estimate', zorder=2)
    if has_live:
        ax_ori.plot(t_live, rot_err_l, color=C_LIVE, linestyle=':',
                    alpha=0.9, label='SW live edge', zorder=3)
    ax_ori.set_xlabel('Time (s)')
    ax_ori.set_ylabel('Orientation error (°)')
    ax_ori.set_title(f'{bag_label} — orientation')
    ax_ori.grid(True, alpha=0.3)
    ax_ori.set_ylim(bottom=0)

fig.tight_layout()

out_pdf = OUT_DIR / 'error_over_time.pdf'
out_png = OUT_DIR / 'error_over_time.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
plt.close(fig)
