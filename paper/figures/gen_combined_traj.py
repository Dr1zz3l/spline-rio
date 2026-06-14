"""
Generate ONE combined full-width trajectory figure (slow / fast / backflips)
for the paper, replacing the three separate per-bag figures.

Layout: 2 rows (X-Y, Y-Z projections) x 3 columns (bags), with a single
shared legend at the bottom and bold column titles on top -- so no space is
wasted on per-subplot legends/titles and the trajectory panels get the room.

Uses the .npz arrays from:
  validate_live_solver.py <bag> --mocap-yaw --cpp --save-arrays --no-plot           (batch)
  validate_live_solver.py <bag> --mocap-yaw --cpp --sliding-window --save-arrays ... (sw)

Usage:  cd analysis/ ; python ../paper/figures/gen_combined_traj.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'analysis'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# (bag_key, column title, show batch estimate)
BAGS = [
    ('slow_racing_best_velocity', 'slow racing', True),
    ('fast_racing_best_velocity', 'fast racing', True),
    ('backflips_best_velocity',   'backflips',   False),  # batch is bistable -> omit
]
MOCAP_TAG  = 'mocap-init_mocap-heading'
PLOTS_ROOT = Path(__file__).parent.parent.parent / 'plots'
OUT_DIR    = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Authored at ~7.2in and displayed at \textwidth (~7.16in) -> ~1:1, so sizes are true pt.
matplotlib.rcParams.update({
    'font.size':       9,
    'axes.labelsize':  9,
    'axes.titlesize':  11,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 9.5,
    'lines.linewidth': 1.3,
    'figure.dpi':      150,
})

# High-contrast, colourblind-safe (Okabe-Ito); the dotted SW live edge
# (headline) is now a strong green instead of low-contrast orange.
C_MOCAP = '#0072B2'   # blue   (ground truth, solid)
C_BATCH = '#D55E00'   # vermillion (batch, dashed)
C_LIVE  = '#009E73'   # green  (SW live edge, dotted)

# rows: (x-index, y-index, xlabel, ylabel)
PROJ = [(0, 1, 'X (m)', 'Y (m)'),
        (1, 2, 'Y (m)', 'Z (m)')]

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15))

for col, (bag, title, show_batch) in enumerate(BAGS):
    bag_dir   = PLOTS_ROOT / bag / 'live_solver'
    b = np.load(bag_dir / f'traj_arrays_{bag}_{MOCAP_TAG}_batch.npz')
    s = np.load(bag_dir / f'traj_arrays_{bag}_{MOCAP_TAG}_sw.npz')
    gt, settled, live = b['mocap'], b['settled'], s['live']

    for row, (xi, yi, xl, yl) in enumerate(PROJ):
        ax = axes[row, col]
        ax.plot(gt[:, xi], gt[:, yi], color=C_MOCAP, lw=1.4, ls='-', zorder=3)
        if show_batch:
            ax.plot(settled[:, xi], settled[:, yi], color=C_BATCH, lw=1.1, ls='--', zorder=2)
        ax.plot(live[:, xi], live[:, yi], color=C_LIVE, lw=1.1, ls=':', alpha=0.9, zorder=4)
        ax.plot(gt[0, xi], gt[0, yi], 'bs', markersize=4.5, zorder=5)
        ax.set_xlabel(xl, labelpad=1)
        ax.set_ylabel(yl, labelpad=1)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.tick_params(length=2, pad=1.5)
        if row == 0:
            ax.set_title(title, fontweight='bold', pad=3)

# single shared legend at the bottom
handles = [Line2D([0], [0], color=C_MOCAP, lw=1.6, ls='-'),
           Line2D([0], [0], color=C_BATCH, lw=1.6, ls='--'),
           Line2D([0], [0], color=C_LIVE,  lw=1.6, ls=':'),
           Line2D([0], [0], color='b', marker='s', ls='none', markersize=5)]
labels  = ['MoCap ground truth', 'Batch estimate (racing only)',
           'SW live edge', 'start']
fig.legend(handles, labels, loc='lower center', ncol=4,
           frameon=False, bbox_to_anchor=(0.5, -0.01),
           columnspacing=1.6, handlelength=2.0)

fig.tight_layout(rect=[0, 0.055, 1, 1], w_pad=1.2, h_pad=0.8)

out = OUT_DIR / 'traj_combined.pdf'
fig.savefig(out, bbox_inches='tight')
fig.savefig(OUT_DIR / 'traj_combined.png', dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.close(fig)
