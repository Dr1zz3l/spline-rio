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
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType, not Type 3 (IEEE PDF eXpress)
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# (bag_key, column title)
BAGS = [
    ('slow_racing_best_velocity', 'slow racing'),
    ('fast_racing_best_velocity', 'fast racing'),
    ('backflips_best_velocity',   'backflips'),
]
MOCAP_TAG  = 'mocap-init_mocap-heading'
PLOTS_ROOT = Path(__file__).parent.parent.parent / 'plots'
OUT_DIR    = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Authored at ~7.2in and displayed at \textwidth (~7.16in) -> ~1:1, so sizes are
# true pt.  Matched to fig 4's on-page sizes (~7.7pt labels, ~7pt ticks/title).
matplotlib.rcParams.update({
    'font.size':       8,
    'axes.labelsize':  8,
    'axes.titlesize':  8,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 8,
    'lines.linewidth': 1.3,
    'figure.dpi':      150,
})

# Two-line, high-contrast, colourblind-safe (Okabe-Ito): ground truth vs
# the SW live edge (the deployed estimate).  Batch is no longer drawn.
C_MOCAP = '#0072B2'   # blue       (ground truth, solid)
C_LIVE  = '#D55E00'   # vermillion (SW live edge, dashed)

# Both rows share Y on the horizontal axis: top is Y-X (X-Y rotated 90 deg so
# Y is at the bottom), bottom is Y-Z.  This makes both rows the same width and
# packs the panels (top row becomes wide, not tall).
# rows: (x-index, y-index, ylabel)
PROJ = [(1, 0, 'X (m)'),
        (1, 2, 'Z (m)')]

# Row heights proportional to the vertical data range (X on top, Z below) so the
# isometric (equal-aspect) boxes pack with little whitespace.
fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.1), sharex='col',
                         gridspec_kw={'height_ratios': [1.6, 1.0]})

for col, (bag, title) in enumerate(BAGS):
    bag_dir   = PLOTS_ROOT / bag / 'live_solver'
    b = np.load(bag_dir / f'traj_arrays_{bag}_{MOCAP_TAG}_batch.npz')
    s = np.load(bag_dir / f'traj_arrays_{bag}_{MOCAP_TAG}_sw.npz')
    gt, live = b['mocap'], s['live']
    allX = np.concatenate([gt[:, 0], live[:, 0]])
    allY = np.concatenate([gt[:, 1], live[:, 1]])
    allZ = np.concatenate([gt[:, 2], live[:, 2]])

    for row, (xi, yi, yl) in enumerate(PROJ):
        ax = axes[row, col]
        ax.plot(gt[:, xi], gt[:, yi], color=C_MOCAP, lw=1.5, ls='-', zorder=3)
        ax.plot(live[:, xi], live[:, yi], color=C_LIVE, lw=1.5, ls='--', zorder=4)
        ax.plot(gt[0, xi], gt[0, yi], color=C_MOCAP, marker='s', markersize=4.5,
                ls='none', zorder=5)
        ax.set_ylabel(yl, labelpad=1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(length=2, pad=1.5)
        if row == 0:
            ax.set_title(title, fontweight='bold', pad=3)
            ax.set_ylim(allX.min() - 0.2, allX.max() + 0.2)
        else:
            ax.set_xlabel('Y (m)', labelpad=1)
            ax.set_ylim(allZ.min() - 0.10, allZ.max() + 0.10)
        # equal data-to-print scaling (isometric): shrink the box to the data so
        # 1 m reads the same horizontally and vertically.  Limits set above are
        # kept; the large shared Y range keeps every panel width-limited (and so
        # the same width), preserving column alignment.
        ax.set_aspect('equal', adjustable='box')
    # widen the shared Y (horizontal) axis by ~0.5 m so the top-row trajectory
    # is not clipped at the loop extremes
    axes[0, col].set_xlim(allY.min() - 0.25, allY.max() + 0.25)

# single shared legend at the bottom
handles = [Line2D([0], [0], color=C_MOCAP, lw=1.8, ls='-'),
           Line2D([0], [0], color=C_LIVE,  lw=1.8, ls='--'),
           Line2D([0], [0], color=C_MOCAP, marker='s', ls='none', markersize=5)]
labels  = ['MoCap ground truth', 'SW live edge', 'start']
fig.legend(handles, labels, loc='lower center', ncol=3,
           frameon=False, bbox_to_anchor=(0.5, -0.01),
           columnspacing=1.8, handlelength=2.2)

fig.tight_layout(rect=[0, 0.055, 1, 1], w_pad=1.2, h_pad=0.8)

out = OUT_DIR / 'traj_combined.pdf'
fig.savefig(out, bbox_inches='tight')
fig.savefig(OUT_DIR / 'traj_combined.png', dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.close(fig)
