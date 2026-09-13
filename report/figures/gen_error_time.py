"""
Fig. 3 -- error over time (position / velocity / orientation) for the three
flights, SW live edge.  Rows = flights (slow, fast, backflips); columns =
error type.  Shares style (fonts, line weights, grid, ticks, palette) with
Fig. 2 via paper_style.py.

Usage:  cd analysis/ ; python ../report/figures/gen_error_time.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'analysis'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import paper_style as ps

ps.apply()

BAGS = [
    ('slow_racing_best_velocity', 'Slow racing error'),
    ('fast_racing_best_velocity', 'Fast racing error'),
    ('backflips_best_velocity',   'Backflips error'),
]
PLOTS_ROOT = Path(__file__).parent.parent.parent / 'plots'
OUT_DIR    = Path(__file__).parent
TAG        = 'mocap-init_mocap-heading'

# Columns: (npz-key, y-label, panel word) \u2014 velocity first (primary metric)
COLS = [('live_vel_errs', 'Velocity (m/s)',  'velocity'),
        ('live_rot_errs', 'Orientation (\u00b0)', 'orientation'),
        ('live_pos_errs', 'Position (%)', 'position')]

# authored at full text width; figure* -> 1:1, rcParams pt are on-page pt
fig, axes = plt.subplots(len(BAGS), 3, figsize=(ps.TEXTWIDTH_IN, 3.25),
                         squeeze=False, layout='constrained')
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.14, hspace=0.14)

for row, (bag_key, bag_label) in enumerate(BAGS):
    sw_npz = PLOTS_ROOT / bag_key / 'live_solver' / f'traj_arrays_{bag_key}_{TAG}_sw.npz'
    if not sw_npz.exists():
        print(f"Missing (re-run with --save-arrays): {sw_npz}")
        continue
    s = np.load(sw_npz)
    t = s['live_t_rel']
    # position error in percent of the flight's ground-truth path length,
    # the normalization of the drift metric in every table (2026-09-08, Timo)
    path_len = float(np.sum(np.linalg.norm(np.diff(s['mocap'], axis=0), axis=1)))
    print(f'{bag_key}: path length {path_len:.1f} m')
    for col, (key, ylab, word) in enumerate(COLS):
        ax = axes[row][col]
        y = 100.0 * s[key] / path_len if key == 'live_pos_errs' else s[key]
        ax.plot(t, y, color=ps.C_LIVE, ls='-', lw=ps.LW_MAIN, zorder=3)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
        # y-label once per column (leftmost handled below); title as row header
        if row == len(BAGS) - 1:
            ax.set_xlabel('Time (s)')
        else:
            ax.tick_params(labelbottom=True)   # keep ticks; axes differ per flight
        ax.set_ylabel(ylab)
    # one bold row header per flight, centred over the middle column, clear of
    # all y-labels and tick numbers
    axes[row][1].set_title(bag_label, fontweight='bold', fontsize=8, pad=4)

out_pdf = OUT_DIR / 'error_over_time.pdf'
fig.savefig(out_pdf)
fig.savefig(OUT_DIR / 'error_over_time.png', dpi=300)
print(f"Saved: {out_pdf}")
plt.close(fig)
