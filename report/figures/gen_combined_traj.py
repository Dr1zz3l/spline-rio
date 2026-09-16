"""
Fig. 2 -- combined trajectory overlays (slow / fast racing, backflips).

Layout: 2 rows (X-Y top, Y-Z bottom) x 3 columns (flights).  Axes are placed
MANUALLY on a single global metres-per-inch scale S, so:
  * every panel has equal aspect (1 m is the same length everywhere -> loops
    keep their true shape),
  * the two panels of a column share the same width and Y-axis extent
    (intra-column horizontal axes are identical size/scale),
  * column widths are proportional to each flight's Y-extent, so equal aspect
    also holds ACROSS columns.
Shared style (fonts, line weights, grid, ticks, palette) comes from
paper_style.py, identical to Fig. 3.

Usage:  cd analysis/ ; python ../report/figures/gen_combined_traj.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'analysis'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import paper_style as ps

ps.apply()

BAGS = [
    ('slow_racing_best_velocity', 'slow racing'),
    ('fast_racing_best_velocity', 'fast racing'),
    ('backflips_best_velocity',   'backflips'),
]
MOCAP_TAG  = 'mocap-init_mocap-heading'
PLOTS_ROOT = Path(__file__).parent.parent.parent / 'plots'
OUT_DIR    = Path(__file__).parent
MARGIN     = 0.15                       # fractional data pad (wider Y ranges fill the panels)

# rows: (x-index, y-index, ylabel)
PROJ = [(1, 0, 'X (m)'),                 # top    : X vs Y
        (1, 2, 'Z (m)')]                 # bottom : Z vs Y

# ---- load data, gather per-column data windows ----------------------------
data = {}
for bag, _ in BAGS:
    d = PLOTS_ROOT / bag / 'live_solver'
    s = np.load(d / f'traj_arrays_{bag}_{MOCAP_TAG}_sw.npz')
    # The live edge only exists once the first window is complete (~2.5 s):
    # before that, the causal output is the first window's estimate.  Draw it
    # as a dotted lead-in from the start marker so the estimate visibly
    # starts where the ground truth starts (start-anchored alignment).
    lead = s['settled'][s['t_rel'] <= s['live_t_rel'][0]]
    data[bag] = (s['mocap'], s['live'], lead)

def padded(a):
    lo, hi = a.min(), a.max(); p = (hi - lo) * MARGIN
    return lo - p, hi + p

# per-column Y window (defines column width) and centres for X, Z
Yspan, Ymid, Xmid, Zmid = {}, {}, {}, {}
Xspan_col, Zspan_col = {}, {}
Xspan_max = Zspan_max = Yspan_max = 0.0
for bag, _ in BAGS:
    P = np.vstack(data[bag])
    ylo, yhi = padded(P[:, 1]); Yspan[bag] = yhi - ylo; Ymid[bag] = 0.5*(ylo+yhi); Yspan_max = max(Yspan_max, yhi-ylo)
    xlo, xhi = padded(P[:, 0]); Xmid[bag] = 0.5*(xlo+xhi); Xspan_col[bag] = xhi-xlo; Xspan_max = max(Xspan_max, xhi-xlo)
    zlo, zhi = padded(P[:, 2]); Zmid[bag] = 0.5*(zlo+zhi); Zspan_col[bag] = zhi-zlo; Zspan_max = max(Zspan_max, zhi-zlo)

# common windows across ALL panels -> one global scale, equal column width
Ywin, Xwin, Zwin = Yspan_max, Xspan_max, Zspan_max

# ---- solve the geometry (all in inches) -----------------------------------
FIG_W   = ps.TEXTWIDTH_IN                # 7.16 in: fill the full page width
L_MARG  = 0.40                           # left: y-label + ticks
COL_GAP = 1.2                            # generous inter-column whitespace
R_MARG  = 0.06
ROW_GAP = 0.26                           # between the two rows
T_MARG  = 0.20                           # titles
B_MARG  = 0.50                           # x-label + shared legend

# Equal column width: all three columns share the same box width, and one
# global inches-per-metre scale S is set from that width and the common Y
# window.  Equal aspect holds everywhere (1 m is the same length in every
# panel -> loops keep their true shape); the widest flight (backflips) fills
# its box, the tighter flights sit with a little margin.
col_w   = (FIG_W - L_MARG - 2*COL_GAP - R_MARG) / 3.0
S       = col_w / Ywin
h0, h1  = Xwin * S, Zwin * S                            # row heights (in)
FIG_H   = T_MARG + h0 + ROW_GAP + h1 + B_MARG

fig = plt.figure(figsize=(FIG_W, FIG_H))

# left edges of each column (inches)
lefts = []
x = L_MARG
for bag, _ in BAGS:
    lefts.append(x); x += col_w + COL_GAP

def add_ax(left_in, bottom_in, w_in, h_in):
    return fig.add_axes([left_in/FIG_W, bottom_in/FIG_H, w_in/FIG_W, h_in/FIG_H])

row_bottoms = [T_MARG + 0.0, T_MARG + h0 + ROW_GAP]      # from TOP; convert below
# bottom coords measured from figure bottom:
b_row0 = FIG_H - T_MARG - h0
b_row1 = FIG_H - T_MARG - h0 - ROW_GAP - h1

for col, (bag, title) in enumerate(BAGS):
    gt, live, lead = data[bag]
    w = col_w                              # equal width for every column
    # Per-column inches-per-metre that fills the fixed box as much as equal
    # aspect allows (bounded by the box width and BOTH row heights, so the
    # trajectory never clips).  Top & bottom of a column share ipm -> equal
    # aspect within the column (loops stay circular) and a common horizontal
    # Y range.  Columns with a smaller flight envelope (racing) thus scale up
    # to fill their boxes instead of sitting inside the backflips-sized window.
    ipm = min(col_w / Yspan[bag], h0 / Xspan_col[bag], h1 / Zspan_col[bag])
    xr  = col_w / ipm                      # x (Y-axis) range for this column
    yr0 = h0 / ipm                         # top-row (X) range
    yr1 = h1 / ipm                         # bottom-row (Z) range
    for row, (xi, yi, yl) in enumerate(PROJ):
        b_in = b_row0 if row == 0 else b_row1
        hh   = h0 if row == 0 else h1
        ax = add_ax(lefts[col], b_in, w, hh)
        ax.plot(gt[:, xi],  gt[:, yi],  color=ps.C_GT,   lw=ps.LW_MAIN, ls='-',  zorder=3)
        ax.plot(lead[:, xi],lead[:, yi],color=ps.C_LIVE, lw=ps.LW_MAIN*0.8, ls=':', zorder=4)
        ax.plot(live[:, xi],live[:, yi],color=ps.C_LIVE, lw=ps.LW_MAIN, ls='--', zorder=4)
        # start: black landmark with white halo -> stands off both traces,
        # matches the caption's black square glyph
        ax.plot(gt[0, xi], gt[0, yi], marker='s', markersize=ps.MS_START,
                markerfacecolor='black', markeredgecolor='white',
                markeredgewidth=0.7, ls='none', zorder=6)
        ax.set_xlim(Ymid[bag] - xr/2, Ymid[bag] + xr/2)
        if row == 0:
            ax.set_ylim(Xmid[bag] - yr0/2, Xmid[bag] + yr0/2)
            ax.set_title(title, fontweight='bold')
            ax.tick_params(labelbottom=False)
        else:
            ax.set_ylim(Zmid[bag] - yr1/2, Zmid[bag] + yr1/2)
            ax.set_xlabel('Y (m)')
        ax.set_ylabel(yl)

# ---- shared legend at the bottom ------------------------------------------
handles = [Line2D([0],[0], color=ps.C_GT,   lw=ps.LW_LEGEND, ls='-'),
           Line2D([0],[0], color=ps.C_LIVE, lw=ps.LW_LEGEND, ls='--'),
           Line2D([0],[0], color=ps.C_LIVE, lw=ps.LW_LEGEND*0.8, ls=':'),
           Line2D([0],[0], marker='s', markerfacecolor='black',
                  markeredgecolor='white', markeredgewidth=0.7,
                  ls='none', markersize=ps.MS_START+0.5)]
labels  = ['MoCap ground truth', 'SW live edge', 'first window (before the live edge)', 'start']
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 0.008), columnspacing=1.4, handlelength=2.2)

out = OUT_DIR / 'traj_combined.pdf'
fig.savefig(out)
fig.savefig(OUT_DIR / 'traj_combined.png', dpi=300)
print(f'Saved: {out}  ({FIG_W:.2f} x {FIG_H:.2f} in, S={S:.3f} in/m)')
plt.close(fig)
