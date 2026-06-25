"""
Real-time operating-point figure: per-window solve time vs sliding-window
length, per flight regime, with the nominal-stride real-time threshold drawn.

A configuration is real-time when its per-window solve time falls below the
stride (the shaded region for the nominal 0.3 s stride). Data from the
2026-06-25 window / knot-density sweep (laptop CPU, RANSAC default).

Usage: cd report/figures && python gen_realtime.py
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType (IEEE PDF eXpress)
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({
    'font.size':       11,
    'axes.labelsize':  12,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# (window_s, solve_s, live_pos_m, live_ori_deg)
fast       = [(1.5, 0.183, 0.485, 2.78), (2.0, 0.268, 0.443, 2.95), (3.0, 0.321, 0.399, 2.88)]
slow_dense = [(2.0, 0.497, 0.329, 2.20), (3.0, 0.724, 0.310, 1.97)]
slow_coarse= [(3.0, 0.544, 0.473, 1.77)]
backflips  = [(2.0, 0.432, 1.456, 6.61), (3.0, 0.589, 1.545, 6.35)]

# Okabe-Ito palette (consistent with the trajectory figure)
C_FAST = '#0072B2'   # blue
C_SLOW = '#D55E00'   # vermillion
C_BACK = '#009E73'   # green

fig, ax = plt.subplots(figsize=(3.45, 2.75))

# real-time region: solve time below the nominal 0.3 s stride
ax.axhspan(0.0, 0.3, color='#cdebc5', alpha=0.55, zorder=0)
ax.axhline(0.3, color='gray', ls='--', lw=1.0, zorder=1)
ax.text(3.18, 0.30, 'real-time\n(0.3 s stride)', fontsize=8, color='dimgray',
        va='top', ha='right')


def plot_line(data, color, label, marker='o', ls='-'):
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    ax.plot(xs, ys, color=color, marker=marker, ls=ls, label=label, lw=1.8, ms=6,
            zorder=3)


plot_line(fast,       C_FAST, 'Fast (40/16 ms knots)')
plot_line(slow_dense, C_SLOW, 'Slow (5/8 ms knots)')
plot_line(backflips,  C_BACK, 'Backflips (10/8 ms)')
# slow with coarse racing knots: single open-square marker
ax.plot([3.0], [0.544], color=C_SLOW, marker='s', ms=8, ls='', mfc='none',
        mew=1.6, label='Slow (40/16 ms knots)', zorder=3)

# annotate the headline / key points with live position RMSE (m)
for (w, t, p, o) in [fast[1], fast[2], slow_dense[1], backflips[1]]:
    ax.annotate(f'{p:.2f} m', (w, t), textcoords='offset points', xytext=(5, 4),
                fontsize=7.5, color='black')

ax.set_xlabel('Window length (s)')
ax.set_ylabel('Per-window solve time (s)')
ax.set_xlim(1.35, 3.25)
ax.set_ylim(0.0, 0.80)
ax.set_xticks([1.5, 2.0, 2.5, 3.0])
ax.legend(loc='upper left', framealpha=0.92, handlelength=1.6)
ax.grid(True, alpha=0.3)

out = Path(__file__).parent
fig.savefig(out / 'realtime_sweep.pdf', bbox_inches='tight')
fig.savefig(out / 'realtime_sweep.png', dpi=150, bbox_inches='tight')
print('wrote realtime_sweep.pdf')
