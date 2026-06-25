"""
Real-time operating-point figure: per-window solve time vs sliding-window
length, per flight regime, with two candidate strides drawn. A configuration
is real-time when its solve time falls below the stride.

3 s / 0.3 s is the unified working point used throughout the paper. Shrinking
the window reaches real time per regime (fast at a 0.3 s stride, slow/backflips
at 0.5 s) at negligible accuracy cost (see the accompanying table). Solve times
carry ~20% run-to-run variance (laptop CPU), so strides are taken with margin.

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

# representative per-window solve time (s) vs window length (s)
fast      = [(1.5, 0.18), (2.0, 0.28), (3.0, 0.32)]
slow      = [(1.5, 0.40), (2.0, 0.52), (3.0, 0.72)]
backflips = [(1.5, 0.31), (2.0, 0.43), (3.0, 0.59)]

# real-time operating points (window, solve) chosen with margin under the stride
rt_points = [(2.0, 0.28), (1.5, 0.40), (2.0, 0.43)]  # fast, slow, backflips

# Okabe-Ito palette (consistent with the trajectory figure)
C_FAST = '#0072B2'   # blue
C_SLOW = '#D55E00'   # vermillion
C_BACK = '#009E73'   # green

fig, ax = plt.subplots(figsize=(3.5, 2.8))

# real-time regions: below the 0.5 s deployable stride, and the tighter 0.3 s
ax.axhspan(0.0, 0.5, color='#eaf4e7', zorder=0)
ax.axhspan(0.0, 0.3, color='#cdebc5', zorder=0)
ax.axhline(0.3, color='gray', ls=':',  lw=1.0, zorder=1)
ax.axhline(0.5, color='gray', ls='--', lw=1.0, zorder=1)
ax.text(3.2, 0.305, '0.3 s stride', fontsize=7.5, color='dimgray', va='bottom', ha='right')
ax.text(3.2, 0.505, '0.5 s stride', fontsize=7.5, color='dimgray', va='bottom', ha='right')


def plot_line(data, color, label):
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    ax.plot(xs, ys, color=color, marker='o', ls='-', label=label, lw=1.8, ms=5,
            zorder=3)


plot_line(fast,      C_FAST, 'Fast')
plot_line(slow,      C_SLOW, 'Slow')
plot_line(backflips, C_BACK, 'Backflips')

# highlight the chosen real-time operating point per regime
for (w, t), c in zip(rt_points, (C_FAST, C_SLOW, C_BACK)):
    ax.plot([w], [t], marker='*', ms=15, color=c, mec='black', mew=0.6, zorder=4)

ax.set_xlabel('Window length (s)')
ax.set_ylabel('Per-window solve time (s)')
ax.set_xlim(1.35, 3.25)
ax.set_ylim(0.0, 0.80)
ax.set_xticks([1.5, 2.0, 2.5, 3.0])
ax.legend(loc='upper left', framealpha=0.92, handlelength=1.4)
ax.grid(True, alpha=0.3)

out = Path(__file__).parent
fig.savefig(out / 'realtime_sweep.pdf', bbox_inches='tight')
fig.savefig(out / 'realtime_sweep.png', dpi=150, bbox_inches='tight')
print('wrote realtime_sweep.pdf  (stars = per-regime real-time operating points)')
