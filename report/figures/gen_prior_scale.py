"""
Generate marg_prior_scale sensitivity figure.

Data hardcoded from CLAUDE.md sweep results:
  slow_racing: scales 2e-4, 1e-6, 1e-7, 1e-8  (live-edge pos + ori)
  fast_racing: scales 2e-4, 1e-5, 1e-7         (live-edge pos + ori, partial)

No re-running required — uses stored sweep values.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    'font.size':       12,
    'axes.labelsize':  12,
    'axes.titlesize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'figure.dpi':      150,
})

# ── Data ────────────────────────────────────────────────────────────────────
# slow_racing sweep (all live-edge)
slow_scales  = np.array([2e-4, 1e-6,  1e-7,  1e-8 ])
slow_ori     = np.array([2.282, 2.393, 2.207, 2.201])
slow_pos     = np.array([0.623, 0.513, 0.383, 0.381])

# fast_racing sweep (partial — only scales that were tested)
fast_scales  = np.array([2e-4, 1e-5,  1e-7 ])
fast_ori     = np.array([4.163, 4.093, 4.57 ])
fast_pos     = np.array([0.877, 0.940, np.nan])   # 1e-7 pos not measured

C_SLOW = '#1f77b4'  # blue
C_FAST = '#d62728'  # red

# ── Plot ────────────────────────────────────────────────────────────────────
fig, (ax_ori, ax_pos) = plt.subplots(1, 2, figsize=(9, 4))

for ax, slow_y, fast_y, ylabel, unit in [
    (ax_ori, slow_ori, fast_ori, 'Live-edge orientation RMSE', '°'),
    (ax_pos, slow_pos, fast_pos, 'Live-edge position RMSE',   'm'),
]:
    ax.semilogx(slow_scales, slow_y, 'o-', color=C_SLOW,
                label='Slow racing', zorder=3)
    mask = ~np.isnan(fast_y)
    ax.semilogx(fast_scales[mask], fast_y[mask], 's--', color=C_FAST,
                label='Fast racing', zorder=3)

    # Mark harmful regime
    ax.axvspan(5e-7, 2e-5, alpha=0.08, color='red', zorder=0)
    ymin, ymax = slow_y.min(), slow_y.max()
    ax.text(2.5e-6, ymin + (ymax - ymin) * 0.05,
            'harmful\nregime', ha='center', va='bottom', fontsize=9,
            color='red', alpha=0.75)

    # Mark defaults
    ax.axvline(1e-7, color=C_SLOW, lw=0.8, linestyle=':', alpha=0.6)
    ax.axvline(2e-4, color=C_FAST, lw=0.8, linestyle=':', alpha=0.6)

    ax.set_xlabel('marg_prior_scale')
    ax.set_ylabel(f'{ylabel} ({unit})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

ax_ori.set_title('Orientation RMSE vs. prior scale')
ax_pos.set_title('Position RMSE vs. prior scale')

fig.tight_layout()

out_pdf = OUT_DIR / 'prior_scale_sensitivity.pdf'
out_png = OUT_DIR / 'prior_scale_sensitivity.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
plt.close(fig)
