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
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType, not Type 3 (IEEE PDF eXpress)
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    # authored ~9in, displayed ~3.45in (~0.38x); fonts sized for ~9-10pt on page
    'font.size':       24,
    'axes.labelsize':  24,
    'axes.titlesize':  23,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 19,
    'lines.linewidth': 2.0,
    'figure.dpi':      150,
})

# ── Data ────────────────────────────────────────────────────────────────────
# slow_racing sweep (all live-edge, 2026-06-01 run)
slow_scales  = np.array([2e-4, 1e-6,  1e-7,  1e-8 ])
slow_ori     = np.array([2.159, 2.210, 2.081, 2.077])
slow_pos     = np.array([0.543, 0.442, 0.338, 0.336])

# fast_racing sweep (live-edge, 2026-06-01 run)
fast_scales  = np.array([2e-4, 1e-5,  1e-6,  1e-7,  1e-8 ])
fast_ori     = np.array([3.644, 3.577, 3.575, 3.574, 3.567])
fast_pos     = np.array([0.825, 0.942, 0.951, 1.279, 1.344])

C_SLOW = '#1f77b4'  # blue
C_FAST = '#d62728'  # red

# ── Plot ────────────────────────────────────────────────────────────────────
fig, (ax_ori, ax_pos) = plt.subplots(1, 2, figsize=(9, 4))

# --- Orientation panel (both series, shared axis) ---
ax_ori.semilogx(slow_scales, slow_ori, 'o-', color=C_SLOW,
                label='Slow racing', zorder=3)
ax_ori.semilogx(fast_scales, fast_ori, 's--', color=C_FAST,
                label='Fast racing', zorder=3)
ax_ori.axvspan(5e-7, 2e-5, alpha=0.08, color='red', zorder=0)
ax_ori.text(2.5e-6, slow_ori.min() + (slow_ori.max() - slow_ori.min()) * 0.1,
            'local max\n(slow ori)', ha='center', va='bottom', fontsize=8,
            color='red', alpha=0.75)
ax_ori.axvline(1e-7, color=C_SLOW, lw=0.8, linestyle=':', alpha=0.6)
ax_ori.axvline(2e-4, color=C_FAST, lw=0.8, linestyle=':', alpha=0.6)
ax_ori.set_xlabel('marg_prior_scale')
ax_ori.set_ylabel('Live-edge orientation RMSE (°)')
ax_ori.set_title('Orientation RMSE vs. prior scale')
ax_ori.legend(loc='upper right')
ax_ori.grid(True, alpha=0.3, which='both')

# --- Position panel: dual y-axes for legibility ---
ax2 = ax_pos.twinx()
ax_pos.semilogx(slow_scales, slow_pos, 'o-', color=C_SLOW,
                label='Slow (left axis)', zorder=3)
ax2.semilogx(fast_scales, fast_pos, 's--', color=C_FAST,
             label='Fast (right axis)', zorder=3)
ax_pos.axvline(1e-7, color=C_SLOW, lw=0.8, linestyle=':', alpha=0.6)
ax_pos.axvline(2e-4, color=C_FAST, lw=0.8, linestyle=':', alpha=0.6)
ax_pos.set_xlabel('marg_prior_scale')
ax_pos.set_ylabel('Live-edge pos. RMSE — Slow racing (m)', color=C_SLOW)
ax2.set_ylabel('Live-edge pos. RMSE — Fast racing (m)', color=C_FAST)
ax_pos.tick_params(axis='y', labelcolor=C_SLOW)
ax2.tick_params(axis='y', labelcolor=C_FAST)
ax_pos.set_title('Position RMSE vs. prior scale')
# Combined legend
lines1, labs1 = ax_pos.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax_pos.legend(lines1 + lines2, labs1 + labs2, loc='upper left', fontsize=9)
ax_pos.grid(True, alpha=0.3, which='both')

fig.tight_layout()

out_pdf = OUT_DIR / 'prior_scale_sensitivity.pdf'
out_png = OUT_DIR / 'prior_scale_sensitivity.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
plt.close(fig)
