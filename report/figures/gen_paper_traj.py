"""
Generate paper-quality trajectory comparison figures (Figs 3, 4, and backflips).

Uses pre-saved .npz array files produced by:
  validate_live_solver.py <bag> --mocap-yaw --cpp --save-arrays --no-plot
  validate_live_solver.py <bag> --mocap-yaw --cpp --sliding-window --save-arrays --no-plot

  # Backflips batch uses bags.yaml overrides (dt_ori=0.0008, locked extrinsics)
  validate_live_solver.py backflips_best_velocity --mocap-yaw --cpp --save-arrays --no-plot
  # Backflips SW uses Phase-3 config
  validate_live_solver.py backflips_best_velocity --mocap-yaw --cpp --sliding-window \
    --set dt_ori=0.008 --set lambda_ori_accel=0.001 --set lock_gyro_bias=0 \
    --set marg_prior_scale=0.0 --set lambda_pos_init_prior=1000.0 --save-arrays --no-plot

Combines:
  - MoCap ground truth  (blue solid)
  - Batch settled        (red dashed)   — smooth, from batch solve
  - SW live edge         (orange dotted) — deployment output, from sliding-window solve

Usage:
  cd analysis/
  python ../report/figures/gen_paper_traj.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'analysis'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BAGS = [
    ('slow_racing_best_velocity', 'mocap-init_mocap-heading'),
    ('fast_racing_best_velocity', 'mocap-init_mocap-heading'),
    ('backflips_best_velocity',   'mocap-init_mocap-heading'),
]

PLOTS_ROOT = Path(__file__).parent.parent.parent / 'plots'
OUT_DIR    = Path(__file__).parent.parent / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper font sizes
matplotlib.rcParams.update({
    # Figure is authored at ~9in then displayed at ~3.45in (\columnwidth),
    # a ~0.38x shrink; source fonts are sized so on-page text is ~9-10pt.
    'font.size':        24,
    'axes.labelsize':   24,
    'axes.titlesize':   23,
    'xtick.labelsize':  20,
    'ytick.labelsize':  20,
    'legend.fontsize':  19,
    'lines.linewidth':  2.0,
    'figure.dpi':       150,
})

C_MOCAP  = '#1f77b4'   # blue
C_BATCH  = '#d62728'   # red
C_LIVE   = 'darkorange'

for bag_key, mocap_tag in BAGS:
    bag_dir   = PLOTS_ROOT / bag_key / 'live_solver'
    batch_npz = bag_dir / f'traj_arrays_{bag_key}_{mocap_tag}_batch.npz'
    sw_npz    = bag_dir / f'traj_arrays_{bag_key}_{mocap_tag}_sw.npz'

    if not batch_npz.exists():
        print(f"Missing: {batch_npz}"); continue
    if not sw_npz.exists():
        print(f"Missing: {sw_npz}"); continue

    b = np.load(batch_npz)
    s = np.load(sw_npz)

    gt      = b['mocap']          # (N,3) MoCap positions
    settled = b['settled']        # (N,3) batch estimate
    live    = s.get('live', None) # (M,3) SW live edge, or None
    if live is None and 'live' in s:
        live = s['live']

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    for i, (ax, (xi, yi, xlabel, ylabel, title)) in enumerate(zip(axes, [
        (0, 1, 'X (m)', 'Y (m)', 'X–Y plane'),
        (1, 2, 'Y (m)', 'Z (m)', 'Y–Z plane'),
    ])):
        ax.plot(gt[:, xi],      gt[:, yi],      color=C_MOCAP, lw=2.0,
                linestyle='-',  label='MoCap',          zorder=3)
        ax.plot(settled[:, xi], settled[:, yi], color=C_BATCH, lw=1.6,
                linestyle='--', label='Batch estimate',  zorder=2)
        if live is not None:
            ax.plot(live[:, xi], live[:, yi],   color=C_LIVE,  lw=1.4,
                    linestyle=':', alpha=0.9, label='SW live edge', zorder=4)
        # start markers for all three traces
        ax.plot(gt[0, xi],      gt[0, yi],      'bs', markersize=7, zorder=5)
        ax.plot(settled[0, xi], settled[0, yi], 'rs', markersize=7, zorder=5)
        if live is not None:
            ax.plot(live[0, xi], live[0, yi],   color=C_LIVE, marker='s',
                    markersize=7, linestyle='none', zorder=5)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title)
        # legend only on the first panel
        if i == 0:
            ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    bag_label = bag_key.replace('_best_velocity', '').replace('_', ' ')
    fig.suptitle(f'{bag_label}', fontsize=26, fontweight='bold')
    fig.tight_layout()

    out = OUT_DIR / f'traj_{bag_key}.pdf'
    fig.savefig(out, bbox_inches='tight')
    out_png = OUT_DIR / f'traj_{bag_key}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    print(f"Saved: {out_png}")
    plt.close(fig)
