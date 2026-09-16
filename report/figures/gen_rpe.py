"""
Generate relative-pose-error (RPE) figure and print drift-% summary (Fig 6 in paper).

KITTI-style translational RPE: for each segment length L, all sub-segments of
that accumulated GT distance are enumerated, and the end-point displacement
error is expressed as % of L.  Alignment-independent: the global start-anchored
alignment (R_align, t_align fixed at frame 0) cancels in relative displacements.

Requires the .npz arrays produced by --save-arrays (already present from
the last eval run):
  plots/<bag>/live_solver/traj_arrays_<bag>_mocap-init_mocap-heading_{batch,sw}.npz

Figure layout: two panels side by side.
  Left  — racing bags: batch (solid) and SW live edge (dashed).
  Right — backflips batch only (separate scale; much larger errors).

Also prints a full numeric summary table to stdout for copying into LaTeX tables.

Usage:
    cd report/figures/
    python gen_rpe.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'analysis', 'lib'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType, not Type 3 (IEEE PDF eXpress)
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from pathlib import Path

from odometry_metrics import (
    path_length, drift_percent, translational_rpe, print_summary_table
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PLOTS_ROOT = Path(__file__).parent.parent.parent / 'plots'
OUT_DIR    = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAG = 'mocap-init_mocap-heading'

# Segment lengths to evaluate (metres)
SEG_LENGTHS = [5, 10, 15, 20, 30, 40]
# Only segments supported by path length of each bag (min ~44m)
SEG_LENGTHS_BACK = [5, 10, 20, 30, 40]  # backflips path ≈54m, same

RACING_BAGS = [
    ('slow_racing_best_velocity', 'Slow racing'),
    ('fast_racing_best_velocity', 'Fast racing'),
]
BACKFLIP_BAG = ('backflips_best_velocity', 'Backflips')

# Colours matching gen_error_time.py convention
C = {
    'slow_batch': '#1f77b4',          # blue (batch)
    'slow_live':  '#aec7e8',          # light blue (SW live)
    'fast_batch': '#d62728',          # red (batch)
    'fast_live':  'darkorange',        # orange (SW live)
    'back_batch': '#2ca02c',          # green (batch)
}

matplotlib.rcParams.update({
    # authored ~11in, displayed ~3.45in (~0.31x); fonts sized for ~7-9pt on page
    'font.size':       20,
    'axes.labelsize':  20,
    'axes.titlesize':  20,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,
    'lines.linewidth': 2.0,
    'figure.dpi':      150,
})

# ---------------------------------------------------------------------------
# Load data helper
# ---------------------------------------------------------------------------

def load_bag(bag_key: str, suffix: str):
    """Return npz data dict or None if file missing."""
    f = PLOTS_ROOT / bag_key / 'live_solver' / \
        f'traj_arrays_{bag_key}_{TAG}_{suffix}.npz'
    if not f.exists():
        print(f"Missing (re-run with --save-arrays): {f}")
        return None
    return np.load(f)


def compute_rpe(d, use_live: bool = False):
    """Compute RPE for a loaded npz dict (batch or SW live edge)."""
    if use_live and 'live' in d and 'live_t_rel' in d:
        pos_est = d['live']
        pos_gt  = d['mocap']   # MoCap at full rate; live is sub-sampled
        # Interpolate GT to live timestamps
        from scipy.interpolate import interp1d
        t_full = d['t_rel']
        t_live = d['live_t_rel']
        gt_interp = interp1d(t_full, pos_gt, axis=0,
                             bounds_error=False, fill_value='extrapolate')
        pos_gt_live = gt_interp(t_live)
        return translational_rpe(pos_est, pos_gt_live, SEG_LENGTHS), pos_gt_live
    else:
        return translational_rpe(d['settled'], d['mocap'], SEG_LENGTHS), d['mocap']


# ---------------------------------------------------------------------------
# Numeric summary (printed to stdout for LaTeX tables)
# ---------------------------------------------------------------------------

print("\n" + "="*72)
print("  DRIFT SUMMARY (all bags, all modes)")
print("="*72)
print(f"  {'Bag':<30} {'Mode':<18} {'Path(m)':>8} {'posRMSE':>8} "
      f"{'Drift%':>7} {'velRMSE':>8} {'oriRMSE':>8}")
print("  " + "-"*70)

for bag_key, label in [*RACING_BAGS, BACKFLIP_BAG]:
    for suffix, mode_label in [('batch', 'Batch settled'),
                                 ('sw',    'SW settled'),
                                 ('sw',    'SW live edge')]:
        d = load_bag(bag_key, suffix)
        if d is None:
            continue
        use_live = (mode_label == 'SW live edge')
        if use_live and 'live' not in d:
            continue

        if use_live:
            # live metrics from npz
            pos_rmse = float(np.sqrt(np.mean(d['live_pos_errs']**2)))
            vel_rmse = float(np.sqrt(np.mean(d['live_vel_errs']**2)))
            ori_rmse = float(np.sqrt(np.mean(d['live_rot_errs']**2)))
            # Path length from full GT
            plen = path_length(d['mocap'])
        else:
            pos_rmse = float(np.sqrt(np.mean(d['pos_errors']**2)))
            vel_rmse = float(np.sqrt(np.mean(d['vel_errors']**2)))
            ori_rmse = float(np.sqrt(np.mean(d['rot_errors']**2)))
            plen = path_length(d['mocap'])

        drift = drift_percent(pos_rmse, plen)
        print(f"  {label:<30} {mode_label:<18} {plen:>8.1f} "
              f"{pos_rmse:>8.3f} {drift:>7.2f} {vel_rmse:>8.3f} {ori_rmse:>8.3f}")

print("="*72)

# KITTI RPE table
print("\n" + "="*72)
print("  KITTI TRANSLATIONAL RPE (% of segment length)")
print("="*72)
hdr = f"  {'Bag':<30} {'Mode':<18} " + \
      "".join(f" {L:>5}m" for L in SEG_LENGTHS)
print(hdr)
print("  " + "-"*72)

for bag_key, label in [*RACING_BAGS, BACKFLIP_BAG]:
    for suffix, mode_label in [('batch', 'Batch'), ('sw', 'SW live')]:
        d = load_bag(bag_key, suffix)
        if d is None:
            continue
        use_live = (mode_label == 'SW live')
        if use_live and 'live' not in d:
            continue

        rpe, pos_gt = compute_rpe(d, use_live=use_live)
        vals = "".join(
            f" {v:>5.2f}%" if not np.isnan(v) else f" {'--':>5} "
            for v in rpe['trans_pct']
        )
        print(f"  {label:<30} {mode_label:<18}{vals}")

print("="*72)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, (ax_race, ax_back) = plt.subplots(1, 2, figsize=(11, 4.2),
                                        gridspec_kw={'width_ratios': [2, 1]})

# ---- Left: racing bags ----
ax = ax_race
for bag_key, label in RACING_BAGS:
    is_slow = 'slow' in bag_key
    c_b = C['slow_batch'] if is_slow else C['fast_batch']
    c_l = C['slow_live']  if is_slow else C['fast_live']

    # Batch
    d = load_bag(bag_key, 'batch')
    if d is not None:
        rpe, _ = compute_rpe(d, use_live=False)
        ax.plot(rpe['seg_lengths'], rpe['trans_pct'],
                color=c_b, linestyle='-', marker='o', markersize=5,
                label=f'{label} batch')

    # SW live edge
    d_sw = load_bag(bag_key, 'sw')
    if d_sw is not None and 'live' in d_sw:
        rpe_l, _ = compute_rpe(d_sw, use_live=True)
        ax.plot(rpe_l['seg_lengths'], rpe_l['trans_pct'],
                color=c_l, linestyle='--', marker='s', markersize=5,
                label=f'{label} SW live')

ax.set_xlabel('Segment length (m)')
ax.set_ylabel('RPE (% of segment)')
ax.set_title('Racing')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

# ---- Right: backflips batch only ----
ax = ax_back
bag_key, label = BACKFLIP_BAG
d = load_bag(bag_key, 'batch')
if d is not None:
    rpe, _ = compute_rpe(d, use_live=False)
    ax.plot(rpe['seg_lengths'], rpe['trans_pct'],
            color=C['back_batch'], linestyle='-', marker='o', markersize=5,
            label=f'{label} batch')

ax.set_xlabel('Segment length (m)')
ax.set_ylabel('RPE (% of segment)')
ax.set_title('Backflips')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

fig.tight_layout()

out_pdf = OUT_DIR / 'rpe_drift.pdf'
out_png = OUT_DIR / 'rpe_drift.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")
plt.close(fig)
