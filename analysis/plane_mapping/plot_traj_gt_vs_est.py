#!/usr/bin/env python3
"""
GT vs iSAM live-edge estimated trajectory, all three bags, x-y and y-z views.
For --isam the estimate IS the live edge (per-knot last value before marginalization),
saved as 'settled' on the same eval-time base as 'mocap' (GT). Run after
validate_live_solver.py ... --isam --save-arrays for each bag.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent / 'plots'
BAGS = [
    ("slow_racing_best_velocity", "slow_racing"),
    ("fast_racing_best_velocity", "fast_racing"),
    ("backflips_best_velocity",   "backflips"),
]
TAG = "_mocap-init_mocap-heading_batch"

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for col, (bag, title) in enumerate(BAGS):
    npz = ROOT / bag / 'live_solver' / f'traj_arrays_{bag}{TAG}.npz'
    d = np.load(npz)
    gt = d['mocap']          # GT (Vicon), aligned eval frame
    est = d['settled']       # iSAM live-edge estimate, aligned to GT
    err = float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))

    for row, (ix, iy, lab) in enumerate([(0, 1, 'x-y (top-down)'), (1, 2, 'y-z (side)')]):
        ax = axes[row, col]
        ax.plot(gt[:, ix],  gt[:, iy],  '-', color='k',       lw=1.6, label='Ground truth (Vicon)')
        ax.plot(est[:, ix], est[:, iy], '-', color='tab:red', lw=1.1, alpha=0.85, label='iSAM live edge')
        ax.plot(gt[0, ix], gt[0, iy], 'o', color='limegreen', ms=7, zorder=5, label='start')
        xl = 'x (m)' if row == 0 else 'y (m)'
        yl = 'y (m)' if row == 0 else 'z (m)'
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(f"{title}\nATE {err:.2f} m   |   {lab}", fontsize=11)
        else:
            ax.set_title(lab, fontsize=10)
        if row == 0 and col == 0:
            ax.legend(fontsize=8, loc='best')

fig.suptitle("Ground truth vs iSAM live-edge trajectory (mocap-aligned) -- floor-plane anchor ON (universal: free-offset + cluster, no per-bag tuning)",
             fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = Path(__file__).parent / "traj_gt_vs_est_all3.png"
plt.savefig(out, dpi=140)
print(f"Saved {out}")
