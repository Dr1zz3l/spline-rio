#!/usr/bin/env python3
"""
Phase 0 (design-study gate): per-axis x/y/z ATE + drift of the iSAM live-edge
estimate vs MoCap, to test the study's premise that x (the long hall axis) is the
dominant unbounded residual. Reads the --save-arrays npz (mocap = GT, settled =
iSAM live edge, both SE3-aligned to GT, so axes are the MoCap world frame:
x = long hall axis ~21 m, y = across-to-wall ~8 m, z = vertical).
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent / 'plots'
TAG = "_mocap-init_mocap-heading_batch"
BAGS = [("slow_racing_best_velocity", "slow"),
        ("fast_racing_best_velocity", "fast"),
        ("backflips_best_velocity",   "backflips")]

label = sys.argv[1] if len(sys.argv) > 1 else "(current arrays)"
print(f"=== Phase 0 per-axis ATE  [{label}] ===")
print(f"{'bag':10s} {'path':>6s} | {'x(long)':>9s} {'y(wall)':>9s} {'z(vert)':>9s} "
      f"{'horiz':>8s} {'total':>8s} | dominant")
for bag, short in BAGS:
    npz = ROOT / bag / 'live_solver' / f'traj_arrays_{bag}{TAG}.npz'
    if not npz.exists():
        print(f"{short:10s}  MISSING {npz}"); continue
    d = np.load(npz)
    gt, est = d['mocap'], d['settled']
    e = est - gt                                   # per-axis error, GT frame
    rms = np.sqrt(np.mean(e**2, axis=0))           # x,y,z RMSE
    horiz = np.sqrt(rms[0]**2 + rms[1]**2)
    total = np.sqrt(np.sum(rms**2))
    path_len = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    dom = ['x(long)', 'y(wall)', 'z(vert)'][int(np.argmax(rms))]
    print(f"{short:10s} {path_len:5.1f}m | {rms[0]:9.3f} {rms[1]:9.3f} {rms[2]:9.3f} "
          f"{horiz:8.3f} {total:8.3f} | {dom}  ({100*rms.max()/total:.0f}% of total)")
