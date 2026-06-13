#!/usr/bin/env python3
"""B1 diagnostic (paper review pass): measure the RAW WLS world-frame ego-velocity
bias on an ICINS flight, isolated from the spline solver, using GT attitude.

For each radar frame we solve the ego-velocity with several front-ends (plain
intensity-WLS = what our pipeline uses; Huber; a reve-style RANSAC), rotate it to
the world frame via the *ground-truth* attitude, and compare to GT world velocity.
The mean per-axis error is the systematic velocity bias; integrated over the flight
it should reproduce the observed position drift. This localises whether the 16.9 m
vertical drift is an input (WLS front-end) systematic and whether outlier rejection
(RANSAC, as reve uses) or tighter elevation gating removes it.

Usage (from analysis/):
  ../.venv/bin/python3 diagnostics/icins_zbias_probe.py icins_flight_1
"""
import sys
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))
from rosbag_loader import load_bag_topics                       # noqa: E402
from radar_velocity_utils import (solve_ego_velocity_weighted,  # noqa: E402
                                   rotation_matrix_from_euler)

REPO = Path(__file__).resolve().parents[2]


def ransac_ego_velocity(P, v, intens, min_intensity=5.0, min_range=0.2,
                        thresh=0.15, iters=150, seed=0):
    """reve-style 3D LSQ RANSAC (inlier_thresh 0.15 m/s as in their config)."""
    m = (intens >= min_intensity) & (np.linalg.norm(P, axis=1) >= min_range)
    P, v = P[m], v[m]
    if len(v) < 5:
        return None
    H = P / np.linalg.norm(P, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    best_in, best_v = None, None
    for _ in range(iters):
        idx = rng.choice(len(v), 3, replace=False)
        try:
            vc = np.linalg.solve(H[idx], v[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(H @ vc - v) < thresh
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best_v = inl, vc
    if best_in is None or best_in.sum() < 5:
        return best_v
    Hi, vi = H[best_in], v[best_in]                     # refit (unweighted LSQ on inliers)
    return np.linalg.solve(Hi.T @ Hi, Hi.T @ vi)


def main():
    alias = sys.argv[1] if len(sys.argv) > 1 else 'icins_flight_1'
    bags = yaml.safe_load((REPO / 'analysis/config/bags.yaml').read_text())
    bag_rel = bags['bags'][alias]
    bag = (REPO / bag_rel).resolve()
    if not bag.exists():
        bag = (REPO / '..' / bag_rel).resolve()
    start_off, dur = bags['timing'][alias]
    # Extrinsics (R_body_from_sensor). Default = ICINS published calib (their horizontal
    # mount, converted). Override with --euler r,p,y for our own pitched mount, e.g.
    # --euler 180,27.5,0 for the racing/backflip bags.
    euler = [-178.501, -0.099, 46.997]
    if '--euler' in sys.argv:
        euler = [float(x) for x in sys.argv[sys.argv.index('--euler') + 1].split(',')]
    R_bs = rotation_matrix_from_euler(*np.radians(euler))     # R_body_from_sensor
    print(f"  extrinsic euler (R_bs) = {euler}")

    bd = load_bag_topics(str(bag), verbose=False)
    t0 = bd.start_time + start_off
    t1 = t0 + dur - 3.0

    gt = [s for s in bd.agiros_state if t0 <= s.timestamp <= t1]
    gt_t = np.array([s.timestamp for s in gt])
    gt_v = np.array([s.velocity for s in gt])                 # world frame
    gt_R = Rotation.from_quat(np.array([s.orientation for s in gt]))  # body->world (xyzw)
    slerp = Slerp(gt_t, gt_R)
    vfun = [lambda tt, d=d: np.interp(tt, gt_t, gt_v[:, d]) for d in range(3)]

    methods = {
        'plain_WLS': dict(use_huber=False),
        'Huber':     dict(use_huber=True, huber_delta=1.0),
        'RANSAC':    'ransac',
    }
    # also test tighter elevation gates applied on top (deg from radar horizon)
    elev_gates = [None, 45.0, 30.0]

    frames = [f for f in bd.radar_velocity
              if t0 <= f.timestamp <= t1 and f.velocities is not None]
    print(f"{alias}: {len(frames)} radar frames in window [{start_off:.0f},"
          f"{start_off+dur-3:.0f}]s, {len(gt_t)} GT samples")
    print(f"{'method':<12}{'gate':>6} {'n':>5}  "
          f"{'bias vx':>8}{'vy':>8}{'vz':>8}  {'drift x':>8}{'y':>8}{'z':>8} (m)")

    for gate in elev_gates:
        for name, cfg in methods.items():
            errs, ts = [], []
            for f in frames:
                P = np.asarray(f.positions, float)
                v = np.asarray(f.velocities, float)
                I = np.asarray(f.intensities, float)
                if gate is not None:
                    rho = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
                    keep = np.degrees(np.abs(np.arctan2(P[:, 2], rho))) < gate
                    P, v, I = P[keep], v[keep], I[keep]
                if len(v) < 5:
                    continue
                if cfg == 'ransac':
                    v_wls = ransac_ego_velocity(P, v, I)
                else:
                    v_wls = solve_ego_velocity_weighted(P, v, I, **cfg)
                if v_wls is None:
                    continue
                R_wb = slerp(min(max(f.timestamp, gt_t[0]), gt_t[-1])).as_matrix()
                v_world = R_wb @ (R_bs @ (-v_wls))
                v_gt = np.array([vfun[d](f.timestamp) for d in range(3)])
                errs.append(v_world - v_gt)
                ts.append(f.timestamp)
            if len(errs) < 10:
                print(f"{name:<12}{str(gate):>6} {len(errs):>5}  (too few)")
                continue
            errs = np.array(errs); ts = np.array(ts)
            bias = errs.mean(axis=0)
            drift = np.array([np.trapz(errs[:, d], ts) for d in range(3)])  # integral of vel error
            g = 'none' if gate is None else f'{gate:.0f}'
            print(f"{name:<12}{g:>6} {len(errs):>5}  "
                  f"{bias[0]:>8.3f}{bias[1]:>8.3f}{bias[2]:>8.3f}  "
                  f"{drift[0]:>8.1f}{drift[1]:>8.1f}{drift[2]:>8.1f}")


if __name__ == '__main__':
    main()
