#!/usr/bin/env python3
"""Ladder rung S2/S3 under two conventions for sparse ("thin") frames.

The deployed front end does NOT delete frames with fewer than five returns: the
RANSAC consensus needs five, so such a frame bypasses it and every one of its
returns is kept (validate_live_solver.py `if n < 5: return ones`). The published
ladder, however, deletes those returns from the statistic before measuring
(notebook 12 cell 12, `SEL2 = SEL1 & ~thin`), so the "after prefilter" row
describes a stream slightly cleaner than the one the solver consumes.

This script measures both, so the report can quote the deployed convention:

  A  published : S2 = clean minus thin-frame returns, S3 = S2 minus consensus
  B  deployed  : S3' = clean minus consensus rejects only; thin frames pass
                 through whole, exactly as the front end lets them

Protocol copied from notebook 12 cells 1, 2, 10 and 12 (same extrinsics, same
timing windows, same ground-truth de-aliasing, same shared rng seeded 0 and
consumed in bag/frame order, so arm A reproduces the published numbers).

    ../.venv/bin/python3 characterize_thin_frame_convention.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = REPO_ROOT / 'analysis'
for p in (ANALYSIS, ANALYSIS / 'lib'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from scipy import stats as sp_stats

from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (compute_doppler_residuals,
                                  rotation_matrix_from_euler, unwrap_doppler)

BAG_KEYS = ['slow_racing_best_velocity', 'fast_racing_best_velocity',
            'backflips_best_velocity']
SHORT = {BAG_KEYS[0]: 'slow racing', BAG_KEYS[1]: 'fast racing',
         BAG_KEYS[2]: 'backflips'}
THRESH = 0.15   # solver.yaml radar_ransac_threshold


def robust_sigma(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def ladder_stats(r):
    """n, sigma_rob, tail mass beyond 3 sigma_rob (%), Q-Q R^2 inside |z|<3."""
    z = (r - np.median(r)) / robust_sigma(r)
    zc = z[np.abs(z) < 3]
    (_, _), (_, _, rv) = sp_stats.probplot(zc, dist='norm', plot=None)
    return len(r), robust_sigma(r), 100 * np.mean(np.abs(z) > 3), rv ** 2


def main():
    cfg = load_config()
    bags, timing = cfg['bags']['bags'], cfg['bags']['timing']
    rc, ext = cfg['bags'].get('radar_config', {}), cfg['extrinsics']
    R_BS = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
    T_BS = np.array(ext['translation_body_m'])
    radar_off = ext['imu_mocap_offset_sec'] - ext['radar_imu_offset_sec']

    rng = np.random.default_rng(0)          # shared, as in the notebook

    def ransac_mask(P, v):
        n = len(v)
        if n < 5:
            return np.ones(n, dtype=bool)
        Hn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
        best = None
        for _ in range(150):
            i3 = rng.choice(n, 3, replace=False)
            try:
                vc = np.linalg.solve(Hn[i3], v[i3])
            except np.linalg.LinAlgError:
                continue
            inl = np.abs(Hn @ vc - v) < THRESH
            if best is None or inl.sum() > best.sum():
                best = inl
        return best if best is not None and best.sum() >= 5 else np.ones(n, bool)

    rows = []
    for key in BAG_KEYS:
        v_max = rc.get('best_velocity' if 'best_velocity' in key
                       else 'default', {}).get('v_max', 4.99)
        data = load_bag_topics(str(REPO_ROOT / bags[key]), verbose=False)
        t0 = data.start_time + timing[key][0]
        t1 = t0 + timing[key][1]
        radar = [f for f in data.radar_velocity
                 if t0 <= f.timestamp <= t1 and f.positions is not None]
        dres = compute_doppler_residuals(data.agiros_state, radar, T_BS, R_BS,
                                         time_offset=radar_off, min_range=0.2)
        meas, pred = dres['measurements'], dres['predictions']
        unw = unwrap_doppler(meas, pred, v_max)
        r = unw - pred
        fi = np.array(dres['frame_indices'])
        pi = np.array(dres['point_indices'])
        aliased = (unw != meas)

        masks = {}
        for i, f in enumerate(radar):
            if f.num_points() < 5:
                continue
            v = np.asarray(f.velocities, float).copy()
            sel = fi == i
            v[pi[sel]] = unw[sel]
            masks[i] = ransac_mask(np.asarray(f.positions, float), v)

        thin = np.array([radar[i].num_points() < 5 for i in fi])
        ok = np.ones(len(r), bool)
        for i, m in masks.items():
            for j in np.where(fi == i)[0]:
                if pi[j] < len(m):
                    ok[j] = m[pi[j]]

        S1 = ~aliased                       # clean population
        S3_pub = S1 & ~thin & ok            # arm A, as published
        S3_dep = S1 & ok                    # arm B, as deployed

        n_frames = len(radar)
        n_thin = sum(1 for f in radar if f.num_points() < 5)
        rows.append((key, r, S1, S3_pub, S3_dep, thin, n_frames, n_thin))
        del data

    print(f"\nRANSAC runs on frames with >=5 returns; thin frames bypass it "
          f"and keep every return.\n")
    print(f"{'bag':<13}{'frames':>7}{'thin':>6}{'runs on':>9}"
          f"{'thin pts':>10}")
    for key, r, S1, _, _, thin, nf, nt in rows:
        print(f"{SHORT[key]:<13}{nf:>7}{nt:>6}{100*(nf-nt)/nf:>8.0f}%"
              f"{100*thin[S1].mean():>9.1f}%")

    for label, idx in (('A  published (thin returns deleted)', 3),
                       ('B  deployed  (thin returns kept, bypass)', 4)):
        print(f"\n{label}")
        print(f"  {'bag':<13}{'n':>7}{'kept':>7}{'removed':>9}"
              f"{'sigma_rob':>11}{'tail':>7}{'QQ R2':>8}")
        for row in rows:
            key, r, S1, = row[0], row[1], row[2]
            SEL = row[idx]
            n, sz, f3, r2 = ladder_stats(r[SEL])
            base = S1.sum()
            print(f"  {SHORT[key]:<13}{n:>7}{100*n/base:>6.0f}%"
                  f"{100*(1-n/base):>8.1f}%{sz:>11.3f}{f3:>6.1f}%{r2:>8.4f}")

    print("\nclean population before the consensus (S1), for the tail-mass ratio")
    print(f"  {'bag':<13}{'n':>7}{'sigma_rob':>11}{'tail':>7}")
    for key, r, S1, _, _, _, _, _ in rows:
        n, sz, f3, _ = ladder_stats(r[S1])
        print(f"  {SHORT[key]:<13}{n:>7}{sz:>11.3f}{f3:>6.1f}%")


if __name__ == '__main__':
    main()
