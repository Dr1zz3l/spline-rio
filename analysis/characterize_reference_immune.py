"""Reference-immune re-measurement of the radar error structure, plus the
selection controls the Gaussian-core ladder needs.

Chapter 5 of the story notebook introduces a two-component decomposition,
r_fj = c_f + e_fj, where c_f is shared by a whole radar frame (reference
velocity error, timestamp error, scene-wide effects) and e_fj is per-point.
Deviations from a frame's own mean cancel c_f EXACTLY, which makes the
within-frame statistic immune to any error of the MoCap reference.

That estimator is currently used for one thing only (the dynamics law).  Every
other per-point effect in the story -- elevation, signal strength, the aliased
error floor, and the identity of the post-ladder tail -- is measured on the
TOTAL residual, which the notebook itself calls an upper bound.  All four of
those quantities vary WITHIN a frame, so all four can be measured with the
reference-immune statistic instead.  This script does that, and adds:

  (1) SELECTION CONTROLS for the ladder.  Rung S3 deletes RANSAC consensus
      rejects.  A consensus inlier test IS a residual test, so some of the
      Gaussianization it produces is selection rather than discovery.  This
      brackets how much, by comparing S3 against two same-rate nulls: random
      deletion (no information) and an oracle |r| cut (the most a pure residual
      cut could ever achieve).

  (2) THE ORIGIN OF sigma_c.  c_f is defined to include timestamp error, and
      the frame timestamps have measured jitter.  A frame arriving dt early or
      late mispredicts every one of its returns by about (radial acceleration)
      x dt, which is exactly a frame-shared error.  This compares the predicted
      timing contribution against the measured sigma_c per bag.

Run from analysis/:  ../.venv/bin/python3 characterize_reference_immune.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy import stats as sp_stats
from scipy.interpolate import interp1d

from config_loader import load_config
from rosbag_loader.loader import (load_bag_topics,
                                  stitch_cpu_counter_resets_improved)
from radar_velocity_utils import (compute_doppler_residuals,
                                  rotation_matrix_from_euler, unwrap_doppler,
                                  quat_to_rotation_matrix)

_cfg = load_config()
BAGS = _cfg['bags']['bags']
TIMING = _cfg['bags']['timing']
RC = _cfg['bags'].get('radar_config', {})
EXT = _cfg['extrinsics']
R_BS = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
T_BS = np.array(EXT['translation_body_m'])
RADAR_OFF = EXT['imu_mocap_offset_sec'] - EXT['radar_imu_offset_sec']
THRESH = 0.15                      # deployed radar_ransac_threshold

BAG_KEYS = ['slow_racing_best_velocity', 'fast_racing_best_velocity',
            'backflips_best_velocity']
SHORT = {'slow_racing_best_velocity': 'slow racing',
         'fast_racing_best_velocity': 'fast racing',
         'backflips_best_velocity': 'backflips'}


def robust_sigma(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def qq_r2(x, zmax=3.0):
    """Straightness of the Q-Q plot inside |z| < zmax (1.0 = Gaussian)."""
    x = np.asarray(x)
    s = robust_sigma(x)
    if not np.isfinite(s) or s <= 0 or len(x) < 50:
        return np.nan
    z = (x - np.median(x)) / s
    zc = z[np.abs(z) < zmax]
    if len(zc) < 50:
        return np.nan
    (_, _), (_, _, r) = sp_stats.probplot(zc, dist='norm', plot=None)
    return r ** 2


def ransac_mask(P, v, rng, thresh=THRESH, iters=150):
    n = len(v)
    if n < 5:
        return np.ones(n, dtype=bool)
    Hn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
    best = None
    for _ in range(iters):
        i3 = rng.choice(n, 3, replace=False)
        try:
            vc = np.linalg.solve(Hn[i3], v[i3])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(Hn @ vc - v) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    return best if best is not None and best.sum() >= 5 else np.ones(n, bool)


def load(key, repo):
    v_max = RC.get('best_velocity', {}).get('v_max', 3.136)
    data = load_bag_topics(str(repo / BAGS[key]), verbose=False)
    t0 = data.start_time + TIMING[key][0]
    t1 = t0 + TIMING[key][1]
    radar = [f for f in data.radar_velocity
             if t0 <= f.timestamp <= t1 and f.positions is not None]
    states = data.agiros_state
    dres = compute_doppler_residuals(states, radar, T_BS, R_BS,
                                     time_offset=RADAR_OFF, min_range=0.2)
    unw = unwrap_doppler(dres['measurements'], dres['predictions'], v_max)
    r = unw - dres['predictions']
    fi = np.array(dres['frame_indices'])
    pi = np.array(dres['point_indices'])
    st_t = np.array([s.timestamp for s in states])

    d = dict(radar=radar, r=r, fi=fi, pi=pi, unw=unw,
             aliased=unw != dres['measurements'],
             inten=np.asarray(dres['intensities'], float),
             u_z=np.array([radar[f].positions[p][2]
                           / np.linalg.norm(radar[f].positions[p])
                           for f, p in zip(fi, pi)]),
             t_pts=np.array([radar[i].timestamp for i in fi]) + RADAR_OFF,
             st_t=st_t, states=states,
             a_i=interp1d(st_t, np.array([s.acceleration for s in states]),
                          axis=0, kind='linear', bounds_error=False,
                          fill_value='extrapolate')
             if states[0].acceleration is not None else None,
             q_i=interp1d(st_t, np.array([s.orientation for s in states]),
                          axis=0, kind='linear', bounds_error=False,
                          fill_value='extrapolate'),
             v_i=interp1d(st_t, np.array([s.velocity for s in states]),
                          axis=0, kind='linear', bounds_error=False,
                          fill_value='extrapolate'),
             radar_full=list(data.radar_velocity))
    rng = np.random.default_rng(0)
    d['masks'] = {i: ransac_mask(np.asarray(f.positions, float),
                                 _unw_frame(d, i), rng)
                  for i, f in enumerate(radar) if f.num_points() >= 5}
    del data
    return d


def _unw_frame(d, i):
    v = np.asarray(d['radar'][i].velocities, float).copy()
    sel = d['fi'] == i
    v[d['pi'][sel]] = d['unw'][sel]
    return v


def point_mask_from_frames(d, frame_masks):
    ok = np.ones(len(d['r']), bool)
    for i, m in frame_masks.items():
        for j in np.flatnonzero(d['fi'] == i):
            if d['pi'][j] < len(m):
                ok[j] = m[d['pi'][j]]
    return ok


def deviations(d, sel, ref_sel=None, min_n=6):
    """Within-frame deviations of the points in `sel`, using the frame mean of
    `ref_sel` (defaults to `sel`).  The frame-shared term cancels exactly."""
    if ref_sel is None:
        ref_sel = sel
    out_d, out_idx = [], []
    for f in np.unique(d['fi']):
        mref = (d['fi'] == f) & ref_sel
        mval = (d['fi'] == f) & sel
        if mref.sum() < min_n or mval.sum() == 0:
            continue
        mu = d['r'][mref].mean()
        n = int(mref.sum())
        # undo the shrinkage a mean over n points imposes on its own members
        scale = np.sqrt(1.0 - 1.0 / n) if np.array_equal(mref, mval) else 1.0
        out_d.append((d['r'][mval] - mu) / scale)
        out_idx.append(np.flatnonzero(mval))
    if not out_d:
        return np.array([]), np.array([], dtype=int)
    return np.concatenate(out_d), np.concatenate(out_idx)


def binned(x, y, edges, min_n=60):
    c, s, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_n:
            c.append(float(x[m].mean()))
            s.append(robust_sigma(y[m]))
            n.append(int(m.sum()))
    return np.array(c), np.array(s), np.array(n)


# ============================================================ (1) ladder nulls
def ladder_controls(d, rng):
    """S2 baseline, S3 (consensus), and two same-rate nulls."""
    thin = np.array([d['radar'][i].num_points() < 5 for i in d['fi']])
    S2 = ~d['aliased'] & ~thin
    S3 = S2 & point_mask_from_frames(d, d['masks'])

    # the per-frame deletion rate S3 actually applied, replayed by two nulls
    rand_ok = np.ones(len(d['r']), bool)
    oracle_ok = np.ones(len(d['r']), bool)
    for f in np.unique(d['fi']):
        idx = np.flatnonzero((d['fi'] == f) & S2)
        if len(idx) == 0:
            continue
        n_drop = int(len(idx) - S3[idx].sum())
        if n_drop <= 0:
            continue
        rand_ok[rng.choice(idx, n_drop, replace=False)] = False
        # oracle: drop the n_drop LARGEST |r| in the frame -- the most any
        # residual-magnitude rule could possibly achieve
        oracle_ok[idx[np.argsort(-np.abs(d['r'][idx]))[:n_drop]]] = False
    return dict(S2=S2, S3=S3, RAND=S2 & rand_ok, ORACLE=S2 & oracle_ok)


def heldout_consensus(d, rng, min_n=10):
    """Split each frame: build the consensus on one half, judge the other half.

    The judged points never influence the model that judges them, so any
    Gaussianization measured on them is not model-fitting on themselves.
    (The inlier test still reads each judged point's own residual -- that is
    what a consensus test IS -- so this controls for model contamination, not
    for residual thresholding, which the ORACLE row brackets.)
    """
    keep, allpts = [], []
    for f in np.unique(d['fi']):
        sel = np.flatnonzero(d['fi'] == f)
        i = int(f)
        if len(sel) < min_n or i not in d['masks']:
            continue
        P = np.asarray(d['radar'][i].positions, float)
        v = _unw_frame(d, i)
        n = len(v)
        perm = rng.permutation(n)
        fit_i, test_i = perm[:n // 2], perm[n // 2:]
        if len(fit_i) < 5:
            continue
        m_fit = ransac_mask(P[fit_i], v[fit_i], rng)
        if m_fit.sum() < 3:
            continue
        Hn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
        A = Hn[fit_i][m_fit]
        vc, *_ = np.linalg.lstsq(A, v[fit_i][m_fit], rcond=None)
        inl = np.abs(Hn[test_i] @ vc - v[test_i]) < THRESH
        for pidx, ok in zip(test_i, inl):
            hit = sel[d['pi'][sel] == pidx]
            if len(hit):
                allpts.append(hit[0])
                if ok:
                    keep.append(hit[0])
    return np.array(allpts, dtype=int), np.array(keep, dtype=int)


def main():
    repo = Path(__file__).resolve().parent.parent
    D = {k: load(k, repo) for k in BAG_KEYS}

    print('=' * 84)
    print('(1) LADDER SELECTION CONTROLS: how much of rung S3 is discovery,')
    print('    how much is a residual cut?  All rows delete the SAME number of')
    print('    points per frame, starting from S2 (alias- and thin-frame-cut).')
    print('=' * 84)
    print(f"{'bag':<13} {'variant':<22} {'kept%':>6} {'sigma_rob':>10} "
          f"{'>3sig':>7} {'QQ R2':>7}")
    for k in BAG_KEYS:
        d = D[k]
        sets = ladder_controls(d, np.random.default_rng(0))
        for tag, label in [('S2', 'S2 (before consensus)'),
                           ('RAND', 'null: random deletion'),
                           ('S3', 'S3 = RANSAC consensus'),
                           ('ORACLE', 'null: oracle |r| cut')]:
            sel = sets[tag]
            rr = d['r'][sel]
            z = (rr - np.median(rr)) / robust_sigma(rr)
            print(f'{SHORT[k]:<13} {label:<22} '
                  f'{100*sel.sum()/len(d["r"]):>5.0f}% {robust_sigma(rr):>10.3f} '
                  f'{100*np.mean(np.abs(z) > 3):>6.1f}% {qq_r2(rr):>7.4f}')
        a, kp = heldout_consensus(d, np.random.default_rng(1))
        if len(kp) > 100:
            print(f'{"":<13} {"held-out consensus":<22} '
                  f'{100*len(kp)/max(len(a),1):>5.0f}% '
                  f'{robust_sigma(d["r"][kp]):>10.3f} '
                  f'{100*np.mean(np.abs((d["r"][kp]-np.median(d["r"][kp]))/robust_sigma(d["r"][kp])) > 3):>6.1f}% '
                  f'{qq_r2(d["r"][kp]):>7.4f}   '
                  f'(judged on {len(a)} held-out points)')
        print()

    print('=' * 84)
    print('(2) WHAT THE POST-LADDER TAIL IS: per-point or frame-shared?')
    print('    A consensus-passing point can only carry a large ground-truth')
    print('    residual if its whole FRAME is off, so the surviving tail should')
    print('    be frame-shared.  If it is, a per-point kernel (Huber) is the')
    print('    wrong tool for it and frame-level robustness is the right one.')
    print('=' * 84)
    print(f"{'bag':<13} {'S3 sigma_i':>11} {'S3 sigma_c':>11} "
          f"{'tail |z|>3':>11} {'of which frame-shared':>22}")
    for k in BAG_KEYS:
        d = D[k]
        sets = ladder_controls(d, np.random.default_rng(0))
        S3 = sets['S3']
        dev, idx = deviations(d, S3)
        if not len(dev):
            continue
        s_i = robust_sigma(dev)
        # frame means of the kept set: their scatter is sigma_c^2 + sigma_i^2/N
        mus, ns = [], []
        for f in np.unique(d['fi']):
            m = (d['fi'] == f) & S3
            if m.sum() >= 6:
                mus.append(d['r'][m].mean())
                ns.append(int(m.sum()))
        mus, ns = np.array(mus), np.array(ns)
        # robust scale on BOTH sides: pairing a MAD-based sigma_i with a
        # plain variance of the frame means inflates sigma_c, because the
        # latter is dragged by the very frames that contain outliers
        s_c = np.sqrt(max(robust_sigma(mus) ** 2 - s_i ** 2 / ns.mean(), 0.0))
        # of the S3 points beyond 3 sigma of the TOTAL residual, how many are
        # explained by their frame's mean rather than their own deviation?
        rr = d['r'][S3]
        z_tot = (rr - np.median(rr)) / robust_sigma(rr)
        tail = np.abs(z_tot) > 3
        z_dev = np.full(len(rr), np.nan)
        pos = {v: i for i, v in enumerate(np.flatnonzero(S3))}
        for val, ii in zip(dev, idx):
            if ii in pos:
                z_dev[pos[ii]] = val / s_i
        both = tail & np.isfinite(z_dev)
        frac = (np.mean(np.abs(z_dev[both]) < 3) if both.sum() else np.nan)
        print(f'{SHORT[k]:<13} {s_i:>11.3f} {s_c:>11.3f} '
              f'{100*np.mean(tail):>10.1f}% {100*frac:>21.0f}%')
    print('  (last column: share of the tail whose OWN within-frame deviation is')
    print('   normal -- i.e. the point is only an outlier because its frame is)')

    print()
    print('=' * 84)
    print('(3) REFERENCE-IMMUNE RE-MEASUREMENT of effects the story currently')
    print('    reads off the total residual')
    print('=' * 84)
    for k in BAG_KEYS:
        d = D[k]
        clean = ~d['aliased']
        dev, idx = deviations(d, clean)
        if not len(dev):
            continue
        print(f'-- {SHORT[k]}')
        c, s, _ = binned(d['u_z'][idx], dev, np.linspace(-1, 1, 9))
        print('   sigma_i vs elevation u_z : '
              + '  '.join(f'{a:+.2f}:{b:.3f}' for a, b in zip(c, s)))
        rel = d['inten'][idx] / np.array(
            [np.median(d['inten'][d['fi'] == f]) for f in d['fi'][idx]])
        e = np.unique(np.percentile(rel, np.linspace(0, 100, 7)))
        c, s, _ = binned(rel, dev, e)
        if len(c) > 2:
            al = np.polyfit(np.log(c), np.log((np.interp(1.0, c, s) / s) ** 2), 1)[0]
            print('   sigma_i vs I/I_med       : '
                  + '  '.join(f'{a:.2f}:{b:.3f}' for a, b in zip(c, s))
                  + f'   -> alpha_hat {al:+.2f}')
        # aliased vs clean at MATCHED speed, both measured against the CLEAN
        # frame mean so the comparison is like-for-like and reference-immune
        v_pts = np.linalg.norm(d['v_i'](d['t_pts']), axis=1)
        dev_al, idx_al = deviations(d, d['aliased'], ref_sel=clean)
        if len(dev_al) > 60:
            edges = np.array([0, 2, 3, 4, 5, 7])
            ca, sa, na = binned(v_pts[idx_al], dev_al, edges, min_n=40)
            cc, sc, nc = binned(v_pts[idx], dev, edges, min_n=40)
            print('   aliased sigma vs |v|     : '
                  + '  '.join(f'{a:.1f}:{b:.3f}' for a, b in zip(ca, sa)))
            print('   clean   sigma vs |v|     : '
                  + '  '.join(f'{a:.1f}:{b:.3f}' for a, b in zip(cc, sc)))

    print()
    print('=' * 84)
    print('(4) WHERE sigma_c COMES FROM: is the frame-shared error just frame')
    print('    timestamp jitter acting on the radial acceleration?')
    print('=' * 84)
    print(f"{'bag':<13} {'jitter sigma':>13} {'rms radial accel':>17} "
          f"{'predicted sigma_c':>18} {'measured sigma_c':>17}")
    for k in BAG_KEYS:
        d = D[k]
        if d['a_i'] is None:
            continue
        stitched, _ = stitch_cpu_counter_resets_improved(d['radar_full'],
                                                         verbose=False)
        tr = np.array([f.timestamp for f in stitched
                       if f.time_cpu_cycles is not None
                       and len(f.time_cpu_cycles) and f.time_cpu_cycles[0] > 0])
        cc = np.array([f.time_cpu_cycles[0] for f in stitched
                       if f.time_cpu_cycles is not None
                       and len(f.time_cpu_cycles) and f.time_cpu_cycles[0] > 0])
        sl, ic, *_ = sp_stats.linregress(cc, tr)[:2] + (0,)
        jit = tr - (sl * cc + ic)
        s_jit = robust_sigma(jit)

        # radial acceleration seen by each return: u . a  (world frame)
        clean = ~d['aliased']
        dev, idx = deviations(d, clean)
        a_w = d['a_i'](d['t_pts'][idx])
        u_s = np.array([d['radar'][f].positions[p]
                        / np.linalg.norm(d['radar'][f].positions[p])
                        for f, p in zip(d['fi'][idx], d['pi'][idx])])
        R_wb = np.array([quat_to_rotation_matrix(q)
                         for q in d['q_i'](d['t_pts'][idx])])
        u_w = np.einsum('nij,nj->ni', R_wb, (R_BS @ u_s.T).T)
        radial_a = np.abs(np.einsum('ni,ni->n', u_w, a_w))
        pred = float(np.sqrt(np.mean(radial_a ** 2))) * s_jit

        s_i = robust_sigma(dev)
        mus, ns = [], []
        for f in np.unique(d['fi']):
            m = (d['fi'] == f) & clean
            if m.sum() >= 6:
                mus.append(d['r'][m].mean())
                ns.append(int(m.sum()))
        mus, ns = np.array(mus), np.array(ns)
        # robust scale on BOTH sides: pairing a MAD-based sigma_i with a
        # plain variance of the frame means inflates sigma_c, because the
        # latter is dragged by the very frames that contain outliers
        s_c = np.sqrt(max(robust_sigma(mus) ** 2 - s_i ** 2 / ns.mean(), 0.0))
        print(f'{SHORT[k]:<13} {1e3*s_jit:>11.2f}ms '
              f'{np.sqrt(np.mean(radial_a**2)):>14.2f}m/s2 '
              f'{pred:>17.3f} {s_c:>17.3f}')
    print('  (robust jitter sigma is used: the raw sigma is dominated by rare')
    print('   one-sided delay spikes, which are not a per-frame random error)')


if __name__ == '__main__':
    main()
