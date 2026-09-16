"""Per-POINT radar noise: does the translational-smear mechanism hold per return?

SUPERSEDED IN PART (2026-08-06, characterize_shape_contradiction.py +
worklog/reports/MECHANISM_VERDICT_2026-08-06.md).  This script's conclusion -- that the noise is
a bearing error and not translational smear -- STANDS, and its exact |v_r| null
is still the sharpest control in the chapter.  Two things below are retracted:
the quadratic speed law quoted in the next paragraph as established (free
exponents on the same statistic give p = 1, not 2), and the fitted constant
dphi = 3.5 deg, which is inflated because NO rung in this ladder lets the ray
error grow off boresight.  With that factor the constant is ~2 deg at boresight.
The ladder here is left as it was: it is a correct record of what these rungs
score, and the newer script reproduces it exactly as its gate.

Chapter 5b establishes that the per-point radar noise grows with speed, and that
the winning form is quadratic:  sigma^2 = sigma_0^2 + (q v^2)^2, q = 0.0104.
The mechanism proposed for that shape is translational smear: while a static
scatterer is coherently integrated over the CPI, the ray direction swings
because the platform translates past it, at

    d v_r / dt = |v|^2 sin^2(theta) / rho,

so the Doppler peak smears by (that rate) x T_CPI.  The catch is that the
mechanism predicts PER-POINT structure -- each return has its own sin(theta)
and its own range -- while 5b tests it on per-FRAME aggregates with the
geometry lumped into the fitted constant q.  5b's one geometry-explicit rung
used the per-frame MEDIAN range and lost to the lumped form, which was then
explained away.  That is not a clean test of the mechanism.

This script runs the clean test.  It keeps the reference-immunity of 5b (all
statistics are taken on within-frame deviations, so anything the whole frame
shares -- including MoCap error -- cancels) but resolves the predictor per
return.  If the geometry-explicit predictor beats the lumped one, the mechanism
is established and the deployed law's constant becomes a PREDICTION:

    coefficient of (v^2 sin^2(theta_j) / rho_j)  ==  kappa * T_CPI,

with T_CPI = 49.92 ms read off the radar configuration (128 Doppler loops x
3-TX TDM x 130 us chirps in 6843AOP_best_velocity.cfg) and kappa an O(1)
processing factor.  If it does NOT beat the lumped form, the v^2 law must be
labelled empirical rather than mechanistic.

Two independent estimators are run and required to agree:
  (A) binned robust-sigma fit -- the 5b protocol, but binned on the per-point
      predictor instead of the per-frame one;
  (B) pairwise composite likelihood -- within a frame, r_j - r_k = e_j - e_k
      cancels the frame-shared term EXACTLY and has variance sigma_j^2 +
      sigma_k^2, giving a per-point likelihood with no reference contamination.

Run from analysis/:  ../.venv/bin/python3 characterize_pointwise_noise.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, minimize

from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (compute_doppler_residuals,
                                  rotation_matrix_from_euler, unwrap_doppler,
                                  quat_to_rotation_matrix)

# --- radar configuration constants, read off 6843AOP_best_velocity.cfg -------
# profileCfg idleTime 115 us + rampEndTime 15 us = 130 us per chirp
# chirpCfg 0..2 = 3-TX TDM  ->  390 us per Doppler sample
# frameCfg numLoops = 128   ->  CPI = 128 * 390 us
T_CPI = 128 * 3 * 130e-6            # 49.92 ms
MIN_RANGE_CLAMP = 0.3               # m, guards 1/rho; sensitivity reported

_cfg = load_config()
BAGS = _cfg['bags']['bags']
TIMING = _cfg['bags']['timing']
RC = _cfg['bags'].get('radar_config', {})
EXT = _cfg['extrinsics']
R_BS = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
T_BS = np.array(EXT['translation_body_m'])
RADAR_OFF = EXT['imu_mocap_offset_sec'] - EXT['radar_imu_offset_sec']

RACING = ['slow_racing_best_velocity', 'fast_racing_best_velocity']
ALL_BAGS = RACING + ['backflips_best_velocity']
SHORT = {'slow_racing_best_velocity': 'slow racing',
         'fast_racing_best_velocity': 'fast racing',
         'backflips_best_velocity': 'backflips'}


def robust_sigma(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def build_point_table(key, repo):
    """Per-point table with within-frame deviations and per-point geometry.

    Returns dict of arrays, one entry per KEPT return:
      d      within-frame deviation, rescaled so its std estimates sigma_j
      fid    frame index (for pair building and bootstrap)
      v      frame speed |v| (m/s)
      vs     frame ANTENNA speed |v + omega x t_bs| (m/s)
      sin2   per-point sin^2(theta) between ray and velocity
      rho    per-point reported range (m)
    """
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
    aliased = unw != dres['measurements']

    st_t = np.array([s.timestamp for s in states])
    v_i = interp1d(st_t, np.array([s.velocity for s in states]), axis=0,
                   kind='linear', bounds_error=False, fill_value='extrapolate')
    w_i = interp1d(st_t, np.array([s.angular_velocity for s in states]), axis=0,
                   kind='linear', bounds_error=False, fill_value='extrapolate')
    q_i = interp1d(st_t, np.array([s.orientation for s in states]), axis=0,
                   kind='linear', bounds_error=False, fill_value='extrapolate')

    # same hygiene as 5b: 6-sigma clip on the bag, alias-cut, N_f >= 6
    keep0 = (np.abs(r - np.median(r)) < 6 * robust_sigma(r)) & ~aliased

    out = {k: [] for k in ('d', 'fid', 'v', 'vs', 'sin2', 'rho')}
    for f in np.unique(fi):
        m = (fi == f) & keep0
        n = int(m.sum())
        if n < 6:
            continue
        fr = radar[int(f)]
        t_f = fr.timestamp + RADAR_OFF
        R_wb = quat_to_rotation_matrix(q_i(t_f))
        v_b = R_wb.T @ v_i(t_f)                       # body-frame velocity
        w_b = w_i(t_f)
        v_s = v_b + np.cross(w_b, T_BS)               # antenna velocity

        P = np.asarray(fr.positions, float)[pi[m]]    # sensor-frame positions
        rho = np.linalg.norm(P, axis=1)
        u_b = (R_BS @ (P / np.maximum(rho, 1e-6)[:, None]).T).T  # body-frame rays

        nv = np.linalg.norm(v_b)
        if nv < 0.2:                                   # sin(theta) undefined
            continue
        sin2 = np.clip(1.0 - (u_b @ v_b / nv) ** 2, 0.0, 1.0)

        # within-frame deviation; c_f cancels exactly.  Var(d) = sigma^2(1-1/N)
        # under equal within-frame variances, so undo that shrinkage.
        rf = r[m]
        d = (rf - rf.mean()) / np.sqrt(1.0 - 1.0 / n)

        out['d'].append(d)
        out['fid'].append(np.full(n, int(f)))
        out['v'].append(np.full(n, nv))
        out['vs'].append(np.full(n, np.linalg.norm(v_s)))
        out['sin2'].append(sin2)
        out['rho'].append(np.maximum(rho, MIN_RANGE_CLAMP))
    del data
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------- predictors
# Each model is  sigma^2 = sigma_0,bag^2 + (coef * X_j)^2, so X is the shape the
# mechanism predicts for sigma itself (not for sigma^2).
#
#   BEARING ERROR.  A ray error of rms angle dphi perturbs u by dphi in a random
#   perpendicular direction; it enters the residual as -du^T v, and the part of v
#   perpendicular to u has magnitude |v| sin(theta).  So
#         sigma = (dphi/sqrt(2)) * |v| sin(theta)      ->  X = v sin(theta),
#   and the fitted coefficient is directly the rms bearing error / sqrt(2).
#   NOTE this is NOT the plain-|v| shape: chapter 5b tests bearing error as
#   sigma ~ |v| and rejects it, but the sin(theta) factor belongs there.
#
#   TEMPORAL SMEAR.  The ray swings during the CPI at dv_r/dt = v^2 sin^2/rho,
#   integrated over T_CPI:
#         sigma = kappa * T_CPI * v^2 sin^2(theta)/rho  ->  X = v^2 sin^2/rho,
#   and the fitted coefficient is kappa * T_CPI with T_CPI known from the config.
def predictors(T):
    """name -> (list of per-point predictors, list of kinds).

    Kinds: 'bearing' (coef = rms bearing error / sqrt(2)), 'smear'
    (coef = kappa * T_CPI), None (shape with no single physical constant).
    Multi-entry lists are QUADRATURE-SUM models, sigma^2 = s0^2 + sum_m
    (coef_m X_m)^2, so competing mechanisms can coexist and the data is free to
    zero the subdominant one -- the same discipline chapter 5b applies.
    """
    sin1 = np.sqrt(T['sin2'])
    cos1 = np.sqrt(np.clip(1.0 - T['sin2'], 0.0, 1.0))
    v, vs, rho = T['v'], T['vs'], T['rho']
    return {
        'floors only               ': ([np.zeros(len(v))], [None]),
        # --- controls that isolate which factor is doing the work
        'sin only     (geometry)   ': ([sin1], [None]),
        'v            (speed only) ': ([v], [None]),
        'v cos = |v_r| (SCALE err) ': ([v * cos1], [None]),
        # --- bearing-error mechanism: sigma = (dphi/sqrt2) |v| sin(theta)
        'v sin        (BEARING)    ': ([v * sin1], ['bearing']),
        'vs sin       (BEARING,ant)': ([vs * sin1], ['bearing']),
        # --- deployed and smear shapes
        'v^2          (5b deployed)': ([v ** 2], [None]),
        'v^2 sin^2                 ': ([v ** 2 * T['sin2']], [None]),
        'v^2 sin^2/rho  (SMEAR)    ': ([v ** 2 * T['sin2'] / rho], ['smear']),
        'vs^2 sin^2/rho (SMEAR,ant)': ([vs ** 2 * T['sin2'] / rho], ['smear']),
        # --- joint: let the data allocate between mechanisms
        'v sin + v^2   (joint)     ': ([v * sin1, v ** 2], ['bearing', None]),
        'v sin + smear (joint)     ': ([v * sin1, v ** 2 * T['sin2'] / rho],
                                       ['bearing', 'smear']),
    }


# --------------------------------------------------- (A) binned robust-sigma
def binned_fit(tabs, Xs, n_bins=10, min_n=120):
    """5b-style joint fit: per-bag floor + shared coefficients, on binned robust
    sigma of the per-point deviations, binned by the FIRST per-point predictor
    (secondary predictors enter through their bin means)."""
    nx = len(next(iter(Xs.values())))
    cen, sig, bagidx = [], [], []
    for bi, key in enumerate(tabs):
        T, XL = tabs[key], Xs[key]
        X0 = XL[0]
        edges = np.unique(np.percentile(X0, np.linspace(0, 100, n_bins + 1)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (X0 >= lo) & (X0 < hi)
            if m.sum() >= min_n:
                cen.append([X[m].mean() for X in XL])
                sig.append(robust_sigma(T['d'][m]))
                bagidx.append(bi)
    c = np.array(cen)                      # (n_bins_total, nx)
    s = np.array(sig)
    b = np.array(bagidx, dtype=int)
    nb = len(tabs)

    def model(th):
        acc = np.abs(th[b]) ** 2
        for m in range(nx):
            acc = acc + (np.abs(th[nb + m]) * c[:, m]) ** 2
        return np.sqrt(acc)

    th0 = np.r_[[np.median(s)] * nb, [1e-3] * nx]
    fsc = max(1.4826 * np.median(np.abs(s - np.median(s))), 1e-4)
    sol = least_squares(lambda th: model(th) - s, th0, loss='soft_l1',
                        f_scale=fsc, max_nfev=40000)
    res = model(sol.x) - s
    rss = float(res @ res)
    return dict(theta=np.abs(sol.x), n_bins=len(s),
                r2=1 - rss / float(((s - s.mean()) ** 2).sum()))


# ------------------------------------------- (B) pairwise composite likelihood
def pair_index(fid, rng, max_pairs_per_frame=60):
    """All within-frame pairs (subsampled), as index arrays into the table."""
    ii, jj = [], []
    order = np.argsort(fid, kind='stable')
    fs = fid[order]
    bounds = np.flatnonzero(np.diff(fs)) + 1
    for grp in np.split(order, bounds):
        n = len(grp)
        a, b = np.triu_indices(n, k=1)
        if len(a) > max_pairs_per_frame:
            sel = rng.choice(len(a), max_pairs_per_frame, replace=False)
            a, b = a[sel], b[sel]
        ii.append(grp[a])
        jj.append(grp[b])
    return np.concatenate(ii), np.concatenate(jj)


def pair_fit(tabs, Xs, rng, trim_z=4.0):
    """Gaussian composite MLE on within-frame pairwise differences.

    Delta_jk = e_j - e_k has variance sigma_j^2 + sigma_k^2 and is exactly free
    of the frame-shared term.  Pairs share points, so this is a COMPOSITE
    likelihood: the point estimate is consistent, but its 'BIC' is not a formal
    model-selection statistic -- the ranking is cross-checked against (A) and
    the CI comes from a bootstrap over frames.
    """
    nx = len(next(iter(Xs.values())))
    D, XI, XJ, bag = [], [], [], []
    for bi, key in enumerate(tabs):
        T, XL = tabs[key], Xs[key]
        i, j = pair_index(T['fid'], rng)
        D.append(T['d'][i] - T['d'][j])
        XI.append(np.column_stack([X[i] for X in XL]))
        XJ.append(np.column_stack([X[j] for X in XL]))
        bag.append(np.full(len(i), bi))
    D = np.concatenate(D)
    XI = np.concatenate(XI)                 # (n_pairs, nx)
    XJ = np.concatenate(XJ)
    bag = np.concatenate(bag).astype(int)
    nb = len(tabs)

    def variance(th):
        var = 2 * np.abs(th[bag]) ** 2
        for m in range(nx):
            var = var + np.abs(th[nb + m]) ** 2 * (XI[:, m] ** 2 + XJ[:, m] ** 2)
        return np.maximum(var, 1e-9)

    def nll(th, mask):
        var = variance(th)[mask]
        return float(np.sum(D[mask] ** 2 / var + np.log(var)))

    mask = np.ones(len(D), bool)
    th = np.r_[[robust_sigma(D) / np.sqrt(2)] * nb, [1e-3] * nx]
    for _ in range(3):                      # trim, refit, retrim
        sol = minimize(nll, th, args=(mask,), method='Nelder-Mead',
                       options=dict(maxiter=40000, maxfev=40000,
                                    xatol=1e-9, fatol=1e-9))
        th = sol.x
        mask = np.abs(D) / np.sqrt(variance(th)) < trim_z
    return dict(theta=np.abs(th), nll=nll(th, mask), n_pairs=int(mask.sum()))


def main():
    repo = Path(__file__).resolve().parent.parent
    rng = np.random.default_rng(0)
    tabs = {k: build_point_table(k, repo) for k in ALL_BAGS}
    fit_bags = {k: tabs[k] for k in RACING}      # same fit set as 5b

    print(f'T_CPI from 6843AOP_best_velocity.cfg = {1e3*T_CPI:.2f} ms '
          f'(128 loops x 3 TX x 130 us)')
    for k in ALL_BAGS:
        print(f'  {SHORT[k]:<12} {len(tabs[k]["d"]):5d} kept returns in '
              f'{len(np.unique(tabs[k]["fid"])):4d} frames')

    def interpret(kinds, coefs):
        """Turn fitted coefficients into the physical constants they predict."""
        out = []
        for kind, coef in zip(kinds, coefs):
            if kind == 'bearing':
                out.append('bearing err '
                           f'{np.degrees(coef * np.sqrt(2)):.1f} deg')
            elif kind == 'smear':
                out.append(f'kappa {coef / T_CPI:.2f}')
        return ', '.join(out)

    def coef_str(coefs):
        return '/'.join(f'{c:.5f}' for c in coefs)

    print()
    print('=' * 96)
    print('PRIMARY RANKING (B): pairwise composite likelihood.  Every model is')
    print('scored on the SAME within-frame pairs, so 2*NLL is comparable across')
    print('rows (it is a composite likelihood: ranking is valid, but the absolute')
    print('gaps are not chi^2-calibrated -- hence the binned cross-check below).')
    print('=' * 96)
    print(f"{'per-point predictor X':<28} {'2*NLL':>11} {'vs floors':>10} "
          f"{'coef(s)':>16} {'floors sl/fa':>14}  physical reading")
    pair_res = {}
    base = None
    for name, (_, kinds) in predictors(tabs[RACING[0]]).items():
        Xs = {k: predictors(tabs[k])[name][0] for k in RACING}
        pf = pair_fit(fit_bags, Xs, np.random.default_rng(1))
        pair_res[name] = pf
        th = pf['theta']
        if base is None:
            base = 2 * pf['nll']
        print(f'{name:<28} {2*pf["nll"]:>11.1f} {2*pf["nll"]-base:>10.1f} '
              f'{coef_str(th[2:]):>16} {th[0]:>6.3f}/{th[1]:<7.3f}  '
              f'{interpret(kinds, th[2:])}')

    print()
    print('CROSS-CHECK (A): binned robust-sigma fit, the chapter-5b protocol but')
    print('binned on the PER-POINT predictor.  MAD-based, so far less sensitive')
    print('to the heavy tail than (B); R^2 is within each model\'s own binning.')
    print(f"{'per-point predictor X':<28} {'R^2':>7} {'coef(s)':>16} "
          f"{'floors sl/fa':>14}  physical reading")
    results = {}
    for name, (_, kinds) in predictors(tabs[RACING[0]]).items():
        if name.startswith('floors'):
            continue
        Xs = {k: predictors(tabs[k])[name][0] for k in RACING}
        fb = binned_fit(fit_bags, Xs)
        results[name] = fb
        th = fb['theta']
        print(f'{name:<28} {fb["r2"]:>7.3f} {coef_str(th[2:]):>16} '
              f'{th[0]:>6.3f}/{th[1]:<7.3f}  {interpret(kinds, th[2:])}')

    # ---- bootstrap-over-frames CI for the two headline forms
    HEAD = ('v^2          (5b deployed)', 'v sin        (BEARING)    ',
            'v^2 sin^2/rho  (SMEAR)    ')
    print()
    print('Bootstrap over frames (200 resamples), binned estimator:')
    for name in HEAD:
        boots = []
        rb = np.random.default_rng(0)
        for _ in range(200):
            tb = {}
            for k in RACING:
                T = tabs[k]
                fr = np.unique(T['fid'])
                pick = rb.choice(fr, len(fr), replace=True)
                idx = np.concatenate([np.flatnonzero(T['fid'] == f)
                                      for f in pick])
                tb[k] = {kk: vv[idx] for kk, vv in T.items()}
            Xb = {k: predictors(tb[k])[name][0] for k in RACING}
            boots.append(binned_fit(tb, Xb)['theta'][2])
        lo, hi = np.percentile(boots, [16, 84])
        extra = ''
        if 'BEARING' in name:
            extra = (f'  -> bearing err {np.degrees(lo*np.sqrt(2)):.1f}'
                     f'-{np.degrees(hi*np.sqrt(2)):.1f} deg')
        print(f'  {name}: coef {results[name]["theta"][2]:.5f} '
              f'68% CI [{lo:.5f}, {hi:.5f}]{extra}')

    # ---- the winner's binned curve, and robustness to support restrictions
    print()
    print('Binned sigma vs the winning predictor X = |v| sin(theta) (m/s):')
    for k in RACING:
        T = tabs[k]
        X = T['v'] * np.sqrt(T['sin2'])
        edges = np.unique(np.percentile(X, np.linspace(0, 100, 9)))
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (X >= lo) & (X < hi)
            if m.sum() >= 60:
                cells.append(f'{X[m].mean():.2f}:{robust_sigma(T["d"][m]):.3f}')
        print(f'  {SHORT[k]:<12} ' + '  '.join(cells))

    print()
    print('Robustness of the BEARING result to support restrictions')
    print('(sin(theta) is computed from the measured ray, so very small values')
    print(' are the least trustworthy; the alias cut also removes low-sin, '
          'high-v points):')
    checks = [
        ('all points            ', lambda T: np.ones(len(T['v']), bool)),
        ('sin(theta) > 0.2      ', lambda T: T['sin2'] > 0.04),
        ('sin(theta) > 0.4      ', lambda T: T['sin2'] > 0.16),
        ('|v| < 3.14 (no alias) ', lambda T: T['v'] < 3.136),
        ('rho > 1 m             ', lambda T: T['rho'] > 1.0),
    ]
    for label, fn in checks:
        sub = {}
        for k in RACING:
            T = tabs[k]
            m = fn(T)
            sub[k] = {kk: vv[m] for kk, vv in T.items()}
        line = []
        for nm in ('v sin        (BEARING)    ', 'v^2          (5b deployed)'):
            Xb = {k: predictors(sub[k])[nm][0] for k in RACING}
            fb = binned_fit(sub, Xb)
            tag = 'bear' if 'BEAR' in nm else 'v^2 '
            deg = (f' ({np.degrees(fb["theta"][2]*np.sqrt(2)):.1f} deg)'
                   if 'BEAR' in nm else '')
            line.append(f'{tag} R2 {fb["r2"]:.3f}{deg}')
        n_tot = sum(int(fn(tabs[k]).sum()) for k in RACING)
        print(f'  {label} n={n_tot:5d}   ' + '   '.join(line))

    # ---- range-clamp sensitivity of the smear coefficient
    print()
    print('Range-clamp sensitivity of the smear coefficient '
          '(1/rho is clamp-sensitive):')
    for clamp in (0.2, 0.3, 0.5, 0.8):
        Xs = {k: [tabs[k]['v'] ** 2 * tabs[k]['sin2']
                  / np.maximum(tabs[k]['rho'], clamp)] for k in RACING}
        fb = binned_fit(fit_bags, Xs)
        print(f'  clamp {clamp:.1f} m: coef {fb["theta"][2]:.5f} '
              f'-> kappa {fb["theta"][2]/T_CPI:.2f}   R^2 {fb["r2"]:.3f}')

    # ---- transfer: freeze the response, refit only the floor, on backflips
    print()
    print('Transfer to backflips (flip regime, NOT in the fit; response frozen,'
          ' floor refit):')
    for name in HEAD:
        coef = results[name]['theta'][2]
        T = tabs['backflips_best_velocity']
        X = predictors(T)[name][0][0]
        edges = np.unique(np.percentile(X, np.linspace(0, 100, 9)))
        c, s = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (X >= lo) & (X < hi)
            if m.sum() >= 60:
                c.append(X[m].mean())
                s.append(robust_sigma(T['d'][m]))
        c, s = np.array(c), np.array(s)
        if len(c) < 3:
            print(f'  {name}: too few populated bins ({len(c)}) to score')
            continue
        sol = least_squares(
            lambda p: np.sqrt(p[0] ** 2 + (coef * c) ** 2) - s, [0.3],
            loss='soft_l1', max_nfev=20000)
        rss = float(np.sum((np.sqrt(sol.x[0] ** 2 + (coef * c) ** 2) - s) ** 2))
        tss = float(((s - s.mean()) ** 2).sum())
        r2 = 1 - rss / tss if tss > 0 else np.nan
        print(f'  {name}: floor {abs(sol.x[0]):.3f} m/s, R^2 {r2:.3f} '
              f'({len(c)} bins)')


if __name__ == '__main__':
    main()
