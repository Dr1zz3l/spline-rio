#!/usr/bin/env python3
"""Speed-law decision experiments E1-E4 (FEEDBACK_PLAN 2026-08-02, approved plan).

E1: which noise-inflation law does the reference-immune within-frame sigma_i
    support (speed forms vs the deployed omega form), and does the winning
    form TRANSFER across flights/platforms (invariance)?
E2: is a causal per-frame speed estimate (post-consensus WLS ego-speed) good
    enough to drive such a law at solve time?
E3: flip-regime omega term on alias-cut backflips + racing consistency.
E4: is sigma_c (frame-shared noise) flat vs speed (whitening assumption)?

Run from analysis/:  ../.venv/bin/python3 characterize_speed_law.py
Optional: --bags key1,key2  (subset), --quick (racing-only E1, skip transfer)

All residuals use the notebook/paper protocol: r = unw - pred against the
MoCap-derived reference through the exact solver forward model; GT-resolved
aliasing with the per-bag v_max; 6-sigma bag-level prefilter; frames with
>= 6 kept returns; within-frame variance with ddof=1.
"""
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = REPO_ROOT / 'analysis'
for p in (ANALYSIS, ANALYSIS / 'lib'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (compute_doppler_residuals,
                                  rotation_matrix_from_euler, unwrap_doppler)

CFG = load_config()
BAGS = CFG['bags']['bags']
TIMING = CFG['bags']['timing']
FLIPPED = set(CFG['bags']['flipped'])
RC = CFG['bags'].get('radar_config', {})
EXT_OVER = CFG['bags'].get('extrinsics_overrides', {})
EXT = CFG['extrinsics']

# Deployed own-platform extrinsics (pitch locked at the measured 27.5 deg)
R_DEPLOYED = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
T_DEPLOYED = np.array(EXT['translation_body_m'])
# ICINS dataset extrinsics (converter-printed; offsets zero, hardware-stamped)
R_ICINS = rotation_matrix_from_euler(*np.radians([-178.501, -0.099, 46.997]))
T_ICINS = np.array([0.01, 0.1, 0.06])
R_YAWFLIP = rotation_matrix_from_euler(0.0, 0.0, np.pi)

RACING = ['slow_racing_best_velocity', 'fast_racing_best_velocity']
BACKFLIPS = 'backflips_best_velocity'
TRANSFER = ['fast_racing_best_velocity_no_clustering', 'circle_best_velocity',
            'circle_fwd', 'loopings',
            'icins_flight_1', 'icins_flight_2', 'icins_flight_3', 'icins_flight_4']
SHORT = {k: k.replace('_best_velocity', '').replace('_flight', '')
         for k in RACING + [BACKFLIPS] + TRANSFER}


def bag_offsets(key):
    ov = EXT_OVER.get(key, {})
    imu_mocap = ov.get('imu_mocap_offset_sec', EXT['imu_mocap_offset_sec'])
    radar_imu = ov.get('radar_imu_offset_sec', EXT['radar_imu_offset_sec'])
    return imu_mocap - radar_imu


def load_bag(key):
    is_icins = key.startswith('icins')
    R_BS = R_ICINS if is_icins else R_DEPLOYED
    T_BS = T_ICINS if is_icins else T_DEPLOYED
    if key in FLIPPED:
        R_BS = R_YAWFLIP @ R_BS
        T_BS = R_YAWFLIP @ T_BS
    v_max = RC.get('best_velocity' if 'best_velocity' in key else 'default',
                   {}).get('v_max', 4.99)
    radar_off = bag_offsets(key)
    data = load_bag_topics(str(REPO_ROOT / BAGS[key]), verbose=False)
    t0 = data.start_time + TIMING[key][0]
    t1 = t0 + TIMING[key][1]
    radar = [f for f in data.radar_velocity
             if t0 <= f.timestamp <= t1 and f.positions is not None]
    states = data.agiros_state
    dres = compute_doppler_residuals(states, radar, T_BS, R_BS,
                                     time_offset=radar_off, min_range=0.2)
    meas, pred = dres['measurements'], dres['predictions']
    unw = unwrap_doppler(meas, pred, v_max)
    st_t = np.array([s.timestamp for s in states])
    d = {
        'radar': radar, 'v_max': v_max, 'radar_off': radar_off,
        'r': unw - pred, 'pred': pred, 'unw': unw, 'meas': meas,
        'aliased': np.abs(unw - meas) > 1e-9,
        'fi': np.array(dres['frame_indices']),
        'pi': np.array(dres['point_indices']),
        'om_interp': interp1d(st_t, np.array([s.angular_velocity for s in states]),
                              axis=0, kind='linear', bounds_error=False,
                              fill_value='extrapolate'),
        'v_interp': interp1d(st_t, np.array([s.velocity for s in states]),
                             axis=0, kind='linear', bounds_error=False,
                             fill_value='extrapolate'),
    }
    return d


def robust_sigma(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def frame_table(d, alias_cut=True, min_kept=6):
    """rows: [s2, |omega|, |v|, median_range, N_kept] per frame."""
    r, fi = d['r'], d['fi']
    keep = np.abs(r - np.median(r)) < 6 * robust_sigma(r)
    if alias_cut:
        keep = keep & ~d['aliased']
    rows = []
    for f in np.unique(fi):
        m = (fi == f) & keep
        if np.sum(m) < min_kept:
            continue
        fr = d['radar'][int(f)]
        t_f = fr.timestamp + d['radar_off']
        P = np.asarray(fr.positions, float)
        rows.append((np.var(r[m], ddof=1),
                     np.linalg.norm(d['om_interp'](t_f)),
                     np.linalg.norm(d['v_interp'](t_f)),
                     np.median(np.linalg.norm(P, axis=1)),
                     int(np.sum(m))))
    return np.array(rows)


# ---------------------------------------------------------------------------
# E1: functional forms on racing (alias-cut), shared response + per-bag floors
# Additive, identifiable parametrizations, fitted on per-frame sigma (sqrt s2)
# with a soft_l1 robust loss (per-frame s2 is chi^2-tailed).
# ---------------------------------------------------------------------------
SPECS = [
    # name, sigma^2 model extra term (shared params p >= 0), n_extra, p0
    ('floors-only', lambda p, om, v, rng: 0.0, 0, []),
    ('v-lin  (+(a v)^2)', lambda p, om, v, rng: (p[0] * v)**2, 1, [0.05]),
    ('om-lin (+(c om)^2)', lambda p, om, v, rng: (p[0] * om)**2, 1, [0.03]),
    ('v+om', lambda p, om, v, rng: (p[0] * v)**2 + (p[1] * om)**2, 2, [0.05, 0.02]),
    ('trans (+(b v^2/r)^2)', lambda p, om, v, rng: (p[0] * v**2 / rng)**2, 1, [0.05]),
    ('v-quad (+(q v^2)^2)', lambda p, om, v, rng: (p[0] * v**2)**2, 1, [0.02]),
]


def fit_joint(tabs, extra_fn, n_extra, p0_extra):
    """theta = [s0 per bag..., extra...]; robust fit on sigma scale."""
    keys = list(tabs)
    sig = np.concatenate([np.sqrt(tabs[k][:, 0]) for k in keys])
    om = np.concatenate([tabs[k][:, 1] for k in keys])
    v = np.concatenate([tabs[k][:, 2] for k in keys])
    rng = np.concatenate([tabs[k][:, 3] for k in keys])
    bag = np.concatenate([np.full(len(tabs[k]), i) for i, k in enumerate(keys)])

    def model_sigma(th):
        s0 = np.abs(th[bag.astype(int)])
        ex = np.abs(th[len(keys):])
        return np.sqrt(s0**2 + extra_fn(ex, om, v, rng))

    th0 = np.r_[[np.sqrt(np.median(tabs[k][:, 0])) for k in keys], p0_extra]
    f_scale = max(1.4826 * np.median(np.abs(sig - np.median(sig))), 1e-3)
    sol = least_squares(lambda th: model_sigma(th) - sig, th0,
                        loss='soft_l1', f_scale=f_scale, max_nfev=40000)
    res = model_sigma(sol.x) - sig
    rss = float(np.sum(res**2))
    n, k = len(sig), len(th0)
    tot = float(np.sum((sig - sig.mean())**2))
    return {'theta': np.abs(sol.x), 'rss': rss, 'r2': 1 - rss / tot,
            'bic': n * np.log(rss / n) + k * np.log(n), 'n': n, 'keys': keys}


def run_e1(tabs_racing, tabs_transfer, quick=False):
    print('\n' + '=' * 74)
    print('E1: functional form on racing (alias-cut sigma_i), joint per-bag floors')
    print('    (robust sigma-scale fits; additive identifiable forms)')
    print('=' * 74)
    results = {}
    for name, fn, n_extra, p0 in SPECS:
        f = fit_joint(tabs_racing, fn, n_extra, p0)
        results[name] = f
        floors = '/'.join(f'{x:.3f}' for x in f['theta'][:len(f['keys'])])
        extra = '/'.join(f'{x:.4f}' for x in f['theta'][len(f['keys']):])
        print(f"  {name:22s} R^2={f['r2']:.3f}  BIC={f['bic']:9.1f}  "
              f"floors={floors}  extra={extra}")

    best = min((k for k in results if k != 'floors-only'),
               key=lambda k: results[k]['bic'])
    print(f'\n  BIC winner: {best}')

    spec = next(sp for sp in SPECS if sp[0] == best)
    rng_bs = np.random.default_rng(0)
    boots = []
    for _ in range(200):
        tb = {}
        for k, t in tabs_racing.items():
            idx = rng_bs.integers(0, len(t), len(t))
            tb[k] = t[idx]
        try:
            fb = fit_joint(tb, spec[1], spec[2], spec[3])
            boots.append(fb['theta'][len(fb['keys']):])
        except Exception:
            continue
    boots = np.array(boots)
    if len(boots):
        lo, hi = np.percentile(boots, [16, 84], axis=0)
        print('  winner shared param 68% CI: '
              + ', '.join(f'[{a:.4f}, {b:.4f}]' for a, b in zip(np.atleast_1d(lo),
                                                                np.atleast_1d(hi))))

    if quick or not tabs_transfer:
        return results, best

    print('\n  TRANSFER (shared response frozen from racing; floor refit per bag)')
    print(f'  {"bag":22s} {"n_frm":>6} {"v range":>11} {"om range":>10} '
          f'{"R2 v-lin":>9} {"R2 om-lin":>10} {"gap v":>6} {"gap om":>7}')
    for hk, ht in tabs_transfer.items():
        if len(ht) < 40:
            print(f'  {SHORT[hk]:22s} {len(ht):>6}   (too few frames)')
            continue
        row = {}
        for form_name in ('v-lin  (+(a v)^2)', 'om-lin (+(c om)^2)'):
            spec_f = next(sp for sp in SPECS if sp[0] == form_name)
            shared = results[form_name]['theta'][2:]
            sig = np.sqrt(ht[:, 0]); om, v, rngm = ht[:, 1], ht[:, 2], ht[:, 3]
            extra_term = spec_f[1](shared, om, v, rngm)
            f_scale = max(1.4826 * np.median(np.abs(sig - np.median(sig))), 1e-3)
            sol = least_squares(
                lambda th: np.sqrt(th[0]**2 + extra_term) - sig,
                [np.median(sig)], loss='soft_l1', f_scale=f_scale)
            res = np.sqrt(sol.x[0]**2 + extra_term) - sig
            rss = np.sum(res**2); tot = np.sum((sig - sig.mean())**2)
            floc = fit_joint({hk: ht}, spec_f[1], spec_f[2], spec_f[3])
            row[form_name] = (1 - rss / tot, rss / max(floc['rss'], 1e-12))
        v_r, om_r = row['v-lin  (+(a v)^2)'], row['om-lin (+(c om)^2)']
        print(f'  {SHORT[hk]:22s} {len(ht):>6} '
              f'{ht[:, 2].min():4.1f}-{ht[:, 2].max():4.1f} '
              f'{ht[:, 1].min():4.1f}-{ht[:, 1].max():4.1f} '
              f'{v_r[0]:>9.3f} {om_r[0]:>10.3f} {v_r[1]:>6.2f} {om_r[1]:>7.2f}')
    return results, best


# ---------------------------------------------------------------------------
# E2: causal speed source: post-consensus WLS ego-speed vs GT speed
# ---------------------------------------------------------------------------
def run_e2(D):
    print('\n' + '=' * 74)
    print('E2: causal speed source: per-frame consensus+LSQ ego-speed vs GT |v|')
    print('=' * 74)
    for key, d in D.items():
        rng = np.random.default_rng(0)
        rows = []
        for i, f in enumerate(d['radar']):
            n = f.num_points()
            if n < 5:
                continue
            sel = d['fi'] == i
            if not np.any(sel):
                continue
            P = np.asarray(f.positions, float)
            v = np.asarray(f.velocities, float).copy()
            v[d['pi'][sel]] = d['unw'][sel]
            Hn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
            cand = []
            for _ in range(150):
                i3 = rng.choice(n, 3, replace=False)
                try:
                    cand.append(np.linalg.solve(Hn[i3], v[i3]))
                except np.linalg.LinAlgError:
                    continue
            if not cand:
                continue
            Rm = np.abs(Hn @ np.array(cand).T - v[:, None]).T
            best = (Rm < 0.15)[np.argmax((Rm < 0.15).sum(axis=1))]
            if best.sum() < 5:
                best = np.ones(n, bool)          # deployed bypass
            v_hat, *_ = np.linalg.lstsq(Hn[best], v[best], rcond=None)
            t_f = f.timestamp + d['radar_off']
            rows.append((np.linalg.norm(v_hat),
                         np.linalg.norm(d['v_interp'](t_f))))
        A = np.array(rows)
        vh, vg = A[:, 0], A[:, 1]
        err = vh - vg
        m_hi = vg > d['v_max']
        line = (f'  {SHORT[key]:22s} n={len(A):4d}  bias={np.median(err):+.3f}  '
                f'rob sigma={robust_sigma(err):.3f}  p95|e|='
                f'{np.percentile(np.abs(err), 95):.3f}')
        if m_hi.sum() > 20:
            line += (f'   [>v_max: n={m_hi.sum()}, bias={np.median(err[m_hi]):+.3f}, '
                     f'rob sigma={robust_sigma(err[m_hi]):.3f}]')
        print(line)


# ---------------------------------------------------------------------------
# E3: flip-regime omega term (alias-cut backflips) + racing consistency
# ---------------------------------------------------------------------------
def run_e3(tab_bf, tabs_racing):
    print('\n' + '=' * 74)
    print('E3: flip-regime term on alias-cut backflips (binned sigma_i fits)')
    print('=' * 74)

    def binned(tab, col, edges, min_frames=8):
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (tab[:, col] >= a) & (tab[:, col] < b)
            if m.sum() >= min_frames:
                out.append((tab[m, col].mean(), np.sqrt(tab[m, 0].mean()),
                            int(m.sum())))
        return np.array(out)

    OM_E = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 13])
    V_E = np.array([2.0, 3.0, 3.5, 3.8, 4.0, 4.2, 4.5, 5.5])
    bo = binned(tab_bf, 1, OM_E)
    bv = binned(tab_bf, 2, V_E)
    print('  sigma_i vs omega bins: '
          + '  '.join(f'{c:.1f}:{sg:.3f}(n{n:.0f})' for c, sg, n in bo))
    print('  sigma_i vs v bins:     '
          + '  '.join(f'{c:.1f}:{sg:.3f}(n{n:.0f})' for c, sg, n in bv))

    for name, col, b in (('omega', 1, bo), ('v', 2, bv)):
        if len(b) < 3:
            continue
        x, sg = b[:, 0], b[:, 1]
        sol = least_squares(
            lambda th: np.sqrt(th[0]**2 + (th[1] * x)**2) - sg,
            [sg.min(), 0.02], max_nfev=20000)
        pred = np.sqrt(sol.x[0]**2 + (sol.x[1] * x)**2)
        r2 = 1 - np.sum((pred - sg)**2) / np.sum((sg - sg.mean())**2)
        print(f'  binned fit sigma^2 = s0^2 + (k*{name})^2: s0={abs(sol.x[0]):.3f} '
              f'k={abs(sol.x[1]):.4f} R^2={r2:.3f}')
        if name == 'omega':
            k = abs(sol.x[1]); s0 = abs(sol.x[0])
            infl5 = np.sqrt(1 + (k * 5.0 / max(s0, 1e-6))**2)
            print(f'    -> implies x{infl5:.2f} inflation at 5 rad/s '
                  f'(racing alias-cut observes ~flat there; consistent if <~1.3)')


# ---------------------------------------------------------------------------
# E4: sigma_c vs speed
# ---------------------------------------------------------------------------
def run_e4(D):
    print('\n' + '=' * 74)
    print('E4: frame-shared sigma_c binned by speed (>= 8 frames/bin)')
    print('=' * 74)
    V_EDGES = np.array([0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0])
    for key, d in D.items():
        r, fi = d['r'], d['fi']
        keep = np.abs(r - np.median(r)) < 6 * robust_sigma(r)
        comps = []
        for f in np.unique(fi):
            m = (fi == f) & keep
            if np.sum(m) >= 6:
                t_f = d['radar'][int(f)].timestamp + d['radar_off']
                comps.append((np.var(r[m], ddof=1), np.mean(r[m]),
                              int(np.sum(m)),
                              np.linalg.norm(d['v_interp'](t_f))))
        C = np.array(comps)
        cells = []
        for a, b in zip(V_EDGES[:-1], V_EDGES[1:]):
            m = (C[:, 3] >= a) & (C[:, 3] < b)
            if m.sum() < 8:
                continue
            s_i2 = C[m, 0].mean()
            s_c2 = max(np.var(C[m, 1], ddof=1) - s_i2 / C[m, 2].mean(), 0.0)
            cells.append((0.5 * (a + b), np.sqrt(s_c2 / s_i2), m.sum()))
        row = '  '.join(f'{c:.2f}:{s:.2f}(n{n})' for c, s, n in cells)
        print(f'  {SHORT[key]:22s} v : sigma_c/sigma_i  {row}   (deployed ratio 0.4)')


def main():
    quick = '--quick' in sys.argv
    subset = None
    for a in sys.argv[1:]:
        if a.startswith('--bags'):
            subset = a.split('=', 1)[1].split(',') if '=' in a else None
    keys = RACING + [BACKFLIPS] + ([] if quick else TRANSFER)
    if subset:
        keys = subset
    D = {}
    for k in keys:
        try:
            D[k] = load_bag(k)
            al = 100 * np.mean(D[k]['aliased'])
            print(f'loaded {SHORT[k]:22s} frames={len(D[k]["radar"]):4d} '
                  f'pts={len(D[k]["r"]):6d} aliased={al:4.1f}% '
                  f'v_max={D[k]["v_max"]:.2f}')
        except Exception as e:
            print(f'LOAD FAILED {k}: {type(e).__name__}: {e}')
    tabs = {k: frame_table(D[k], alias_cut=True) for k in D}
    for k, t in tabs.items():
        print(f'  {SHORT[k]:22s} frames(alias-cut, >=6 kept) = {len(t)}')

    tabs_racing = {k: tabs[k] for k in RACING if k in tabs}
    tabs_transfer = {k: tabs[k] for k in TRANSFER if k in tabs}
    results, best = run_e1(tabs_racing, tabs_transfer, quick=quick)
    run_e2({k: D[k] for k in D if not k.startswith('icins')})
    if BACKFLIPS in tabs:
        run_e3(tabs[BACKFLIPS], tabs_racing)
    run_e4(D)

    out = {k: {'theta': list(map(float, v['theta'])), 'r2': v['r2'],
               'bic': v['bic']} for k, v in results.items()}
    out_path = ANALYSIS / 'speed_law_fit_results.json'
    out_path.write_text(json.dumps({'winner': best, 'fits': out}, indent=1))
    print(f'\nfit results -> {out_path}')


if __name__ == '__main__':
    main()
