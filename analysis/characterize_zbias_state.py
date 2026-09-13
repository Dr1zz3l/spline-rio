#!/usr/bin/env python3
"""Per-frame radar elevation-bias (b_z) series: the measurement behind the
planned estimated z-bias STATE (worklog/2026-08-11_bearing-weight-pathways.md,
pathway 1).

Estimator-free: residuals are formed against MoCap/GT through the FULL forward
model (incl. lever arm), never against the solver's own output. Stream matches
deployment: GT-aided unwrap (own bags), RANSAC prefilter (thresh 0.15, seeded
default_rng(0), <5-return bypass), min_range 0.2, no intensity filter.

Per frame f and candidate frame c in {sensor, body, world}:
    r_i = v_meas_i - v_pred_i(GT)          (i = kept returns)
    b_c(f) = sum(uz_i r_i) / sum(uz_i^2)   (1-dof WLS elevation coefficient)
with uz_i the z-component of the unit ray in frame c. Under the bias model
v_meas = v_true + b * uz the coefficient IS the bias, and the sign matches the
solver correction v_corr = v_meas - b * uz (retired radar_zbias_fixed, and the
planned state).

Outputs per bag x frame-choice: series mean/std (Phase-H closure: sigma_z
0.26-0.64 m/s, ICINS means +0.08/+0.13), measurement floor (median per-frame
se), excess process spread, pooled R^2, correlation time tau (integrated ACF,
corr_time_irregular protocol from derive_stream_weights.py), regime-proxy
correlations corr(b,|v|), corr(b,|omega|), and the derived random-walk
increment per 0.3 s stride under an OU model:
    rw_sigma_stride = sigma_sig * sqrt(2 * stride / tau).

Frame decision (which uz the solver state should use): highest pooled R^2 +
most persistent series (|mean|/std, tau) wins; printed in the summary.

Usage (from analysis/):
  ../.venv/bin/python3 characterize_zbias_state.py            # 8-flight battery set
  ../.venv/bin/python3 characterize_zbias_state.py <alias>...
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from rosbag_loader import load_bag_topics                    # noqa: E402
from radar_velocity_utils import (predict_doppler_velocity,  # noqa: E402
                                  rotation_matrix_from_euler)

REPO = Path(__file__).resolve().parents[1]
BATTERY = ['slow_racing_best_velocity', 'fast_racing_best_velocity',
           'backflips_best_velocity', 'fast_racing_best_velocity_no_clustering',
           'icins_flight_1', 'icins_flight_2', 'icins_flight_3', 'icins_flight_4']
# Deployed own-platform extrinsics (locked pitch 27.5, NOT the yaml's 25.5 seed)
OWN_EULER = [180.0, 27.5, 0.0]
OWN_TRANS = None    # filled from extrinsics.yaml translation_m
ICINS_EULER = [-178.501, -0.099, 46.997]   # converter output (see bags.yaml)
FRAMES = ('sensor', 'body', 'world')
STRIDE = 0.3        # deployed SW stride, for the rw-increment suggestion
WINDOW = 3.0        # deployed SW window, for the window-level regression
MIN_N = 8           # minimum kept returns for a frame to contribute
MIN_UZ2 = 0.5       # identifiability floor on sum(uz^2)
# candidate-J bearing weight, variant 'b' (DIRCOS), exactly the deployed form
# (validate_live_solver _bearing_point_weights; X is sign-invariant in vv so
# the GT sensor velocity can be used directly)
BW_S0, BW_COEF, BW_COS_FLOOR = 0.0827, 0.03291, 0.15


def ransac_mask(P, v, rng, thresh=0.15, iters=150):
    """Deployment-semantics reve 3-point consensus (validate_live_solver
    _ransac_mask, sans the opt-in sigma gate): <5 returns bypass, all-true
    fallback when consensus is <5."""
    n = len(v)
    if n < 5:
        return np.ones(n, dtype=bool)
    Hn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            vc = np.linalg.solve(Hn[idx], v[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(Hn @ vc - v) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    return best if best is not None and best.sum() >= 5 else np.ones(n, dtype=bool)


def corr_time_irregular(t, x, max_lag=30.0, bin_w=0.5):
    """derive_stream_weights.py protocol (integrated positive-part ACF, first
    sustained zero), coarser bins: the series is per-frame (~10-30 Hz) and the
    bias process is slow, so 0.5 s bins / 30 s horizon."""
    x = x - x.mean()
    var = np.mean(x ** 2)
    if var <= 0 or len(t) < 20:
        return np.nan, None, np.nan
    order = np.argsort(t)
    t, x = t[order], x[order]
    n = len(t)
    nbins = int(max_lag / bin_w)
    num = np.zeros(nbins); cnt = np.zeros(nbins)
    j_hi = 0
    for i in range(n):
        while j_hi < n and t[j_hi] - t[i] < max_lag:
            j_hi += 1
        for j in range(i + 1, j_hi):
            b = int((t[j] - t[i]) / bin_w)
            num[b] += x[i] * x[j]; cnt[b] += 1
    rho = np.where(cnt > 20, num / np.maximum(cnt, 1) / var, 0.0)
    rho = np.clip(rho, 0.0, None)
    tau = 0.0
    for b in range(nbins):
        if rho[b] <= 0 and b > 2:
            break
        tau += 2 * rho[b] * bin_w
    rate = n / (t[-1] - t[0])
    return 1.0 / rate + tau, rho, rate


def bag_config(alias, bags, ext_cfg):
    if alias.startswith('icins'):
        return ICINS_EULER, np.array([0.01, 0.1, 0.06]), 0.0, None
    ov = (bags.get('extrinsics_overrides') or {}).get(alias, {})
    imu_mocap = ov.get('imu_mocap_offset_sec', ext_cfg['imu_mocap_offset_sec'])
    radar_imu = ov.get('radar_imu_offset_sec', ext_cfg['radar_imu_offset_sec'])
    vmax = 3.136 if 'best_velocity' in alias else 4.99
    return OWN_EULER, np.asarray(ext_cfg['translation_body_m'], float), \
        imu_mocap - radar_imu, vmax


def run_bag(alias, bags, ext_cfg):
    euler, trans, offset, vmax = bag_config(alias, bags, ext_cfg)
    R_bs = rotation_matrix_from_euler(*np.radians(euler))
    # none of the battery bags are in the flipped set (verified vs bags.yaml)

    bag_rel = bags['bags'][alias]
    bag = (REPO / bag_rel).resolve()
    if not bag.exists():
        bag = (REPO / '..' / bag_rel).resolve()
    start_off, dur = bags['timing'][alias]
    bd = load_bag_topics(str(bag), verbose=False)
    t0 = bd.start_time + start_off
    t1 = t0 + dur - 3.0

    gt = [s for s in bd.agiros_state if t0 - 1.0 <= s.timestamp <= t1 + 1.0]
    gt_t = np.array([s.timestamp for s in gt])
    gt_v = np.array([s.velocity for s in gt])
    gt_w = np.array([s.angular_velocity for s in gt])
    gt_R = Rotation.from_quat(np.array([s.orientation for s in gt]))
    if len(gt_t) > 27:   # low-pass GT velocity like the eval (Butter 4, 10 Hz)
        dt = np.median(np.diff(gt_t)); fs = 1.0 / dt; fc = min(10.0, fs * 0.4)
        b, a = butter(4, fc / (fs / 2), btype='low')
        for d in range(3):
            gt_v[:, d] = filtfilt(b, a, gt_v[:, d])
    slerp = Slerp(gt_t, gt_R)

    rng = np.random.default_rng(0)   # one seeded stream per bag, like the driver
    series = {c: {'t': [], 'b': [], 'se': [], 'spd': [], 'omg': []} for c in FRAMES}
    n_frames = n_used = 0
    pooled = {c: [0.0, 0.0] for c in FRAMES}     # [sum r^2 after fit, sum r^2]
    recs = {k: [] for k in ('t', 'r', 'uzs', 'uzb', 'uzw', 'wb', 'uw3')}
    for f in bd.radar_velocity:
        if f.velocities is None or f.positions is None:
            continue
        ts = f.timestamp + offset
        if not (t0 <= ts <= t1) or ts < gt_t[0] or ts > gt_t[-1]:
            continue
        n_frames += 1
        P = np.asarray(f.positions, float)
        v = np.asarray(f.velocities, float)
        rr = np.linalg.norm(P, axis=1)
        m = rr >= 0.2
        if m.sum() < MIN_N:
            continue
        P, v = P[m], v[m]
        v_world = np.array([np.interp(ts, gt_t, gt_v[:, d]) for d in range(3)])
        omega = np.array([np.interp(ts, gt_t, gt_w[:, d]) for d in range(3)])
        R_wb = slerp(ts).as_matrix()
        v_pred = predict_doppler_velocity(v_world, omega, R_wb, P, trans, R_bs)
        clean = np.ones(len(v), bool)
        if vmax is not None:                     # GT-aided unwrap (own bags)
            kk = np.round((v_pred - v) / (2 * vmax))
            v = v + kk * (2 * vmax)
            # aliased returns carry a broken on-chip bearing (TDM); deployment
            # weights them w_al=0.019, so the elevation regression drops them
            clean = kk == 0
        keep = ransac_mask(P, v, rng) & clean
        if keep.sum() < MIN_N:
            continue
        u_s = (P / np.maximum(rr[m][:, None], 1e-9))[keep]
        r = (v - v_pred)[keep]
        u_b = u_s @ R_bs.T
        u_w = u_b @ R_wb.T
        # candidate-J bearing weight from the GT sensor-frame velocity
        # (X is invariant to the vv sign convention)
        vv = R_bs.T @ (R_wb.T @ v_world + np.cross(omega, trans))
        cphi = np.maximum(u_s[:, 0], BW_COS_FLOOR)
        X = np.sqrt((vv[1] - u_s[:, 1] * vv[0] / cphi) ** 2
                    + (vv[2] - u_s[:, 2] * vv[0] / cphi) ** 2)
        wb = 1.0 / (1.0 + (BW_COEF * X / BW_S0) ** 2)
        recs['t'].append(np.full(keep.sum(), ts)); recs['r'].append(r)
        recs['uzs'].append(u_s[:, 2]); recs['uzb'].append(u_b[:, 2])
        recs['uzw'].append(u_w[:, 2]); recs['wb'].append(wb)
        recs['uw3'].append(u_w)
        spd = np.linalg.norm(v_world); omg = np.linalg.norm(omega)
        used = False
        for c, u in zip(FRAMES, (u_s, u_b, u_w)):
            uz = u[:, 2]
            s2 = float(uz @ uz)
            if s2 < MIN_UZ2:
                continue
            bz = float(uz @ r) / s2
            res = r - bz * uz
            sig2 = float(res @ res) / max(len(r) - 1, 1)
            series[c]['t'].append(ts); series[c]['b'].append(bz)
            series[c]['se'].append(np.sqrt(sig2 / s2))
            series[c]['spd'].append(spd); series[c]['omg'].append(omg)
            pooled[c][0] += float(res @ res); pooled[c][1] += float(r @ r)
            used = True
        n_used += used
    recs = {k: (np.concatenate(vs) if vs else np.array([]))
            for k, vs in recs.items()}     # uw3 concatenates to (N,3)
    return series, pooled, n_frames, n_used, recs, (t0, t1)


def window_series(ts, uz, r, w, t0, t1):
    """Windowed weighted elevation coefficient at the deployed geometry
    (3.0 s window, 0.3 s stride): the statistic an in-solver b_z state would
    estimate. Returns (t_end, b, se) arrays."""
    tw, bw, sew = [], [], []
    for te in np.arange(t0 + WINDOW, t1 + 1e-9, STRIDE):
        m = (ts > te - WINDOW) & (ts <= te)
        if m.sum() < 30:
            continue
        u, rr, ww = uz[m], r[m], w[m]
        s2 = float(np.sum(ww * u * u))
        if s2 < MIN_UZ2:
            continue
        b = float(np.sum(ww * u * rr)) / s2
        res = rr - b * u
        sig2 = float(np.sum(ww * res * res)) / max(int(m.sum()) - 1, 1)
        tw.append(te); bw.append(b); sew.append(np.sqrt(sig2 / s2))
    return np.array(tw), np.array(bw), np.array(sew)


def window_series_joint(ts, uw3, uzb, r, w, t0, t1):
    """Solver-faithful variant: per window jointly fit a world-frame velocity
    error (3 dof, what the spline can absorb) AND the body-frame elevation
    coefficient (what the b_z state absorbs):  r_i = -u_world,i . dv + b uzb_i.
    In level flight the b column is near-collinear with the world columns
    (the state is weakly observable, by construction of the model); the
    returned se carries that. Returns (t_end, b, se) arrays."""
    tw, bw, sew = [], [], []
    for te in np.arange(t0 + WINDOW, t1 + 1e-9, STRIDE):
        m = (ts > te - WINDOW) & (ts <= te)
        n = int(m.sum())
        if n < 30:
            continue
        A = np.column_stack([-uw3[m], uzb[m]])
        ww = w[m]; rr = r[m]
        Aw = A * ww[:, None]
        N = A.T @ Aw
        try:
            Ninv = np.linalg.inv(N + 1e-9 * np.eye(4))
        except np.linalg.LinAlgError:
            continue
        x = Ninv @ (Aw.T @ rr)
        res = rr - A @ x
        sig2 = float(np.sum(ww * res * res)) / max(n - 4, 1)
        se = np.sqrt(max(sig2 * Ninv[3, 3], 0.0))
        if se > 1.0:      # unobservable window (level flight): skip
            continue
        tw.append(te); bw.append(float(x[3])); sew.append(se)
    return np.array(tw), np.array(bw), np.array(sew)


def main():
    aliases = [a for a in sys.argv[1:] if not a.startswith('-')] or BATTERY
    bags = yaml.safe_load((REPO / 'analysis/config/bags.yaml').read_text())
    ext_cfg = yaml.safe_load((REPO / 'analysis/config/extrinsics.yaml').read_text())

    hdr = (f"{'bag':<12}{'frame':<8}{'n':>5} {'wmean':>7} {'std*':>6} {'MADs':>6} "
           f"{'floor':>6} {'sig':>6} {'R2p':>6} {'tau':>6} {'rw.3':>6} "
           f"{'c|v|':>6} {'c|w|':>6}")
    print(hdr); print('-' * len(hdr))
    summary = {}
    win_rows = []
    for alias in aliases:
        series, pooled, n_frames, n_used, recs, (t0, t1) = run_bag(alias, bags, ext_cfg)
        short = alias.replace('_best_velocity', '').replace('_racing', '') \
                     .replace('fast_no_clustering', 'fast2').replace('_flight_', '')
        for c in FRAMES:
            s = series[c]
            if len(s['b']) < 20:
                print(f"{short:<12}{c:<8} (too few frames: {len(s['b'])})")
                continue
            t = np.array(s['t']); b = np.array(s['b']); se = np.array(s['se'])
            spd = np.array(s['spd']); omg = np.array(s['omg'])
            w = 1.0 / np.maximum(se, 1e-3) ** 2
            wmean = float(np.sum(w * b) / np.sum(w))
            mads = 1.4826 * np.median(np.abs(b - np.median(b)))
            # winsorize at 5 robust sigmas: single low-diversity frames throw
            # the moment stats and the ACF; the bias process lives in the bulk
            bw = np.clip(b, np.median(b) - 5 * mads, np.median(b) + 5 * mads)
            std = bw.std(ddof=1)
            floor = float(np.median(se))
            sig = np.sqrt(max(std ** 2 - floor ** 2, 0.0))
            r2p = 1.0 - pooled[c][0] / pooled[c][1] if pooled[c][1] > 0 else np.nan
            tau, _, _ = corr_time_irregular(t, bw)
            rw = sig * np.sqrt(min(2 * STRIDE / tau, 2.0)) if tau and tau > 0 else np.nan
            cv = np.corrcoef(bw, spd)[0, 1] if spd.std() > 0 else np.nan
            cw = np.corrcoef(bw, omg)[0, 1] if omg.std() > 0 else np.nan
            print(f"{short:<12}{c:<8}{len(b):>5} {wmean:>+7.3f} {std:>6.3f} "
                  f"{mads:>6.3f} {floor:>6.3f} {sig:>6.3f} {r2p:>6.3f} "
                  f"{tau:>6.1f} {rw:>6.3f} {cv:>+6.2f} {cw:>+6.2f}")
            summary.setdefault(c, []).append((alias, wmean, sig, tau, r2p))
        print(f"{'':<12}(frames in window {n_frames}, used {n_used})")
        # window-level pass (deployed 3.0 s / 0.3 s geometry), unit vs J weights
        if len(recs['t']):
            ones = np.ones_like(recs['r'])
            for c, uzk in (('body', 'uzb'), ('world', 'uzw')):
                for ch, w in (('ols', ones), ('bwJ', recs['wb'])):
                    tw, bw, sew = window_series(recs['t'], recs[uzk],
                                                recs['r'], w, t0, t1)
                    if len(bw) < 8:
                        continue
                    dif = np.diff(bw)[np.abs(np.diff(tw) - STRIDE) < 0.05]
                    win_rows.append(
                        (short, c, ch, len(bw), bw.mean(), bw.std(ddof=1),
                         float(np.median(sew)),
                         dif.std(ddof=1) if len(dif) > 6 else np.nan))
            # solver-faithful joint fit: world-velocity dof + BODY b_z column
            for ch, w in (('ols', ones), ('bwJ', recs['wb'])):
                tw, bw, sew = window_series_joint(recs['t'], recs['uw3'],
                                                  recs['uzb'], recs['r'],
                                                  w, t0, t1)
                if len(bw) < 8:
                    win_rows.append((short, 'joint', ch, len(bw),
                                     np.nan, np.nan, np.nan, np.nan))
                    continue
                dif = np.diff(bw)[np.abs(np.diff(tw) - STRIDE) < 0.05]
                win_rows.append(
                    (short, 'joint', ch, len(bw), bw.mean(), bw.std(ddof=1),
                     float(np.median(sew)),
                     dif.std(ddof=1) if len(dif) > 6 else np.nan))

    if win_rows:
        print("\n=== window-level b_z (3.0s/0.3s, deployment geometry) ===")
        print(f"{'bag':<12}{'frame':<7}{'chan':<5}{'nwin':>5} {'mean':>7} "
              f"{'std':>6} {'floor':>6} {'d.3':>6}")
        for row in win_rows:
            print(f"{row[0]:<12}{row[1]:<7}{row[2]:<5}{row[3]:>5} "
                  f"{row[4]:>+7.3f} {row[5]:>6.3f} {row[6]:>6.3f} {row[7]:>6.3f}")
        print("(mean = persistent component the state would absorb; d.3 = std of "
              "consecutive-window diffs = direct rw increment per 0.3 s stride; "
              "bwJ = candidate-J bearing weights, ols = unit weights)")

    print("\n=== frame decision (pooled R^2 / persistence across bags) ===")
    for c in FRAMES:
        rows = summary.get(c, [])
        if not rows:
            continue
        r2s = np.array([r[4] for r in rows])
        means = np.array([r[1] for r in rows])
        sigs = np.array([r[2] for r in rows])
        taus = np.array([r[3] for r in rows if np.isfinite(r[3])])
        print(f"{c:<8} R2p geomean {np.exp(np.mean(np.log(np.maximum(r2s, 1e-6)))):.4f}  "
              f"|mean| rms {np.sqrt(np.mean(means ** 2)):.3f}  "
              f"sig med {np.median(sigs):.3f}  tau med {np.median(taus):.1f}s")
    print("\nPrior/rw suggestion (per chosen frame): prior sigma ~ rms(|mean|) + med(sig); "
          "rw sigma per 0.3 s stride = sig * sqrt(2*0.3/tau) (OU), printed per bag as rw.3.")


if __name__ == '__main__':
    main()
