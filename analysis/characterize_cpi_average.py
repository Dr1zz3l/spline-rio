#!/usr/bin/env python3
"""Estimator-free check of the CPI-averaged Doppler model (BACKFLIPS_PLAN.md
experiment 3, 2026-09-16), run BEFORE any solver change.

Hypothesis: a frame integrates the CPI (49.92 ms, `radar_config_spec.py`);
at 10 rad/s the bearing sweeps 29 deg and the lever-arm term omega x t_bs
turns inside the CPI, so the instantaneous forward model at the frame time
is wrong even with a perfect clock.  Test: replace the instantaneous
prediction v_pred(t_f) by the mean over K quadrature points in
[t_f - CPI/2, t_f + CPI/2] of the SAME forward model driven by the
pose-derived reference, and compare both against the measured Doppler.

Two things are printed per bag (backflips is the target; the racing bags
and fast2 / circle are the controls):
  1. the size of the correction itself, rms(v_mean - v_inst), by |omega| bin
     (if this is far below the per-return noise floor, no solver can see it);
  2. the residual rms of each model against the measurement at EACH MODEL'S
     OWN best lag (the deployed latency was calibrated with the instantaneous
     model on |a| > 4 frames; a model change can move the stamp, so both are
     re-scanned), on the |a| > 4 frames and on all frames.
First-order expectation (worked out in the worklog entry): for a symmetric
window the mean equals the centre value plus O(CPI^2) curvature terms of
order omega^2 CPI^2 / 24 x |omega x t_bs| ~ 1 cm/s at 10 rad/s, so the
correction should be ~1 % of the flip residual.  This script measures it.

Run from analysis/:  ../.venv/bin/python3 characterize_cpi_average.py [bag ...]
"""
import sys
from pathlib import Path
sys.path.insert(0, 'lib'); sys.path.insert(0, '.')
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import predict_doppler_velocity, unwrap_doppler, rotation_matrix_from_euler

_cfg = load_config()
BAGS = _cfg['bags']['bags']; TIMING = _cfg['bags']['timing']; EXT = _cfg['extrinsics']
FLIPPED = set(_cfg['bags'].get('flipped', []) or []); MIN_RANGE = _cfg['solver']['min_range']
RC = _cfg['bags'].get('radar_config', {})
R_BASE = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0])); T_BASE = np.array(EXT['translation_body_m'])
OFF = EXT['imu_mocap_offset_sec'] - EXT['radar_imu_offset_sec']
FLIGHTS = sys.argv[1:] or ['backflips_best_velocity', 'slow_racing_best_velocity', 'fast_racing_best_velocity',
                           'fast_racing_best_velocity_no_clustering', 'circle_best_velocity']
CPI = 0.04992
K = 9
TAUS = (np.arange(K) - (K - 1) / 2) / (K - 1) * CPI      # uniform, includes the end points
LAGS = np.arange(-0.16, -0.0799, 0.001)
OMEGA_BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 11), (11, 20)]
rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def refine(cost):
    k = int(np.nanargmin(cost))
    if 0 < k < len(cost) - 1:
        y0, y1, y2 = cost[k - 1:k + 2]; d = y0 - 2 * y1 + y2
        return LAGS[k] + (LAGS[1] - LAGS[0]) * 0.5 * (y0 - y2) / d if d > 0 else LAGS[k]
    return LAGS[k]


def run(key):
    d = load_bag_topics(str(Path('..') / BAGS[key]), verbose=False)
    t0 = d.start_time + TIMING[key][0]; t1 = t0 + TIMING[key][1]
    S = [x for x in d.agiros_state if t0 - 0.6 <= x.timestamp <= t1 + 0.6]
    tm = np.array([x.timestamp for x in S]); keep = np.ones(len(tm), bool); keep[1:] = np.diff(tm) > 1e-3
    tm = tm[keep]; S = [x for x, k in zip(S, keep) if k]
    Q = np.array([x.orientation for x in S]); okq = np.ones(len(tm), bool); okq[1:] = np.abs(np.diff(Q, axis=0)).max(axis=1) > 1e-12
    sl = Slerp(tm[okq], Rotation.from_quat(Q[okq]))
    v_i = interp1d(tm, np.array([x.velocity for x in S]), axis=0, bounds_error=False, fill_value=np.nan)
    a_i = interp1d(tm, np.array([x.acceleration for x in S]), axis=0, bounds_error=False, fill_value=np.nan)
    # body rate from the GYRO (1 kHz, the reference rate is 15 Hz-smoothed): inside a 50 ms window this matters
    imu = [m for m in d.imu_data if t0 - 1 <= m.timestamp <= t1 + 1]
    ti = np.array([m.timestamp for m in imu]) + EXT['imu_mocap_offset_sec']
    g_i = interp1d(ti, np.array([m.angular_velocity for m in imu]), axis=0, bounds_error=False, fill_value=np.nan)
    R_s, T_s = R_BASE, T_BASE.copy()
    if key in FLIPPED:
        Rz = rotation_matrix_from_euler(0.0, 0.0, np.pi); R_s = Rz @ R_BASE; T_s = Rz @ T_BASE
    vmax = (RC.get('best_velocity', {}) if 'best_velocity' in key else RC.get('default', {})).get('v_max', 4.99)

    def predict(t, P):
        vw, ob = v_i(t), g_i(t)
        if not (np.all(np.isfinite(vw)) and np.all(np.isfinite(ob))): return None
        return predict_doppler_velocity(vw, ob, sl(np.clip(t, tm[okq][0], tm[okq][-1])).as_matrix(), P, T_s, R_s)

    # 1. correction size at the deployed lag, by |omega| bin; 2. per-frame lag-scan costs for both models
    corr = {b: [] for b in OMEGA_BINS}; res_inst = {b: [] for b in OMEGA_BINS}; res_mean = {b: [] for b in OMEGA_BINS}
    scan = []   # (|a|, ss_inst[L], n_inst[L], ss_mean[L], n_mean[L])
    for fr in d.radar_velocity:
        if not (t0 <= fr.timestamp <= t1): continue
        P = np.asarray(fr.positions); rng = np.asarray(fr.ranges) if fr.ranges is not None else np.linalg.norm(P, axis=1)
        k = rng >= MIN_RANGE
        if k.sum() == 0: continue
        P = P[k]; vm = np.asarray(fr.velocities)[k]
        tf = fr.timestamp + OFF
        aw = a_i(tf); ob = g_i(tf)
        if not (np.all(np.isfinite(aw)) and np.all(np.isfinite(ob))): continue
        p0 = predict(tf, P)
        pk = [predict(tf + tau, P) for tau in TAUS]
        if p0 is None or any(p is None for p in pk): continue
        pm = np.mean(pk, axis=0)
        om = np.linalg.norm(ob)
        for b in OMEGA_BINS:
            if b[0] <= om < b[1]:
                meas = unwrap_doppler(vm, p0, vmax); r0 = meas - p0; rm = meas - pm
                inl = np.abs(r0) < 1.0
                corr[b].append(pm[inl] - p0[inl]); res_inst[b].append(r0[inl]); res_mean[b].append(rm[inl])
        ssi = np.full(len(LAGS), np.nan); ni = np.zeros(len(LAGS), int); ssm = ssi.copy(); nm = ni.copy()
        for li, L in enumerate(LAGS):
            t = fr.timestamp + L
            q0 = predict(t, P)
            if q0 is None: continue
            qk = [predict(t + tau, P) for tau in TAUS]
            if any(q is None for q in qk): continue
            qm = np.mean(qk, axis=0)
            r0 = unwrap_doppler(vm, q0, vmax) - q0; r0 = r0[np.abs(r0) < 1.0]
            rm = unwrap_doppler(vm, qm, vmax) - qm; rm = rm[np.abs(rm) < 1.0]
            ssi[li] = np.sum(r0 ** 2); ni[li] = len(r0); ssm[li] = np.sum(rm ** 2); nm[li] = len(rm)
        scan.append((np.linalg.norm(aw), ssi, ni, ssm, nm))

    print(f"-- {key}  (frames {len(scan)}, K={K} points over +-{CPI/2*1e3:.1f} ms, deployed lag {OFF*1e3:+.0f} ms)")
    print("   |omega| bin : n returns | rms(v_mean - v_inst) | residual rms instantaneous -> CPI-mean")
    for b in OMEGA_BINS:
        if not corr[b]: continue
        c = np.concatenate(corr[b]); r0 = np.concatenate(res_inst[b]); rm = np.concatenate(res_mean[b])
        print(f"   {b[0]:2d}-{b[1]:<2d}      : {len(c):6d} | {rms(c):.4f} (max {np.abs(c).max():.3f}) | {rms(r0):.4f} -> {rms(rm):.4f}  ({(rms(rm)/rms(r0)-1)*100:+.2f} %)")
    c = np.concatenate([np.concatenate(v) for v in corr.values() if v]); r0 = np.concatenate([np.concatenate(v) for v in res_inst.values() if v]); rm = np.concatenate([np.concatenate(v) for v in res_mean.values() if v])
    print(f"   all         : {len(c):6d} | {rms(c):.4f} | {rms(r0):.4f} -> {rms(rm):.4f}  ({(rms(rm)/rms(r0)-1)*100:+.2f} %)")
    for name, sel in (("|a| > 4 frames (deployed protocol)", lambda s: s[0] > 4.0), ("all frames", lambda s: True)):
        sub = [s for s in scan if sel(s)]
        if len(sub) < 8: continue
        ci = np.sqrt(np.nansum([s[1] for s in sub], axis=0) / np.maximum(np.sum([s[2] for s in sub], axis=0), 1))
        cm = np.sqrt(np.nansum([s[3] for s in sub], axis=0) / np.maximum(np.sum([s[4] for s in sub], axis=0), 1))
        print(f"   lag scan, {name} (n={len(sub)}): instantaneous best {refine(ci)*1e3:+.1f} ms rms {np.nanmin(ci):.4f};"
              f" CPI-mean best {refine(cm)*1e3:+.1f} ms rms {np.nanmin(cm):.4f}  ({(np.nanmin(cm)/np.nanmin(ci)-1)*100:+.2f} %)")


print(f"CPI {CPI*1e3:.2f} ms; quadrature K={K} uniform incl. end points; forward model: pose-derived velocity, raw-pose attitude, gyro rate; deployed extrinsics")
for key in FLIGHTS:
    run(key)
