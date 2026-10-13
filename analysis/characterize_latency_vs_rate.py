#!/usr/bin/env python3
"""Radar latency as a function of the body rate and the acceleration
(BACKFLIPS_PLAN.md experiment 2, 2026-09-16).

Deployed: ONE radar latency per day and chirp configuration (-122 ms total,
`radar_imu_offset_sec` 0.127), measured on the |a| > 4 m/s^2 frames of the
three racing-configuration bags (`characterize_time_offsets.py`).  The
backflips bag alone reads -136 ms on all frames but -123 on |a| > 7
(`worklog/2026-09-11_radar-clock.md` sec. 3), a regime dependence with no
cause.  Question here: is the lag a function of the body rate |omega| (or of
|a|) that the bags SHARE in their overlapping bins (fast racing turns reach
8 rad/s, flips 10-16), or a backflips-only offset?

Protocol = the deployed one per frame (per-return Doppler residual against
the forward model driven by the pose-derived reference at the sensor-clock
frame stamp + lag; unwrap toward the prediction; gross outliers |r| > 1 m/s
dropped; plain rms), pooled per bin over the racing-configuration bags,
best lag per bin by parabolic refinement of the pooled cost.  Bins are
assigned at the deployed lag.  Printed per bag and pooled, for every bin:
count, best lag, and the cost at the best lag vs at -122 (the depth of the
minimum tells whether the bin can see a lag at all).  fast2 (no clustering)
and circle (old chirp config) are printed as held-out bags.

Run from analysis/:  ../.venv/bin/python3 characterize_latency_vs_rate.py [bag ...]
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
OFF_DEPLOYED = EXT['imu_mocap_offset_sec'] - EXT['radar_imu_offset_sec']
POOL = ['slow_racing_best_velocity', 'fast_racing_best_velocity', 'backflips_best_velocity']
HELD = ['fast_racing_best_velocity_no_clustering', 'circle_best_velocity']
FLIGHTS = sys.argv[1:] or POOL + HELD
LAGS = np.arange(-0.20, -0.0599, 0.001)
OMEGA_BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 11), (11, 20)]
ACC_BINS = [(0, 4), (4, 7), (7, 10), (10, 40)]
i122 = int(np.argmin(np.abs(LAGS + 0.122)))


def refine(cost):
    if np.all(np.isnan(cost)): return np.nan
    k = int(np.nanargmin(cost))
    if 0 < k < len(cost) - 1:
        y0, y1, y2 = cost[k - 1:k + 2]
        d = y0 - 2 * y1 + y2
        return LAGS[k] + (LAGS[1] - LAGS[0]) * 0.5 * (y0 - y2) / d if d > 0 else LAGS[k]
    return LAGS[k]


def frame_costs(key):
    """Per frame: (|omega|, |a|, sum of squared residuals per lag, count per lag)."""
    d = load_bag_topics(str(Path('..') / BAGS[key]), verbose=False)
    t0 = d.start_time + TIMING[key][0]; t1 = t0 + TIMING[key][1]
    S = [x for x in d.agiros_state if t0 - 0.6 <= x.timestamp <= t1 + 0.6]
    tm = np.array([x.timestamp for x in S]); keep = np.ones(len(tm), bool); keep[1:] = np.diff(tm) > 1e-3
    tm = tm[keep]; S = [x for x, k in zip(S, keep) if k]
    Q = np.array([x.orientation for x in S]); okq = np.ones(len(tm), bool); okq[1:] = np.abs(np.diff(Q, axis=0)).max(axis=1) > 1e-12
    sl = Slerp(tm[okq], Rotation.from_quat(Q[okq]))
    om_i = interp1d(tm, np.array([x.angular_velocity for x in S]), axis=0, bounds_error=False, fill_value=np.nan)
    v_i = interp1d(tm, np.array([x.velocity for x in S]), axis=0, bounds_error=False, fill_value=np.nan)
    a_i = interp1d(tm, np.array([x.acceleration for x in S]), axis=0, bounds_error=False, fill_value=np.nan)
    R_s, T_s = R_BASE, T_BASE.copy()
    if key in FLIPPED:
        Rz = rotation_matrix_from_euler(0.0, 0.0, np.pi); R_s = Rz @ R_BASE; T_s = Rz @ T_BASE
    vmax = (RC.get('best_velocity', {}) if 'best_velocity' in key else RC.get('default', {})).get('v_max', 4.99)
    out = []
    for fr in d.radar_velocity:
        if not (t0 <= fr.timestamp <= t1): continue
        P = np.asarray(fr.positions); rng = np.asarray(fr.ranges) if fr.ranges is not None else np.linalg.norm(P, axis=1)
        k = rng >= MIN_RANGE
        if k.sum() == 0: continue
        P = P[k]; vm = np.asarray(fr.velocities)[k]
        td = fr.timestamp + OFF_DEPLOYED
        ob0, aw0 = om_i(td), a_i(td)
        if not (np.all(np.isfinite(ob0)) and np.all(np.isfinite(aw0))): continue
        ss = np.full(len(LAGS), np.nan); nn = np.zeros(len(LAGS), int)
        tl = fr.timestamp + LAGS
        vw, ob = v_i(tl), om_i(tl)
        fin = np.all(np.isfinite(vw), axis=1) & np.all(np.isfinite(ob), axis=1)
        if not fin.any(): continue
        Rl = sl(np.clip(tl[fin], tm[okq][0], tm[okq][-1])).as_matrix()
        for j, i in enumerate(np.where(fin)[0]):
            pred = predict_doppler_velocity(vw[i], ob[i], Rl[j], P, T_s, R_s)
            r = unwrap_doppler(vm, pred, vmax) - pred; r = r[np.abs(r) < 1.0]
            ss[i] = np.sum(r ** 2); nn[i] = len(r)
        out.append((np.linalg.norm(ob0), np.linalg.norm(aw0), ss, nn))
    return out


def table(rows, label):
    """rows: list of (|omega|, |a|, ss, nn).  Best lag per bin (pooled rms)."""
    def line(name, sel):
        sub = [r for r in rows if sel(r)]
        if len(sub) < 8: return f"   {name:<22s} n={len(sub):4d}  (too few)"
        ss = np.nansum([r[2] for r in sub], axis=0); nn = np.sum([r[3] for r in sub], axis=0)
        cost = np.sqrt(ss / np.maximum(nn, 1)); cost[nn < 50] = np.nan
        best = refine(cost); k = int(np.nanargmin(cost))
        depth = (cost[i122] - cost[k]) / cost[k] * 100
        return f"   {name:<22s} n={len(sub):4d}  best {best*1e3:+7.1f} ms  rms {cost[k]:.3f} (at -122: {cost[i122]:.3f}, +{depth:.1f} %)"
    print(f"-- {label}")
    print("   by |omega| (all frames)")
    for lo, hi in OMEGA_BINS: print(line(f"|omega| {lo}-{hi}", lambda r, lo=lo, hi=hi: lo <= r[0] < hi))
    print("   by |omega|, frames with |a| > 4 only (the deployed selection)")
    for lo, hi in OMEGA_BINS: print(line(f"|omega| {lo}-{hi}", lambda r, lo=lo, hi=hi: lo <= r[0] < hi and r[1] > 4.0))
    print("   by |a| (all frames)")
    for lo, hi in ACC_BINS: print(line(f"|a| {lo}-{hi}", lambda r, lo=lo, hi=hi: lo <= r[1] < hi))
    print("   by |a|, frames with |omega| < 4 only")
    for lo, hi in ACC_BINS: print(line(f"|a| {lo}-{hi}", lambda r, lo=lo, hi=hi: lo <= r[1] < hi and r[0] < 4.0))
    print(line("all frames", lambda r: True)); print(line("|a| > 4 (deployed)", lambda r: r[1] > 4.0))
    # bootstrap over frames: how well determined is the best lag of a selection?
    rng = np.random.default_rng(0)
    def boot(name, sel, B=300):
        sub = [r for r in rows if sel(r)]
        if len(sub) < 8: return
        SS = np.array([r[2] for r in sub]); NN = np.array([r[3] for r in sub]); best = []
        for _ in range(B):
            i = rng.integers(0, len(sub), len(sub))
            ss = np.nansum(SS[i], axis=0); nn = NN[i].sum(axis=0)
            cost = np.sqrt(ss / np.maximum(nn, 1)); cost[nn < 50] = np.nan
            best.append(refine(cost))
        b = np.array(best) * 1e3; b = b[np.isfinite(b)]
        if len(b) < 10: return
        print(f"   bootstrap {name:<18s} n={len(sub):4d}  best lag median {np.median(b):+7.1f} ms, 68 % interval [{np.percentile(b, 16):+.1f}, {np.percentile(b, 84):+.1f}], 95 % [{np.percentile(b, 2.5):+.1f}, {np.percentile(b, 97.5):+.1f}]")
    boot("|a| > 4 (deployed)", lambda r: r[1] > 4.0); boot("|a| > 7", lambda r: r[1] > 7.0)
    for lo, hi in OMEGA_BINS: boot(f"|omega| {lo}-{hi}", lambda r, lo=lo, hi=hi: lo <= r[0] < hi)


print(f"deployed radar total offset {OFF_DEPLOYED*1e3:+.0f} ms (bins assigned there); lags {LAGS[0]*1e3:.0f}..{LAGS[-1]*1e3:.0f} ms; sensor-clock frame stamps; pose-derived reference")
data = {}
for key in FLIGHTS:
    data[key] = frame_costs(key); table(data[key], key)
pool = [k for k in FLIGHTS if k in POOL]
if len(pool) > 1:
    table([r for k in pool for r in data[k]], f"POOLED {pool}")
