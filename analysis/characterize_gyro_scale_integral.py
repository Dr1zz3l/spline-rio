#!/usr/bin/env python3
"""Gyro scale factor from the TOTAL SWEPT ANGLE (BACKFLIPS_PLAN.md experiment 4
without the bench, 2026-09-16).

The rate-vs-rate fit (`characterize_gyro_scale.py`, 2026-09-10) is limited
by the MoCap attitude-rate noise (0.3-0.5 rad/s at 5 ms baselines) and by the
flip-phase tracking glitches, and came out inconclusive at the sub-percent
level.  A scale error integrates COHERENTLY with the swept angle while the
MoCap attitude noise at the segment end points does not, so the integral is
the sensitive estimator: over a 3 s stretch of continuous flipping (~25 rad
about the flip axis) a 0.5 % scale error is 0.12 rad = 7 deg against an
end-point attitude noise of ~0.01 rad.  The backflips bag is the "many full
revolutions" recording the plan asks for, about the body y axis; the racing
and circle bags supply the z axis; nothing sustains a roll (x stays weak).

Per segment [ta, tb] (non-overlapping, length L, end points required to be
free of MoCap tracking glitches): integrate the gyro from R = I with
  omega_corr = (I + diag(s)) exp([c]_x) (z - b),   t_imu + imu_mocap_offset + eps
and compare with the MoCap relative rotation R_a^T R_b:
  e = Log(Delta_mocap^-1 Delta_gyro)   (3, body frame at tb).
Unknowns: per-axis scale s (3), gyro bias b (3, seeded by the pre-flight
stationary median), IMU-to-body misalignment c (3), clock residual eps (1).
Jacobians by central finite differences (re-integration), robust least
squares over all segments; per bag and pooled (s, c, eps shared; bias per
bag).  The pre-flight stationary bias alone (b fixed) is printed too, since
in continuous flipping s*theta and b*Delta t are nearly collinear.

Run from analysis/:  ../.venv/bin/python3 characterize_gyro_scale_integral.py [bag ...]
"""
import sys
from pathlib import Path
sys.path.insert(0, 'lib'); sys.path.insert(0, '.')
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics

_cfg = load_config()
BAGS = _cfg['bags']['bags']; TIMING = _cfg['bags']['timing']; EXT = _cfg['extrinsics']
IMU_OFF = EXT['imu_mocap_offset_sec']
FLIGHTS = sys.argv[1:] or ['backflips_best_velocity', 'fast_racing_best_velocity', 'slow_racing_best_velocity']
SEG_LEN = [1.0, 2.0]
SEG_STRIDE = 0.5            # overlapping windows; SEs scaled by sqrt(L / stride)
MAX_RATE, MAX_SPEED = 25.0, 12.0
GLITCH_MARGIN = 0.04


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def load(key):
    d = load_bag_topics(str(Path('..') / BAGS[key]), verbose=False)
    t0 = d.start_time + TIMING[key][0]; t1 = t0 + TIMING[key][1]
    ti = np.array([m.timestamp for m in d.imu_data]); gz = np.array([m.angular_velocity for m in d.imu_data]); az = np.array([m.linear_acceleration for m in d.imu_data])
    # pre-flight bias: the motors-OFF ground stretch (as characterize_imu_noise.py
    # find_ground_windows: grow from the bag start while the 99th percentile of
    # |gyro| per 0.1 s stays below 0.15 rad/s, before the MoCap take-off)
    tmp = np.array([p.timestamp for p in d.mocap_pose]); zp = np.array([p.position[2] for p in d.mocap_pose])
    z0 = float(np.median(zp[tmp < tmp[0] + 1.0])); air = tmp[zp > z0 + 0.15]
    t_takeoff = float(air.min()) if len(air) else float(tmp.max())
    gn = np.linalg.norm(gz, axis=1); lo = max(ti.min(), d.start_time + 0.15); hi = lo
    while hi + 0.1 < t_takeoff:
        m = (ti >= lo) & (ti < hi + 0.1)
        if m.sum() > 20 and np.percentile(gn[m], 99) > 0.15: break
        hi += 0.1
    if hi - lo >= 0.8:
        m = (ti >= lo) & (ti < hi); b0 = np.median(gz[m], axis=0); b0_info = f"motors-off {hi-lo:.1f} s from t={lo-d.start_time:.1f} s, take-off at {t_takeoff-d.start_time:.1f} s"
    else:
        b0 = np.zeros(3); b0_info = "NO motors-off stretch found, bias seed 0"
    # raw MoCap pose, glitch flags
    tm = np.array([p.timestamp for p in d.mocap_pose]); q = np.array([p.orientation for p in d.mocap_pose]); P = np.array([p.position for p in d.mocap_pose])
    keep = np.ones(len(tm), bool); keep[1:] = (np.diff(tm) > 1e-3) & (np.abs(np.diff(q, axis=0)).max(axis=1) > 1e-12)
    tm, q, P = tm[keep], q[keep], P[keep]
    R = Rotation.from_quat(q); st = np.diff(tm)
    rate = (R[:-1].inv() * R[1:]).magnitude() / st; speed = np.linalg.norm(np.diff(P, axis=0), axis=1) / st
    bad_t = tm[1:][(rate > MAX_RATE) | (speed > MAX_SPEED)]
    sl = Slerp(tm, R)
    m = (ti + IMU_OFF >= t0 - 0.5) & (ti + IMU_OFF <= t1 + 0.5)
    return dict(key=key, t0=t0, t1=t1, ti=ti[m], gz=gz[m], b0=b0, b0_info=b0_info, tm=tm, sl=sl, bad_t=bad_t, start=d.start_time)


def integrate(D, ta, tb, s, b, c, eps):
    """Gyro-integrated relative rotation over [ta, tb] (MoCap clock), midpoint rule."""
    t = D['ti'] + IMU_OFF + eps
    m = (t >= ta) & (t <= tb)
    tt = t[m]; z = D['gz'][m]
    if len(tt) < 10: return None
    C = Rotation.from_rotvec(c).as_matrix()
    w = (1.0 + s)[None, :] * (C @ (z - b).T).T
    dt = np.diff(tt); wm = 0.5 * (w[1:] + w[:-1])
    # end corrections so the integral spans exactly [ta, tb]
    rv = np.concatenate([[(tt[0] - ta) * w[0]], wm * dt[:, None], [(tb - tt[-1]) * w[-1]]])
    R = Rotation.from_rotvec(rv)
    while len(R) > 1:              # ordered pairwise tree product (vectorised)
        n = len(R)
        if n % 2:
            R = Rotation.concatenate([R[0:n - 1:2] * R[1:n:2], R[n - 1:]])
        else:
            R = R[0::2] * R[1::2]
    return R[0]


def segments(D, L):
    segs = []
    ta = D['t0'] + 0.3
    while ta + L <= D['t1'] - 0.3:
        tb = ta + L
        ok = all(np.min(np.abs(D['bad_t'] - x)) > GLITCH_MARGIN for x in (ta, tb)) if len(D['bad_t']) else True
        if ok: segs.append((ta, tb))
        ta += SEG_STRIDE
    return segs


def residual_vec(D, segs, s, b, c, eps):
    out = []
    for ta, tb in segs:
        Dg = integrate(D, ta, tb, s, b, c, eps)
        Dm = D['sl'](ta).inv() * D['sl'](tb)
        out.append((Dm.inv() * Dg).as_rotvec() if Dg is not None else np.zeros(3))
    return np.concatenate(out)


def swept(D, segs):
    """Integrated |omega| per axis (the swept angle, unwrapped) per segment."""
    t = D['ti'] + IMU_OFF; out = []
    for ta, tb in segs:
        m = (t >= ta) & (t <= tb); out.append(np.trapz(np.abs(D['gz'][m] - D['b0']), t[m], axis=0))
    return np.array(out)


def fit(bags, segs_by_bag, which, L):
    """which: subset of ('s','b','c','eps'); bias per bag, others shared."""
    nb = len(bags)
    def unpack(x):
        i = 0; s = np.zeros(3); c = np.zeros(3); eps = 0.0; bs = [D['b0'].copy() for D in bags]
        if 's' in which: s = x[i:i+3]; i += 3
        if 'c' in which: c = x[i:i+3]; i += 3
        if 'eps' in which: eps = x[i]; i += 1
        if 'b' in which:
            for k in range(nb): bs[k] = bags[k]['b0'] + x[i:i+3]; i += 3
        return s, c, eps, bs
    n = 3 * ('s' in which) + 3 * ('c' in which) + ('eps' in which) + 3 * nb * ('b' in which)
    def res(x):
        s, c, eps, bs = unpack(x)
        return np.concatenate([residual_vec(D, sg, s, bs[k], c, eps) for k, (D, sg) in enumerate(zip(bags, segs_by_bag))])
    x0 = np.zeros(n)
    r0 = res(x0)
    sol = least_squares(res, x0, loss='huber', f_scale=0.03, diff_step=np.full(n, 1e-3), x_scale=np.concatenate([np.full(n - 3*nb*('b' in which), 1e-2), np.full(3*nb*('b' in which), 1e-3)]) if n else None)
    r1 = res(sol.x)
    J = sol.jac; sig = 1.4826 * np.median(np.abs(r1 - np.median(r1))) if len(r1) > 3 else np.nan
    cov = np.linalg.pinv(J.T @ J) * sig ** 2 * (L / SEG_STRIDE); se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    s, c, eps, bs = unpack(sol.x); _, _, _, _ = unpack(se)
    i = 0; txt = []
    if 's' in which: txt.append(f"s = [{', '.join(f'{100*v:+.3f}' for v in s)}] +- [{', '.join(f'{100*v:.3f}' for v in se[i:i+3])}] %"); i += 3
    if 'c' in which: txt.append(f"c = {np.degrees(c).round(2)} +- {np.degrees(se[i:i+3]).round(2)} deg"); i += 3
    if 'eps' in which: txt.append(f"eps = {eps*1e3:+.2f} +- {se[i]*1e3:.2f} ms"); i += 1
    if 'b' in which:
        for k in range(nb): txt.append(f"db[{bags[k]['key'][:5]}] = {(1e3*x_to_db(sol.x, i, k)).round(2)} +- {(1e3*se[i+3*k:i+3*k+3]).round(2)} mrad/s")
    return f"rms {np.sqrt(np.mean(r0**2)):.4f} -> {np.sqrt(np.mean(r1**2)):.4f} rad (robust {sig:.4f}); " + '; '.join(txt)


def x_to_db(x, i, k):
    return x[i + 3 * k:i + 3 * k + 3]


print(f"IMU offset {IMU_OFF*1e3:+.1f} ms; windows at a {SEG_STRIDE} s stride (overlapping), end points > {GLITCH_MARGIN*1e3:.0f} ms from any MoCap glitch\n")
data = {k: load(k) for k in FLIGHTS}
for k in FLIGHTS:
    D = data[k]
    print(f"== {k}: bias seed {D['b0'].round(4)} rad/s ({D['b0_info']}); MoCap glitches in flight: {int(((D['bad_t'] > D['t0']) & (D['bad_t'] < D['t1'])).sum())}")
for L in SEG_LEN:
    print(f"\n================ segment length {L:.0f} s ================")
    segs = {k: segments(data[k], L) for k in FLIGHTS}
    for k in FLIGHTS:
        D = data[k]; sg = segs[k]
        if len(sg) < 3: print(f"-- {k}: {len(sg)} segments, skipped"); continue
        th = swept(D, sg)
        e0 = residual_vec(D, sg, np.zeros(3), D['b0'], np.zeros(3), 0.0).reshape(-1, 3)
        print(f"-- {k}: {len(sg)} segments; swept angle per segment (mean |theta| per axis) = {th.mean(axis=0).round(1)} rad; raw end-point error rms per axis {np.sqrt(np.mean(e0**2, axis=0)).round(4)} rad")
        for which in (('s',), ('s', 'eps'), ('s', 'c', 'eps'), ('s', 'b', 'c', 'eps')):
            print(f"   fit {'+'.join(which):<12s}: {fit([D], [sg], which, L)}")
    pool = [k for k in FLIGHTS if len(segs[k]) >= 3]
    if len(pool) > 1:
        print(f"-- POOLED {pool} (s, c, eps shared; bias per bag)")
        for which in (('s', 'c', 'eps'), ('s', 'b', 'c', 'eps')):
            print(f"   fit {'+'.join(which):<12s}: {fit([data[k] for k in pool], [segs[k] for k in pool], which, L)}")
        for k in pool:
            others = [o for o in pool if o != k]
            print(f"   without {k[:9]:<10s} (s+c+eps): {fit([data[o] for o in others], [segs[o] for o in others], ('s', 'c', 'eps'), L)}")
