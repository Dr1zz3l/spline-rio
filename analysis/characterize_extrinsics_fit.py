#!/usr/bin/env python3
"""Estimator-free fit of the radar extrinsics (R_bs as a small rotation about
the deployed euler(180, 27.5, 0), t_bs) from all per-return Dopplers against
the pose-derived reference, at the deployed clocks (2026-09-12).

Motivation: the radar ego-velocity direction is biased against the reference
body velocity by -3.3 / -2.2 / -1.2 deg on slow / fast / backflips (2026-09-11),
same sign, regime-dependent; the GP path on slow racing is rotated -1.6 deg
about the start.  The deployed extrinsics are eyeballed (translation) and
roughly measured (pitch by inclinometer, yaw assumed 0).  A lever-arm error
enters through omega x t and scales with the body rate; a mount rotation is
constant.  Fitting both jointly over the racing-configuration bags, held out
per bag, separates them without touching the estimator.

Run from analysis/:  ../.venv/bin/python3 characterize_extrinsics_fit.py
"""
import sys
from pathlib import Path
sys.path.insert(0, 'lib'); sys.path.insert(0, '.')
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import predict_doppler_velocity, unwrap_doppler, rotation_matrix_from_euler

_cfg = load_config()
BAGS = _cfg['bags']['bags']; TIMING = _cfg['bags']['timing']; EXT = _cfg['extrinsics']
RC = _cfg['bags'].get('radar_config', {}); MIN_RANGE = _cfg['solver']['min_range']
R0 = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
T0 = np.array(EXT['translation_body_m'])
OFF = EXT['imu_mocap_offset_sec'] - EXT['radar_imu_offset_sec']
import os as _os
if _os.environ.get('RIO_RADAR_OFF_MS'):      # 2026-09-17: per-bag latency sensitivity of the fit (e.g. fast2's own -128)
    OFF = float(_os.environ['RIO_RADAR_OFF_MS']) / 1000.0
REPAIR = '--alias-repair' in sys.argv          # 2026-09-17: repair the aliased bearings (three AOP lobes, measured offsets) before the fit
FLIGHTS = [a for a in sys.argv[1:] if not a.startswith('--')] or ['slow_racing_best_velocity', 'fast_racing_best_velocity', 'backflips_best_velocity']
_LOBES = np.array([(-0.271, 0.271), (-0.177, -0.448), (0.448, 0.177)])


def _repair(P, vm, pred, vmax):
    """Multiplets by geometry (distinct lobes), singletons by the reference Doppler; returns (P2, v_unw, keep)."""
    import itertools
    vu = unwrap_doppler(vm, pred, vmax); k = np.round((vu - vm) / (2 * vmax)).astype(int)
    rng = np.linalg.norm(P, axis=1); u = P / rng[:, None]; P2 = P.copy(); keep = np.ones(len(P), bool)
    def cands(i):
        c1 = -u[i, 1] - k[i] * _LOBES[:, 0]; c2 = u[i, 2] + k[i] * _LOBES[:, 1]
        c1 = (c1 + 1) % 2 - 1; c2 = (c2 + 1) % 2 - 1; s2 = c1**2 + c2**2
        U = np.stack([np.sqrt(np.clip(1 - s2, 0, None)), -c1, c2], axis=1); U[s2 > 1] = np.nan; return U
    al = [i for i in range(len(P)) if k[i] != 0]; C = {i: cands(i) for i in al}; done = np.zeros(len(P), bool)
    for a in al:
        if done[a]: continue
        grp = [b for b in al if not done[b] and abs(rng[b] - rng[a]) < 0.03 and abs(vm[b] - vm[a]) < 0.01 and k[b] == k[a]]
        if len(grp) >= 2:
            best = None
            for combo in itertools.product(range(3), repeat=len(grp)):
                if len(set(combo)) < len(combo): continue
                V = np.array([C[g][c] for g, c in zip(grp, combo)])
                if np.isnan(V).any(): continue
                spread = max(np.linalg.norm(V[p] - V[q]) for p in range(len(grp)) for q in range(p + 1, len(grp)))
                if best is None or spread < best[0]: best = (spread, V.mean(axis=0))
            if best is not None and best[0] < 0.15:
                um = best[1] / np.linalg.norm(best[1]); P2[grp[0]] = rng[grp[0]] * um
                for g in grp[1:]: keep[g] = False
                for g in grp: done[g] = True
    return P2, vu, keep, done, al, C, k, rng

rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))


def load(key):
    d = load_bag_topics(str(Path('..') / BAGS[key]), verbose=False)
    t0 = d.start_time + TIMING[key][0]; t1 = t0 + TIMING[key][1]
    S = [s for s in d.agiros_state if t0 - 1 <= s.timestamp <= t1 + 1]
    ts = np.array([s.timestamp for s in S]); keep = np.ones(len(ts), bool); keep[1:] = np.diff(ts) > 1e-3
    ts = ts[keep]; S = [s for s, k in zip(S, keep) if k]
    vi = interp1d(ts, np.array([s.velocity for s in S]), axis=0, bounds_error=False, fill_value=np.nan)
    wi = interp1d(ts, np.array([s.angular_velocity for s in S]), axis=0, bounds_error=False, fill_value=np.nan)
    sl = Slerp(ts, Rotation.from_quat(np.array([s.orientation for s in S])))
    vmax = (RC.get('best_velocity', {}) if 'best_velocity' in key else RC.get('default', {})).get('v_max', 4.99)
    rows = []   # per frame: (v_world, omega_body, R_wb, P_s, v_meas)
    for fr in d.radar_velocity:
        if not (t0 <= fr.timestamp <= t1): continue
        t = fr.timestamp + OFF; vw, ob = vi(t), wi(t)
        if not (np.all(np.isfinite(vw)) and np.all(np.isfinite(ob))): continue
        R_wb = sl(np.clip(t, ts[0], ts[-1])).as_matrix()
        P = np.asarray(fr.positions); rng = np.asarray(fr.ranges) if fr.ranges is not None else np.linalg.norm(P, axis=1)
        k = rng >= MIN_RANGE
        if k.sum() < 3: continue
        P = P[k]; vm = np.asarray(fr.velocities)[k]
        if REPAIR:
            pred0 = predict_doppler_velocity(vw, ob, R_wb, P, T0, R0)
            P2, vu, keep, done, al, C, kk, rg = _repair(P, vm, pred0, vmax)
            # singletons: the lobe whose bearing best matches the reference prediction (the fit's own reference)
            v_ant = R0.T @ (R_wb.T @ vw + np.cross(ob, T0)); sgn = -1.0
            for a in al:
                if done[a]: continue
                U = C[a]; ok = np.isfinite(U[:, 0])
                if not ok.any(): continue
                err = np.abs(sgn * (U @ v_ant) - vu[a]); err[~ok] = np.inf
                P2[a] = rg[a] * U[int(np.argmin(err))]
            rows.append((vw, ob, R_wb, P2[keep], vu[keep], vmax))
        else:
            rows.append((vw, ob, R_wb, P, vm, vmax))
    return rows


def residuals(x, rows, unwrap_ref):
    """x = [dphi(3) small rotation applied to R0 (body side), dt(3)]"""
    R_s = Rotation.from_rotvec(x[:3]).as_matrix() @ R0
    T_s = T0 + x[3:6]
    out = []
    for (vw, ob, R_wb, P, vm, vmax), pref in zip(rows, unwrap_ref):
        pred = predict_doppler_velocity(vw, ob, R_wb, P, T_s, R_s)
        r = unwrap_doppler(vm, pref, vmax) - pred   # unwrap against the DEPLOYED prediction (fixed)
        out.append(r)
    r = np.concatenate(out)
    return r


def fit(rows_list, name):
    rows = [r for rl in rows_list for r in rl]
    # unwrap references at the deployed extrinsics (fixed), drop gross outliers by the deployed residual
    pref = [predict_doppler_velocity(vw, ob, R_wb, P, T0, R0) for (vw, ob, R_wb, P, vm, vmax) in rows]
    keep_rows, keep_pref = [], []
    for (vw, ob, R_wb, P, vm, vmax), p in zip(rows, pref):
        r = unwrap_doppler(vm, p, vmax) - p; k = np.abs(r) < 1.0
        if k.sum() >= 3:
            keep_rows.append((vw, ob, R_wb, P[k], vm[k], vmax)); keep_pref.append(p[k])
    r0 = residuals(np.zeros(6), keep_rows, keep_pref)
    sol = least_squares(residuals, np.zeros(6), args=(keep_rows, keep_pref), loss='soft_l1', f_scale=0.15)
    r1 = sol.fun if sol.success else residuals(sol.x, keep_rows, keep_pref)
    J = sol.jac; cov = np.linalg.pinv(J.T @ J) * (np.median(np.abs(r1)) * 1.4826) ** 2
    se = np.sqrt(np.diag(cov))
    print(f"  {name:<28s} n={len(r0):6d}  rms {np.sqrt(np.mean(r0**2)):.4f} -> {np.sqrt(np.mean(r1**2)):.4f}  robust {rs(r0):.4f} -> {rs(r1):.4f}")
    print(f"    d(roll,pitch,yaw) [deg] = {np.degrees(sol.x[:3]).round(2)} +- {np.degrees(se[:3]).round(2)}   dt [cm] = {(100*sol.x[3:]).round(1)} +- {(100*se[3:]).round(1)}")
    return sol.x, keep_rows, keep_pref


print(f"deployed: R0 = euler(180, 27.5, 0), t0 = {T0} m, radar total offset {OFF*1e3:+.0f} ms; alias repair {'ON' if REPAIR else 'off'}")
data = {k: load(k) for k in FLIGHTS}
print("\n== per bag")
per = {k: fit([data[k]], k)[0] for k in FLIGHTS}
print("\n== joint (all three racing-configuration bags)")
xj, rows_all, pref_all = fit([data[k] for k in FLIGHTS], 'joint')
print("\n== held out: fit on two, residual on the third (deployed -> fitted)")
for k in FLIGHTS:
    others = [data[o] for o in FLIGHTS if o != k]
    x, _, _ = fit(others, f"fit without {k[:9]}")
    rows = data[k]
    pref = [predict_doppler_velocity(vw, ob, R_wb, P, T0, R0) for (vw, ob, R_wb, P, vm, vmax) in rows]
    kr, kp = [], []
    for (vw, ob, R_wb, P, vm, vmax), p in zip(rows, pref):
        r = unwrap_doppler(vm, p, vmax) - p; m = np.abs(r) < 1.0
        if m.sum() >= 3: kr.append((vw, ob, R_wb, P[m], vm[m], vmax)); kp.append(p[m])
    r0 = residuals(np.zeros(6), kr, kp); r1 = residuals(x, kr, kp)
    print(f"    -> {k[:9]} held out: rms {np.sqrt(np.mean(r0**2)):.4f} -> {np.sqrt(np.mean(r1**2)):.4f}, robust {rs(r0):.4f} -> {rs(r1):.4f}")
print("\n== rotation-only and translation-only joint fits (which parameters carry the improvement)")
for name, mask in (('rotation only', [1,1,1,0,0,0]), ('translation only', [0,0,0,1,1,1]), ('yaw only', [0,0,1,0,0,0])):
    m = np.array(mask, bool)
    f = lambda y: residuals(np.where(m, np.pad(y, (0, 0)) if False else _expand(y, m), 0.0), rows_all, pref_all)
    def _expand(y, m):
        x = np.zeros(6); x[m] = y; return x
    sol = least_squares(lambda y: residuals(_expand(y, m), rows_all, pref_all), np.zeros(m.sum()), loss='soft_l1', f_scale=0.15)
    x = _expand(sol.x, m); r1 = residuals(x, rows_all, pref_all)
    print(f"  {name:<18s} rms -> {np.sqrt(np.mean(r1**2)):.4f}  robust -> {rs(r1):.4f}   d(rpy) {np.degrees(x[:3]).round(2)} deg, dt {(100*x[3:]).round(1)} cm")
