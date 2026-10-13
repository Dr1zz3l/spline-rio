#!/usr/bin/env python3
"""Test of the THREE-LOBE model of the aliased bearing (2026-09-16).

`simulate_aop_alias_bearing.py` (AOP geometry from TI SWRU546E Fig. 4-8,
chirp order TX1/TX2/TX3, residual TDM phase 2 pi k t / 3 on the t-th TX
group): the 2D angle spectrum of an aliased return of order k has three
lobes of nearly equal height (0.62 / 0.61 / 0.61 of the clean peak) at
FIXED direction-cosine offsets from the true bearing,
    D1 = (-0.28, +0.28), D2 = (-0.17, -0.45), D3 = (+0.45, +0.18)   (k = +1;
negated for k = -1), constant to 0.007 over the field of view.  The chip
reports one of the three.  So the reported bearing is the true bearing plus
one of three known offsets: a per-return three-hypothesis problem, not a
map, and not scatter.

Tests, all estimator-free (pose-derived reference, deployed extrinsics and
lag, k from the deployed unwrapper):
 1. Doppler: for every aliased return, the residual under each of the three
    candidate corrections (reported - k D_i, wrapped to the visible range);
    the best of the three vs the uncorrected residual.  The sign conventions
    of the two cosine axes are unknown a priori, so all four (sx, sz) axis
    sign variants are tried; the data pick one.  Control: clean returns with
    a fictitious k = +1 get the same best-of-three freedom.
 2. Floor: fraction of aliased returns that some candidate puts on the floor
    (|z| < 0.25 m) vs the same for the control.
 3. Multiplets: with `multiObjBeamForming -1 1 0.5` the chip emits a second
    point when a second angle peak exceeds half the maximum; aliased
    detections should then often appear as 2-3 points at the SAME range and
    Doppler with bearings differing by D_i - D_j.  Count same-frame groups
    of equal (range, Doppler) for aliased vs clean returns and print the
    cosine differences inside aliased groups.

Run from analysis/:  ../.venv/bin/python3 characterize_alias_three_lobes.py [bag ...]
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
FLIGHTS = sys.argv[1:] or ['fast_racing_best_velocity_no_clustering', 'circle_best_velocity', 'fast_racing_best_velocity',
                           'backflips_best_velocity', 'slow_racing_best_velocity']
LOBES = np.array([(-0.28, 0.28), (-0.17, -0.45), (0.45, 0.18)])
rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def load(key):
    d = load_bag_topics(str(Path('..') / BAGS[key]), verbose=False)
    t0 = d.start_time + TIMING[key][0]; t1 = t0 + TIMING[key][1]
    S = [x for x in d.agiros_state if t0 - 0.6 <= x.timestamp <= t1 + 0.6]
    tm = np.array([x.timestamp for x in S]); keep = np.ones(len(tm), bool); keep[1:] = np.diff(tm) > 1e-3
    tm = tm[keep]; S = [x for x, k in zip(S, keep) if k]
    Q = np.array([x.orientation for x in S]); okq = np.ones(len(tm), bool); okq[1:] = np.abs(np.diff(Q, axis=0)).max(axis=1) > 1e-12
    sl = Slerp(tm[okq], Rotation.from_quat(Q[okq]))
    p_i = interp1d(tm, np.array([x.position for x in S]), axis=0, bounds_error=False, fill_value=np.nan)
    om_i = interp1d(tm, np.array([x.angular_velocity for x in S]), axis=0, bounds_error=False, fill_value=np.nan)
    v_i = interp1d(tm, np.array([x.velocity for x in S]), axis=0, bounds_error=False, fill_value=np.nan)
    R_s, T_s = R_BASE, T_BASE.copy()
    if key in FLIPPED:
        Rz = rotation_matrix_from_euler(0.0, 0.0, np.pi); R_s = Rz @ R_BASE; T_s = Rz @ T_BASE
    vmax = (RC.get('best_velocity', {}) if 'best_velocity' in key else RC.get('default', {})).get('v_max', 4.99)
    U, VANT, VU, K, RNG, VM, FR, RWS, PSW = [], [], [], [], [], [], [], [], []
    sgn = None
    for fi, fr in enumerate(d.radar_velocity):
        if not (t0 <= fr.timestamp <= t1): continue
        t = fr.timestamp + OFF; vw, ob, pw = v_i(t), om_i(t), p_i(t)
        if not (np.all(np.isfinite(vw)) and np.all(np.isfinite(ob)) and np.all(np.isfinite(pw))): continue
        R_wb = sl(np.clip(t, tm[okq][0], tm[okq][-1])).as_matrix()
        P = np.asarray(fr.positions); rng = np.linalg.norm(P, axis=1)
        m = rng >= MIN_RANGE
        if m.sum() == 0: continue
        P = P[m]; vm = np.asarray(fr.velocities)[m]; rng = rng[m]
        u = P / rng[:, None]
        pred = predict_doppler_velocity(vw, ob, R_wb, P, T_s, R_s)
        vu = unwrap_doppler(vm, pred, vmax); k = np.round((vu - vm) / (2 * vmax)).astype(int)
        v_ant_s = R_s.T @ (R_wb.T @ vw + np.cross(ob, T_s))
        if sgn is None:
            sgn = -1.0 if np.allclose(pred, -(u @ v_ant_s), atol=1e-6) else 1.0
        n = len(u)
        U.append(u); VANT.append(np.tile(v_ant_s, (n, 1))); VU.append(vu); K.append(k); RNG.append(rng); VM.append(vm)
        FR.append(np.full(n, fi)); RWS.append(np.tile(R_wb @ R_s, (n, 1, 1))); PSW.append(np.tile(pw + R_wb @ T_s, (n, 1)))
    cat = lambda L: np.concatenate(L)
    return dict(key=key, U=cat(U), VANT=cat(VANT), VU=cat(VU), K=cat(K), RNG=cat(RNG), VM=cat(VM), FR=cat(FR), RWS=cat(RWS), PSW=cat(PSW), sgn=sgn, vmax=vmax)


def candidates(u, k, sx, sz, i):
    """Corrected unit vectors (ROS sensor frame) for lobe i; NaN rows where outside the visible range."""
    c1 = -u[:, 1] - k * sx * LOBES[i, 0]; c2 = u[:, 2] - k * sz * LOBES[i, 1]
    c1 = (c1 + 1.0) % 2.0 - 1.0; c2 = (c2 + 1.0) % 2.0 - 1.0
    s2 = c1 ** 2 + c2 ** 2
    un = np.stack([np.sqrt(np.clip(1.0 - s2, 0.0, None)), -c1, c2], axis=1)
    un[s2 > 1.0] = np.nan
    return un


def best_of_three(D, sel, sx, sz, k_override=None):
    u = D['U'][sel]; k = D['K'][sel] if k_override is None else np.full(sel.sum(), k_override)
    R = []
    for i in range(3):
        un = candidates(u, k, sx, sz, i)
        R.append(D['VU'][sel] - D['sgn'] * np.einsum('ij,ij->i', un, D['VANT'][sel]))
    R = np.array(R)                         # (3, n), NaN where invalid
    ib = np.nanargmin(np.abs(np.where(np.isnan(R), np.inf, R)), axis=0)
    rb = R[ib, np.arange(R.shape[1])]
    return rb, ib


def floor_any(D, sel, sx, sz, k_override=None, tol=0.25):
    u = D['U'][sel]; k = D['K'][sel] if k_override is None else np.full(sel.sum(), k_override)
    hit = np.zeros(sel.sum(), bool); z_best = np.full(sel.sum(), np.inf)
    for i in range(3):
        un = candidates(u, k, sx, sz, i)
        pw = D['PSW'][sel] + np.einsum('nij,nj->ni', D['RWS'][sel], D['RNG'][sel][:, None] * un)
        z = pw[:, 2]; ok = np.isfinite(z)
        hit |= ok & (np.abs(z) < tol); z_best = np.where(ok & (np.abs(z) < np.abs(z_best)), z, z_best)
    return hit


for key in FLIGHTS:
    D = load(key)
    al = D['K'] != 0; cl = D['K'] == 0
    print(f"-- {key}: {len(D['K'])} returns, {al.sum()} aliased (k=-1: {(D['K']==-1).sum()}, k=+1: {(D['K']==1).sum()})")
    if al.sum() < 30: print("   too few aliased"); continue
    r0 = D['VU'][al] - D['sgn'] * np.einsum('ij,ij->i', D['U'][al], D['VANT'][al])
    rc = D['VU'][cl] - D['sgn'] * np.einsum('ij,ij->i', D['U'][cl], D['VANT'][cl])
    print(f"   Doppler residual as reported: aliased rms {rms(r0):.3f} (|r|<0.25: {100*np.mean(np.abs(r0)<0.25):.0f} %), clean rms {rms(rc):.3f} (|r|<0.25: {100*np.mean(np.abs(rc)<0.25):.0f} %)")
    print("   best-of-three corrections (Doppler), aliased vs control (clean with fictitious k=+1, same freedom):")
    for sx in (1, -1):
        for sz in (1, -1):
            rb, ib = best_of_three(D, al, sx, sz); rcb, _ = best_of_three(D, cl, sx, sz, k_override=1)
            fa = np.isfinite(rb); fc = np.isfinite(rcb)
            print(f"     signs ({sx:+d},{sz:+d}): aliased rms {rms(rb[fa]):.3f}, |r|<0.25: {100*np.mean(np.abs(rb[fa])<0.25):.0f} %  (lobe picks {np.bincount(ib, minlength=3).tolist()})   | control rms {rms(rcb[fc]):.3f} -> |r|<0.25: {100*np.mean(np.abs(rcb[fc])<0.25):.0f} % (from {100*np.mean(np.abs(rc)<0.25):.0f} %)")
    print("   floor test: fraction of returns with some candidate within 0.25 m of the floor:")
    z0a = (D['PSW'][al] + np.einsum('nij,nj->ni', D['RWS'][al], D['RNG'][al][:, None] * D['U'][al]))[:, 2]
    z0c = (D['PSW'][cl] + np.einsum('nij,nj->ni', D['RWS'][cl], D['RNG'][cl][:, None] * D['U'][cl]))[:, 2]
    print(f"     as reported: aliased {100*np.mean(np.abs(z0a)<0.25):.0f} %, clean {100*np.mean(np.abs(z0c)<0.25):.0f} %")
    for sx in (1, -1):
        for sz in (1, -1):
            print(f"     signs ({sx:+d},{sz:+d}): aliased any-of-three {100*np.mean(floor_any(D, al, sx, sz)):.0f} %   | control (clean, fictitious k=+1) any-of-three {100*np.mean(floor_any(D, cl, sx, sz, 1)):.0f} %")
    # multiplets: same frame, same range (< 0.03 m) and same reported Doppler (< 0.01 m/s)
    groups_al, groups_cl, diffs = [], [], []
    for f in np.unique(D['FR']):
        idx = np.where(D['FR'] == f)[0]
        used = np.zeros(len(idx), bool)
        for a in range(len(idx)):
            if used[a]: continue
            same = [idx[b] for b in range(len(idx)) if not used[b] and abs(D['RNG'][idx[b]] - D['RNG'][idx[a]]) < 0.03 and abs(D['VM'][idx[b]] - D['VM'][idx[a]]) < 0.01]
            for b in range(len(idx)):
                if idx[b] in same: used[b] = True
            (groups_al if D['K'][idx[a]] != 0 else groups_cl).append(len(same))
            if D['K'][idx[a]] != 0 and len(same) >= 2:
                for p in range(len(same)):
                    for q in range(p + 1, len(same)):
                        up, uq = D['U'][same[p]], D['U'][same[q]]
                        diffs.append((-(uq[1] - up[1]), uq[2] - up[2]))
    ga, gc = np.array(groups_al), np.array(groups_cl)
    print(f"   multiplets (same frame, range < 3 cm, Doppler < 1 cm/s): aliased returns in groups of >= 2: {100*np.mean(ga>=2):.0f} % (n={len(ga)}); clean: {100*np.mean(gc>=2):.0f} % (n={len(gc)})")
    if diffs:
        dd = np.abs(np.array(diffs))
        print(f"     |cosine difference| inside aliased groups: median ({np.median(dd[:,0]):.2f}, {np.median(dd[:,1]):.2f}); predicted |D_i - D_j| = (0.11,0.73), (0.73,0.10), (0.62,0.63)")
