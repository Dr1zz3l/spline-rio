#!/usr/bin/env python3
"""Is the bearing of an aliased return corrupted at all, judged WITHOUT the
Doppler (2026-09-16, double check of `characterize_alias_bearing_wrap.py`)?

Every return is a static scatterer in an arena whose dominant structure is
the floor.  Project each reported point into the world frame with the
pose-derived reference pose at the frame time (deployed extrinsics and lag)
and compare the world height distribution of aliased and clean returns.
If the aliased bearings are corrupted, their world heights scatter while the
clean ones cluster at the floor; if they were a deterministic wrap, the
aliased returns would cluster at a DIFFERENT height.  As a second, also
Doppler-free, check the same direction-cosine offset search as the wrap
script is run on the floor criterion: which offset (a, b) applied as
k (a, b) brings the aliased returns to the floor?  Control: clean returns
with a fictitious k = +1.

Run from analysis/:  ../.venv/bin/python3 characterize_alias_world_position.py [bag ...]
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
FLIGHTS = sys.argv[1:] or ['circle_best_velocity', 'fast_racing_best_velocity', 'backflips_best_velocity',
                           'fast_racing_best_velocity_no_clustering', 'slow_racing_best_velocity']
GRID = np.arange(-1.0, 1.0, 1.0 / 24)
FLOOR_TOL = 0.25


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
    rows = []   # per return: u (sensor), range, R_ws (3x3), p_s_world (3), k
    for fr in d.radar_velocity:
        if not (t0 <= fr.timestamp <= t1): continue
        t = fr.timestamp + OFF; vw, ob, pw = v_i(t), om_i(t), p_i(t)
        if not (np.all(np.isfinite(vw)) and np.all(np.isfinite(ob)) and np.all(np.isfinite(pw))): continue
        R_wb = sl(np.clip(t, tm[okq][0], tm[okq][-1])).as_matrix()
        P = np.asarray(fr.positions); rng = np.linalg.norm(P, axis=1)
        m = rng >= MIN_RANGE
        if m.sum() == 0: continue
        P = P[m]; vm = np.asarray(fr.velocities)[m]; rng = rng[m]
        pred = predict_doppler_velocity(vw, ob, R_wb, P, T_s, R_s)
        k = np.round((unwrap_doppler(vm, pred, vmax) - vm) / (2 * vmax)).astype(int)
        R_ws = R_wb @ R_s; p_sw = pw + R_wb @ T_s
        for j in range(len(P)):
            rows.append((P[j] / rng[j], rng[j], R_ws, p_sw, k[j], np.linalg.norm(vw)))
    return rows, vmax


def world_z(rows, a, b, k_override=None):
    out = np.full(len(rows), np.nan)
    for i, (u, r, R_ws, p_sw, k, _) in enumerate(rows):
        kk = k if k_override is None else k_override
        c1 = (-u[1] + a * kk + 1.0) % 2.0 - 1.0; c2 = (u[2] + b * kk + 1.0) % 2.0 - 1.0
        s2 = c1 * c1 + c2 * c2
        if s2 > 1.0: continue
        un = np.array([np.sqrt(1.0 - s2), -c1, c2])
        out[i] = (p_sw + R_ws @ (r * un))[2]
    return out


def describe(z, name):
    z = z[np.isfinite(z)]
    if len(z) == 0: return f"   {name}: none"
    h, e = np.histogram(z, bins=np.arange(-3.0, 4.01, 0.25))
    top = np.argsort(h)[::-1][:3]
    peaks = ', '.join(f"[{e[i]:+.2f},{e[i+1]:+.2f}) {100*h[i]/len(z):.0f} %" for i in top)
    return (f"   {name:<34s} n={len(z):5d}  median {np.median(z):+.2f}  IQR {np.percentile(z,75)-np.percentile(z,25):.2f}  "
            f"within +-{FLOOR_TOL} of floor (z=0): {100*np.mean(np.abs(z) < FLOOR_TOL):.0f} %  | top bins: {peaks}")


for key in FLIGHTS:
    rows, vmax = load(key)
    K = np.array([r[4] for r in rows]); al = K != 0; cl = K == 0
    print(f"-- {key}: {len(rows)} returns, {al.sum()} aliased; flight height median {np.median([r[3][2] for r in rows]):.2f} m")
    z0 = world_z(rows, 0.0, 0.0)
    print(describe(z0[cl], "clean, as reported"))
    print(describe(z0[al], "aliased, as reported"))
    for kk in (-1, 1):
        m = K == kk
        if m.sum() >= 20: print(describe(z0[m], f"aliased k={kk:+d}, as reported"))
    if al.sum() < 30: continue
    # floor-criterion search over the direction-cosine offsets (Doppler-free)
    ral = [r for r in rows if r[4] != 0]; rcl = [r for r in rows if r[4] == 0]
    def floor_frac(rs, a, b, ko=None):
        z = world_z(rs, a, b, ko); f = np.isfinite(z)
        return np.mean(np.abs(z[f]) < FLOOR_TOL) if f.sum() > 0.5 * len(z) else np.nan
    base = floor_frac(ral, 0, 0)
    best = (-1, 0, 0)
    for a in GRID:
        for b in GRID:
            f = floor_frac(ral, a, b)
            if np.isfinite(f) and f > best[0]: best = (f, a, b)
    # per alias order: a TDM phase effect flips sign with k, a geometric selection effect does not
    for kk in (-1, 1):
        rk = [r for r in rows if r[4] == kk]
        if len(rk) < 60: continue
        bk = (-1, 0, 0)
        for a in GRID:
            for b in GRID:
                f = floor_frac(rk, a, b)
                if np.isfinite(f) and f > bk[0]: bk = (f, a, b)
        print(f"   k = {kk:+d} (n={len(rk)}): floor fraction as reported {100*floor_frac(rk, 0, 0):.0f} %, best {100*bk[0]:.0f} % at (a,b)=({bk[1]:+.3f},{bk[2]:+.3f}) -> EFFECTIVE shift k*(a,b) = ({kk*bk[1]:+.3f},{kk*bk[2]:+.3f})")
    cands = [(0, 2/3), (0, -2/3), (2/3, 0), (-2/3, 0), (0, 1/3), (0, -1/3), (1/3, 0), (-1/3, 0), (0, 1/6), (0, -1/6), (1/6, 0), (-1/6, 0)]
    print(f"   floor fraction of aliased returns: as reported {100*base:.0f} %, best over the grid {100*best[0]:.0f} % at (a,b)=({best[1]:+.3f},{best[2]:+.3f}); candidates: "
          + ', '.join(f"({a:+.2f},{b:+.2f}) {100*floor_frac(ral, a, b):.0f} %" for a, b in cands))
    bc = (-1, 0, 0)
    for a in GRID:
        for b in GRID:
            f = floor_frac(rcl, a, b, 1)
            if np.isfinite(f) and f > bc[0]: bc = (f, a, b)
    print(f"   CONTROL clean returns with fictitious k=+1: floor fraction as reported {100*floor_frac(rcl, 0, 0, 1):.0f} %, best {100*bc[0]:.0f} % at ({bc[1]:+.3f},{bc[2]:+.3f})")
