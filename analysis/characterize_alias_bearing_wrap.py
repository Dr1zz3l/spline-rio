#!/usr/bin/env python3
"""Is the bearing of an ALIASED return a deterministic WRAP in direction-cosine
space (2026-09-16, Timo: "are you 100 % sure we can't recover bearing under
alias?")?

Physics.  3 TX in TDM (chirpCfg 0/1/2 -> TX0/TX1/TX2, chirp period T_c, per-TX
period 3 T_c).  A target with true radial velocity v accumulates the Doppler
phase phi_d = 4 pi v (3 T_c) / lambda per TX period; the Doppler FFT sees it
modulo 2 pi, i.e. v modulo 2 v_max.  The SDK compensates the inter-TX phase of
the virtual array with the MEASURED Doppler bin (phi_hat/3 on TX1's four
channels, 2 phi_hat/3 on TX2's).  For an aliased return with true
phi_d = phi_hat + 2 pi k, the compensation leaves a residual phase of
2 pi k / 3 on TX1's channels and 4 pi k / 3 on TX2's: a fixed staircase across
the three sub-arrays, deterministic in the alias order k.  Wherever the angle
estimate uses an inter-TX baseline of length d, that residual phase moves the
estimated DIRECTION COSINE along that baseline by (k / 3) (lambda / d) modulo
lambda / d: for d = lambda / 2 by 2k/3 modulo 2, a WRAP of two thirds of the
visible range, not a small shift.  The 2026-07-19 test
(`worklog/2026-07-19_alias-bearing-repair.md`) regressed a constant ANGULAR
shift per k and found scatter with R^2 <= 0.14; a linear shift model cannot
represent a wrap (half the points move one way, the rest wrap back), so that
negative does not exclude this mechanism.  This script tests it directly and
estimator-free, without assuming the AOP array geometry: search the two
direction-cosine offsets (a, b) applied as k * (a, b) modulo 2 to the reported
(lateral, elevation) cosines of every aliased return, and ask whether some
(a, b) brings the aliased returns' Doppler residual against the pose-derived
reference down to the clean floor.  Control: the same search on the
non-aliased returns with a fictitious k = +1 must find its minimum at (0, 0).

Frames: ROS sensor frame of the driver (x boresight = TI y, y = -TI x
(lateral), z = TI z (elevation)).  k from the deployed unwrapper against the
reference prediction WITH the reported bearing (as the July test); the
unwrapped Doppler is held fixed while the bearing moves.

Run from analysis/:  ../.venv/bin/python3 characterize_alias_bearing_wrap.py [bag ...]
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
GRID = np.arange(-1.0, 1.0, 1.0 / 48)
rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def load(key):
    d = load_bag_topics(str(Path('..') / BAGS[key]), verbose=False)
    t0 = d.start_time + TIMING[key][0]; t1 = t0 + TIMING[key][1]
    S = [x for x in d.agiros_state if t0 - 0.6 <= x.timestamp <= t1 + 0.6]
    tm = np.array([x.timestamp for x in S]); keep = np.ones(len(tm), bool); keep[1:] = np.diff(tm) > 1e-3
    tm = tm[keep]; S = [x for x, k in zip(S, keep) if k]
    Q = np.array([x.orientation for x in S]); okq = np.ones(len(tm), bool); okq[1:] = np.abs(np.diff(Q, axis=0)).max(axis=1) > 1e-12
    sl = Slerp(tm[okq], Rotation.from_quat(Q[okq]))
    om_i = interp1d(tm, np.array([x.angular_velocity for x in S]), axis=0, bounds_error=False, fill_value=np.nan)
    v_i = interp1d(tm, np.array([x.velocity for x in S]), axis=0, bounds_error=False, fill_value=np.nan)
    R_s, T_s = R_BASE, T_BASE.copy()
    if key in FLIPPED:
        Rz = rotation_matrix_from_euler(0.0, 0.0, np.pi); R_s = Rz @ R_BASE; T_s = Rz @ T_BASE
    vmax = (RC.get('best_velocity', {}) if 'best_velocity' in key else RC.get('default', {})).get('v_max', 4.99)
    U, VANT, VM, VU, K, SPD = [], [], [], [], [], []
    for fr in d.radar_velocity:
        if not (t0 <= fr.timestamp <= t1): continue
        t = fr.timestamp + OFF; vw, ob = v_i(t), om_i(t)
        if not (np.all(np.isfinite(vw)) and np.all(np.isfinite(ob))): continue
        R_wb = sl(np.clip(t, tm[okq][0], tm[okq][-1])).as_matrix()
        P = np.asarray(fr.positions); rng = np.asarray(fr.ranges) if fr.ranges is not None else np.linalg.norm(P, axis=1)
        m = rng >= MIN_RANGE
        if m.sum() == 0: continue
        P = P[m]; vm = np.asarray(fr.velocities)[m]
        u = P / np.linalg.norm(P, axis=1, keepdims=True)
        pred = predict_doppler_velocity(vw, ob, R_wb, P, T_s, R_s)
        vu = unwrap_doppler(vm, pred, vmax)
        k = np.round((vu - vm) / (2 * vmax)).astype(int)
        # antenna velocity in the SENSOR frame; sign calibrated against predict_doppler_velocity below
        v_ant_body = R_wb.T @ vw + np.cross(ob, T_s)
        v_ant_s = R_s.T @ v_ant_body
        U.append(u); VANT.append(np.tile(v_ant_s, (len(u), 1))); VM.append(vm); VU.append(vu); K.append(k)
        SPD.append(np.full(len(u), np.linalg.norm(v_ant_body)))
        if len(U) == 1:
            sgn = -1.0 if np.allclose(pred, -(u @ v_ant_s), atol=1e-6) else 1.0
            assert np.allclose(pred, sgn * (u @ v_ant_s), atol=1e-6), "sign convention mismatch"
    U = np.concatenate(U); VANT = np.concatenate(VANT); VM = np.concatenate(VM); VU = np.concatenate(VU); K = np.concatenate(K); SPD = np.concatenate(SPD)
    return dict(key=key, U=U, VANT=VANT, VM=VM, VU=VU, K=K, SPD=SPD, vmax=vmax, sgn=sgn)


def residual(D, sel, a, b, k_override=None):
    """Doppler residual of the selected returns after moving their (lateral,
    elevation) direction cosines by k*(a, b) modulo 2 (the visible range)."""
    u = D['U'][sel]; k = D['K'][sel] if k_override is None else np.full(sel.sum(), k_override)
    c1 = -u[:, 1] + a * k; c2 = u[:, 2] + b * k               # TI x (lateral), TI z (elevation)
    c1 = (c1 + 1.0) % 2.0 - 1.0; c2 = (c2 + 1.0) % 2.0 - 1.0
    s2 = c1 ** 2 + c2 ** 2
    ok = s2 <= 1.0
    un = np.stack([np.sqrt(np.clip(1.0 - s2, 0.0, None)), -c1, c2], axis=1)
    pred = D['sgn'] * np.einsum('ij,ij->i', un, D['VANT'][sel])
    r = D['VU'][sel] - pred
    r[~ok] = np.nan
    return r


def scan(D, sel, k_override=None):
    best = (np.inf, 0, 0); table = np.full((len(GRID), len(GRID)), np.nan)
    for i, a in enumerate(GRID):
        for j, b in enumerate(GRID):
            r = residual(D, sel, a, b, k_override)
            fin = np.isfinite(r)
            if fin.sum() < 0.5 * len(r): continue
            c = np.sqrt(np.nanmean(np.minimum(np.abs(r), 1.5) ** 2))     # capped rms: robust to the few that cannot be fixed
            table[i, j] = c
            if c < best[0]: best = (c, a, b)
    return best, table


for key in FLIGHTS:
    D = load(key)
    al = D['K'] != 0; cl = D['K'] == 0
    print(f"-- {key}: {len(D['K'])} returns, {al.sum()} aliased ({100*al.mean():.0f} %; k = -1: {(D['K']==-1).sum()}, +1: {(D['K']==1).sum()}, |k|>1: {(np.abs(D['K'])>1).sum()}), v_max {D['vmax']}")
    if al.sum() < 30:
        print("   too few aliased returns"); continue
    r0 = residual(D, al, 0.0, 0.0); rc = residual(D, cl, 0.0, 0.0)
    print(f"   residual rms as reported: aliased {rms(r0):.3f} (|r|<0.25: {100*np.mean(np.abs(r0)<0.25):.0f} %), clean {rms(rc):.3f} (|r|<0.25: {100*np.mean(np.abs(rc)<0.25):.0f} %)")
    best, tab = scan(D, al)
    r1 = residual(D, al, best[1], best[2])
    print(f"   best wrap over the grid: (a, b) = ({best[1]:+.3f}, {best[2]:+.3f}) -> capped rms {best[0]:.3f} (from {tab[len(GRID)//2, len(GRID)//2]:.3f} at (0,0)); plain rms {np.nanrms if False else rms(r1[np.isfinite(r1)]):.3f}, |r|<0.25: {100*np.nanmean(np.abs(r1)<0.25):.0f} %")
    # physically motivated candidates: +-1/3, +-2/3 on either axis
    cands = [(0, 2/3), (0, -2/3), (2/3, 0), (-2/3, 0), (0, 1/3), (0, -1/3), (1/3, 0), (-1/3, 0), (2/3, 2/3), (-2/3, -2/3), (2/3, -2/3), (-2/3, 2/3),
             (0.375, 0.5), (0.208, 0.417), (0.25, 0.292), (0.083, -0.583), (-0.083, -0.375)]   # the floor-criterion optima of characterize_alias_world_position.py
    for kk in (-1, 1):
        m = D['K'] == kk
        if m.sum() >= 60:
            bk, _ = scan(D, m)
            print(f"   k = {kk:+d} grid: best (a,b) = ({bk[1]:+.3f},{bk[2]:+.3f}) capped rms {bk[0]:.3f} vs {np.sqrt(np.nanmean(np.minimum(np.abs(residual(D, m, 0, 0)), 1.5)**2)):.3f} at (0,0) -> EFFECTIVE shift ({kk*bk[1]:+.3f},{kk*bk[2]:+.3f})")
    print("   candidates (a, b) -> capped rms: " + ', '.join(f"({a:+.2f},{b:+.2f}) {np.sqrt(np.nanmean(np.minimum(np.abs(residual(D, al, a, b)), 1.5) ** 2)):.3f}" for a, b in cands))
    # per-k split at the best offset
    for kk in (-1, 1):
        m = D['K'] == kk
        if m.sum() >= 20:
            rk0 = residual(D, m, 0.0, 0.0); rk1 = residual(D, m, best[1], best[2])
            print(f"   k = {kk:+d} (n={m.sum()}): rms {rms(rk0):.3f} -> {rms(rk1[np.isfinite(rk1)]):.3f}, |r|<0.25 {100*np.mean(np.abs(rk0)<0.25):.0f} % -> {100*np.nanmean(np.abs(rk1)<0.25):.0f} %")
    # control: clean returns with a fictitious k = +1 must prefer (0, 0)
    bc, tc = scan(D, cl, k_override=1)
    print(f"   CONTROL (clean returns, fictitious k=+1): best (a, b) = ({bc[1]:+.3f}, {bc[2]:+.3f}) capped rms {bc[0]:.3f} vs {tc[len(GRID)//2, len(GRID)//2]:.3f} at (0,0)")
    # how deep is the aliased minimum relative to its surroundings?
    finite = tab[np.isfinite(tab)]
    print(f"   aliased grid: min {np.nanmin(tab):.3f}, median {np.median(finite):.3f}, at (0,0) {tab[len(GRID)//2, len(GRID)//2]:.3f}")
