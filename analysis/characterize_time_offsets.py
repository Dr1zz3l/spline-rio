#!/usr/bin/env python3
"""Estimator-free time-offset calibration against the pose-derived reference
(2026-09-10/12; supersedes the 'empirically tuned' extrinsics.yaml values).

  imu_mocap_offset_sec : lag that aligns the Pixhawk gyro with the reference
                         body rate (the loader's pose-derived rate), rms fit on
                         |omega| > 1 rad/s, 1 ms steps, parabolic refinement.  Positive = ADD to raw
                         IMU timestamps (IMU clock behind MoCap).
  radar total offset   : lag that aligns the raw per-return Doppler with the
                         forward model driven by the reference velocity, body
                         rate and attitude (deployed extrinsics: 27.5 deg locked
                         pitch, lever arm), measurement unwrapped toward the
                         prediction.  DEPLOYED PROTOCOL (2026-09-11): plain rms
                         (gross outliers |r| > 1 m/s dropped) over the frames
                         whose reference acceleration exceeds 4 m/s^2, where a
                         clock error actually shows; the robust width over all
                         frames (also printed) is flat to +-10 ms.  Frames carry
                         the sensor-clock stamps of the loader.  Positive = ADD
                         to radar timestamps.  radar_imu_offset_sec = imu_mocap
                         - total, ONE value per day and chirp configuration,
                         pooled over the bags of that configuration (never per
                         bag).

Run from analysis/:  ../.venv/bin/python3 characterize_time_offsets.py [bag ...]
"""
import sys
from pathlib import Path
sys.path.insert(0, 'lib'); sys.path.insert(0, '.')
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (predict_doppler_velocity, unwrap_doppler,
                                  rotation_matrix_from_euler)

_cfg = load_config()
BAGS = _cfg['bags']['bags']; TIMING = _cfg['bags']['timing']; EXT = _cfg['extrinsics']
FLIPPED = set(_cfg['bags'].get('flipped', []) or [])
MIN_RANGE = _cfg['solver']['min_range']
RC = _cfg['bags'].get('radar_config', {})
R_BASE = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
T_BASE = np.array(EXT['translation_body_m'])
rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))

FLIGHTS = sys.argv[1:] or ['slow_racing_best_velocity', 'fast_racing_best_velocity',
                           'backflips_best_velocity', 'fast_racing_best_velocity_no_clustering',
                           'circle_best_velocity']


def mocap_rate_grid(poses, t0, t1):
    """Body rate from the RAW MoCap pose for the gyro lag: deduped slerp grid,
    +-5 ms centred difference, 30 Hz zero-phase low-pass.  The loader's
    pose-derived rate (15 Hz, median-filtered) is too smooth for a sharp lag
    fit; the attitude-increment method (2026-09-12) agrees with this one to
    within 2 ms on the racing flights."""
    tm = np.array([p.timestamp for p in poses]); q = np.array([p.orientation for p in poses])
    m = (tm >= t0 - 0.5) & (tm <= t1 + 0.5); tm, q = tm[m], q[m]
    keep = np.ones(len(tm), bool)
    keep[1:] = (np.diff(tm) > 1e-3) & (np.abs(np.diff(q, axis=0)).max(axis=1) > 1e-9)
    tm, q = tm[keep], q[keep]
    sl = Slerp(tm, Rotation.from_quat(q))
    tg = np.arange(tm[0] + 0.02, tm[-1] - 0.02, 1.0 / 200.0); h = 0.005
    om = (sl(tg - h).inv() * sl(tg + h)).as_rotvec() / (2 * h)
    b, a = butter(4, 30.0 / 100.0, btype='low')
    return tg, filtfilt(b, a, om, axis=0)


def reference_grid(states, t0, t1):
    """Reference kinematics from the loader's agiros state: the pose as
    recorded, velocity / body rate / acceleration derived from it by the loader
    (`derive_pose_reference`, zero-phase 15 / 5 Hz), on the state's own stamps.
    The raw /mocap pose stream is the same pose to within 1-2 ms; the
    pilot's velocity / rate / acceleration fields are NOT used."""
    S = [x for x in states if t0 - 0.6 <= x.timestamp <= t1 + 0.6]
    tm = np.array([x.timestamp for x in S]); keep = np.ones(len(tm), bool); keep[1:] = np.diff(tm) > 1e-3
    tm = tm[keep]; S = [x for x, k in zip(S, keep) if k]
    q = np.array([x.orientation for x in S])
    sl = Slerp(tm, Rotation.from_quat(q))
    om = np.array([x.angular_velocity for x in S]); v = np.array([x.velocity for x in S])
    acc = np.array([x.acceleration for x in S])
    return tm, sl, om, v, acc


def refine(lags, cost):
    k = int(np.nanargmin(cost))
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = cost[k - 1:k + 2]
        return lags[k] + (lags[1] - lags[0]) * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2), cost[k]
    return lags[k], cost[k]


for key in FLIGHTS:
    data = load_bag_topics(str(Path('..') / BAGS[key]), verbose=False)
    t0 = data.start_time + TIMING[key][0]; t1 = t0 + TIMING[key][1]
    tg, sl, om, v, acc = reference_grid(data.agiros_state, t0, t1)
    om_i = interp1d(tg, om, axis=0, bounds_error=False, fill_value=np.nan)
    v_i = interp1d(tg, v, axis=0, bounds_error=False, fill_value=np.nan)
    # ---- IMU lag (raw MoCap rate) ----
    tgr, omr = mocap_rate_grid(data.mocap_pose, t0, t1)
    omr_i = interp1d(tgr, omr, axis=0, bounds_error=False, fill_value=np.nan)
    imu = [m for m in data.imu_data if t0 <= m.timestamp <= t1]
    ti = np.array([m.timestamp for m in imu]); z = np.array([m.angular_velocity for m in imu])
    lags = np.arange(-0.04, 0.0401, 0.001); cost = []
    for L in lags:
        o = omr_i(ti + L); ok = np.all(np.isfinite(o), axis=1) & (np.linalg.norm(o, axis=1) > 1.0)
        cost.append(np.sqrt(np.mean((z[ok] - o[ok]) ** 2)) if ok.sum() > 100 else np.nan)
    imu_lag, imu_rms = refine(lags, np.array(cost))
    # ---- radar lag ----
    R_s = R_BASE; T_s = T_BASE.copy()
    if key in FLIPPED:
        Rz = rotation_matrix_from_euler(0.0, 0.0, np.pi); R_s = Rz @ R_BASE; T_s = Rz @ T_BASE
    vmax = (RC.get('best_velocity', {}) if 'best_velocity' in key else RC.get('default', {})).get('v_max', 4.99)
    frames = [f for f in data.radar_velocity if t0 <= f.timestamp <= t1]
    a_i = interp1d(tg, acc, axis=0, bounds_error=False, fill_value=np.nan)
    lags_r = np.arange(-0.20, -0.0399, 0.001); cost_r = []; cost_hi = []; nres = []; nhi = []
    for L in lags_r:
        res = []; res_hi = []; n_hi = 0
        for fr in frames:
            t = fr.timestamp + L
            vw, ob, aw = v_i(t), om_i(t), a_i(t)
            if not (np.all(np.isfinite(vw)) and np.all(np.isfinite(ob))): continue
            R_wb = sl(np.clip(t, tg[0], tg[-1])).as_matrix()
            P = np.asarray(fr.positions); rng = np.asarray(fr.ranges) if fr.ranges is not None else np.linalg.norm(P, axis=1)
            keep = rng >= MIN_RANGE
            if keep.sum() == 0: continue
            pred = predict_doppler_velocity(vw, ob, R_wb, P[keep], T_s, R_s)
            meas = unwrap_doppler(np.asarray(fr.velocities)[keep], pred, vmax)
            r = meas - pred
            res.append(r)
            if np.all(np.isfinite(aw)) and np.linalg.norm(aw) > 4.0:
                res_hi.append(r[np.abs(r) < 1.0]); n_hi += 1
        res = np.concatenate(res) if res else np.array([np.nan])
        cost_r.append(rs(res)); nres.append(len(res))
        rh = np.concatenate(res_hi) if res_hi else np.array([np.nan])
        cost_hi.append(np.sqrt(np.nanmean(rh ** 2)) if len(rh) > 50 else np.nan); nhi.append(n_hi)
    rad_lag_rob, rad_rms_rob = refine(lags_r, np.array(cost_r))
    rad_lag, rad_rms = refine(lags_r, np.array(cost_hi))
    cr = np.array(cost_hi)
    print(f"{key}:")
    print(f"  IMU->MoCap lag  {imu_lag*1e3:+6.1f} ms  (gyro rms {imu_rms:.3f} rad/s; at 0/+15 ms {cost[40]:.3f}/{cost[55]:.3f})")
    print(f"  radar total lag {rad_lag*1e3:+6.1f} ms  DEPLOYED PROTOCOL: plain rms on |a| > 4 m/s^2 frames ({nhi[len(nhi)//2]} frames; rms {rad_rms:.3f};"
          f" at -120/-130 ms {cr[np.argmin(np.abs(lags_r+0.120))]:.3f}/{cr[np.argmin(np.abs(lags_r+0.130))]:.3f})")
    print(f"                  {rad_lag_rob*1e3:+6.1f} ms  robust width over all frames (flat; {nres[len(nres)//2]} returns, rms {rad_rms_rob:.3f}) -- for reference only")
    print(f"  -> imu_mocap_offset_sec {imu_lag:+.4f}, radar_imu_offset_sec {imu_lag - rad_lag:.4f}"
          f"   (deployed 0.0050 / 0.1270 = total -0.1220)")
