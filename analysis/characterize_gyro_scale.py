#!/usr/bin/env python3
"""Measure the gyroscope scale-factor error against the MoCap attitude,
estimator-free, to set the prior of the GP's gyro scale-factor state
(GpConfig::gyro_scale_prior; 2026-09-10 flip-attitude NEES study).

Body rate from MoCap: omega_gt = Log(R_i^T R_{i+1}) / dt (right-tangent, body
frame), low-passed at 30 Hz, interpolated to the IMU timestamps (shifted by
the calibrated imu_mocap_offset_sec).  Per axis, z_i = (1 + s_i) omega_i + b_i
is fitted by Huber least squares on samples with |omega| > 0.5 rad/s; a full
3x3 fit z = M omega + b reports the off-diagonal (misalignment) terms next to
the diagonal scale; the offset sensitivity (+-2 ms) shows the slope is not a
timing artifact.  The residual is binned by |omega| to show what white
residue remains once the scale is a state.

Run from analysis/:  ../.venv/bin/python3 characterize_gyro_scale.py
"""
import sys
from pathlib import Path
sys.path.insert(0, 'lib'); sys.path.insert(0, '.')
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics

_cfg = load_config()
BAGS = _cfg['bags']['bags']; TIMING = _cfg['bags']['timing']; EXT = _cfg['extrinsics']
IMU_OFF = EXT['imu_mocap_offset_sec']
FLIGHTS = ['slow_racing_best_velocity', 'fast_racing_best_velocity', 'backflips_best_velocity']
rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))


def gt_rate(poses, t0, t1):
    """Body rate from MoCap on a uniform 200 Hz grid: dedupe held/repeated
    samples, slerp, centred difference over +-5 ms, 30 Hz low-pass."""
    from scipy.spatial.transform import Slerp
    tm = np.array([p.timestamp for p in poses]); q = np.array([p.orientation for p in poses])
    m = (tm >= t0 - 0.5) & (tm <= t1 + 0.5); tm, q = tm[m], q[m]
    keep = np.ones(len(tm), bool)
    keep[1:] = (np.diff(tm) > 1e-3) & (np.abs(np.diff(q, axis=0)).max(axis=1) > 1e-9)
    tm, q = tm[keep], q[keep]
    sl = Slerp(tm, Rotation.from_quat(q))
    tg = np.arange(tm[0] + 0.01, tm[-1] - 0.01, 1.0 / 200.0)
    h = 0.005
    dR = sl(tg - h).inv() * sl(tg + h)
    om = dR.as_rotvec() / (2 * h)
    b, a = butter(4, 30.0 / 100.0, btype='low')
    om = filtfilt(b, a, om, axis=0)
    return tg, om, 1.0 / np.median(np.diff(tm))


def fit_axis(om, z):
    f = lambda p: np.concatenate([(1 + p[0]) * om + p[1] - z])
    r = least_squares(f, [0.0, 0.0], loss='huber', f_scale=0.2)
    return r.x


for key in FLIGHTS:
    data = load_bag_topics(str(Path('..') / BAGS[key]), verbose=False)
    t0 = data.start_time + TIMING[key][0]; t1 = t0 + TIMING[key][1]
    imu = [m for m in data.imu_data if t0 <= m.timestamp <= t1]
    t_imu = np.array([m.timestamp for m in imu]) + IMU_OFF
    z = np.array([m.angular_velocity for m in imu])
    tg, omg, fs = gt_rate(data.mocap_pose, t0, t1)
    print(f"\n===== {key}: {len(imu)} IMU samples, MoCap {fs:.0f} Hz, |omega| max {np.linalg.norm(omg, axis=1).max():.1f} rad/s")
    res = {}
    for off in (-0.002, 0.0, 0.002):
        om = interp1d(tg, omg, axis=0, bounds_error=False, fill_value=np.nan)(t_imu + off)
        ok = np.all(np.isfinite(om), axis=1) & (np.linalg.norm(om, axis=1) > 1.0)
        s = [fit_axis(om[ok, i], z[ok, i]) for i in range(3)]
        res[off] = (om, ok, s)
        print(f"  offset {off*1e3:+.0f} ms: n={ok.sum():6d}  scale s = [{s[0][0]:+.5f}, {s[1][0]:+.5f}, {s[2][0]:+.5f}]"
              f"  bias = [{s[0][1]:+.4f}, {s[1][1]:+.4f}, {s[2][1]:+.4f}]")
    om, ok, s = res[0.0]
    # full 3x3 (scale + misalignment) fit
    A = np.hstack([om[ok], np.ones((ok.sum(), 1))])
    M = np.linalg.lstsq(A, z[ok], rcond=None)[0]        # (4,3): rows omega_x,y,z,1 ; cols z_x,y,z
    Mm = M[:3].T - np.eye(3)
    print("  full 3x3 fit (M - I), rows = gyro axis, cols = GT axis:")
    for i in range(3):
        print("    [" + " ".join(f"{Mm[i, j]:+.5f}" for j in range(3)) + "]")
    # residual after the per-axis scale + bias, binned by |omega|
    r = np.stack([z[ok, i] - (1 + s[i][0]) * om[ok, i] - s[i][1] for i in range(3)], axis=1)
    on = np.linalg.norm(om[ok], axis=1)
    print("  residual after scale+bias, per-axis robust sigma [rad/s] by |omega|:")
    for lo, hi in [(0.5, 1), (1, 2), (2, 4), (4, 8), (8, 20)]:
        m = (on >= lo) & (on < hi)
        if m.sum() > 200:
            sig = [rs(r[m, i]) for i in range(3)]
            print(f"    [{lo:>3},{hi:>3}) n={m.sum():6d}  sigma = [{sig[0]:.4f}, {sig[1]:.4f}, {sig[2]:.4f}]"
                  f"   mean/|omega| = {np.mean(sig)/np.mean(on[m]):.4f}")
    # same residual WITHOUT the scale (bias only): what the white model has to absorb
    r0 = np.stack([z[ok, i] - om[ok, i] - np.median(z[ok, i] - om[ok, i]) for i in range(3)], axis=1)
    m = on >= 4
    if m.sum() > 200:
        print(f"  without scale, |omega|>=4: per-axis robust sigma = "
              f"[{rs(r0[m,0]):.4f}, {rs(r0[m,1]):.4f}, {rs(r0[m,2]):.4f}]  (with scale: "
              f"[{rs(r[m,0]):.4f}, {rs(r[m,1]):.4f}, {rs(r[m,2]):.4f}])")
