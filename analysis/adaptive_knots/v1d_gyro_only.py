"""V1d — bound what "trust the gyro" could achieve on backflips.

V1c showed: no spline-bandwidth deficit (omega residual flat 0.295 rad/s for
dt=16..4ms, flip==quiet) -> the solver's ~9 deg ori error is an optimization /
weighting failure. This script bounds the orientation accuracy achievable from
the gyro stream alone over the clean window [0.3, 17.1] s:

  1. Honest scale/misalignment estimate: mask mocap glitches, interp to IMU
     grid, IDENTICAL low-pass on both, then (a) Procrustes rotation-only
     alignment, (b) full linear A on top. (V1b/V1c used mismatched filters ->
     fake -17%.)
  2. Gyro-only dead-reckoning vs mocap with: zero bias | stationary-estimated
     bias | oracle constant bias (LS) | oracle bias + scale/misalign correction.
     Reports ori error vs time and RMSE/final.

Run:  cd analysis && ../.venv/bin/python3 adaptive_knots/v1d_gyro_only.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from nonuniform_bspline import so3_exp, so3_log
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

BAG_KEY = 'backflips_best_velocity'
T_CLEAN = (0.3, 17.1)

cfg = load_config()
bag_path = str(Path(__file__).parent.parent.parent / cfg['bags']['bags'][BAG_KEY])
t_off, dur = cfg['bags']['timing'][BAG_KEY]
print(f"Loading {bag_path} ...")
bag = load_bag_topics(bag_path, verbose=False)
t0 = bag.start_time + t_off

imu = [x for x in bag.imu_data if t0 - 1.0 <= x.timestamp <= t0 + dur + 0.5]
t_imu = np.array([x.timestamp for x in imu]) - t0
gyro = np.array([x.angular_velocity for x in imu])
fs = 1.0 / np.median(np.diff(t_imu))

poses = [p for p in bag.mocap_pose if t0 - 1.0 <= p.timestamp <= t0 + dur + 0.5]
tp = np.array([p.timestamp for p in poses]) - t0
quats = np.array([p.orientation for p in poses])
keep = np.concatenate([[True], np.diff(tp) > 1e-6])
tp, quats = tp[keep], quats[keep]
rots = Rotation.from_quat(quats)
slerp = Slerp(tp, rots)
Rm = rots.as_matrix()
om_m = np.zeros((len(tp) - 1, 3))
for i in range(len(tp) - 1):
    om_m[i] = so3_log(Rm[i].T @ Rm[i + 1]) / (tp[i + 1] - tp[i])
t_omm = 0.5 * (tp[:-1] + tp[1:])

# glitch mask + interp mocap omega to IMU grid
good = (np.linalg.norm(om_m, axis=1) < 25.0) & ~(np.diff(tp) > 0.02)
good &= (t_omm > T_CLEAN[0] - 0.2) & (t_omm < T_CLEAN[1] + 0.2)
sel = (t_imu > T_CLEAN[0]) & (t_imu < T_CLEAN[1])
om_i = np.column_stack([np.interp(t_imu[sel], t_omm[good], om_m[good, k]) for k in range(3)])

# time-align: scan offset on |omega| (both smoothed identically afterwards)
sos = butter(4, 8.0, 'low', fs=fs, output='sos')
g_f = sosfiltfilt(sos, gyro[sel], axis=0)


def filt(x):
    return sosfiltfilt(sos, x, axis=0)


best = (0.0, -2.0)
nm_g = np.linalg.norm(g_f, axis=1)
for off in np.arange(-0.05, 0.05, 0.0005):
    mi = np.column_stack([np.interp(t_imu[sel] + off, t_omm[good], om_m[good, k]) for k in range(3)])
    c = np.corrcoef(nm_g, np.linalg.norm(filt(mi), axis=1))[0, 1]
    if c > best[1]:
        best = (off, c)
off, corr0 = best
om_i = np.column_stack([np.interp(t_imu[sel] + off, t_omm[good], om_m[good, k]) for k in range(3)])
m_f = filt(om_i)
print(f"\n== 1. scale/misalignment (identical filtering, 8 Hz) ==")
print(f"  time offset {off*1e3:+.1f} ms (corr {corr0:.4f})")

# (a) Procrustes rotation-only:  g ~ R m
H = m_f.T @ g_f
U, S, Vt = np.linalg.svd(H)
D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
R_al = Vt.T @ D @ U.T
ang = np.degrees(np.linalg.norm(so3_log(R_al)))
r_rot = g_f - m_f @ R_al.T
print(f"  rotation-only alignment: {ang:.3f} deg; residual RMS "
      f"{np.sqrt(np.mean((g_f - m_f)**2)):.4f} -> {np.sqrt(np.mean(r_rot**2)):.4f} rad/s")

# (b) full linear on top of rotation: r_rot ~ A*(R m) + b
mr = m_f @ R_al.T
X = np.column_stack([mr, np.ones(len(mr))])
coef, *_ = np.linalg.lstsq(X, r_rot, rcond=None)
A, b = coef[:3].T, coef[3]
r_full = r_rot - X @ coef
print(f"  + linear A (percent):\n{np.array2string(100*A, precision=3, suppress_small=True)}")
print(f"  bias b = {np.array2string(b, precision=4)} rad/s")
print(f"  residual RMS after A: {np.sqrt(np.mean(r_full**2)):.4f} rad/s")
nm = np.linalg.norm(m_f, axis=1)
print(f"  corr(|r|,|omega|): rot-only {np.corrcoef(np.linalg.norm(r_rot,axis=1), nm)[0,1]:+.3f} "
      f"-> after A {np.corrcoef(np.linalg.norm(r_full,axis=1), nm)[0,1]:+.3f}")

# ---------------------------------------------------------------------------
# 2. gyro-only dead-reckoning vs mocap
# ---------------------------------------------------------------------------
print(f"\n== 2. gyro-only dead-reckoning, clean window {T_CLEAN} ==")
tc = t_imu[sel]
gc = gyro[sel]

# stationary bias: use the pre-window second if quiet, else first 0.5 s of window
pre = (t_imu > -0.8) & (t_imu < T_CLEAN[0])
b_stat = gyro[pre].mean(axis=0) if np.linalg.norm(gyro[pre], axis=1).max() < 0.2 else None
if b_stat is None:
    q = np.linalg.norm(gc, axis=1) < 0.5
    b_stat = gc[q][:500].mean(axis=0) if q.sum() > 100 else np.zeros(3)
    print(f"  (no stationary pre-segment; bias from quiet in-window samples)")
print(f"  stationary bias estimate: {np.array2string(b_stat, precision=4)} rad/s")


def dead_reckon(gyro_corr):
    R = slerp(np.clip(tc[0] + off, tp[0], tp[-1])).as_matrix()  # start at mocap attitude
    out = [R]
    for i in range(1, len(tc)):
        w = 0.5 * (gyro_corr[i - 1] + gyro_corr[i])
        out.append(out[-1] @ so3_exp(w * (tc[i] - tc[i - 1])))
    return out


def eval_dr(R_dr, label):
    errs = []
    for i in range(0, len(tc), 25):
        t_q = np.clip(tc[i] + off, tp[0], tp[-1])
        R_gt = slerp(t_q).as_matrix()
        errs.append(np.degrees(np.linalg.norm(so3_log(R_dr[i].T @ R_gt))))
    errs = np.array(errs)
    print(f"  {label:34s}: ori RMSE {np.sqrt(np.mean(errs**2)):7.2f} deg | "
          f"median {np.median(errs):6.2f} | final {errs[-1]:7.2f}")
    return errs


eval_dr(dead_reckon(gc), "zero bias")
eval_dr(dead_reckon(gc - b_stat), "stationary bias")

# oracle constant bias: coarse 3D search around b_stat via final-error... use LS on
# lowpassed residual vs mocap omega instead (bias term from part 1 regression)
b_oracle = b_stat + b  # part-1 b is the residual bias after rotation alignment
eval_dr(dead_reckon(gc - b_oracle), "oracle bias (regression)")

# oracle bias + scale/misalignment correction: w_corr = (I+A)^-1 R_al^T ... careful:
# regression was g ~ R_al m + A(R_al m) + b  =>  m = R_al^T (I+A)^{-1} (g - b)
M = np.linalg.inv(np.eye(3) + A)
g_corr = ((gc - b) @ M.T) @ R_al  # row-vec: m = R_al^T (I+A)^{-1} (g - b)
eval_dr(dead_reckon(g_corr), "oracle bias + scale/misalign")

print("\nIf 'stationary bias' RMSE << 9 deg: the gyro stream alone out-tracks the")
print("solver -> the fix is weighting/robustness (trust gyro through flips), not knots.")
