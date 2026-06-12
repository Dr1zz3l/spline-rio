"""V1b — decompose the rate-correlated gyro residual on backflips.

V1a found uniform-8ms already represents MoCap R(t) to ~1.3 deg RMSE during
flips (vs the 8.3-9.2 deg solver error) -> raw R-bandwidth is NOT the dominant
failure. This script tests the competing explanations for
corr(|r_gyro|,|omega|) = +0.94:

  A. MoCap breakdown: |omega|_mocap spikes to 780 rad/s and error spikes 70 deg
     after t~17.3s. Compare |omega| from gyro vs mocap-FD over time. If mocap
     diverges where gyro stays sane, the tail of the eval window corrupts BOTH
     the placement signal and (worse) the solver's reported RMSE itself.
  B. Vibration vs coherent rotation: band-split the gyro at the 8ms-spline
     bandwidth. If the above-bandwidth power during flips is absent from the
     mocap-derived omega (245 Hz, FD bandwidth ~50 Hz), it is vibration ->
     denser knots would chase noise (adaptive knots NO-GO; robust gyro loss
     instead).
  C. Gyro scale/misalignment: fit r = z_gyro - omega_mocap ~ A*omega_mocap + b.
     Scale errors are rate-proportional and reproduce the +0.94 correlation.
     Report ||A|| and residual reduction.
  D. Spline omega-tracking: fit the SO(3) spline to gyro dead-reckoned R at
     {8, 4, 2 ms} and measure the omega residual RMS per grid (what the gyro
     factors see) on a clean mid-flight segment.

Run:  cd analysis && ../.venv/bin/python3 adaptive_knots/v1b_gyro_analysis.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from nonuniform_bspline import (NonUniformSO3Spline, extend_knots, so3_exp,
                                so3_log)
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from scipy.spatial.transform import Rotation, Slerp

BAG_KEY = 'backflips_best_velocity'

cfg = load_config()
bag_path = str(Path(__file__).parent.parent.parent / cfg['bags']['bags'][BAG_KEY])
t_off, dur = cfg['bags']['timing'][BAG_KEY]

print(f"Loading {bag_path} ...")
bag = load_bag_topics(bag_path, verbose=False)
t0 = bag.start_time + t_off
t1 = t0 + dur

imu = [d for d in bag.imu_data if t0 - 0.5 <= d.timestamp <= t1 + 0.5]
t_imu = np.array([d.timestamp for d in imu]) - t0
gyro = np.array([d.angular_velocity for d in imu])
print(f"  {len(imu)} IMU samples ({1.0/np.median(np.diff(t_imu)):.0f} Hz)")

poses = [p for p in bag.mocap_pose if t0 - 0.5 <= p.timestamp <= t1 + 0.5]
tp = np.array([p.timestamp for p in poses]) - t0
quats = np.array([p.orientation for p in poses])
keep = np.concatenate([[True], np.diff(tp) > 1e-6])
tp, quats = tp[keep], quats[keep]
rots = Rotation.from_quat(quats)
slerp = Slerp(tp, rots)
print(f"  {len(tp)} mocap poses ({1.0/np.median(np.diff(tp)):.0f} Hz)")

# mocap-FD omega on the native mocap timestamps (body frame)
Rm = rots.as_matrix()
om_mocap = np.zeros((len(tp) - 1, 3))
for i in range(len(tp) - 1):
    om_mocap[i] = so3_log(Rm[i].T @ Rm[i + 1]) / (tp[i + 1] - tp[i])
t_omm = 0.5 * (tp[:-1] + tp[1:])

# ---------------------------------------------------------------------------
# A. mocap health: |omega| gyro vs mocap over time
# ---------------------------------------------------------------------------
print("\n== A. MoCap health check (|omega| gyro vs mocap-FD) ==")
nm_g = np.linalg.norm(gyro, axis=1)
nm_m = np.linalg.norm(om_mocap, axis=1)
# compare on 0.1 s bins
bins = np.arange(0.0, dur, 0.1)
gi = np.clip(np.digitize(t_imu, bins) - 1, 0, len(bins) - 1)
mi = np.clip(np.digitize(t_omm, bins) - 1, 0, len(bins) - 1)
g_bin = np.array([np.median(nm_g[gi == b]) if (gi == b).any() else np.nan for b in range(len(bins))])
m_bin = np.array([np.median(nm_m[mi == b]) if (mi == b).any() else np.nan for b in range(len(bins))])
disagree = np.abs(m_bin - g_bin)
bad = bins[np.nan_to_num(disagree) > 2.0]
print(f"  max |omega|: gyro {nm_g.max():.1f}, mocap-FD {nm_m.max():.1f} rad/s")
print(f"  bins with |median omega| disagreement > 2 rad/s: {len(bad)} "
      f"{('-> t = ' + ', '.join(f'{b:.1f}' for b in bad[:12])) if len(bad) else ''}")
# mocap sample-interval gaps (dropouts)
gaps = np.diff(tp)
big = np.where(gaps > 0.02)[0]
print(f"  mocap gaps > 20ms: {len(big)}"
      + (f" at t = {', '.join(f'{tp[i]:.2f}({gaps[i]*1e3:.0f}ms)' for i in big[:12])}" if len(big) else ""))

# define CLEAN window: drop leading/trailing bins where mocap disagrees
clean_mask_bins = np.nan_to_num(disagree, nan=np.inf) < 2.0
# find largest contiguous clean run
runs, start = [], None
for b in range(len(bins)):
    if clean_mask_bins[b] and start is None:
        start = b
    elif not clean_mask_bins[b] and start is not None:
        runs.append((start, b)); start = None
if start is not None:
    runs.append((start, len(bins)))
s, e = max(runs, key=lambda r: r[1] - r[0])
T0c, T1c = bins[s] + 0.05, bins[e - 1] + 0.05
print(f"  largest clean segment: [{T0c:.1f}, {T1c:.1f}] s ({T1c-T0c:.1f}s of {dur:.1f}s)")

# ---------------------------------------------------------------------------
# B. vibration vs coherent rotation (band-split at 8ms-spline bandwidth)
# ---------------------------------------------------------------------------
print("\n== B. above-bandwidth gyro content: vibration or rotation? ==")
from scipy.signal import butter, sosfiltfilt, welch

fs = 1.0 / np.median(np.diff(t_imu))
# cubic spline with dt=8ms tracks roughly up to ~1/(4*dt) ~ 31 Hz; use 25 Hz split
f_split = 25.0
sos_lo = butter(4, f_split, 'low', fs=fs, output='sos')
sel_c = (t_imu > T0c) & (t_imu < T1c)
g_c = gyro[sel_c]
g_lo = sosfiltfilt(sos_lo, g_c, axis=0)
g_hi = g_c - g_lo
om_at_imu = np.interp(t_imu[sel_c], t_omm, np.convolve(nm_m, np.ones(25) / 25, mode='same'))
flip = om_at_imu > 4.0
for name, msk in (("flip", flip), ("quiet", ~flip)):
    if msk.sum() < 100:
        print(f"  {name}: too few samples"); continue
    p_hi = np.sqrt(np.mean(g_hi[msk] ** 2))
    p_lo = np.sqrt(np.mean(g_lo[msk] ** 2))
    print(f"  {name:6s}: RMS gyro >{f_split:.0f}Hz = {p_hi:.3f} rad/s | <{f_split:.0f}Hz = {p_lo:.3f} rad/s")

# is the >25Hz content present in mocap omega? PSD comparison 25-100 Hz
f_g, P_g = welch(np.linalg.norm(g_c, axis=1), fs=fs, nperseg=4096)
fs_m = 1.0 / np.median(np.diff(t_omm))
sel_m = (t_omm > T0c) & (t_omm < T1c)
f_m, P_m = welch(np.linalg.norm(om_mocap[sel_m], axis=1), fs=fs_m, nperseg=2048)
band = (f_g > 25) & (f_g < min(100, fs_m / 2 - 5))
band_m = (f_m > 25) & (f_m < min(100, fs_m / 2 - 5))
print(f"  PSD power 25-100Hz: gyro {np.trapz(P_g[band], f_g[band]):.4f} | "
      f"mocap-FD {np.trapz(P_m[band_m], f_m[band_m]):.4f} (rad/s)^2 "
      f"(mocap-FD noise floor inflates its value; if gyro >> mocap it is real but untracked by mocap; "
      f"if comparable, mocap-FD is mostly noise there)")

# ---------------------------------------------------------------------------
# C. gyro scale/misalignment regression on the clean segment
# ---------------------------------------------------------------------------
print("\n== C. r_gyro ~ A*omega + b (scale/misalignment) ==")
# time-align gyro to mocap omega: scan offset maximizing corr of |omega|
nm_g_s = np.convolve(nm_g, np.ones(11) / 11, mode='same')
offsets = np.arange(-0.05, 0.05, 0.001)
best = (None, -2)
for off in offsets:
    m_interp = np.interp(t_imu[sel_c] + off, t_omm, nm_m)
    c = np.corrcoef(nm_g_s[sel_c], m_interp)[0, 1]
    if c > best[1]:
        best = (off, c)
off, c = best
print(f"  imu->mocap time offset: {off*1e3:+.1f} ms (corr {c:.4f})")

# lowpass both to 15 Hz to suppress vibration + mocap FD noise, then regress
sos15 = butter(4, 15.0, 'low', fs=fs, output='sos')
g15 = sosfiltfilt(sos15, g_c, axis=0)
om_m_i = np.column_stack([np.interp(t_imu[sel_c] + off, t_omm, om_mocap[:, k]) for k in range(3)])
sos15m = butter(4, min(15.0, 0.45 * fs_m), 'low', fs=fs, output='sos')  # already interp to imu rate
om15 = sosfiltfilt(sos15m, om_m_i, axis=0)
r = g15 - om15
X = np.column_stack([om15, np.ones(len(om15))])
coef, *_ = np.linalg.lstsq(X, r, rcond=None)
A, b = coef[:3].T, coef[3]
r_after = r - X @ coef
print(f"  bias b = {np.array2string(b, precision=4)} rad/s")
print(f"  A (scale/misalign, percent):\n{np.array2string(100*A, precision=2, suppress_small=True)}")
print(f"  residual RMS: before {np.sqrt(np.mean(r**2)):.4f} -> after {np.sqrt(np.mean(r_after**2)):.4f} rad/s")
rr = np.linalg.norm(r, axis=1)
print(f"  corr(|r|, |omega|) before regression: "
      f"{np.corrcoef(rr, np.linalg.norm(om15, axis=1))[0,1]:+.3f}, after: "
      f"{np.corrcoef(np.linalg.norm(r_after, axis=1), np.linalg.norm(om15, axis=1))[0,1]:+.3f}")

# ---------------------------------------------------------------------------
# D. spline omega-tracking vs grid density (gyro dead-reckoned reference)
# ---------------------------------------------------------------------------
print("\n== D. spline omega residual vs knot density (clean segment) ==")
selc = (t_imu > T0c) & (t_imu < T1c)
tc = t_imu[selc]
gc = gyro[selc]
# dead-reckon R from gyro (midpoint rule)
R_dr = [np.eye(3)]
for i in range(1, len(tc)):
    w_mid = 0.5 * (gc[i - 1] + gc[i])
    R_dr.append(R_dr[-1] @ so3_exp(w_mid * (tc[i] - tc[i - 1])))
R_dr = np.array(R_dr)


def fit_and_omega_residual(dt):
    kt = np.arange(tc[0], tc[-1] + 1e-9, dt)
    ext = extend_knots(kt)
    # Greville sampling from dead-reckoned R (nearest sample)
    R_k = np.empty((len(kt), 3, 3))
    for j in range(len(kt)):
        xi = np.clip((ext[j + 4] + ext[j + 5] + ext[j + 6]) / 3.0, tc[0], tc[-1])
        R_k[j] = R_dr[np.searchsorted(tc, xi).clip(0, len(tc) - 1)]
    sp = NonUniformSO3Spline(kt, R_k)
    # refine 4x at Greville points
    for _ in range(4):
        R_new = sp.R.copy()
        for j in range(len(kt)):
            xi = np.clip((ext[j + 4] + ext[j + 5] + ext[j + 6]) / 3.0,
                         sp.t_start, sp.t_end - 1e-9)
            R_s, _ = sp.evaluate(xi)
            R_new[j] = sp.R[j] @ (R_s.T @ R_dr[np.searchsorted(tc, xi).clip(0, len(tc) - 1)])
        sp = NonUniformSO3Spline(kt, R_new)
    # omega residual at IMU samples (subsample x4 for speed)
    res = []
    om_n = []
    for i in range(0, len(tc), 4):
        if not (sp.t_start + 0.02 < tc[i] < sp.t_end - 0.02):
            continue
        _, w = sp.evaluate(tc[i])
        res.append(gc[i] - w)
        om_n.append(np.linalg.norm(gc[i]))
    res = np.array(res); om_n = np.array(om_n)
    rn = np.linalg.norm(res, axis=1)
    fl = om_n > 4.0
    print(f"  dt={dt*1e3:4.1f}ms (n={len(kt):5d}): omega residual RMS "
          f"all {np.sqrt(np.mean(rn**2)):.4f} | flip {np.sqrt(np.mean(rn[fl]**2)):.4f} "
          f"| quiet {np.sqrt(np.mean(rn[~fl]**2)):.4f} rad/s "
          f"| corr(|r|,|om|) {np.corrcoef(rn, om_n)[0,1]:+.3f}")


for dt in (0.016, 0.008, 0.004, 0.002):
    fit_and_omega_residual(dt)

print("\nInterpretation guide: if omega residual during flips stays >> quiet even at"
      "\n2ms (and C shows scale/misalign explains it), denser knots chase noise ->"
      "\nadaptive knots demoted; gyro robust loss / scale calibration promoted.")
