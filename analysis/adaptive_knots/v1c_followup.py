"""V1c — follow-ups from V1b:

 1. Tail contamination of the REPORTED backflips RMSE: mocap is broken for
    t > ~17.2 s (dropouts + 48k rad/s FD spikes). Recompute settled/live
    pos+ori RMSE from the saved best-run npz excluding the broken tail
    (and the smaller dropouts at 7.58 / 14.57 s).
 2. Gyro scale/misalignment regression (V1b-C redone robustly): mask mocap
    glitches BEFORE filtering, align via smoothed |omega|, then regress.
 3. Spline omega-tracking vs grid (V1b-D redone): slerp-interpolated
    dead-reckoned reference (V1b used nearest-sample -> time-quantization
    noise ~|omega|*0.5ms/dt dominated dense grids).

Run:  cd analysis && ../.venv/bin/python3 adaptive_knots/v1c_followup.py
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
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

BAG_KEY = 'backflips_best_velocity'
T_CLEAN_END = 17.2   # from V1b-A
plots = Path(__file__).parent.parent.parent / 'plots' / BAG_KEY / 'live_solver'

# ---------------------------------------------------------------------------
# 1. tail contamination of reported RMSE
# ---------------------------------------------------------------------------
print("== 1. reported RMSE vs mocap-clean RMSE (best run tether10_zbneg1) ==")
d = np.load(plots / 'traj_arrays_tether10_zbneg1_sw.npz')
for tag, terr, rerr, tt in (("settled", d['pos_errors'], d['rot_errors'], d['t_rel']),
                            ("live   ", d['live_pos_errs'], d['live_rot_errs'], d['live_t_rel'])):
    full_p = np.sqrt(np.mean(terr ** 2))
    full_r = np.sqrt(np.mean(rerr ** 2))
    m = tt < T_CLEAN_END
    # also drop +-0.1s around the small mid-flight dropouts
    for tg in (7.58, 14.59):
        m &= ~((tt > tg - 0.1) & (tt < tg + 0.15))
    cl_p = np.sqrt(np.mean(terr[m] ** 2))
    cl_r = np.sqrt(np.mean(rerr[m] ** 2))
    tail_r = np.sqrt(np.mean(rerr[~(tt < T_CLEAN_END)] ** 2)) if (~(tt < T_CLEAN_END)).any() else 0
    print(f"  {tag}: pos {full_p:.3f} -> clean {cl_p:.3f} m | "
          f"ori {full_r:.2f} -> clean {cl_r:.2f} deg "
          f"(tail-only ori {tail_r:.1f} deg, {100*(~m).mean():.1f}% samples dropped)")

# ---------------------------------------------------------------------------
# load bag for 2./3.
# ---------------------------------------------------------------------------
cfg = load_config()
bag_path = str(Path(__file__).parent.parent.parent / cfg['bags']['bags'][BAG_KEY])
t_off, dur = cfg['bags']['timing'][BAG_KEY]
print(f"\nLoading {bag_path} ...")
bag = load_bag_topics(bag_path, verbose=False)
t0 = bag.start_time + t_off

imu = [x for x in bag.imu_data if t0 - 0.5 <= x.timestamp <= t0 + dur + 0.5]
t_imu = np.array([x.timestamp for x in imu]) - t0
gyro = np.array([x.angular_velocity for x in imu])
fs = 1.0 / np.median(np.diff(t_imu))

poses = [p for p in bag.mocap_pose if t0 - 0.5 <= p.timestamp <= t0 + dur + 0.5]
tp = np.array([p.timestamp for p in poses]) - t0
quats = np.array([p.orientation for p in poses])
keep = np.concatenate([[True], np.diff(tp) > 1e-6])
tp, quats = tp[keep], quats[keep]
Rm = Rotation.from_quat(quats).as_matrix()
om_mocap = np.zeros((len(tp) - 1, 3))
for i in range(len(tp) - 1):
    om_mocap[i] = so3_log(Rm[i].T @ Rm[i + 1]) / (tp[i + 1] - tp[i])
t_omm = 0.5 * (tp[:-1] + tp[1:])

# ---------------------------------------------------------------------------
# 2. robust gyro scale/misalignment regression
# ---------------------------------------------------------------------------
print("\n== 2. gyro scale/misalignment (robust) ==")
# clean window + glitch mask (|omega| sane, no gap straddle)
good = (np.linalg.norm(om_mocap, axis=1) < 25.0)
gap = np.diff(tp) > 0.02
good &= ~gap
good &= (t_omm > 0.2) & (t_omm < T_CLEAN_END)
print(f"  mocap omega samples kept: {good.sum()}/{len(good)}")

tg, og = t_omm[good], om_mocap[good]
# smooth both signals to ~10 Hz equivalent before alignment/regression
w_m = max(1, int(round(0.05 / np.median(np.diff(tg)))))
og_s = np.column_stack([np.convolve(og[:, k], np.ones(w_m) / w_m, 'same') for k in range(3)])
sos = butter(4, 10.0, 'low', fs=fs, output='sos')
g_s = sosfiltfilt(sos, gyro, axis=0)
sel = (t_imu > 0.3) & (t_imu < T_CLEAN_END - 0.1)
nm_g = np.linalg.norm(g_s, axis=1)
nm_m = np.linalg.norm(og_s, axis=1)

best = (0.0, -2.0)
for off in np.arange(-0.06, 0.06, 0.001):
    mi = np.column_stack([np.interp(t_imu[sel] + off, tg, og_s[:, k]) for k in range(3)])
    c = np.corrcoef(nm_g[sel], np.linalg.norm(mi, axis=1))[0, 1]
    if c > best[1]:
        best = (off, c)
off, c = best
print(f"  imu->mocap time offset: {off*1e3:+.1f} ms (corr {c:.4f})")

om_i = np.column_stack([np.interp(t_imu[sel] + off, tg, og_s[:, k]) for k in range(3)])
r = g_s[sel] - om_i
rms0 = np.sqrt(np.mean(r ** 2))
X = np.column_stack([om_i, np.ones(len(om_i))])
coef, *_ = np.linalg.lstsq(X, r, rcond=None)
A, b = coef[:3].T, coef[3]
r_a = r - X @ coef
print(f"  bias b = {np.array2string(b, precision=4)} rad/s")
print(f"  A (percent):\n{np.array2string(100*A, precision=2, suppress_small=True)}")
print(f"  residual RMS: {rms0:.4f} -> {np.sqrt(np.mean(r_a**2)):.4f} rad/s")
nm = np.linalg.norm(om_i, axis=1)
print(f"  corr(|r|,|omega|): before {np.corrcoef(np.linalg.norm(r,axis=1), nm)[0,1]:+.3f}, "
      f"after {np.corrcoef(np.linalg.norm(r_a,axis=1), nm)[0,1]:+.3f}")

# ---------------------------------------------------------------------------
# 3. spline omega-tracking vs grid density (slerp-interpolated reference)
# ---------------------------------------------------------------------------
print("\n== 3. spline omega residual vs knot density (slerp-fixed) ==")
selc = (t_imu > 0.3) & (t_imu < T_CLEAN_END - 0.1)
tc = t_imu[selc]
gc = gyro[selc]
R_dr = [np.eye(3)]
for i in range(1, len(tc)):
    R_dr.append(R_dr[-1] @ so3_exp(0.5 * (gc[i - 1] + gc[i]) * (tc[i] - tc[i - 1])))
R_dr = np.array(R_dr)
qs = Rotation.from_matrix(R_dr)
dr_slerp = Slerp(tc, qs)


def R_ref(t):
    return dr_slerp(np.clip(t, tc[0], tc[-1])).as_matrix()


for dt in (0.016, 0.008, 0.004, 0.002):
    kt = np.arange(tc[0], tc[-1] + 1e-9, dt)
    ext = extend_knots(kt)
    xi_all = np.clip((ext[4:4+len(kt)] + ext[5:5+len(kt)] + ext[6:6+len(kt)]) / 3.0,
                     tc[0], tc[-1])
    R_k = dr_slerp(xi_all).as_matrix()
    sp = NonUniformSO3Spline(kt, R_k)
    for _ in range(4):
        R_new = sp.R.copy()
        xi_cl = np.clip(xi_all, sp.t_start, sp.t_end - 1e-9)
        for j in range(len(kt)):
            R_s, _ = sp.evaluate(xi_cl[j])
            R_new[j] = sp.R[j] @ (R_s.T @ R_ref(xi_cl[j]))
        sp = NonUniformSO3Spline(kt, R_new)
    res, om_n = [], []
    for i in range(0, len(tc), 4):
        if not (sp.t_start + 0.02 < tc[i] < sp.t_end - 0.02):
            continue
        _, w = sp.evaluate(tc[i])
        res.append(gc[i] - w)
        om_n.append(np.linalg.norm(gc[i]))
    res = np.array(res); om_n = np.array(om_n)
    rn = np.linalg.norm(res, axis=1)
    fl = om_n > 4.0
    print(f"  dt={dt*1e3:4.1f}ms (n={len(kt):5d}): omega res RMS "
          f"all {np.sqrt(np.mean(rn**2)):.4f} | flip {np.sqrt(np.mean(rn[fl]**2)):.4f} "
          f"| quiet {np.sqrt(np.mean(rn[~fl]**2)):.4f} rad/s "
          f"| corr {np.corrcoef(rn, om_n)[0,1]:+.3f}")
print("\n(>25Hz vibration floor from V1b-B: ~0.17 rad/s — residuals near that floor")
print(" mean the grid is NOT the limiter; the gyro factors are noise-limited.)")
