"""NEES consistency evaluation of the SW solver's live-edge covariance (ROADMAP 4.1).

Usage:
  1. Run the solver with covariance collection:
       ../.venv/bin/python3 validate_live_solver.py <bag> ... --set nees=1
     -> saves ../plots/nees_last_run.npz (t, vel, quat, Sigma 6x6 per window)
  2. ../.venv/bin/python3 nees_eval.py <bag_key> [npz_path]

Computes per-window NEES of (v_world, right-tangent ori) at the live edge vs
MoCap GT.  For a consistent estimator NEES_vel ~ chi2(3) (mean 3), same for
ori; the empirical inflation factor mean(NEES)/dof is the covariance
calibration result (sigma_scale = sqrt(inflation)).

GT: pure Vicon mocap (slerp for R; smoothed position FD for velocity).  The
solver time base is already mocap-aligned (driver shifts IMU by
imu_mocap_offset at load), so npz timestamps compare directly.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from scipy.spatial.transform import Rotation, Slerp

bag_key = sys.argv[1]
npz_path = sys.argv[2] if len(sys.argv) > 2 else '../plots/nees_last_run.npz'

d = np.load(npz_path)
t_w, vel_w, quat_w, Sig_w = d['t'], d['vel'], d['quat'], d['Sigma']
print(f"{len(t_w)} windows from {npz_path}")

cfg = load_config()
bag_path = str(Path('..') / cfg['bags']['bags'][bag_key])
bag = load_bag_topics(bag_path, verbose=False)

poses = bag.mocap_pose
tp = np.array([p.timestamp for p in poses])
keep = np.concatenate([[True], np.diff(tp) > 1e-6])
tp = tp[keep]
pos = np.array([p.position for p in poses])[keep]
quat = np.array([p.orientation for p in poses])[keep]
rots = Rotation.from_quat(quat)
slerp = Slerp(tp, rots)

t_v = 0.5 * (tp[:-1] + tp[1:])   # mocap-FD midpoints, used only for the ori glitch mask
# GT velocity: the Agiros state velocity (Vicon-fed), Butterworth-10 Hz filtfilt --
# the SAME clean reference the headline velocity RMSE uses (NOT a finite difference
# of mocap position, whose near-duplicate-timestamp glitches were inflating the mean).
from scipy.signal import butter, filtfilt
_ag  = bag.agiros_state
t_ag = np.array([s.timestamp for s in _ag])
_ak  = np.concatenate([[True], np.diff(t_ag) > 1e-6])
t_ag = t_ag[_ak]
v_ag = np.array([s.velocity for s in _ag])[_ak]
_fs  = 1.0 / np.median(np.diff(t_ag)); _fc = min(10.0, _fs * 0.4)
_bb, _aa = butter(4, _fc / (_fs / 2), btype='low')
if len(v_ag) > 27:
    v_ag = np.column_stack([filtfilt(_bb, _aa, v_ag[:, k]) for k in range(3)])

# GT-quality mask: both channels fail only at mocap dropouts / large angular
# glitches (the clean Agiros velocity reference has no FD spikes to mask).
om = np.linalg.norm(
    Rotation.from_matrix(
        np.einsum('nij,njk->nik', rots[:-1].as_matrix().transpose(0, 2, 1),
                  rots[1:].as_matrix())).as_rotvec(), axis=1) / np.diff(tp)
gap = np.diff(tp) > 0.02
bad_o_t = t_v[(om > 25.0) | gap]            # GT-bad windows (mocap dropouts/glitches)


def _clean(t, bad):
    return not len(bad) or np.min(np.abs(bad - t)) > 0.15


from scipy.stats import chi2 as _chi2
nees_v, nees_o, nees_f = [], [], []
err_v, err_o = [], []
for i, t in enumerate(t_w):
    if not (tp[0] + 0.05 < t < tp[-1] - 0.05):
        continue
    R_gt = slerp(t).as_matrix()
    v_gt = np.array([np.interp(t, t_ag, v_ag[:, k]) for k in range(3)])
    R_est = Rotation.from_quat(quat_w[i]).as_matrix()
    e_v = vel_w[i] - v_gt
    e_o = Rotation.from_matrix(R_est.T @ R_gt).as_rotvec()  # right tangent
    S = Sig_w[i]
    Svv, Soo = S[:3, :3], S[3:, 3:]
    v_ok = _clean(t, bad_o_t)
    try:
        if _clean(t, bad_o_t):
            nees_o.append(float(e_o @ np.linalg.solve(Soo, e_o)))
            err_o.append(np.degrees(np.linalg.norm(e_o)))
        if v_ok:
            nees_v.append(float(e_v @ np.linalg.solve(Svv, e_v)))
            err_v.append(np.linalg.norm(e_v))
            e6 = np.concatenate([e_v, e_o])
            nees_f.append(float(e6 @ np.linalg.solve(S, e6)))
    except np.linalg.LinAlgError:
        continue

nees_v, nees_o, nees_f = map(np.array, (nees_v, nees_o, nees_f))
print(f"\nbag {bag_key}: {len(t_w)} windows; kept ori={len(nees_o)} vel={len(nees_v)} "
      f"(per-channel GT-quality mask)")
print(f"  raw errors (kept): vel RMSE {np.sqrt(np.mean(np.array(err_v)**2)):.3f} m/s, "
      f"ori RMSE {np.sqrt(np.mean(np.array(err_o)**2)):.2f} deg")
for name, n, dof in (("vel", nees_v, 3), ("ori", nees_o, 3), ("full", nees_f, 6)):
    N = len(n)
    # 95% interval for the mean NEES of N consistent dof-NEES samples: chi2_{N*dof}/N
    lo, hi = _chi2.ppf([0.025, 0.975], N * dof) / N
    flag = "OK" if lo <= n.mean() <= hi else ("OVERCONF" if n.mean() > hi else "CONSERV")
    print(f"  NEES {name:4s}: MASKED MEAN {n.mean():6.2f}  (median {np.median(n):5.2f}) "
          f"vs {dof}  95%CI[{lo:.2f},{hi:.2f}]  {flag}  sigma_scale x{np.sqrt(n.mean()/dof):.2f}")
