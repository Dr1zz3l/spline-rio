import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""Test Doppler sign convention across all bags."""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import predict_doppler_velocity
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d

def eval_bag(path, yaw, pitch, roll, negate_pred=False, t_start=5, t_end=55):
    bag = load_bag_topics(path, verbose=False)
    states = bag.agiros_state
    radar = bag.radar_pcl
    t0 = states[0].timestamp
    ts = np.array([s.timestamp for s in states])
    vels = np.array([s.velocity for s in states])
    quats = np.array([s.orientation for s in states])
    omegas = np.array([s.angular_velocity for s in states])
    vi = [interp1d(ts, vels[:, i], bounds_error=False, fill_value='extrapolate') for i in range(3)]
    oi = [interp1d(ts, omegas[:, i], bounds_error=False, fill_value='extrapolate') for i in range(3)]
    T = np.array([0.07, 0, 0])
    R_bs = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
    preds, meas = [], []
    for r in radar:
        t_rel = r.timestamp - t0
        if not (t_start <= t_rel < t_end):
            continue
        if r.velocities is None or len(r.velocities) == 0:
            continue
        t = r.timestamp
        if t < ts[0] or t > ts[-1]:
            continue
        v_w = np.array([vi[i](t) for i in range(3)])
        om = np.array([oi[i](t) for i in range(3)])
        idx = np.argmin(np.abs(ts - t))
        R_wb = Rotation.from_quat(quats[idx]).as_matrix()
        pts = r.positions
        rng = np.linalg.norm(pts, axis=1)
        mask = rng > 0.2
        pts, v_m = pts[mask], r.velocities[mask]
        if len(pts) == 0:
            continue
        v_p = predict_doppler_velocity(v_w, om, R_wb, pts, T, R_bs)
        if negate_pred:
            v_p = -v_p
        preds.extend(v_p)
        meas.extend(v_m)
    preds, meas = np.array(preds), np.array(meas)
    corr = np.corrcoef(preds, meas)[0, 1] if len(preds) > 10 else 0
    rmse = np.sqrt(np.mean((preds - meas) ** 2)) if len(preds) > 10 else 999
    return corr, rmse, len(preds)

bags = [
    ('original',    'rosbags/2025-12-17-16-02-22.bag'),
    ('circle',      'rosbags/circle_2025-12-17-17-21-37.bag'),
    ('circle_fast', 'rosbags/circle_fast_2025-12-17-17-25-34.bag'),
    ('circle_fwd',  'rosbags/circle_forward_2025-12-17-17-37-38.bag'),
    ('backflips',   'rosbags/backflips_2025-12-17-17-41-24.bag'),
    ('loopings',    'rosbags/circle_fast_forward_2025-12-17-17-39-49.bag'),
]

configs = [
    ("pitch=+30, normal",    0, +30, 0, False),
    ("pitch=+30, negated",   0, +30, 0, True),
    ("pitch=-30, normal",    0, -30, 0, False),
    ("pitch=-30, negated",   0, -30, 0, True),
]

for cfg_name, yaw, pitch, roll, negate in configs:
    print(f"\n=== {cfg_name} ===")
    for name, path in bags:
        c, r, n = eval_bag(path, yaw, pitch, roll, negate_pred=negate)
        print(f"  {name:15s}  corr={c:+.4f}  RMSE={r:.3f}  n={n}")
