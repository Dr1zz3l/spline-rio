#!/usr/bin/env python3
"""
Decisive diagnostic for the floor-plane factor: using MoCap truth, compute the EXACT
statistic the factor classifies on -- z_pred_world = (R_wb * p_body).z + p_drone_z
in the SOLVER (start-recentred) frame -- for every static return, and check:
  (1) where do TRUE floor returns (mocap world-z ~ 0) land? -> the correct floor_z
  (2) are floor vs non-floor returns SEPARABLE by this statistic? (classifiability)
  (3) for true floor returns, residual spread = achievable vertical anchor precision.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import rotation_matrix_from_euler, predict_doppler_velocity
from config_loader import load_config

bag_key = sys.argv[1] if len(sys.argv) > 1 else "slow_racing_best_velocity"
GATE = 0.5
_cfg = load_config(); _EXT = _cfg['extrinsics']
BAG = _cfg['bags']['bags'].get(bag_key, bag_key)
R_bs = rotation_matrix_from_euler(np.radians(180.0), np.radians(27.5), 0.0)
t_bs = np.array(_EXT['translation_body_m'])
off = _EXT['imu_mocap_offset_sec'] - _EXT['radar_imu_offset_sec']

bag = load_bag_topics(BAG, verbose=False)
rf = bag.radar_velocity
for f in rf: f.timestamp += off
st = bag.agiros_state
filt = [st[0]]
for s in st[1:]:
    if s.timestamp - filt[-1].timestamp >= 1e-3: filt.append(s)
st = filt
mt = np.array([s.timestamp for s in st]); mp = np.array([s.position for s in st])
mq = np.array([s.orientation for s in st]); mv = np.array([s.velocity for s in st])
mw = np.array([s.angular_velocity for s in st])
ok = np.linalg.norm(mq, axis=1) > 0.1
mt, mp, mq, mv, mw = mt[ok], mp[ok], mq[ok], mv[ok], mw[ok]
for i in range(1, len(mq)):
    if np.dot(mq[i], mq[i-1]) < 0: mq[i] *= -1
slerp = Slerp(mt, Rotation.from_quat(mq))
pos_i = interp1d(mt, mp, axis=0, kind='cubic', fill_value='extrapolate')
vel_i = interp1d(mt, mv, axis=0, kind='linear', fill_value='extrapolate')
omg_i = interp1d(mt, mw, axis=0, kind='linear', fill_value='extrapolate')
t0, t1 = mt[0], mt[-1]
start_z = pos_i(t0)[2]
print(f"bag={bag_key}  start mocap z = {start_z:.3f} m  -> floor at solver_z ~ {-start_z:.3f}")

zpred, zworld_true, zoff = [], [], []
for f in rf:
    t = f.timestamp
    if t < t0 or t > t1 or f.num_points() == 0: continue
    R_wb = slerp(t).as_matrix(); pw = pos_i(t)
    pts = np.array(f.positions)
    pred = predict_doppler_velocity(v_body_world=vel_i(t), omega_body=omg_i(t),
        R_world_from_body=R_wb, radar_positions_sensor=pts,
        T_body_from_sensor=t_bs, R_body_from_sensor=R_bs)
    vel = np.array(f.velocities)
    keep = np.abs(vel - pred) <= GATE
    if not np.any(keep): continue
    pb = (R_bs @ pts[keep].T).T + t_bs           # body-frame return pos
    zo = (R_wb @ pb.T).T[:, 2]                    # (R_wb * p_body).z  == z_off
    pdrone_solver = pw[2] - start_z              # drone z in solver frame
    zp = zo + pdrone_solver                       # factor's z_pred_world
    # true world z (mocap frame) of the return
    pwld = (R_wb @ pb.T).T + pw
    zoff.extend(zo); zpred.extend(zp); zworld_true.extend(pwld[:, 2])

zpred = np.array(zpred); zworld_true = np.array(zworld_true)
is_floor = np.abs(zworld_true) < 0.35          # truly on the ground (mocap z~0)
print(f"\nstatic returns: {len(zpred)}   true-floor (|mocap_z|<0.35): {is_floor.sum()} "
      f"({100*is_floor.mean():.0f}%)")
print(f"factor z_pred_world for TRUE-FLOOR returns: "
      f"mean {zpred[is_floor].mean():+.3f}  std {zpred[is_floor].std():.3f}  "
      f"[p10 {np.percentile(zpred[is_floor],10):+.2f}, p90 {np.percentile(zpred[is_floor],90):+.2f}]")
print(f"factor z_pred_world for NON-floor returns:   "
      f"mean {zpred[~is_floor].mean():+.3f}  std {zpred[~is_floor].std():.3f}  "
      f"[p10 {np.percentile(zpred[~is_floor],10):+.2f}, p90 {np.percentile(zpred[~is_floor],90):+.2f}]")
# separability: if we classify floor by |z_pred - floor_z|<band, with floor_z = -start_z
fz = -start_z
for band in (0.2, 0.4, 0.6):
    sel = np.abs(zpred - fz) < band
    purity = is_floor[sel].mean() if sel.sum() else 0
    recall = (sel & is_floor).sum() / max(is_floor.sum(), 1)
    print(f"  band {band}: selected {sel.sum():4d}  purity {100*purity:3.0f}%  "
          f"recall {100*recall:3.0f}%")
