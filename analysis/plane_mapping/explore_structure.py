#!/usr/bin/env python3
"""
Feasibility spike for FEATURE/PLANE TRACKING -> absolute position.

The deployed RIO uses radar only for Doppler VELOCITY; the point POSITIONS (x,y,z)
inform only the lever-arm direction, so position is pure dead-reckoning (drifts).
The hall is a structured cuboid (walls/floor/roof) + an arena rope-mesh, so the
STATIC returns lie on planes. If those planes are observable per-frame, a
point-to-plane factor anchors ABSOLUTE position and kills drift.

This script does NOT change the solver. It uses MoCap pose (ground truth) as an
oracle to answer the only question that decides feasibility:

  Per radar frame, how many STATIC returns do we get, and do they form
  DETECTABLE, STABLE planes with enough returns to constrain position?

Static-return selection = |v_meas - v_pred_egomotion| < gate (the world is
stationary, so a static scatterer's Doppler must match the predicted ego-motion
Doppler). Same trick as viz/plot_radar_map.py with MAX_DOPPLER_ERROR low.

Usage: ../.venv/bin/python3 plane_mapping/explore_structure.py [bag] [doppler_gate]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import open3d as o3d

from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import rotation_matrix_from_euler, predict_doppler_velocity
from config_loader import load_config

# ---------------- config ----------------
bag_key = sys.argv[1] if len(sys.argv) > 1 else "slow_racing_best_velocity"
DOPPLER_GATE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # m/s, static-ness
MIN_RANGE, MAX_RANGE, MIN_INTENSITY = 0.2, 80.0, 1.0

_cfg = load_config()
BAGS = _cfg['bags']['bags']
_EXT = _cfg['extrinsics']
BAG_PATH = BAGS.get(bag_key, bag_key)

# deployed extrinsics: pitch 27.5 frozen (yaml keeps 25.5 stale seed)
R_bs = rotation_matrix_from_euler(np.radians(180.0), np.radians(27.5), np.radians(0.0))
t_bs = np.array(_EXT['translation_body_m'])
IMU_MOCAP_OFFSET = _EXT['imu_mocap_offset_sec']
RADAR_IMU_OFFSET = _EXT['radar_imu_offset_sec']

print("=" * 78)
print(f"PLANE-FEATURE FEASIBILITY  bag={bag_key}  doppler_gate={DOPPLER_GATE} m/s")
print("=" * 78)

# ---------------- load ----------------
bag = load_bag_topics(BAG_PATH, verbose=False)
radar_frames = bag.radar_velocity
states = bag.agiros_state

radar_total_offset = IMU_MOCAP_OFFSET - RADAR_IMU_OFFSET
for f in radar_frames:
    f.timestamp += radar_total_offset

# dedupe near-duplicate mocap stamps
filt = [states[0]]
for s in states[1:]:
    if s.timestamp - filt[-1].timestamp >= 1e-3:
        filt.append(s)
states = filt

mt = np.array([s.timestamp for s in states])
mp = np.array([s.position for s in states])
mq = np.array([s.orientation for s in states])
mv = np.array([s.velocity for s in states])
mw = np.array([s.angular_velocity for s in states])
valid = np.linalg.norm(mq, axis=1) > 0.1
mt, mp, mq, mv, mw = mt[valid], mp[valid], mq[valid], mv[valid], mw[valid]
for i in range(1, len(mq)):
    if np.dot(mq[i], mq[i - 1]) < 0:
        mq[i] *= -1

slerp = Slerp(mt, Rotation.from_quat(mq))
pos_i = interp1d(mt, mp, axis=0, kind='cubic', fill_value='extrapolate')
vel_i = interp1d(mt, mv, axis=0, kind='linear', fill_value='extrapolate')
omg_i = interp1d(mt, mw, axis=0, kind='linear', fill_value='extrapolate')
t0, t1 = mt[0], mt[-1]

# ---------------- per-frame static return extraction ----------------
all_world, all_inten = [], []
per_frame_static = []     # count of static returns per frame
per_frame_world = []      # list of (t, Nx3 world static pts) for observability pass
n_total_returns = 0
n_in_window = 0

for f in radar_frames:
    t = f.timestamp
    if t < t0 or t > t1 or f.num_points() == 0:
        continue
    n_in_window += 1
    R_wb = slerp(t).as_matrix()
    pw0 = pos_i(t)
    pts_s = np.array(f.positions)
    pred = predict_doppler_velocity(
        v_body_world=vel_i(t), omega_body=omg_i(t), R_world_from_body=R_wb,
        radar_positions_sensor=pts_s, T_body_from_sensor=t_bs, R_body_from_sensor=R_bs)
    rng = f.ranges if f.ranges is not None else np.linalg.norm(pts_s, axis=1)
    inten = f.intensities if f.intensities is not None else np.full(len(pts_s), 100.0)
    vel = np.array(f.velocities)
    n_total_returns += len(pts_s)

    keep = ((rng >= MIN_RANGE) & (rng <= MAX_RANGE) & (inten >= MIN_INTENSITY) &
            (np.abs(vel - pred) <= DOPPLER_GATE))
    if not np.any(keep):
        per_frame_static.append(0)
        continue
    p_body = (R_bs @ pts_s[keep].T).T + t_bs
    p_world = (R_wb @ p_body.T).T + pw0
    # room sanity bounds
    inb = ((np.abs(p_world[:, 0]) <= 12) & (np.abs(p_world[:, 1]) <= 12) &
           (p_world[:, 2] >= -2) & (p_world[:, 2] <= 6))
    p_world = p_world[inb]
    per_frame_static.append(len(p_world))
    if len(p_world):
        per_frame_world.append((t, p_world))
        all_world.append(p_world)
        all_inten.append(inten[keep][inb])

all_world = np.vstack(all_world)
all_inten = np.concatenate(all_inten)
per_frame_static = np.array(per_frame_static)

print(f"\nRadar frames in window: {n_in_window}")
print(f"Total returns: {n_total_returns}   static (gate {DOPPLER_GATE}): {len(all_world)} "
      f"({100*len(all_world)/max(n_total_returns,1):.1f}%)")
print(f"Static returns / frame: mean {per_frame_static.mean():.1f}  median "
      f"{np.median(per_frame_static):.0f}  p10 {np.percentile(per_frame_static,10):.0f}  "
      f"max {per_frame_static.max()}")
print(f"Frames with 0 static: {np.sum(per_frame_static==0)}/{len(per_frame_static)} "
      f"({100*np.mean(per_frame_static==0):.0f}%)   "
      f">=3 static: {100*np.mean(per_frame_static>=3):.0f}%   "
      f">=10: {100*np.mean(per_frame_static>=10):.0f}%")

# ---------------- global plane detection (the MAP an oracle could build) ----------------
print("\n--- dominant planes in the aggregate static cloud (Open3D RANSAC) ---")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_world)
planes = []
work = pcd
remaining = np.asarray(work.points)
for pi in range(8):
    if len(remaining) < 200:
        break
    work = o3d.geometry.PointCloud()
    work.points = o3d.utility.Vector3dVector(remaining)
    model, inliers = work.segment_plane(distance_threshold=0.15,
                                         ransac_n=3, num_iterations=2000)
    if len(inliers) < 150:
        break
    a, b, c, d = model
    n = np.array([a, b, c]); n /= np.linalg.norm(n)
    inl = remaining[inliers]
    # classify
    if abs(n[2]) > 0.85:
        kind = "FLOOR" if inl[:, 2].mean() < 1.0 else "ROOF"
    elif abs(n[2]) < 0.35:
        kind = "WALL"
    else:
        kind = "slanted"
    planes.append((model, n, inl, kind))
    print(f"  plane {pi}: {kind:7s} n=[{n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}] "
          f"d={d:+.2f}  inliers={len(inliers)}  "
          f"extent x[{inl[:,0].min():.1f},{inl[:,0].max():.1f}] "
          f"y[{inl[:,1].min():.1f},{inl[:,1].max():.1f}] "
          f"z[{inl[:,2].min():.1f},{inl[:,2].max():.1f}]")
    remaining = np.delete(remaining, inliers, axis=0)

print(f"\n{len(planes)} planes capture "
      f"{100*(len(all_world)-len(remaining))/len(all_world):.0f}% of static returns")

# ---------------- per-frame observability against the global planes ----------------
# For each frame's static returns, count how many global planes get >=K returns
# (a plane is "observable this frame" only if enough points hit it).
K = 3
plane_models = [p[0] for p in planes]
def plane_dist(pts, model):
    a, b, c, d = model
    nn = np.sqrt(a*a + b*b + c*c)
    return np.abs(pts @ np.array([a, b, c]) + d) / nn

obs_planes_per_frame = []
floor_obs = 0
for t, pw in per_frame_world:
    hit = 0
    for (model, n, _, kind) in planes:
        on = plane_dist(pw, model) <= 0.20
        if np.sum(on) >= K:
            hit += 1
            if kind == "FLOOR":
                floor_obs += 1
    obs_planes_per_frame.append(hit)
obs_planes_per_frame = np.array(obs_planes_per_frame)
nf = len(per_frame_world)
print(f"\nPer-frame plane observability (>= {K} returns on a global plane):")
print(f"  >=1 plane: {100*np.mean(obs_planes_per_frame>=1):.0f}%   "
      f">=2 (translation-observable): {100*np.mean(obs_planes_per_frame>=2):.0f}%   "
      f">=3 (full 3D): {100*np.mean(obs_planes_per_frame>=3):.0f}%")
print(f"  FLOOR observable (vertical anchor): {100*floor_obs/max(nf,1):.0f}% of nonempty frames")

# ---------------- figure ----------------
fig = plt.figure(figsize=(14, 5))
colors = plt.get_cmap('tab10')
ax1 = fig.add_subplot(131)
for i, (_, _, inl, kind) in enumerate(planes):
    ax1.scatter(inl[:, 0], inl[:, 1], s=2, color=colors(i % 10), label=f"{kind} {i}")
ax1.scatter(mp[:, 0], mp[:, 1], s=1, color='k', alpha=0.3)
ax1.set_title('top-down (xy): planes + traj'); ax1.set_xlabel('x'); ax1.set_ylabel('y')
ax1.axis('equal'); ax1.legend(fontsize=6, markerscale=3)
ax2 = fig.add_subplot(132)
for i, (_, _, inl, kind) in enumerate(planes):
    ax2.scatter(inl[:, 0], inl[:, 2], s=2, color=colors(i % 10))
ax2.scatter(mp[:, 0], mp[:, 2], s=1, color='k', alpha=0.3)
ax2.set_title('side (xz)'); ax2.set_xlabel('x'); ax2.set_ylabel('z'); ax2.axis('equal')
ax3 = fig.add_subplot(133)
ax3.hist(per_frame_static, bins=range(0, max(per_frame_static.max(), 1) + 2),
         color='steelblue')
ax3.set_title('static returns / frame'); ax3.set_xlabel('count'); ax3.set_ylabel('frames')
plt.tight_layout()
out = Path(__file__).parent / f"structure_{bag_key}.png"
plt.savefig(out, dpi=130)
print(f"\nSaved {out}")
