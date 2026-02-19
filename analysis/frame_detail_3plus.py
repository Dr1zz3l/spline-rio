"""
Detailed frame-by-frame analysis at 3+ m/s on circle_fwd.
Check what's happening at the WORST frames.
"""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import predict_doppler_velocity
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

bag = load_bag_topics('rosbags/circle_forward_2025-12-17-17-37-38.bag', verbose=False)
states = bag.agiros_state
radar = bag.radar_pcl

t0 = states[0].timestamp
ts = np.array([s.timestamp for s in states])
vels = np.array([s.velocity for s in states])
quats = np.array([s.orientation for s in states])
omegas = np.array([s.angular_velocity for s in states])

# Filter duplicates
dt = np.diff(ts)
mask = np.concatenate([[True], dt > 0.001])
ts = ts[mask]; vels = vels[mask]; quats = quats[mask]; omegas = omegas[mask]

T_sensor = np.array([0.07, 0, 0])
R_bs_p30 = Rotation.from_euler('ZYX', [0, 30, 0], degrees=True).as_matrix()

# Raw velocity distribution
all_vel = []
for r in radar:
    if r.velocities is not None:
        all_vel.extend(r.velocities)
all_vel = np.array(all_vel)
print(f"RAW VELOCITY DISTRIBUTION (circle_fwd)")
print(f"  Total points: {len(all_vel)}")
print(f"  Range: [{all_vel.min():.3f}, {all_vel.max():.3f}]")
print(f"  Mean: {all_vel.mean():.3f}")
print(f"  Positive: {(all_vel > 0.1).sum()} ({(all_vel > 0.1).mean():.1%})")
print(f"  Negative: {(all_vel < -0.1).sum()} ({(all_vel < -0.1).mean():.1%})")
unique_v = np.unique(np.round(all_vel, 2))
print(f"  Unique values: {unique_v.tolist()}")

print(f"\n{'='*100}")
print(f"FRAME-BY-FRAME AT 3+ m/s (pitch=+30°)")
print(f"{'='*100}")

frame_count = 0
all_pred_3plus = []
all_meas_3plus = []

for r in radar:
    if r.velocities is None or len(r.velocities) < 1:
        continue
    
    t_rel = r.timestamp - t0
    t = r.timestamp  # no offset
    
    if t < ts[0] or t > ts[-1]:
        continue
    
    idx = np.searchsorted(ts, t)
    idx = np.clip(idx, 1, len(ts) - 1)
    
    if abs(ts[idx] - t) < abs(ts[idx-1] - t):
        q = quats[idx]; omega = omegas[idx]
    else:
        q = quats[idx-1]; omega = omegas[idx-1]
    
    alpha = np.clip((t - ts[idx-1]) / (ts[idx] - ts[idx-1]), 0, 1)
    v_world = (1 - alpha) * vels[idx-1] + alpha * vels[idx]
    speed = np.linalg.norm(v_world)
    
    if speed < 3.0:
        continue
    
    R_wb = Rotation.from_quat(q).as_matrix()
    R_bw = R_wb.T
    v_body = R_bw @ v_world
    euler = Rotation.from_quat(q).as_euler('ZYX', degrees=True)
    
    v_pred = predict_doppler_velocity(v_world, omega, R_wb, r.positions, T_sensor, R_bs_p30)
    v_meas = r.velocities
    
    all_pred_3plus.extend(v_pred)
    all_meas_3plus.extend(v_meas)
    
    # Show first 10 and every 20th frame after that
    frame_count += 1
    if frame_count <= 10 or frame_count % 20 == 0:
        n_same = np.sum(np.sign(v_pred) == np.sign(v_meas))
        n_total = len(v_pred)
        
        print(f"\nt={t_rel:.2f}s speed={speed:.2f} yaw={euler[0]:.1f}° pitch={euler[1]:.1f}° roll={euler[2]:.1f}°")
        print(f"  v_world=[{v_world[0]:+.2f},{v_world[1]:+.2f},{v_world[2]:+.2f}]")
        print(f"  v_body =[{v_body[0]:+.2f},{v_body[1]:+.2f},{v_body[2]:+.2f}]")
        print(f"  omega  =[{omega[0]:+.2f},{omega[1]:+.2f},{omega[2]:+.2f}]")
        print(f"  {n_total} points, {n_same}/{n_total} same sign")
        
        for j in range(min(3, len(v_pred))):
            rng = np.linalg.norm(r.positions[j])
            u_s = r.positions[j] / rng
            u_b = R_bs_p30 @ u_s
            
            print(f"    pt{j}: range={rng:.2f}m"
                  f" u_b=[{u_b[0]:+.3f},{u_b[1]:+.3f},{u_b[2]:+.3f}]"
                  f" pred={v_pred[j]:+.3f} meas={v_meas[j]:+.3f}"
                  f" {'SAME' if np.sign(v_pred[j]) == np.sign(v_meas[j]) else 'FLIP'}")

print(f"\n{'='*100}")
print(f"AGGREGATE at 3+ m/s: {len(all_pred_3plus)} points in {frame_count} frames")
p = np.array(all_pred_3plus)
m = np.array(all_meas_3plus)
sig = np.abs(m) > 0.3
print(f"  pred_mean={p[sig].mean():+.3f}, meas_mean={m[sig].mean():+.3f}")
print(f"  corr={np.corrcoef(p[sig], m[sig])[0,1]:+.3f}")
print(f"  sign agree={((np.sign(p[sig]) == np.sign(m[sig])).mean()):.1%}")

# Check what v_body looks like during these frames
print(f"\n  v_body_x stats: mean={np.mean([p for p in all_pred_3plus]):+.3f}")

# Also check: do we see the same issue WITHOUT applying body rotation?
# I.e., if we use v_world directly instead of v_body
print(f"\n{'='*100}")
print(f"SANITY CHECK: v_body during 3+ m/s frames")
v_body_x_list = []
for r in radar:
    if r.velocities is None or len(r.velocities) < 1:
        continue
    t = r.timestamp
    if t < ts[0] or t > ts[-1]:
        continue
    idx = np.searchsorted(ts, t)
    idx = np.clip(idx, 1, len(ts) - 1)
    if abs(ts[idx] - t) < abs(ts[idx-1] - t):
        q = quats[idx]
    else:
        q = quats[idx-1]
    alpha = np.clip((t - ts[idx-1]) / (ts[idx] - ts[idx-1]), 0, 1)
    v_world = (1 - alpha) * vels[idx-1] + alpha * vels[idx]
    speed = np.linalg.norm(v_world)
    if speed < 3.0:
        continue
    R_wb = Rotation.from_quat(q).as_matrix()
    v_body = R_wb.T @ v_world
    v_body_x_list.append(v_body)

vbx = np.array(v_body_x_list)
print(f"  v_body_x: mean={vbx[:,0].mean():+.3f} std={vbx[:,0].std():.3f} min={vbx[:,0].min():+.3f} max={vbx[:,0].max():+.3f}")
print(f"  v_body_y: mean={vbx[:,1].mean():+.3f} std={vbx[:,1].std():.3f}")
print(f"  v_body_z: mean={vbx[:,2].mean():+.3f} std={vbx[:,2].std():.3f}")

# Check v_world distribution
print(f"\n  v_world direction:")
for r in radar:
    if r.velocities is None or len(r.velocities) < 1:
        continue
    t = r.timestamp
    if t < ts[0] or t > ts[-1]:
        continue
    idx = np.searchsorted(ts, t)
    idx = np.clip(idx, 1, len(ts) - 1)
    alpha = np.clip((t - ts[idx-1]) / (ts[idx] - ts[idx-1]), 0, 1)
    v_world = (1 - alpha) * vels[idx-1] + alpha * vels[idx]
    speed = np.linalg.norm(v_world)
    if speed >= 3.0:
        t_rel = r.timestamp - t0
        # Just print first 5
        if t_rel < 15 or t_rel > 14.5 and t_rel < 15.5:
            print(f"    t={t_rel:.2f}  v_world=[{v_world[0]:+.2f},{v_world[1]:+.2f},{v_world[2]:+.2f}]  speed={speed:.2f}")
