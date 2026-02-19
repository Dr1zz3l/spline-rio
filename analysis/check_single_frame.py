"""Concrete single-frame Doppler analysis to verify sign convention."""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import predict_doppler_velocity
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d

# Use circle_fwd for clearest forward motion
bag = load_bag_topics('rosbags/circle_forward_2025-12-17-17-37-38.bag', verbose=False)
states = bag.agiros_state
radar = bag.radar_pcl
t0 = states[0].timestamp
ts = np.array([s.timestamp for s in states])
vels = np.array([s.velocity for s in states])
quats = np.array([s.orientation for s in states])
omegas = np.array([s.angular_velocity for s in states])

# Find a frame where drone is moving fast with clear forward velocity
for r in radar:
    t_rel = r.timestamp - t0
    if t_rel < 20 or t_rel > 40:
        continue
    if r.velocities is None or len(r.velocities) < 3:
        continue
    
    t = r.timestamp
    idx = np.argmin(np.abs(ts - t))
    v_world = vels[idx]
    q = quats[idx]
    omega = omegas[idx]
    speed = np.linalg.norm(v_world)
    
    if speed < 2.0:
        continue
    
    R_wb = Rotation.from_quat(q).as_matrix()
    R_bw = R_wb.T
    v_body = R_bw @ v_world
    euler = Rotation.from_quat(q).as_euler('ZYX', degrees=True)
    
    print(f"=== t={t_rel:.2f}s ===")
    print(f"  v_world = [{v_world[0]:+.3f}, {v_world[1]:+.3f}, {v_world[2]:+.3f}]  speed={speed:.2f}")
    print(f"  v_body  = [{v_body[0]:+.3f}, {v_body[1]:+.3f}, {v_body[2]:+.3f}]")
    print(f"  euler (ZYX) = yaw={euler[0]:.1f}° pitch={euler[1]:.1f}° roll={euler[2]:.1f}°")
    print(f"  omega = [{omega[0]:+.3f}, {omega[1]:+.3f}, {omega[2]:+.3f}]")
    
    # Analyze each radar point
    pts = r.positions
    v_meas = r.velocities
    print(f"  Radar points: {len(pts)}")
    
    T = np.array([0.07, 0, 0])
    for sign_label, pitch in [("pitch=+30", 30), ("pitch=-30", -30)]:
        R_bs = Rotation.from_euler('ZYX', [0, pitch, 0], degrees=True).as_matrix()
        v_pred = predict_doppler_velocity(v_world, omega, R_wb, pts, T, R_bs)
        
        print(f"\n  {sign_label}:")
        for j in range(min(5, len(pts))):
            rng = np.linalg.norm(pts[j])
            u_s = pts[j] / rng
            u_b = R_bs @ u_s
            
            # Compute lever arm
            v_body_in_body = R_bw @ v_world
            lever = np.cross(omega, T)
            v_ant = v_body_in_body + lever
            
            dp = np.dot(u_b, v_ant)
            
            print(f"    pt{j}: pos_s=[{pts[j][0]:+.2f},{pts[j][1]:+.2f},{pts[j][2]:+.2f}]"
                  f" u_b=[{u_b[0]:+.3f},{u_b[1]:+.3f},{u_b[2]:+.3f}]"
                  f" v_pred={dp:+.3f}  v_meas={v_meas[j]:+.3f}"
                  f"  {'SAME' if np.sign(dp) == np.sign(v_meas[j]) else 'FLIP' if abs(v_meas[j]) > 0.1 else '~0'}")
    
    print()
    # Only show first 3 matching frames
    if t_rel > 25:
        break

# Now check the ORIGINAL bag in the same way
print("\n" + "="*80)
print("=== ORIGINAL BAG ===")
bag2 = load_bag_topics('rosbags/2025-12-17-16-02-22.bag', verbose=False)
states2 = bag2.agiros_state
radar2 = bag2.radar_pcl
t02 = states2[0].timestamp
ts2 = np.array([s.timestamp for s in states2])
vels2 = np.array([s.velocity for s in states2])
quats2 = np.array([s.orientation for s in states2])
omegas2 = np.array([s.angular_velocity for s in states2])

count = 0
for r in radar2:
    t_rel = r.timestamp - t02
    if t_rel < 10 or t_rel > 30:
        continue
    if r.velocities is None or len(r.velocities) < 2:
        continue
    
    t = r.timestamp
    idx = np.argmin(np.abs(ts2 - t))
    v_world = vels2[idx]
    speed = np.linalg.norm(v_world)
    
    if speed < 0.5:
        continue
    
    q = quats2[idx]
    omega = omegas2[idx]
    R_wb = Rotation.from_quat(q).as_matrix()
    R_bw = R_wb.T
    v_body = R_bw @ v_world
    
    pts = r.positions
    v_meas = r.velocities
    
    T = np.array([0.07, 0, 0])
    R_bs_neg = Rotation.from_euler('ZYX', [0, -30, 0], degrees=True).as_matrix()
    v_pred_neg = predict_doppler_velocity(v_world, omega, R_wb, pts, T, R_bs_neg)
    
    print(f"t={t_rel:.2f}s  speed={speed:.2f}  v_body=[{v_body[0]:+.2f},{v_body[1]:+.2f},{v_body[2]:+.2f}]", end="")
    for j in range(min(3, len(pts))):
        print(f"  [{v_pred_neg[j]:+.2f} vs {v_meas[j]:+.2f}]", end="")
    print()
    
    count += 1
    if count >= 15:
        break
