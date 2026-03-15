import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""
Verify agiros state data integrity:
1. Does diff(position)/dt match velocity?  
2. Does the quaternion give correct body velocity?
3. Check on both gentle (original) and aggressive (circle_fwd) bags
"""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics
from scipy.spatial.transform import Rotation

for name, path in [
    ('original', 'rosbags/2025-12-17-16-02-22.bag'),
    ('circle_fwd', 'rosbags/circle_forward_2025-12-17-17-37-38.bag'),
    ('backflips', 'rosbags/backflips_2025-12-17-17-41-24.bag'),
]:
    print(f"\n{'='*80}")
    print(f"BAG: {name}")
    print(f"{'='*80}")
    
    bag = load_bag_topics(path, verbose=False)
    states = bag.agiros_state
    
    ts = np.array([s.timestamp for s in states])
    pos = np.array([s.position for s in states])
    vel = np.array([s.velocity for s in states])
    quats = np.array([s.orientation for s in states])
    omegas = np.array([s.angular_velocity for s in states])
    
    # Filter duplicates
    dt = np.diff(ts)
    mask = np.concatenate([[True], dt > 0.001])
    ts = ts[mask]; pos = pos[mask]; vel = vel[mask]; quats = quats[mask]; omegas = omegas[mask]
    
    # 1. Check velocity = d(position)/dt
    dt = np.diff(ts)
    v_diff = np.diff(pos, axis=0) / dt[:, None]
    v_mid = (vel[:-1] + vel[1:]) / 2  # average velocity at midpoints
    
    # Correlation between v_diff and v_mid for each axis
    for ax, label in enumerate(['x', 'y', 'z']):
        c = np.corrcoef(v_diff[:, ax], v_mid[:, ax])[0, 1]
        rmse = np.sqrt(np.mean((v_diff[:, ax] - v_mid[:, ax])**2))
        print(f"  vel_{label} check: corr(d_pos/dt, vel) = {c:.6f}, RMSE = {rmse:.4f} m/s")
    
    # 2. Check body velocity makes sense
    # During forward flight, v_body_x should be positive and largest component
    speeds = np.linalg.norm(vel, axis=1)
    fast = speeds > 1.5
    
    if fast.sum() > 10:
        v_body_all = np.zeros_like(vel)
        for i in range(len(vel)):
            R_wb = Rotation.from_quat(quats[i]).as_matrix()
            v_body_all[i] = R_wb.T @ vel[i]
        
        vb_fast = v_body_all[fast]
        print(f"\n  v_body during fast flight (speed > 1.5, n={fast.sum()}):")
        print(f"    v_body_x: mean={vb_fast[:,0].mean():+.3f} std={vb_fast[:,0].std():.3f} "
              f"min={vb_fast[:,0].min():+.3f} max={vb_fast[:,0].max():+.3f}")
        print(f"    v_body_y: mean={vb_fast[:,1].mean():+.3f} std={vb_fast[:,1].std():.3f}")
        print(f"    v_body_z: mean={vb_fast[:,2].mean():+.3f} std={vb_fast[:,2].std():.3f}")
        
        # 3. Check for orientation discontinuities
        # Compute angular rate from quaternion differentiation
        q = quats.copy()
        
        # Quaternion sign consistency (make sure consecutive quats are on same hemisphere)
        for i in range(1, len(q)):
            if np.dot(q[i], q[i-1]) < 0:
                q[i] = -q[i]
        
        # Check for large jumps in euler angles
        eulers = np.array([Rotation.from_quat(qi).as_euler('ZYX', degrees=True) for qi in quats])
        d_euler = np.diff(eulers, axis=0)
        dt_euler = np.diff(ts)
        euler_rate = d_euler / dt_euler[:, None]
        
        # Find moments of large euler rate
        large_rate_mask = np.any(np.abs(euler_rate) > 500, axis=1)  # > 500 deg/s
        print(f"\n  Quaternion continuity:")
        print(f"    Large euler rate (>500°/s) moments: {large_rate_mask.sum()} / {len(large_rate_mask)}")
        
        if large_rate_mask.sum() > 0:
            first_jump = np.where(large_rate_mask)[0][0]
            t_jump = ts[first_jump] - ts[0]
            print(f"    First jump at t={t_jump:.2f}s")
            print(f"    Euler rates: yaw={euler_rate[first_jump,0]:.0f}°/s "
                  f"pitch={euler_rate[first_jump,1]:.0f}°/s "
                  f"roll={euler_rate[first_jump,2]:.0f}°/s")
    
    # 4. Check raw world velocity: during circles, velocity direction should rotate
    # Just show first few seconds of fast flight
    radar = bag.radar_pcl
    t0 = ts[0]
    
    print(f"\n  Sample frames during fast flight:")
    count = 0
    for r in radar:
        if r.velocities is None or len(r.velocities) < 2:
            continue
        t_rel = r.timestamp - t0
        idx = np.argmin(np.abs(ts - r.timestamp))
        v_world = vel[idx]
        speed = np.linalg.norm(v_world)
        
        if speed < 2.0:
            continue
        
        q = quats[idx]
        R_wb = Rotation.from_quat(q).as_matrix()
        v_body = R_wb.T @ v_world
        euler = Rotation.from_quat(q).as_euler('ZYX', degrees=True)
        
        mean_doppler = np.mean(r.velocities)
        
        # Check: forward direction of body in world frame
        body_fwd_world = R_wb @ np.array([1, 0, 0])
        # v_world projected onto body forward (in world frame)
        v_along_fwd = np.dot(v_world, body_fwd_world)
        
        print(f"    t={t_rel:.2f}s speed={speed:.2f}"
              f" yaw={euler[0]:+7.1f}° v_body_x={v_body[0]:+.2f}"
              f" mean_dop={mean_doppler:+.2f}"
              f" v_along_fwd={v_along_fwd:+.2f}"
              f" body_fwd_world=[{body_fwd_world[0]:+.2f},{body_fwd_world[1]:+.2f},{body_fwd_world[2]:+.2f}]")
        
        count += 1
        if count >= 15:
            break
