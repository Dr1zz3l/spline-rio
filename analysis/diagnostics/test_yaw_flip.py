import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""
Test the 180° yaw flip hypothesis.

Theory: In some trajectory profiles, the agiros body frame x-axis is 
rotated 180° relative to the physical drone. If the radar is at a fixed 
physical location, this means in the "flipped" body frame:
  - T_sensor goes from [+0.07, 0, 0] to [-0.07, 0, 0]
  - R_body_from_sensor gains an extra R_z(180°) pre-rotation

We test this by running both "normal" and "flipped" extrinsics on all bags.
If the hypothesis is correct, each bag should work with exactly ONE of the two.
"""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import predict_doppler_velocity
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

BAGS = {
    'original':    'rosbags/2025-12-17-16-02-22.bag',
    'circle':      'rosbags/circle_2025-12-17-17-21-37.bag',
    'circle_fast': 'rosbags/circle_fast_2025-12-17-17-25-34.bag',
    'circle_fwd':  'rosbags/circle_forward_2025-12-17-17-37-38.bag',
    'backflips':   'rosbags/backflips_2025-12-17-17-41-24.bag',
    'loopings':    'rosbags/circle_fast_forward_2025-12-17-17-39-49.bag',
}

# Normal extrinsics: radar at body +x, tilted 30° down
T_normal = np.array([0.07, 0, 0])
R_bs_normal = Rotation.from_euler('ZYX', [0, 30, 0], degrees=True).as_matrix()

# Flipped extrinsics: body frame rotated 180° yaw relative to physical drone
# In the flipped body frame, sensor is at -x:
T_flipped = np.array([-0.07, 0, 0])
# R_bs_flipped = R_z(180°) @ R_y(+30°)
R_z180 = Rotation.from_euler('z', 180, degrees=True).as_matrix()
R_bs_flipped = R_z180 @ R_bs_normal

# Verify the boresight directions
boresight_s = np.array([1, 0, 0])  # sensor boresight
bore_normal = R_bs_normal @ boresight_s
bore_flipped = R_bs_flipped @ boresight_s
print(f"Normal boresight in body:  {bore_normal}  (forward + down) ✓")
print(f"Flipped boresight in body: {bore_flipped}  (backward + down) ✓")
print()

TIME_OFFSET = -0.02

def analyze_bag(name, path, T, R_bs, label):
    bag = load_bag_topics(path, verbose=False)
    states = bag.agiros_state
    radar = bag.radar_pcl
    
    ts = np.array([s.timestamp for s in states])
    vels = np.array([s.velocity for s in states])
    quats = np.array([s.orientation for s in states])
    omegas = np.array([s.angular_velocity for s in states])
    
    dt = np.diff(ts)
    mask = np.concatenate([[True], dt > 0.001])
    ts = ts[mask]; vels = vels[mask]; quats = quats[mask]; omegas = omegas[mask]
    
    for i in range(3):
        vels[:, i] = savgol_filter(vels[:, i], min(15, len(vels)//2*2-1), 3)
    
    all_pred = []
    all_meas = []
    all_speeds = []
    
    for r in radar:
        if r.velocities is None or len(r.velocities) < 1:
            continue
        t = r.timestamp + TIME_OFFSET
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
        
        R_wb = Rotation.from_quat(q).as_matrix()
        v_pred = predict_doppler_velocity(v_world, omega, R_wb, r.positions, T, R_bs)
        
        all_pred.extend(v_pred)
        all_meas.extend(r.velocities)
        all_speeds.extend([speed] * len(v_pred))
    
    pred = np.array(all_pred)
    meas = np.array(all_meas)
    speeds = np.array(all_speeds)
    
    sig = np.abs(meas) > 0.3
    p, m, s = pred[sig], meas[sig], speeds[sig]
    
    corr = np.corrcoef(p, m)[0, 1] if len(p) > 10 else float('nan')
    sign_agree = (np.sign(p) == np.sign(m)).mean() if len(p) > 0 else 0
    rmse = np.sqrt(np.mean((p - m)**2)) if len(p) > 0 else float('nan')
    
    # Fast subset
    fast = s >= 1.5
    if fast.sum() > 10:
        fast_corr = np.corrcoef(p[fast], m[fast])[0, 1]
        fast_sign = (np.sign(p[fast]) == np.sign(m[fast])).mean()
    else:
        fast_corr = float('nan')
        fast_sign = float('nan')
    
    return corr, sign_agree, rmse, fast_corr, fast_sign

print("=" * 110)
print("HYPOTHESIS TEST: 180° Body Frame Yaw Flip")
print("=" * 110)
print(f"{'bag':>15s} | {'normal corr':>11s} {'sign%':>6s} {'RMSE':>6s} {'fast_corr':>9s} | "
      f"{'flipped corr':>12s} {'sign%':>6s} {'RMSE':>6s} {'fast_corr':>9s} | {'winner':>8s}")
print("-" * 110)

for name, path in BAGS.items():
    bag = load_bag_topics(path, verbose=False)
    
    cn, sn, rn, fcn, fsn = analyze_bag(name, path, T_normal, R_bs_normal, "normal")
    cf, sf, rf, fcf, fsf = analyze_bag(name, path, T_flipped, R_bs_flipped, "flipped")
    
    winner = "normal" if cn > cf else "FLIPPED"
    
    print(f"{name:>15s} | {cn:+11.3f} {sn:6.1%} {rn:6.2f} {fcn:+9.3f} | "
          f"{cf:+12.3f} {sf:6.1%} {rf:6.2f} {fcf:+9.3f} | {winner:>8s}")

# Also check: what's v_body_x during fast flight for each bag?
print(f"\n{'=' * 110}")
print("Body x-velocity during fast flight (speed > 1.5 m/s)")
print(f"{'=' * 110}")

for name, path in BAGS.items():
    bag = load_bag_topics(path, verbose=False)
    states = bag.agiros_state
    ts = np.array([s.timestamp for s in states])
    vels = np.array([s.velocity for s in states])
    quats = np.array([s.orientation for s in states])
    
    dt = np.diff(ts)
    mask = np.concatenate([[True], dt > 0.001])
    ts = ts[mask]; vels = vels[mask]; quats = quats[mask]
    
    speeds = np.linalg.norm(vels, axis=1)
    fast = speeds > 1.5
    
    if fast.sum() < 10:
        print(f"  {name:>15s}: insufficient fast frames")
        continue
    
    v_body_all = np.zeros_like(vels[fast])
    for i, idx in enumerate(np.where(fast)[0]):
        R_wb = Rotation.from_quat(quats[idx]).as_matrix()
        v_body_all[i] = R_wb.T @ vels[idx]
    
    # Is v_body_x predominantly positive or negative?
    fwd_pct = (v_body_all[:, 0] > 0).mean()
    print(f"  {name:>15s}: v_body_x mean={v_body_all[:,0].mean():+.2f}"
          f"  std={v_body_all[:,0].std():.2f}"
          f"  forward%={fwd_pct:.1%}"
          f"  → {'FORWARD' if fwd_pct > 0.7 else 'BACKWARD' if fwd_pct < 0.3 else 'MIXED'} flight")
