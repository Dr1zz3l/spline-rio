import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""
Test the effect of TIME OFFSET on sign agreement.
Check if radar-MoCap offset differs from IMU-MoCap offset.
Also verify the raw Doppler distribution per bag.
"""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import predict_doppler_velocity
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

BAGS = {
    'circle':     'rosbags/circle_2025-12-17-17-21-37.bag',
    'circle_fwd': 'rosbags/circle_forward_2025-12-17-17-37-38.bag',
}

T_sensor = np.array([0.07, 0, 0])
R_bs_p30 = Rotation.from_euler('ZYX', [0, 30, 0], degrees=True).as_matrix()

def test_offset(name, path, offset):
    bag = load_bag_topics(path, verbose=False)
    states = bag.agiros_state
    radar = bag.radar_pcl
    
    ts = np.array([s.timestamp for s in states])
    vels = np.array([s.velocity for s in states])
    quats = np.array([s.orientation for s in states])
    omegas = np.array([s.angular_velocity for s in states])
    
    # Filter duplicates
    dt = np.diff(ts)
    mask = np.concatenate([[True], dt > 0.001])
    ts = ts[mask]; vels = vels[mask]; quats = quats[mask]; omegas = omegas[mask]
    
    # Smoothing
    for i in range(3):
        vels[:, i] = savgol_filter(vels[:, i], min(15, len(vels)//2*2-1), 3)
    
    all_pred = []
    all_meas = []
    all_speeds = []
    
    for r in radar:
        if r.velocities is None or len(r.velocities) < 1:
            continue
        t = r.timestamp + offset
        if t < ts[0] or t > ts[-1]:
            continue
        
        idx = np.searchsorted(ts, t)
        idx = np.clip(idx, 1, len(ts) - 1)
        
        if abs(ts[idx] - t) < abs(ts[idx-1] - t):
            q = quats[idx]; omega = omegas[idx]
        else:
            q = quats[idx-1]; omega = omegas[idx-1]
        
        alpha = (t - ts[idx-1]) / (ts[idx] - ts[idx-1])
        v_world = (1 - alpha) * vels[idx-1] + alpha * vels[idx]
        speed = np.linalg.norm(v_world)
        
        R_wb = Rotation.from_quat(q).as_matrix()
        v_pred = predict_doppler_velocity(v_world, omega, R_wb, r.positions, T_sensor, R_bs_p30)
        
        all_pred.extend(v_pred)
        all_meas.extend(r.velocities)
        all_speeds.extend([speed] * len(v_pred))
    
    pred = np.array(all_pred)
    meas = np.array(all_meas)
    speeds = np.array(all_speeds)
    
    sig = np.abs(meas) > 0.3
    p, m, s = pred[sig], meas[sig], speeds[sig]
    
    corr = np.corrcoef(p, m)[0, 1] if len(p) > 10 else float('nan')
    same_pct = (np.sign(p) == np.sign(m)).mean() if len(p) > 0 else 0
    
    # Speed > 2 m/s  
    fast = s >= 2.0
    if fast.sum() > 10:
        fast_corr = np.corrcoef(p[fast], m[fast])[0, 1]
        fast_same = (np.sign(p[fast]) == np.sign(m[fast])).mean()
        fast_pred_mean = np.mean(p[fast])
        fast_meas_mean = np.mean(m[fast])
    else:
        fast_corr = fast_same = fast_pred_mean = fast_meas_mean = 0
    
    return corr, same_pct, fast_corr, fast_same, fast_pred_mean, fast_meas_mean

print("TIME OFFSET SWEEP (pitch=+30°)")
print("="*100)

offsets = [0.0, -0.01, -0.02, -0.03, -0.04, -0.05, +0.01, +0.02]

for name, path in BAGS.items():
    print(f"\n--- {name} ---")
    print(f"{'offset_ms':>10s}  {'all_corr':>8s}  {'all_same':>8s}  {'fast_corr':>9s}  {'fast_same':>9s}  {'fast_pred':>9s}  {'fast_meas':>9s}")
    
    for offset in sorted(offsets):
        c, s, fc, fs, fp, fm = test_offset(name, path, offset)
        print(f"{offset*1000:>10.0f}  {c:+8.3f}  {s:8.1%}  {fc:+9.3f}  {fs:9.1%}  {fp:+9.3f}  {fm:+9.3f}")

# Also check raw velocity distributions
print("\n\n" + "="*100)
print("RAW VELOCITY DISTRIBUTIONS")
print("="*100)

for name, path in BAGS.items():
    bag = load_bag_topics(path, verbose=False)
    radar = bag.radar_pcl
    
    all_vel = []
    for r in radar:
        if r.velocities is not None:
            all_vel.extend(r.velocities)
    all_vel = np.array(all_vel)
    
    # Unique quantized values
    unique_v = np.unique(np.round(all_vel, 2))
    
    print(f"\n{name}: {len(all_vel)} total points")
    print(f"  Velocity range: [{all_vel.min():.3f}, {all_vel.max():.3f}]")
    print(f"  Mean: {all_vel.mean():.3f}, Std: {all_vel.std():.3f}")
    print(f"  Unique values ({len(unique_v)}): {unique_v[:20].tolist()}")
    
    # Histogram of velocity signs
    n_pos = (all_vel > 0.1).sum()
    n_neg = (all_vel < -0.1).sum()
    n_zero = ((all_vel >= -0.1) & (all_vel <= 0.1)).sum()
    print(f"  Positive: {n_pos} ({n_pos/len(all_vel):.1%}), Negative: {n_neg} ({n_neg/len(all_vel):.1%}), ~Zero: {n_zero} ({n_zero/len(all_vel):.1%})")
    
    # Histogram by velocity bin
    bins = np.arange(-5, 5.5, 0.604)
    counts, edges = np.histogram(all_vel, bins=bins)
    top_bins = np.argsort(counts)[-5:][::-1]
    print(f"  Top 5 velocity bins:")
    for b in top_bins:
        center = (edges[b] + edges[b+1]) / 2
        print(f"    [{edges[b]:.2f}, {edges[b+1]:.2f}) center={center:.2f}: {counts[b]} ({counts[b]/len(all_vel):.1%})")
