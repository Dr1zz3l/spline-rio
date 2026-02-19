"""
Comprehensive per-point sign agreement test across ALL frames and ALL bags.

For each radar point, we check whether pred and meas have the same sign.
We bin by drone speed and report statistics.

Also tests the NEGATED model: v_pred = -(u · v_ant) for the case where
the radar convention might be "positive = receding" instead of "positive = approaching".
"""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import predict_doppler_velocity
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d

BAGS = {
    'original':    'rosbags/2025-12-17-16-02-22.bag',
    'circle':      'rosbags/circle_2025-12-17-17-21-37.bag',
    'circle_fast': 'rosbags/circle_fast_2025-12-17-17-25-34.bag',
    'circle_fwd':  'rosbags/circle_forward_2025-12-17-17-37-38.bag',
    'backflips':   'rosbags/backflips_2025-12-17-17-41-24.bag',
    'loopings':    'rosbags/circle_fast_forward_2025-12-17-17-39-49.bag',
}

T = np.array([0.07, 0, 0])
PITCH_VALUES = [+30, -30]

TIME_OFFSET = -0.02  # MoCap - radar offset

def analyze_bag(name, path):
    bag = load_bag_topics(path, verbose=False)
    states = bag.agiros_state
    radar = bag.radar_pcl
    
    ts = np.array([s.timestamp for s in states])
    vels = np.array([s.velocity for s in states])
    quats = np.array([s.orientation for s in states])
    omegas = np.array([s.angular_velocity for s in states])
    
    # Build interpolators for smooth velocity
    from scipy.signal import savgol_filter
    # Filter duplicate timestamps
    dt = np.diff(ts)
    mask = np.concatenate([[True], dt > 0.001])
    ts_f = ts[mask]
    vels_f = vels[mask]
    quats_f = quats[mask]
    omegas_f = omegas[mask]
    
    # Apply SavGol smoothing to velocity
    for i in range(3):
        vels_f[:, i] = savgol_filter(vels_f[:, i], min(15, len(vels_f)//2*2-1), 3)
    
    print(f"\n{'='*70}")
    print(f"BAG: {name} ({len(radar)} radar frames)")
    print(f"{'='*70}")
    
    results = {}
    
    for pitch in PITCH_VALUES:
        R_bs = Rotation.from_euler('ZYX', [0, pitch, 0], degrees=True).as_matrix()
        
        all_pred = []
        all_meas = []
        all_speeds = []
        
        for r in radar:
            if r.velocities is None or len(r.velocities) < 1:
                continue
            
            t_radar = r.timestamp + TIME_OFFSET
            
            if t_radar < ts_f[0] or t_radar > ts_f[-1]:
                continue
            
            # Interpolate state at radar time
            idx = np.searchsorted(ts_f, t_radar)
            idx = np.clip(idx, 1, len(ts_f) - 1)
            
            # Simple nearest neighbor for quaternion
            if abs(ts_f[idx] - t_radar) < abs(ts_f[idx-1] - t_radar):
                q = quats_f[idx]
                omega = omegas_f[idx]
            else:
                q = quats_f[idx-1]
                omega = omegas_f[idx-1]
            
            # Linear interp for velocity
            alpha = (t_radar - ts_f[idx-1]) / (ts_f[idx] - ts_f[idx-1])
            v_world = (1 - alpha) * vels_f[idx-1] + alpha * vels_f[idx]
            speed = np.linalg.norm(v_world)
            
            R_wb = Rotation.from_quat(q).as_matrix()
            
            v_pred = predict_doppler_velocity(v_world, omega, R_wb, r.positions, T, R_bs)
            v_meas = r.velocities
            
            all_pred.extend(v_pred)
            all_meas.extend(v_meas)
            all_speeds.extend([speed] * len(v_pred))
        
        all_pred = np.array(all_pred)
        all_meas = np.array(all_meas)
        all_speeds = np.array(all_speeds)
        
        # Filter out near-zero measurements (quantization at 0)
        sig_mask = np.abs(all_meas) > 0.3
        pred_s = all_pred[sig_mask]
        meas_s = all_meas[sig_mask]
        speeds_s = all_speeds[sig_mask]
        
        # Per-point sign agreement
        same_sign = np.sign(pred_s) == np.sign(meas_s)
        
        # Overall correlation
        if len(pred_s) > 10:
            corr = np.corrcoef(pred_s, meas_s)[0, 1]
        else:
            corr = float('nan')
        
        # Same but negated model
        neg_same_sign = np.sign(-pred_s) == np.sign(meas_s)
        if len(pred_s) > 10:
            neg_corr = np.corrcoef(-pred_s, meas_s)[0, 1]
        else:
            neg_corr = float('nan')
        
        # Speed-binned analysis
        speed_bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 10.0)]
        
        print(f"\n  pitch={pitch:+d}° | total points: {len(pred_s)} | "
              f"sign agree: {same_sign.mean():.1%} | corr: {corr:+.3f} | "
              f"neg sign agree: {neg_same_sign.mean():.1%} | neg corr: {neg_corr:+.3f}")
        
        print(f"    {'speed bin':>12s}  {'N':>6s}  {'same%':>6s}  {'neg_same%':>8s}  "
              f"{'corr':>7s}  {'neg_corr':>8s}  {'pred_mean':>9s}  {'meas_mean':>9s}")
        
        for lo, hi in speed_bins:
            mask = (speeds_s >= lo) & (speeds_s < hi)
            n = mask.sum()
            if n < 10:
                continue
            
            p = pred_s[mask]
            m = meas_s[mask]
            
            ss = (np.sign(p) == np.sign(m)).mean()
            ns = (np.sign(-p) == np.sign(m)).mean()
            c = np.corrcoef(p, m)[0, 1]
            nc = np.corrcoef(-p, m)[0, 1]
            
            print(f"    {lo:.1f}-{hi:.1f} m/s  {n:6d}  {ss:6.1%}  {ns:8.1%}  "
                  f"{c:+7.3f}  {nc:+8.3f}  {np.mean(p):+9.3f}  {np.mean(m):+9.3f}")
        
        results[pitch] = {
            'corr': corr, 'sign_agree': same_sign.mean(),
            'neg_corr': neg_corr, 'neg_sign_agree': neg_same_sign.mean()
        }
    
    return results

# Run analysis
print("COMPREHENSIVE SIGN AGREEMENT ANALYSIS")
print("="*70)
print(f"Time offset: {TIME_OFFSET*1000:.0f}ms | Translation: {T}")
print(f"Model: v_pred = u_body · v_ant_body")
print(f"Negated: v_neg  = -(u_body · v_ant_body)")

all_results = {}
for name, path in BAGS.items():
    try:
        all_results[name] = analyze_bag(name, path)
    except Exception as e:
        print(f"\nERROR on {name}: {e}")
        import traceback
        traceback.print_exc()

# Summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Bag':>15s} | {'p=+30 corr':>10s} {'p=+30 sign%':>11s} | "
      f"{'p=-30 corr':>10s} {'p=-30 sign%':>11s} | "
      f"{'p=+30 neg_corr':>14s} {'p=+30 neg%':>10s}")

for name in BAGS:
    if name not in all_results:
        continue
    r = all_results[name]
    r30 = r[30]
    rm30 = r[-30]
    print(f"{name:>15s} | {r30['corr']:+10.3f} {r30['sign_agree']:11.1%} | "
          f"{rm30['corr']:+10.3f} {rm30['sign_agree']:11.1%} | "
          f"{r30['neg_corr']:+14.3f} {r30['neg_sign_agree']:10.1%}")
