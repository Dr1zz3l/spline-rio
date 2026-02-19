"""
Compare raw velocity distributions across ALL bags.
If some bags have negated Doppler, this will be obvious.
"""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics

BAGS = {
    'original':    'rosbags/2025-12-17-16-02-22.bag',
    'circle':      'rosbags/circle_2025-12-17-17-21-37.bag',
    'circle_fast': 'rosbags/circle_fast_2025-12-17-17-25-34.bag',
    'circle_fwd':  'rosbags/circle_forward_2025-12-17-17-37-38.bag',
    'backflips':   'rosbags/backflips_2025-12-17-17-41-24.bag',
    'loopings':    'rosbags/circle_fast_forward_2025-12-17-17-39-49.bag',
}

print(f"{'bag':>15s} | {'N':>6s} | {'min':>7s} {'max':>7s} | {'mean':>7s} {'std':>7s} | {'pos%':>6s} {'neg%':>6s} {'zero%':>6s} | unique values")
print("-"*140)

for name, path in BAGS.items():
    bag = load_bag_topics(path, verbose=False)
    radar = bag.radar_pcl
    
    all_vel = []
    for r in radar:
        if r.velocities is not None:
            all_vel.extend(r.velocities)
    v = np.array(all_vel)
    
    unique = np.unique(np.round(v, 2))
    n_pos = (v > 0.1).sum()
    n_neg = (v < -0.1).sum()
    n_z = len(v) - n_pos - n_neg
    
    print(f"{name:>15s} | {len(v):6d} | {v.min():+7.2f} {v.max():+7.2f} | "
          f"{v.mean():+7.2f} {v.std():7.2f} | "
          f"{n_pos/len(v):6.1%} {n_neg/len(v):6.1%} {n_z/len(v):6.1%} | "
          f"{unique.tolist()}")

# Now check: do we see the pattern that FORWARD-moving drone → negative Doppler?
# Let's look at a simple metric: correlation between world-frame forward velocity
# and mean Doppler per frame
print("\n\n" + "="*100)
print("PER-FRAME: mean_Doppler vs drone_speed_body_x")
print("="*100)

from scipy.spatial.transform import Rotation

for name, path in BAGS.items():
    bag = load_bag_topics(path, verbose=False)
    states = bag.agiros_state
    radar = bag.radar_pcl
    
    ts = np.array([s.timestamp for s in states])
    vels = np.array([s.velocity for s in states])
    quats = np.array([s.orientation for s in states])
    
    frame_vbx = []
    frame_mean_doppler = []
    
    for r in radar:
        if r.velocities is None or len(r.velocities) < 1:
            continue
        t = r.timestamp
        idx = np.argmin(np.abs(ts - t))
        v_world = vels[idx]
        q = quats[idx]
        R_wb = Rotation.from_quat(q).as_matrix()
        v_body = R_wb.T @ v_world
        
        frame_vbx.append(v_body[0])
        frame_mean_doppler.append(np.mean(r.velocities))
    
    vbx = np.array(frame_vbx)
    md = np.array(frame_mean_doppler)
    
    # Correlation
    corr = np.corrcoef(vbx, md)[0, 1] if len(vbx) > 10 else 0
    
    # Check sign: when body moves forward (vbx > 1), what's mean Doppler?
    fast_mask = vbx > 1.0
    if fast_mask.sum() > 5:
        fast_mean_doppler = md[fast_mask].mean()
        fast_mean_vbx = vbx[fast_mask].mean()
    else:
        fast_mean_doppler = 0
        fast_mean_vbx = 0
    
    print(f"  {name:>15s}: corr(v_body_x, mean_doppler) = {corr:+.3f} | "
          f"fast: v_bx_mean={fast_mean_vbx:+.2f}, doppler_mean={fast_mean_doppler:+.2f}")
