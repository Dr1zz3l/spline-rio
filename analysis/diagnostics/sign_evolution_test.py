import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""
Simple hovering/slow test: when drone is slow (speed < 0.3), 
most Dopplers should be ~0 (static clutter removed).
When drone starts moving, check sign of first non-zero Dopplers.

Also: check the ONSET of flight in each bag to see if the sign
convention changes at some point within a bag.
"""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics
from scipy.spatial.transform import Rotation

BAGS = {
    'original':    'rosbags/2025-12-17-16-02-22.bag',
    'circle':      'rosbags/circle_2025-12-17-17-21-37.bag',
    'circle_fast': 'rosbags/circle_fast_2025-12-17-17-25-34.bag',
    'circle_fwd':  'rosbags/circle_forward_2025-12-17-17-37-38.bag',
    'backflips':   'rosbags/backflips_2025-12-17-17-41-24.bag',
    'loopings':    'rosbags/circle_fast_forward_2025-12-17-17-39-49.bag',
}

for name, path in BAGS.items():
    bag = load_bag_topics(path, verbose=False)
    states = bag.agiros_state
    radar = bag.radar_pcl
    
    ts = np.array([s.timestamp for s in states])
    vels = np.array([s.velocity for s in states])
    quats = np.array([s.orientation for s in states])
    t0 = ts[0]
    
    print(f"\n{'='*80}")
    print(f"BAG: {name}")
    print(f"{'='*80}")
    
    # Find onset of motion and first radar frames during motion
    print(f"  First 20 radar frames with speed > 0.3:")
    count = 0
    for r in radar:
        if r.velocities is None or len(r.velocities) < 1:
            continue
        t_rel = r.timestamp - t0
        idx = np.argmin(np.abs(ts - r.timestamp))
        v_world = vels[idx]
        speed = np.linalg.norm(v_world)
        
        if speed < 0.3:
            continue
        
        q = quats[idx]
        R_wb = Rotation.from_quat(q).as_matrix()
        v_body = R_wb.T @ v_world
        
        mean_d = np.mean(r.velocities)
        n_pts = len(r.velocities)
        v_vals = np.unique(np.round(r.velocities, 2))
        
        # Simple prediction: body-forward velocity * cos(30°) 
        # (rough Doppler for boresight target with pitch=+30°)
        v_boresight_pred = v_body[0] * 0.866 + v_body[2] * (-0.5)
        
        sign_agree = "✓" if np.sign(v_boresight_pred) == np.sign(mean_d) or abs(mean_d) < 0.1 else "✗"
        
        print(f"    t={t_rel:6.2f}s speed={speed:.2f} v_bx={v_body[0]:+.2f} "
              f"pred_bore={v_boresight_pred:+.2f} "
              f"mean_dop={mean_d:+.2f} pts={n_pts} {sign_agree} "
              f"vals={v_vals.tolist()}")
        
        count += 1
        if count >= 20:
            break
    
    # Now check: time-evolution of sign convention within the bag
    # Split into 5-second windows and compute correlation in each
    print(f"\n  Time-windowed corr(v_bx, mean_doppler):")
    
    frame_data = []
    for r in radar:
        if r.velocities is None or len(r.velocities) < 1:
            continue
        t_rel = r.timestamp - t0
        idx = np.argmin(np.abs(ts - r.timestamp))
        v_world = vels[idx]
        q = quats[idx]
        R_wb = Rotation.from_quat(q).as_matrix()
        v_body = R_wb.T @ v_world
        speed = np.linalg.norm(v_world)
        mean_d = np.mean(r.velocities)
        frame_data.append((t_rel, v_body[0], mean_d, speed))
    
    fd = np.array(frame_data)
    if len(fd) < 5:
        continue
    
    # Windows
    t_max = fd[:, 0].max()
    window = 5.0
    for t_start in np.arange(0, t_max, window):
        t_end = t_start + window
        mask = (fd[:, 0] >= t_start) & (fd[:, 0] < t_end)
        n = mask.sum()
        if n < 5:
            continue
        vbx = fd[mask, 1]
        md = fd[mask, 2]
        spd = fd[mask, 3]
        
        # Only consider frames with some motion
        fast_mask = spd > 0.5
        if fast_mask.sum() < 3:
            continue
        
        c = np.corrcoef(vbx[fast_mask], md[fast_mask])[0, 1]
        print(f"    t=[{t_start:.0f},{t_end:.0f})s: n={fast_mask.sum():3d} "
              f"corr={c:+.3f} "
              f"mean_vbx={vbx[fast_mask].mean():+.2f} "
              f"mean_dop={md[fast_mask].mean():+.2f}")
