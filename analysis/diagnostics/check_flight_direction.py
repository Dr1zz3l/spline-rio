import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""Check body-frame velocity direction for each bag to understand flight orientation patterns."""
import numpy as np
import sys
sys.path.insert(0, '.')
from rosbag_loader.loader import load_bag_topics
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

bags = [
    ('original',    '../rosbags/2025-12-17-16-02-22.bag'),
    ('circle',      '../rosbags/circle_2025-12-17-17-21-37.bag'),
    ('circle_fast', '../rosbags/circle_fast_2025-12-17-17-25-34.bag'),
    ('circle_fwd',  '../rosbags/circle_forward_2025-12-17-17-37-38.bag'),
    ('backflips',   '../rosbags/backflips_2025-12-17-17-41-24.bag'),
    ('loopings',    '../rosbags/circle_fast_forward_2025-12-17-17-39-49.bag'),
]

# Known yaw flip test results
winners = {
    'original':    'normal',
    'circle':      'normal',
    'circle_fast': 'normal',
    'circle_fwd':  'FLIPPED',
    'backflips':   'FLIPPED',
    'loopings':    'FLIPPED',
}

header = f"{'bag':>15s} | {'mean_vbx':>8s} | {'fwd%':>6s} | {'speed':>5s} | {'winner':>8s}"
print(header)
print('-' * len(header))

for name, path in bags:
    bag = load_bag_topics(path, verbose=False)
    s = bag.agiros_state
    ts = np.array([x.timestamp for x in s])
    v = np.array([x.velocity for x in s])
    q = np.array([x.orientation for x in s])
    
    # Clean duplicate timestamps
    dt = np.diff(ts)
    msk = np.concatenate([[True], dt > 0.001])
    ts, v, q = ts[msk], v[msk], q[msk]
    
    # Smooth velocities
    for i in range(3):
        v[:, i] = savgol_filter(v[:, i], min(15, len(v) // 2 * 2 - 1), 3)
    
    # Convert to body frame
    vb = np.array([Rotation.from_quat(qi).inv().apply(vi) for qi, vi in zip(q, v)])
    spd = np.linalg.norm(vb, axis=1)
    
    fast = spd > 1.0
    if fast.sum() > 0:
        vbx_fast = vb[fast, 0]
        fwd_pct = (vbx_fast > 0).mean()
        print(f"{name:>15s} | {vbx_fast.mean():+8.2f} | {fwd_pct:5.1%} | {spd[fast].mean():5.2f} | {winners[name]:>8s}")
    else:
        print(f"{name:>15s} | N/A")

print()
print("Interpretation:")
print("  If fwd% > 50%: drone predominantly flies FORWARD in body frame (v_body_x > 0)")
print("  If fwd% < 50%: drone predominantly flies BACKWARD in body frame (v_body_x < 0)")
print()
print("  'normal'  winner = radar at body +x (front), standard [0,+30,0] extrinsics")
print("  'FLIPPED' winner = radar at body -x (back), needs 180° yaw correction")
