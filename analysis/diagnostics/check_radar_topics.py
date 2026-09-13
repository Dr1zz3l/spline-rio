import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""Compare radar_pcl vs radar_velocity topic velocities."""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics

bag = load_bag_topics('rosbags/2025-12-17-16-02-22.bag', verbose=False)
pcl = bag.radar_pcl
vel = bag.radar_velocity

print(f"radar_pcl frames: {len(pcl)}, radar_velocity frames: {len(vel)}")

# Find matching frames and compare
print("\n=== Matching frames velocity comparison ===")
matches = 0
sign_matches = 0
sign_flips = 0

for p in pcl[:200]:
    if p.velocities is None or len(p.velocities) == 0:
        continue
    best_v = None
    best_dt = 0.05
    for v in vel:
        dt = abs(v.timestamp - p.timestamp)
        if dt < best_dt:
            best_dt = dt
            best_v = v
    if best_v is None or best_v.velocities is None:
        continue
    if len(best_v.velocities) != len(p.velocities):
        continue
    
    pcl_v = p.velocities
    vel_v = best_v.velocities
    
    if matches < 10:
        ratio_str = ""
        for i in range(min(3, len(pcl_v))):
            if abs(pcl_v[i]) > 0.01:
                ratio_str += f" r={vel_v[i]/pcl_v[i]:.3f}"
        print(f"  dt={best_dt*1000:.1f}ms  n={len(pcl_v)}  pcl={pcl_v[:3].round(3)}  vel={vel_v[:3].round(3)}{ratio_str}")
    
    for i in range(len(pcl_v)):
        if abs(pcl_v[i]) > 0.01:
            if np.sign(pcl_v[i]) == np.sign(vel_v[i]):
                sign_matches += 1
            else:
                sign_flips += 1
    matches += 1

print(f"\nTotal matched frames: {matches}")
print(f"Sign comparison: same={sign_matches}, flipped={sign_flips}")
if sign_matches + sign_flips > 0:
    print(f"Sign agreement: {100*sign_matches/(sign_matches+sign_flips):.1f}%")

# Also check: does radar_velocity have positions?
if vel and vel[0].positions is not None:
    print(f"\nradar_velocity has positions: shape={vel[0].positions.shape}")
    print(f"  First positions: {vel[0].positions[:3]}")
    if pcl:
        print(f"  First pcl positions: {pcl[0].positions[:3]}")
