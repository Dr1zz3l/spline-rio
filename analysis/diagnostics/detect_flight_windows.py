import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""Detect flight windows for each rosbag by analyzing MoCap velocity and radar data density."""

import numpy as np
import sys
sys.path.insert(0, "analysis")
from rosbag_loader.loader import load_bag_topics

BAGS = {
    # "circle":       "rosbags/circle_2025-12-17-17-21-37.bag",
    # "circle_fast":  "rosbags/circle_fast_2025-12-17-17-25-34.bag",
    # "circle_fwd":   "rosbags/circle_forward_2025-12-17-17-37-38.bag",
    # "loopings":     "rosbags/circle_fast_forward_2025-12-17-17-39-49.bag",
    # "backflips":    "rosbags/backflips_2025-12-17-17-41-24.bag",
    "backflips_best_velocity": "rosbags/Wed_11032026_1503/backflip_oldconfig_2026-03-11-16-40-51.bag",
    "circle_best_velocity": "rosbags/Wed_11032026_1503/circle_5mps_oldconfig_2026-03-11-16-38-36.bag",
    "fast_racing_1_velocity_no_clustering": "rosbags/Wed_11032026_1503/fast_racing_2026-03-11-15-19-58.bag",
    "fast_racing_best_velocity_crash": "rosbags/Wed_11032026_1503/fastracing_oldconfig_2026-03-11-16-52-09.bag",
    "fast_racing_best_velocity": "rosbags/Wed_11032026_1503/fastracing_oldconfig_2026-03-11-17-20-18.bag",
    "slow_racing_best_velocity": "rosbags/Wed_11032026_1503/slowracing_oldconfig_2026-03-11-17-18-43.bag"
}

VELOCITY_THRESHOLD = 0.5  # m/s — consider "in flight" above this

for bag_key, bag_path in BAGS.items():
    print(f"\n{'='*70}")
    print(f"BAG: {bag_key} ({bag_path})")
    print(f"{'='*70}")
    
    data = load_bag_topics(bag_path, verbose=False)
    
    # MoCap velocity analysis
    states = data.agiros_state
    if not states:
        print("  No agiros_state data!")
        continue
    
    times = np.array([s.timestamp for s in states])
    vels = np.array([s.velocity for s in states])
    speeds = np.linalg.norm(vels, axis=1)
    
    t0 = times[0]
    t_rel = times - t0
    
    print(f"  Total duration: {t_rel[-1]:.1f}s ({len(states)} MoCap samples)")
    print(f"  Speed: min={speeds.min():.2f} max={speeds.max():.2f} mean={speeds.mean():.2f} m/s")
    
    # Find contiguous flight windows where speed > threshold
    flying = speeds > VELOCITY_THRESHOLD
    transitions = np.diff(flying.astype(int))
    starts = np.where(transitions == 1)[0] + 1  # indices where flight begins
    ends = np.where(transitions == -1)[0]        # indices where flight ends
    
    # Handle edge cases
    if flying[0]:
        starts = np.insert(starts, 0, 0)
    if flying[-1]:
        ends = np.append(ends, len(flying) - 1)
    
    print(f"\n  Flight windows (speed > {VELOCITY_THRESHOLD} m/s):")
    for i, (si, ei) in enumerate(zip(starts, ends)):
        t_start = t_rel[si]
        t_end = t_rel[ei]
        dur = t_end - t_start
        mean_spd = speeds[si:ei+1].mean()
        max_spd = speeds[si:ei+1].max()
        print(f"    [{i}] t={t_start:.1f}s to {t_end:.1f}s (dur={dur:.1f}s, mean_speed={mean_spd:.1f} m/s, max={max_spd:.1f} m/s)")
    
    # Radar data density
    radar = data.radar_velocity
    if radar:
        radar_times = np.array([f.timestamp for f in radar]) - t0
        # Count points per 1s bin
        bins = np.arange(0, t_rel[-1] + 1, 1.0)
        counts, _ = np.histogram(radar_times, bins=bins)
        
        # Find radar-active windows (>5 points/sec)
        active = counts > 5
        active_bins = np.where(active)[0]
        if len(active_bins) > 0:
            print(f"\n  Radar active: t={bins[active_bins[0]]:.0f}s to {bins[active_bins[-1]+1]:.0f}s")
            print(f"  Radar points/sec (active bins): min={counts[active].min()} max={counts[active].max()} mean={counts[active].mean():.0f}")
        else:
            print(f"\n  Radar: NO active windows (max pts/sec = {counts.max()})")
    
    # Recommend: largest flight window that overlaps with radar
    if len(starts) > 0 and radar:
        best_dur = 0
        best_start = 0
        for si, ei in zip(starts, ends):
            dur = t_rel[ei] - t_rel[si]
            if dur > best_dur:
                best_dur = dur
                best_start = t_rel[si]
        # Add 0.5s margin
        rec_start = max(0, best_start - 0.5)
        rec_dur = min(best_dur + 1.0, t_rel[-1] - rec_start)
        print(f"\n  >>> RECOMMENDED: START_TIME_OFFSET={rec_start:.1f}  DURATION={rec_dur:.1f}")
