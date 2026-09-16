"""Variance components per omega-bin: is the rate degradation in sigma_c?"""
import sys
from pathlib import Path
sys.path.insert(0, '/home/mouse/MyData/radar-iwr6843-driver/analysis/lib')
sys.path.insert(0, '/home/mouse/MyData/radar-iwr6843-driver/analysis')
import numpy as np
from scipy.interpolate import interp1d
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (compute_doppler_residuals,
                                  rotation_matrix_from_euler, unwrap_doppler)
_cfg = load_config()
BAGS = _cfg['bags']['bags']; TIMING = _cfg['bags']['timing']
RC = _cfg['bags'].get('radar_config', {}); EXT = _cfg['extrinsics']
R_BS = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
T_BS = np.array(EXT['translation_body_m'])
OFF = EXT['imu_mocap_offset_sec'] - EXT['radar_imu_offset_sec']
EDGES = [0, 1, 2, 3, 4, 6, 8, 10, 14]

for bag_key in ['slow_racing_best_velocity', 'fast_racing_best_velocity',
                'backflips_best_velocity']:
    v_max = RC.get('best_velocity' if 'best_velocity' in bag_key else 'default',
                   {}).get('v_max', 4.99)
    data = load_bag_topics(str(Path('/home/mouse/MyData/radar-iwr6843-driver')
                               / BAGS[bag_key]), verbose=False)
    t0 = data.start_time + TIMING[bag_key][0]; t1 = t0 + TIMING[bag_key][1]
    radar = [f for f in data.radar_velocity
             if t0 <= f.timestamp <= t1 and f.positions is not None]
    states = data.agiros_state
    d = compute_doppler_residuals(states, radar, T_BS, R_BS,
                                  time_offset=OFF, min_range=0.2)
    r = unwrap_doppler(d['measurements'], d['predictions'], v_max) - d['predictions']
    fi = np.array(d['frame_indices'])
    st_t = np.array([s.timestamp for s in states])
    om_i = interp1d(st_t, np.array([s.angular_velocity for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    s_all = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r - np.median(r)) < 6 * s_all

    # per-frame stats + frame omega
    frames = {}
    for f in np.unique(fi):
        m = (fi == f) & keep
        if np.sum(m) >= 6:
            om = np.linalg.norm(om_i(radar[int(f)].timestamp + OFF))
            frames[f] = (np.var(r[m], ddof=1), np.mean(r[m]), int(np.sum(m)), om)
    print(f"\n=== {bag_key}")
    print(f"  {'omega bin':>10} {'frames':>7} {'sigma_i':>8} {'sigma_c':>8} {'ratio':>6}")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = [v for v in frames.values() if lo <= v[3] < hi]
        if len(sel) < 8: continue
        wi = np.array([v[0] for v in sel]); mu = np.array([v[1] for v in sel])
        ns = np.array([v[2] for v in sel])
        s_i2 = np.mean(wi)
        s_c2 = max(np.var(mu, ddof=1) - s_i2 / np.mean(ns), 0.0)
        print(f"  {f'{lo}-{hi}':>10} {len(sel):>7} {np.sqrt(s_i2):>8.3f} "
              f"{np.sqrt(s_c2):>8.3f} {np.sqrt(s_c2/s_i2):>6.2f}")
