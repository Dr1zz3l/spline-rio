"""Intra-frame correlation of radar Doppler residuals + in-band IMU noise.

Variance-components decomposition per frame: within-frame scatter = sigma_i
(independent part), frame-mean scatter = sigma_c^2 + sigma_i^2/N (shared
part). rho = sigma_c^2/(sigma_c^2+sigma_i^2); N_eff = N/(1+(N-1)rho).
Also: gyro/accel residual sigma after 10 Hz low-pass (spline-band noise).
"""
import sys
from pathlib import Path
sys.path.insert(0, '/home/mouse/MyData/radar-iwr6843-driver/analysis/lib')
sys.path.insert(0, '/home/mouse/MyData/radar-iwr6843-driver/analysis')
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
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
IMU_OFF = EXT['imu_mocap_offset_sec']

def robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))

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

    # variance components over frames with >=6 points, tails clipped at 6*MAD
    s_all = robust_sigma(r)
    keep = np.abs(r - np.median(r)) < 6 * s_all
    within, means, ns = [], [], []
    for f in np.unique(fi):
        m = (fi == f) & keep
        n = int(np.sum(m))
        if n >= 6:
            within.append(np.var(r[m], ddof=1)); means.append(np.mean(r[m])); ns.append(n)
    within = np.array(within); means = np.array(means); ns = np.array(ns)
    s_i2 = np.mean(within)
    s_mean2 = np.var(means, ddof=1)
    s_c2 = max(s_mean2 - s_i2 / np.mean(ns), 0.0)
    rho = s_c2 / (s_c2 + s_i2)
    N = np.mean(ns)
    n_eff = N / (1 + (N - 1) * rho)

    # in-band gyro/accel noise (10 Hz low-pass on residual vs agiros ref)
    imu = [m for m in data.imu_data if t0 <= m.timestamp <= t1]
    t_imu = np.array([m.timestamp for m in imu]) + IMU_OFF
    st_t = np.array([s.timestamp for s in states])
    om_i = interp1d(st_t, np.array([s.angular_velocity for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    g_res = np.array([m.angular_velocity for m in imu]) - om_i(t_imu)
    g_res -= np.median(g_res, axis=0)
    fs = 1.0 / np.median(np.diff(t_imu))
    b, a = butter(2, 10.0 / (fs / 2), btype='low')
    g_lp = filtfilt(b, a, g_res, axis=0)
    s_g_raw = robust_sigma(g_res.ravel()); s_g_ib = robust_sigma(g_lp.ravel())

    # implied weights relative to radar (per-frame information counting)
    s_r = np.sqrt(s_i2 + s_c2)          # per-point marginal sigma
    lam_r_naive = 1 / s_r**2            # per point, naive
    lam_r_eff = lam_r_naive * n_eff / N # per point, correlation-corrected
    lam_g_ib = 1 / s_g_ib**2
    print(f"\n=== {bag_key}")
    print(f"  frames used {len(ns)}, mean N {N:.1f}")
    print(f"  sigma_i {np.sqrt(s_i2):.3f}  sigma_c {np.sqrt(s_c2):.3f}  "
          f"rho {rho:.2f}  N_eff {n_eff:.1f} (of {N:.1f})")
    print(f"  gyro sigma raw {s_g_raw:.3f} -> in-band(10Hz) {s_g_ib:.3f} rad/s")
    print(f"  implied lambda_gyro/lambda_radar: naive "
          f"{(s_r/ s_g_raw)**2:.2f} -> corrected "
          f"{lam_g_ib/lam_r_eff:.2f}   (deployed: 4.0)")

# --- accel in-band (appended pass) ---
from radar_velocity_utils import quat_to_rotation_matrix
G = np.array([0.0, 0.0, -9.81])
for bag_key in ['slow_racing_best_velocity', 'fast_racing_best_velocity']:
    data = load_bag_topics(str(Path('/home/mouse/MyData/radar-iwr6843-driver')
                               / BAGS[bag_key]), verbose=False)
    t0 = data.start_time + TIMING[bag_key][0]; t1 = t0 + TIMING[bag_key][1]
    imu = [m for m in data.imu_data if t0 <= m.timestamp <= t1]
    states = data.agiros_state
    t_imu = np.array([m.timestamp for m in imu]) + IMU_OFF
    st_t = np.array([s.timestamp for s in states])
    ac_i = interp1d(st_t, np.array([s.acceleration for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    qu_i = interp1d(st_t, np.array([s.orientation for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    acc = np.array([m.linear_acceleration for m in imu])
    a_w = ac_i(t_imu); q = qu_i(t_imu)
    res = np.empty_like(acc)
    for i in range(len(t_imu)):
        res[i] = acc[i] - quat_to_rotation_matrix(q[i]).T @ (a_w[i] - G)
    res -= np.median(res, axis=0)
    fs = 1.0 / np.median(np.diff(t_imu))
    b, a = butter(2, 10.0 / (fs / 2), btype='low')
    lp = filtfilt(b, a, res, axis=0)
    print(f"{bag_key}: accel sigma raw {robust_sigma(res.ravel()):.3f} -> "
          f"in-band {robust_sigma(lp.ravel()):.3f} m/s^2")
