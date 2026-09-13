#!/usr/bin/env python3
"""Physical accelerometer error model (branch principled-weighting).

Regresses the in-band accelerometer residual (z_a - R_bw(a_w - g) - bias)
onto physical regressors, per bag:

  1. constant bias B (3)
  2. IMU lever arm r off the CoM:  a_imu = a_com + domega x r + omega x (omega x r)
     -> linear in r via  Omega(t) = [domega]_x + [omega]_x [omega]_x   (3)
  3. per-axis scale error:  diag(k) . f_body   (3)
  4. thrust proxy: mean-removed motor-speed sum m(t) along body z (1, if
     motor data present)

Joint linear least squares; reports the residual sigma before/after each
component and the fitted parameters (r in cm -> physically checkable).
If the lever arm + scale explain most of the accel systematic, lambda_a and
the omega_a curve inherit a physical basis (lever-arm error grows ~ omega^2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import quat_to_rotation_matrix

_cfg = load_config()
BAGS = _cfg['bags']['bags']; TIMING = _cfg['bags']['timing']
IMU_OFF = _cfg['extrinsics']['imu_mocap_offset_sec']
G = np.array([0.0, 0.0, -9.81])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def lowpass(x, fs, fc=10.0):
    b, a = butter(2, fc / (fs / 2), btype='low')
    return filtfilt(b, a, x, axis=0)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


for bag_key in ['slow_racing_best_velocity', 'fast_racing_best_velocity']:
    data = load_bag_topics(str(Path(__file__).parent.parent / BAGS[bag_key]),
                           verbose=False)
    t0 = data.start_time + TIMING[bag_key][0]
    t1 = t0 + TIMING[bag_key][1]
    imu = [m for m in data.imu_data if t0 <= m.timestamp <= t1]
    states = data.agiros_state
    st_t = np.array([s.timestamp for s in states])
    t = np.array([m.timestamp for m in imu]) + IMU_OFF
    fs = 1.0 / np.median(np.diff(t))

    acc = np.array([m.linear_acceleration for m in imu])
    gyr = np.array([m.angular_velocity for m in imu])
    ac_i = interp1d(st_t, np.array([s.acceleration for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    qu_i = interp1d(st_t, np.array([s.orientation for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    a_w = ac_i(t); q = qu_i(t)

    # specific-force prediction and raw residual
    f_pred = np.empty_like(acc)
    for i in range(len(t)):
        f_pred[i] = quat_to_rotation_matrix(q[i]).T @ (a_w[i] - G)
    res = acc - f_pred

    # in-band everything (systematics are low-frequency; kills vibration)
    res_lp = lowpass(res, fs)
    gyr_lp = lowpass(gyr, fs)
    f_lp = lowpass(f_pred, fs)
    dgyr = np.gradient(gyr_lp, t, axis=0)

    # motor thrust proxy (mean-removed motor-speed sum), if available
    motors = None
    if states[0].motors is not None:
        mo_i = interp1d(st_t, np.array([s.motors for s in states]), axis=0,
                        kind='linear', bounds_error=False,
                        fill_value='extrapolate')
        m_sum = mo_i(t).sum(axis=1)
        motors = lowpass(m_sum - np.mean(m_sum), fs)

    n = len(t)
    # design matrix per sample: res_i (3) = B + Omega_i r + diag(k) f_i [+ c*m_i e_z]
    n_par = 3 + 3 + 3 + (1 if motors is not None else 0)
    A = np.zeros((3 * n, n_par))
    y = res_lp.reshape(-1)
    for i in range(n):
        Omega = skew(dgyr[i]) + skew(gyr_lp[i]) @ skew(gyr_lp[i])
        A[3*i:3*i+3, 0:3] = np.eye(3)
        A[3*i:3*i+3, 3:6] = Omega
        A[3*i:3*i+3, 6:9] = np.diag(f_lp[i])
        if motors is not None:
            A[3*i+2, 9] = motors[i]

    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = (A @ theta).reshape(-1, 3)

    # incremental attribution: sigma after removing components cumulatively
    def sig_after(cols):
        th = np.zeros_like(theta)
        th[cols] = theta[cols]
        return rms(y - A @ th)

    print(f"\n=== {bag_key}  (n={n}, in-band 10 Hz)")
    print(f"  raw in-band residual rms      : {rms(y):.3f} m/s^2")
    print(f"  after bias                    : {sig_after(range(0,3)):.3f}")
    print(f"  after bias+lever              : {sig_after(range(0,6)):.3f}")
    print(f"  after bias+lever+scale        : {sig_after(range(0,9)):.3f}")
    if motors is not None:
        print(f"  after all (+thrust proxy)     : {sig_after(range(0,n_par)):.3f}")
    print(f"  fitted bias  B [m/s^2]        : {theta[0:3].round(3)}")
    print(f"  fitted lever r [cm]           : {(100*theta[3:6]).round(1)}")
    print(f"  fitted scale k [%]            : {(100*theta[6:9]).round(2)}")
    if motors is not None:
        print(f"  thrust-proxy coeff            : {theta[9]:.2e}")
    # rate dependence of the unexplained part
    resid = (y - A @ theta).reshape(-1, 3)
    om = np.linalg.norm(gyr_lp, axis=1)
    for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 10)]:
        m = (om >= lo) & (om < hi)
        if np.sum(m) > 200:
            print(f"    unexplained rms @ |omega| {lo}-{hi}: "
                  f"{rms(resid[m]):.3f} m/s^2  (n={int(np.sum(m))})")

# ---- drag hypothesis: residual_xy ~ -c_d * v_body_xy (appended pass) ------
print("\n\n===== DRAG MODEL TEST: accel_xy = -c_d * v_body_xy =====")
for bag_key in ['slow_racing_best_velocity', 'fast_racing_best_velocity']:
    data = load_bag_topics(str(Path(__file__).parent.parent / BAGS[bag_key]),
                           verbose=False)
    t0 = data.start_time + TIMING[bag_key][0]
    t1 = t0 + TIMING[bag_key][1]
    imu = [m for m in data.imu_data if t0 <= m.timestamp <= t1]
    states = data.agiros_state
    st_t = np.array([s.timestamp for s in states])
    t = np.array([m.timestamp for m in imu]) + IMU_OFF
    fs = 1.0 / np.median(np.diff(t))
    acc = np.array([m.linear_acceleration for m in imu])
    ac_i = interp1d(st_t, np.array([s.acceleration for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    qu_i = interp1d(st_t, np.array([s.orientation for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    ve_i = interp1d(st_t, np.array([s.velocity for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    a_w = ac_i(t); q = qu_i(t); v_w = ve_i(t)
    f_pred = np.empty_like(acc); v_b = np.empty_like(acc)
    for i in range(len(t)):
        R = quat_to_rotation_matrix(q[i])
        f_pred[i] = R.T @ (a_w[i] - G)
        v_b[i] = R.T @ v_w[i]
    acc_lp = lowpass(acc, fs); f_lp = lowpass(f_pred, fs)
    vb_lp = lowpass(v_b, fs)
    for ax, name in [(0, 'x'), (1, 'y')]:
        # direct measurement model: z_ax ~ -c_d v_ax + b (skip prediction!)
        A = np.stack([vb_lp[:, ax], np.ones(len(t))], axis=1)
        cfit, *_ = np.linalg.lstsq(A, acc_lp[:, ax], rcond=None)
        pred = A @ cfit
        print(f"  {bag_key[:4]} axis {name}: z_a = {cfit[0]:+.3f}*v_b {cfit[1]:+.2f}; "
              f"rms(z_a)={rms(acc_lp[:,ax]):.3f} -> resid {rms(acc_lp[:,ax]-pred):.3f} "
              f"m/s^2 | vs standard-model resid {rms(acc_lp[:,ax]-f_lp[:,ax]):.3f}")
