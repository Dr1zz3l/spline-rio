"""Reference-free IMU noise from the in-bag static segments.

Every headline bag starts with the vehicle sitting on the ground: first a
motors-OFF stretch, then (on some bags) a motors-ON stretch before takeoff.
In both the true body rate is zero and the true specific force is g, so the
whole IMU residual is sensor noise plus vibration -- no MoCap reference error,
no spline representation error.  That is the one place in this dataset where
the IMU can be characterized without a reference.

Why it matters.  `derive_stream_weights.py` (and notebook chapter 5e) derive
the deployed static weights from lambda* = 1/(sigma^2 * f_s * tau) using the
IN-FLIGHT in-band residual against the MoCap reference.  Those per-bag lambda*
spread by 8x for the gyro, which reads like an unexplained instability of a
"derived" constant.  This script shows the spread is not the sensor: the
motors-off floor is bag-independent, and the in-flight residual sits far above
both it and the measured vibration.  What lambda* encodes is therefore the
TOTAL residual budget (sensor + vibration + reference + representation), which
is the right quantity for a factor-graph weight but must be named correctly.

What this CANNOT do: the true rate is zero throughout, so nothing here founds
the rate-conditioned accel gate (`accel_soft_sigma`).  That needs a bench
measurement; see documentation/todo/TODO_MEASUREMENTS.md.

Run from analysis/:  ../.venv/bin/python3 characterize_imu_noise.py
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
BAGS = _cfg['bags']['bags']
TIMING = _cfg['bags']['timing']
EXT = _cfg['extrinsics']
IMU_OFF = EXT['imu_mocap_offset_sec']
G = np.array([0.0, 0.0, -9.81])
BAND_HZ = 10.0          # spline band edge, same convention as 5e / derive_stream_weights

BAG_KEYS = ['slow_racing_best_velocity', 'fast_racing_best_velocity',
            'backflips_best_velocity']
SHORT = {'slow_racing_best_velocity': 'slow racing',
         'fast_racing_best_velocity': 'fast racing',
         'backflips_best_velocity': 'backflips'}


def robust_sigma(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def corr_time_reg(x, dt):
    """Integrated autocorrelation time, summed to the first zero crossing."""
    x = x - x.mean()
    ac = np.correlate(x, x, 'full')[len(x) - 1:]
    ac = ac / ac[0]
    tau = dt
    for k in range(1, min(len(ac), 8000)):
        if ac[k] <= 0:
            break
        tau += 2 * ac[k] * dt
    return tau


def quantization_step(x3):
    """Smallest reported increment, i.e. one LSB of the logged stream."""
    steps = []
    for k in range(3):
        u = np.unique(np.round(x3[:, k], 9))
        du = np.diff(u)
        du = du[du > 1e-12]
        if len(du):
            steps.append(float(np.median(du)))
    return float(np.median(steps)) if steps else np.nan


def budget(res3, fs, band=BAND_HZ):
    """sigma / tau / lambda* for a (N,3) residual, raw and in-band."""
    out = {}
    b, a = butter(2, band / (fs / 2), btype='low')
    for tag, x in [('raw', res3), ('inband', filtfilt(b, a, res3, axis=0))]:
        x = x - np.median(x, axis=0)
        sig = robust_sigma(x.ravel())
        tau = float(np.mean([corr_time_reg(x[:, k], 1 / fs) for k in range(3)]))
        out[tag] = dict(sigma=sig, tau=tau, lam=1.0 / (sig ** 2 * fs * tau))
    return out


def find_ground_windows(mocap, ti, gyro, min_len=0.8):
    """(motors_off, motors_on, t_takeoff) as (lo, hi) seconds from bag start.

    Motors-off: the initial stretch where the gyro is at its quietest and the
    MoCap position does not move.  Motors-on: after that, still before takeoff.
    Both are identified WITHOUT looking at the IMU residual we then measure --
    the gate is the MoCap position and the gyro magnitude, not the noise.
    """
    tm = mocap[:, 0]
    z0 = float(np.median(mocap[tm < 1.0, 3]))
    air = tm[mocap[:, 3] > z0 + 0.15]
    t_takeoff = float(air.min()) if len(air) else float(tm.max())

    gn = np.linalg.norm(gyro, axis=1)
    step = 0.1
    # motors OFF: grow from bag start while the rate stays at hover-floor level
    lo = float(max(ti.min(), 0.15))
    hi = lo
    while hi + step < t_takeoff:
        m = (ti >= lo) & (ti < hi + step)
        if m.sum() > 20 and np.percentile(gn[m], 99) > 0.15:
            break
        hi += step
    off = (lo, hi) if hi - lo >= min_len else None

    # motors ON, still on the ground: after the off window, before takeoff,
    # requiring the MoCap position to stay put
    on = None
    if off is not None:
        lo2 = hi + 0.5           # skip the spin-up transient
        hi2 = t_takeoff - 0.3
        if hi2 - lo2 >= min_len:
            mz = (tm >= lo2) & (tm < hi2)
            if mz.sum() > 5 and np.ptp(mocap[mz, 1:3], axis=0).max() < 0.05:
                on = (lo2, hi2)
    return off, on, t_takeoff


def main():
    repo = Path(__file__).resolve().parent.parent
    rows = {}
    for key in BAG_KEYS:
        data = load_bag_topics(str(repo / BAGS[key]), verbose=False)
        t0 = data.start_time
        ti = np.array([m.timestamp for m in data.imu_data]) - t0
        gyro = np.array([m.angular_velocity for m in data.imu_data])
        acc = np.array([m.linear_acceleration for m in data.imu_data])
        mocap = np.array([[p.timestamp - t0] + list(p.position)
                          for p in data.mocap_pose])
        fs = 1.0 / float(np.median(np.diff(ti)))

        off, on, t_takeoff = find_ground_windows(mocap, ti, gyro)
        r = dict(fs=fs, takeoff=t_takeoff, off=off, on=on,
                 q_gyro=quantization_step(gyro), q_accel=quantization_step(acc))

        # ---- ground windows: true omega = 0, true specific force = g, so the
        #      residual IS the sensor (+ vibration).  No reference enters.
        for tag, win in [('off', off), ('on', on)]:
            if win is None:
                continue
            m = (ti >= win[0]) & (ti < win[1])
            r[f'gyro_{tag}'] = budget(gyro[m], fs)
            r[f'accel_{tag}'] = budget(acc[m], fs)

        # ---- reference's OWN noise in the motors-off window: the flight
        #      controller state should report exactly zero rate and zero
        #      acceleration there, so any spread is reference noise.
        if off is not None and len(data.agiros_state):
            st_t = np.array([s.timestamp for s in data.agiros_state]) - t0
            ms = (st_t >= off[0]) & (st_t < off[1])
            if ms.sum() > 20:
                w_ref = np.array([s.angular_velocity
                                  for s in data.agiros_state])[ms]
                r['ref_gyro_sigma'] = robust_sigma(w_ref.ravel())
                if data.agiros_state[0].acceleration is not None:
                    a_ref = np.array([s.acceleration
                                      for s in data.agiros_state])[ms]
                    r['ref_accel_sigma'] = robust_sigma(
                        (a_ref - np.median(a_ref, axis=0)).ravel())

        # ---- in-flight in-band residual, the 5e protocol, for comparison
        st_t = np.array([s.timestamp for s in data.agiros_state])
        f0 = t0 + TIMING[key][0]
        f1 = f0 + TIMING[key][1]
        mf = (ti + t0 >= f0) & (ti + t0 <= f1)
        t_f = ti[mf] + t0 + IMU_OFF
        om_i = interp1d(st_t, np.array([s.angular_velocity
                                        for s in data.agiros_state]), axis=0,
                        kind='linear', bounds_error=False,
                        fill_value='extrapolate')
        g_res = gyro[mf] - om_i(t_f)
        r['gyro_flight'] = budget(g_res, fs)
        if data.agiros_state[0].acceleration is not None:
            ac_i = interp1d(st_t, np.array([s.acceleration
                                            for s in data.agiros_state]),
                            axis=0, kind='linear', bounds_error=False,
                            fill_value='extrapolate')
            qu_i = interp1d(st_t, np.array([s.orientation
                                            for s in data.agiros_state]),
                            axis=0, kind='linear', bounds_error=False,
                            fill_value='extrapolate')
            a_w = ac_i(t_f)
            qq = qu_i(t_f)
            a_res = np.array([acc[mf][i] - quat_to_rotation_matrix(qq[i]).T
                              @ (a_w[i] - G) for i in range(len(t_f))])
            r['accel_flight'] = budget(a_res, fs)

        rows[key] = r
        del data

    # ------------------------------------------------------------------ report
    print('=' * 78)
    print('WINDOWS (seconds from bag start; all identified from MoCap position '
          'and\ngyro magnitude, never from the residual being measured)')
    print('=' * 78)
    print(f"{'bag':<14} {'motors OFF':>16} {'motors ON (ground)':>20} "
          f"{'takeoff':>9}")
    for key in BAG_KEYS:
        r = rows[key]
        fo = f'[{r["off"][0]:.1f},{r["off"][1]:.1f})' if r['off'] else 'none'
        fn = f'[{r["on"][0]:.1f},{r["on"][1]:.1f})' if r['on'] else 'none'
        print(f'{SHORT[key]:<14} {fo:>16} {fn:>20} {r["takeoff"]:>8.1f}s')

    for stream in ('gyro', 'accel'):
        unit = 'rad/s' if stream == 'gyro' else 'm/s^2'
        print()
        print('=' * 78)
        print(f'{stream.upper()} ({unit}): sensor floor vs vibration vs '
              f'in-flight residual')
        print('=' * 78)
        print(f"{'bag':<14} {'condition':<20} {'sigma raw':>10} "
              f"{'sigma in-band':>14} {'tau_ib':>8} {'lambda*_ib':>11}")
        for key in BAG_KEYS:
            r = rows[key]
            for tag, label in [('off', 'ground, motors OFF'),
                               ('on', 'ground, motors ON'),
                               ('flight', 'IN FLIGHT (vs MoCap)')]:
                d = r.get(f'{stream}_{tag}')
                if d is None:
                    continue
                print(f'{SHORT[key]:<14} {label:<20} {d["raw"]["sigma"]:>10.4f} '
                      f'{d["inband"]["sigma"]:>14.4f} '
                      f'{1e3*d["inband"]["tau"]:>7.1f}ms '
                      f'{d["inband"]["lam"]:>11.4g}')

    print()
    print('=' * 78)
    print('WHAT THE IN-FLIGHT RESIDUAL IS MADE OF')
    print('=' * 78)
    print(f"{'bag':<14} {'stream':<7} {'in-flight/sensor':>17} "
          f"{'in-flight/vibration':>20}   (in-band sigma ratios)")
    for key in BAG_KEYS:
        r = rows[key]
        for stream in ('gyro', 'accel'):
            fl = r.get(f'{stream}_flight')
            of = r.get(f'{stream}_off')
            on = r.get(f'{stream}_on')
            if fl is None or of is None:
                continue
            v_off = fl['inband']['sigma'] / of['inband']['sigma']
            v_on = (fl['inband']['sigma'] / on['inband']['sigma']
                    if on else np.nan)
            print(f'{SHORT[key]:<14} {stream:<7} {v_off:>16.1f}x '
                  f'{v_on:>19.1f}x')

    print()
    print('=' * 78)
    print('QUANTIZATION OF THE LOGGED IMU (one LSB of the recorded stream)')
    print('=' * 78)
    for key in BAG_KEYS:
        r = rows[key]
        sig_off = r['accel_off']['raw']['sigma']
        print(f'  {SHORT[key]:<14} gyro {r["q_gyro"]:.5f} rad/s   '
              f'accel {r["q_accel"]:.5f} m/s^2 (= {r["q_accel"]/9.81*1e3:.2f} mg)'
              f'   at-rest accel MAD/LSB = {sig_off/1.4826/r["q_accel"]:.2f}')
    print('  The at-rest accel MAD is exactly ONE quantization step on every bag')
    print('  and axis, so the accelerometer\'s true noise floor is BELOW 1 mg and')
    print('  cannot be resolved from this log: the accel "sensor floor" above is')
    print('  an UPPER bound, and the in-flight/sensor ratios are LOWER bounds.')
    print('  Quantization itself contributes only q/sqrt(12) = '
          f'{rows[BAG_KEYS[0]]["q_accel"]/12**0.5:.4f} m/s^2, negligible against')
    print('  the 0.45-1.11 m/s^2 in-flight in-band residual that sets the weight.')

    print()
    print('Reference stream\'s OWN noise while the vehicle is provably static')
    print('(agiros_state should report exactly zero there):')
    for key in BAG_KEYS:
        r = rows[key]
        if 'ref_gyro_sigma' in r:
            extra = (f"  accel {r['ref_accel_sigma']:.4f} m/s^2"
                     if 'ref_accel_sigma' in r else '')
            print(f'  {SHORT[key]:<14} gyro {r["ref_gyro_sigma"]:.4f} rad/s'
                  + extra)

    print()
    print('Deployed (solver_cpp.yaml): lambda_gyro = 3.13, '
          'lambda_accel = 0.00392')
    print('Chapter 5e in-flight in-band inputs: gyro 0.024 / 0.051 rad/s '
          '-> lambda* 8.96 / 1.10')
    print('                                     accel 0.447 / 1.106 m/s^2 '
          '-> lambda* 0.00838 / 0.00183')


if __name__ == '__main__':
    main()
