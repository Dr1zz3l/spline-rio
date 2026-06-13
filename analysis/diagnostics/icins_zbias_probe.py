#!/usr/bin/env python3
"""B1 diagnostic: per-frame radar ego-velocity bias (rotated to world via GT attitude),
under plain WLS / Huber / RANSAC front-ends, isolated from the spline solver.

Generalised to run correctly on BOTH the converted ICINS bags (positive control) and our
own bags. The own-bag fixes (vs the first version, which returned a spurious horizontal
bias): apply the radar->GT time offset, use each bag's extrinsics + flip set, GT-aided
Doppler unwrap, and low-pass GT velocity like the eval.

Conventions mirror analysis/validate_live_solver.py:
  radar_total_offset = imu_mocap_offset_sec - radar_imu_offset_sec   (added to radar stamps)
  v_world = R_wb @ (R_bs @ (-v_wls));   R_bs = R_yaw_flip @ R_base  if bag in flipped set

Usage (from analysis/):
  ../.venv/bin/python3 diagnostics/icins_zbias_probe.py icins_flight_1 slow_racing_best_velocity \
      fast_racing_best_velocity circle
"""
import sys
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))
from rosbag_loader import load_bag_topics                       # noqa: E402
from radar_velocity_utils import (solve_ego_velocity_weighted,  # noqa: E402
                                   rotation_matrix_from_euler)

REPO = Path(__file__).resolve().parents[2]
ICINS_EULER = [-178.501, -0.099, 46.997]   # their calib, converted (offset 0)


def ransac_ego_velocity(P, v, intens, min_intensity=5.0, min_range=0.2,
                        thresh=0.15, iters=150, seed=0):
    m = (intens >= min_intensity) & (np.linalg.norm(P, axis=1) >= min_range)
    P, v = P[m], v[m]
    if len(v) < 5:
        return None
    H = P / np.linalg.norm(P, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    best_in, best_v = None, None
    for _ in range(iters):
        idx = rng.choice(len(v), 3, replace=False)
        try:
            vc = np.linalg.solve(H[idx], v[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(H @ vc - v) < thresh
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best_v = inl, vc
    if best_in is None or best_in.sum() < 5:
        return best_v
    Hi, vi = H[best_in], v[best_in]
    return np.linalg.solve(Hi.T @ Hi, Hi.T @ vi)


def gt_aided_unwrap(P, v, R_bs, R_wb, v_world_gt, v_max):
    """Pick the Doppler alias branch nearest the GT-predicted radial velocity
    (mirrors preunwrap_radar_frames with mocap_vel_fn supplied)."""
    u = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    v_sensor_gt = R_bs.T @ R_wb.T @ v_world_gt        # GT body/sensor velocity
    v_pred = -(u @ v_sensor_gt)                       # predicted radial (TI sign)
    period = 2.0 * v_max
    k = np.round((v_pred - v) / period)
    return v + k * period


def bag_config(alias, bags, ext_cfg, flipped):
    """Return (euler, radar_total_offset, flip, v_max_or_None) for a bag."""
    if alias.startswith('icins'):
        return ICINS_EULER, 0.0, False, None
    euler = ext_cfg['rotation_euler_deg']                       # our default [180,25.5,0]
    ov = (bags.get('extrinsics_overrides') or {}).get(alias, {})
    imu_mocap = ov.get('imu_mocap_offset_sec', ext_cfg['imu_mocap_offset_sec'])
    radar_imu = ov.get('radar_imu_offset_sec', ext_cfg['radar_imu_offset_sec'])
    offset = imu_mocap - radar_imu
    vmax = 3.136 if 'best_velocity' in alias else 4.99          # radar_config v_max
    return euler, offset, (alias in flipped), vmax


def run_bag(alias, bags, ext_cfg, flipped):
    euler, offset, flip, vmax = bag_config(alias, bags, ext_cfg, flipped)
    R_base = rotation_matrix_from_euler(*np.radians(euler))
    R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)
    R_bs = (R_yaw_flip @ R_base) if flip else R_base

    bag_rel = bags['bags'][alias]
    bag = (REPO / bag_rel).resolve()
    if not bag.exists():
        bag = (REPO / '..' / bag_rel).resolve()
    start_off, dur = bags['timing'][alias]
    bd = load_bag_topics(str(bag), verbose=False)
    t0 = bd.start_time + start_off
    t1 = t0 + dur - 3.0

    gt = [s for s in bd.agiros_state if t0 <= s.timestamp <= t1]
    gt_t = np.array([s.timestamp for s in gt])
    gt_v = np.array([s.velocity for s in gt])                  # world frame
    gt_R = Rotation.from_quat(np.array([s.orientation for s in gt]))
    # low-pass GT velocity (4th-order Butterworth, 10 Hz) as the eval does
    if len(gt_t) > 27:
        dt = np.median(np.diff(gt_t)); fs = 1.0 / dt; fc = min(10.0, fs * 0.4)
        b, a = butter(4, fc / (fs / 2), btype='low')
        for d in range(3):
            gt_v[:, d] = filtfilt(b, a, gt_v[:, d])
    slerp = Slerp(gt_t, gt_R)
    vfun = [lambda tt, d=d: np.interp(tt, gt_t, gt_v[:, d]) for d in range(3)]

    frames = [f for f in bd.radar_velocity if f.velocities is not None]
    methods = {'plain_WLS': dict(use_huber=False),
               'Huber': dict(use_huber=True, huber_delta=1.0),
               'RANSAC': 'ransac'}
    rows, nfit = {}, 0
    dur_eval = None
    for name, cfg in methods.items():
        errs, ts = [], []
        for f in frames:
            t_shift = f.timestamp + offset                     # -> GT clock
            if not (t0 <= t_shift <= t1):
                continue
            P = np.asarray(f.positions, float)
            v = np.asarray(f.velocities, float)
            I = np.asarray(f.intensities, float)
            tg = min(max(t_shift, gt_t[0]), gt_t[-1])
            R_wb = slerp(tg).as_matrix()
            v_gt = np.array([vfun[d](t_shift) for d in range(3)])
            if vmax is not None:                               # GT-aided unwrap (our bags)
                v = gt_aided_unwrap(P, v, R_bs, R_wb, v_gt, vmax)
            if cfg == 'ransac':
                v_wls = ransac_ego_velocity(P, v, I)
            else:
                v_wls = solve_ego_velocity_weighted(P, v, I, **cfg)
            if v_wls is None:
                continue
            v_world = R_wb @ (R_bs @ (-v_wls))
            errs.append(v_world - v_gt); ts.append(t_shift)
        if len(errs) < 10:
            rows[name] = None; continue
        errs = np.array(errs); ts = np.array(ts)
        bias = errs.mean(axis=0)
        drift = np.array([np.trapz(errs[:, d], ts) for d in range(3)])
        rows[name] = (len(errs), bias, drift)
        nfit = len(errs); dur_eval = ts[-1] - ts[0]
    return rows, nfit, dur_eval, dict(euler=euler, offset=offset, flip=flip, vmax=vmax)


def main():
    aliases = [a for a in sys.argv[1:] if not a.startswith('-')]
    if not aliases:
        aliases = ['icins_flight_1', 'slow_racing_best_velocity',
                   'fast_racing_best_velocity', 'circle']
    bags = yaml.safe_load((REPO / 'analysis/config/bags.yaml').read_text())
    ext_cfg = yaml.safe_load((REPO / 'analysis/config/extrinsics.yaml').read_text())
    flipped = set(bags.get('flipped', []))

    print(f"{'bag':<26}{'method':<10}{'n':>5}  "
          f"{'bias x':>8}{'y':>8}{'z':>8}   {'drift x':>8}{'y':>8}{'z':>8}  win[s]")
    for alias in aliases:
        try:
            rows, nfit, dur_eval, cfg = run_bag(alias, bags, ext_cfg, flipped)
        except Exception as e:
            print(f"{alias:<26} ERROR: {e}"); continue
        tag = f"{alias[:24]}"
        print(f"# {tag}  euler={cfg['euler']} offset={cfg['offset']*1000:+.0f}ms "
              f"flip={cfg['flip']} vmax={cfg['vmax']}")
        for name in ('plain_WLS', 'Huber', 'RANSAC'):
            r = rows.get(name)
            if r is None:
                print(f"{'':<26}{name:<10} (too few)"); continue
            n, bias, drift = r
            print(f"{'':<26}{name:<10}{n:>5}  "
                  f"{bias[0]:>8.3f}{bias[1]:>8.3f}{bias[2]:>8.3f}   "
                  f"{drift[0]:>8.1f}{drift[1]:>8.1f}{drift[2]:>8.1f}  {dur_eval:>5.0f}")


if __name__ == '__main__':
    main()
