"""What can ONE radar frame actually observe?

A single frame of N returns gives N Doppler scalars and N bearings.  Together
they determine the 3-DoF ego-velocity through v_r,j = -u_j^T v, i.e. a linear
system H v = -v_r with H the stacked unit bearings.  How well that system is
conditioned -- and which body axis is worst -- is a property of the SENSOR and
the scene geometry alone.  It is the first thing that has to be answered before
asking whether radar is a reasonable odometry sensor, because it sets the floor
on what any estimator downstream can do.

This is the piece the characterization was missing.  Half of every radar
measurement is the ray u, whose nominal angular resolution on this AOP part is
only 30 deg, and the vertical channel is known to be the weak one (the driver
notes "limited elevation diversity"), but neither had been measured.

Everything here is computed on the DEPLOYED stream: unwrapped Doppler, RANSAC
consensus applied, thin frames bypassed, exactly what the solver consumes.

Run from analysis/:  ../.venv/bin/python3 characterize_frame_observability.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (compute_doppler_residuals,
                                  rotation_matrix_from_euler, unwrap_doppler)

_cfg = load_config()
BAGS = _cfg['bags']['bags']
TIMING = _cfg['bags']['timing']
RC = _cfg['bags'].get('radar_config', {})
EXT = _cfg['extrinsics']
R_BS = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
T_BS = np.array(EXT['translation_body_m'])
RADAR_OFF = EXT['imu_mocap_offset_sec'] - EXT['radar_imu_offset_sec']

BAG_KEYS = ['slow_racing_best_velocity', 'fast_racing_best_velocity',
            'backflips_best_velocity']
SHORT = {'slow_racing_best_velocity': 'slow racing',
         'fast_racing_best_velocity': 'fast racing',
         'backflips_best_velocity': 'backflips'}
AXES = ('x fwd', 'y left', 'z up')


def ransac_mask(P, v, rng, thresh=0.15, iters=150):
    n = len(v)
    if n < 5:
        return np.ones(n, dtype=bool)
    Hn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
    best = None
    for _ in range(iters):
        i3 = rng.choice(n, 3, replace=False)
        try:
            vc = np.linalg.solve(Hn[i3], v[i3])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(Hn @ vc - v) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    return best if best is not None and best.sum() >= 5 else np.ones(n, bool)


def frame_geometry(key, repo):
    """Per-frame bearing geometry of the deployed (consensus-kept) stream."""
    v_max = RC.get('best_velocity', {}).get('v_max', 3.136)
    data = load_bag_topics(str(repo / BAGS[key]), verbose=False)
    t0 = data.start_time + TIMING[key][0]
    t1 = t0 + TIMING[key][1]
    radar = [f for f in data.radar_velocity
             if t0 <= f.timestamp <= t1 and f.positions is not None]
    dres = compute_doppler_residuals(data.agiros_state, radar, T_BS, R_BS,
                                     time_offset=RADAR_OFF, min_range=0.2)
    unw = unwrap_doppler(dres['measurements'], dres['predictions'], v_max)
    fi = np.array(dres['frame_indices'])
    pi = np.array(dres['point_indices'])

    rng = np.random.default_rng(0)
    rows = []
    for i, f in enumerate(radar):
        P = np.asarray(f.positions, float)
        n0 = len(P)
        if n0 < 3:
            continue
        v = np.asarray(f.velocities, float).copy()
        sel = fi == i
        v[pi[sel]] = unw[sel]
        keep = (ransac_mask(P, v, rng) if n0 >= 5
                else np.ones(n0, bool))          # deployed bypass for thin frames
        Pk = P[keep]
        if len(Pk) < 3:
            continue
        # unit rays in the BODY frame, so the axes below are the vehicle's
        u = (R_BS @ (Pk / np.linalg.norm(Pk, axis=1, keepdims=True)).T).T
        M = u.T @ u                                # information, up to 1/sigma^2
        sv = np.linalg.svd(M, compute_uv=False)
        if sv[-1] <= 1e-12:
            rows.append((len(Pk), np.inf, np.inf, np.inf, np.inf,
                         np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        C = np.linalg.inv(M)                       # covariance, up to sigma^2
        amp = np.sqrt(np.diag(C))                  # per-axis error amplification
        # a frame-COMMON Doppler offset c shifts every v_r by c, and the
        # least-squares solution by g*c with g = (H^T H)^-1 H^T 1.  This is the
        # direction the frame-shared error sigma_c corrupts, and unlike the
        # per-point term it does NOT average down with N.
        g = C @ (u.T @ np.ones(len(u)))
        el = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))
        rows.append((len(Pk), sv[0] / sv[-1], amp[0], amp[1], amp[2],
                     float(np.ptp(el)), float(np.std(el)),
                     g[0], g[1], g[2]))
    del data
    return np.array(rows, dtype=float)


def main():
    repo = Path(__file__).resolve().parent.parent
    print('Per-frame ego-velocity observability on the DEPLOYED stream')
    print('(unit bearings in the body frame; amplification = sqrt(diag((H^T H)^-1)),')
    print(' i.e. the factor by which per-point Doppler noise becomes velocity')
    print(' error on that axis, before the IMU or the window contribute anything)')
    print()
    G = {}
    for key in BAG_KEYS:
        G[key] = frame_geometry(key, repo)

    print('=' * 80)
    print('CONDITIONING')
    print('=' * 80)
    print(f"{'bag':<13} {'frames':>7} {'kept pts':>9} "
          f"{'cond(H^T H) med':>16} {'p90':>8} {'>100':>7}")
    for key in BAG_KEYS:
        A = G[key]
        cond = A[:, 1]
        fin = np.isfinite(cond)
        print(f'{SHORT[key]:<13} {len(A):>7} {np.median(A[:, 0]):>9.0f} '
              f'{np.median(cond[fin]):>16.1f} {np.percentile(cond[fin], 90):>8.1f} '
              f'{100*np.mean(cond > 100):>6.0f}%')

    print()
    print('=' * 80)
    print('PER-AXIS NOISE AMPLIFICATION  (median over frames; 1/sqrt(N) would be')
    print('the ideal for N well-spread rays -- larger means that axis is weakly')
    print('constrained by the frame geometry)')
    print('=' * 80)
    print(f"{'bag':<13} " + ' '.join(f'{a:>10}' for a in AXES)
          + f" {'z / horizontal':>16} {'ideal 1/sqrt(N)':>17}")
    for key in BAG_KEYS:
        A = G[key]
        med = [np.median(A[:, 2 + k]) for k in range(3)]
        ideal = 1.0 / np.sqrt(np.median(A[:, 0]))
        print(f'{SHORT[key]:<13} ' + ' '.join(f'{m:>10.2f}' for m in med)
              + f' {med[2]/np.mean(med[:2]):>16.1f}x {ideal:>17.2f}')

    print()
    print('=' * 80)
    print('ELEVATION DIVERSITY  (spread of ray elevations within a frame -- the')
    print('geometric reason the vertical channel is the weak one)')
    print('=' * 80)
    print(f"{'bag':<13} {'elev spread med':>16} {'elev std med':>14} "
          f"{'frames with std < 15 deg':>25}")
    for key in BAG_KEYS:
        A = G[key]
        print(f'{SHORT[key]:<13} {np.nanmedian(A[:, 5]):>15.0f}d '
              f'{np.nanmedian(A[:, 6]):>13.0f}d '
              f'{100*np.nanmean(A[:, 6] < 15):>24.0f}%')

    print()
    print('=' * 80)
    print('WHAT THIS COSTS IN VELOCITY, at the measured per-point Doppler noise')
    print('=' * 80)
    # measured per-point and frame-shared scales on the SAME deployed stream
    # (characterize_reference_immune.py, S3-kept set)
    sigma_pt = {'slow_racing_best_velocity': 0.071,
                'fast_racing_best_velocity': 0.093,
                'backflips_best_velocity': 0.150}
    sigma_c = {'slow_racing_best_velocity': 0.058,
               'fast_racing_best_velocity': 0.047,
               'backflips_best_velocity': 0.146}
    print(f"{'bag':<13} {'contribution':<18} "
          + ' '.join(f'{a:>10}' for a in AXES) + '   (m/s)')
    for key in BAG_KEYS:
        A = G[key]
        s_i, s_c = sigma_pt[key], sigma_c[key]
        per = [s_i * np.nanmedian(A[:, 2 + k]) for k in range(3)]
        sha = [s_c * np.nanmedian(np.abs(A[:, 7 + k])) for k in range(3)]
        tot = [np.hypot(a, b) for a, b in zip(per, sha)]
        print(f'{SHORT[key]:<13} {"per-point sigma_i":<18} '
              + ' '.join(f'{m:>10.3f}' for m in per))
        print(f'{"":<13} {"frame-shared s_c":<18} '
              + ' '.join(f'{m:>10.3f}' for m in sha))
        print(f'{"":<13} {"TOTAL":<18} '
              + ' '.join(f'{m:>10.3f}' for m in tot))
    print()
    print('Three readings, all worth stating:')
    print(' 1. Geometry is NOT the bottleneck on racing.  A single 10 Hz frame')
    print('    already pins the velocity to well under the deployed live-edge')
    print('    velocity RMSE (0.18-0.30 m/s), so the radar carries more velocity')
    print('    information per frame than the fused estimate extracts; what the')
    print('    fusion buys is continuity and the unobservable directions, not')
    print('    raw per-frame velocity accuracy.')
    print(' 2. The two error components load on DIFFERENT axes.  The per-point')
    print('    term is worst on z (limited elevation diversity: the frames that')
    print('    matter span ~25 deg of elevation std, and the worst 10-35% span')
    print('    under 15 deg).  The frame-shared term is worst on x, because every')
    print('    ray shares a forward-looking mean direction, so a Doppler offset')
    print('    common to the frame reads as forward motion.')
    print(' 3. Only the per-point part averages down with more returns.  The')
    print('    frame-shared part does not, at any N -- which is precisely the')
    print('    structure the sigma_c whitening exists to model, and it is the')
    print('    dominant single-frame error on the forward axis in flips.')


if __name__ == '__main__':
    main()
