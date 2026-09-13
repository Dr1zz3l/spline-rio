"""How reference-immune is the within-frame statistic, really?

The story notebook leans on a decomposition r_fj = c_f + e_fj in which the
frame-shared term is a SCALAR, and concludes that subtracting a frame's own mean
makes the per-point width sigma_i immune to reference error.  That is true for a
scalar offset common to every ray.  It is NOT true for the error that actually
matters here.

A reference VELOCITY error dv is a 3-vector and enters return j as

    r_j = -u_j^T dv,

which varies from ray to ray.  Subtracting the frame mean removes only the part
of dv along the frame's mean ray; two of its three degrees of freedom survive
into what we call the reference-immune sigma_i.  This script bounds that: it
removes a full per-frame 3-DoF velocity by least squares (with the proper
degrees-of-freedom correction) and reports how much sigma_i drops.

It also runs a second, more actionable diagnostic.  A CONSTANT radar-mounting
rotation error eps produces

    r_j = -eps . (u_j x v),

which follows the same |v| sin(theta) law as a random bearing error but is
coherent, and therefore removable by calibration rather than by weighting.  If a
coherent component is present, the rms bearing error quoted in chapter 8b is an
upper bound, and the residual roll/yaw of the radar mount -- which the solver
never estimates (`optimize_pitch_only: true`) -- is a free accuracy lever.

Run from analysis/:  ../.venv/bin/python3 characterize_reference_immunity_bound.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.interpolate import interp1d

from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (compute_doppler_residuals,
                                  rotation_matrix_from_euler, unwrap_doppler,
                                  quat_to_rotation_matrix)

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
MIN_N = 8


def robust_sigma(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def frames(key, repo):
    """Per-frame (residuals, body-frame unit rays, body velocity), alias-cut."""
    v_max = RC.get('best_velocity', {}).get('v_max', 3.136)
    data = load_bag_topics(str(repo / BAGS[key]), verbose=False)
    t0 = data.start_time + TIMING[key][0]
    t1 = t0 + TIMING[key][1]
    radar = [f for f in data.radar_velocity
             if t0 <= f.timestamp <= t1 and f.positions is not None]
    states = data.agiros_state
    dres = compute_doppler_residuals(states, radar, T_BS, R_BS,
                                     time_offset=RADAR_OFF, min_range=0.2)
    unw = unwrap_doppler(dres['measurements'], dres['predictions'], v_max)
    r = unw - dres['predictions']
    fi = np.array(dres['frame_indices'])
    pi = np.array(dres['point_indices'])
    keep = (np.abs(r - np.median(r)) < 6 * robust_sigma(r)) \
        & (unw == dres['measurements'])

    st_t = np.array([s.timestamp for s in states])
    v_i = interp1d(st_t, np.array([s.velocity for s in states]), axis=0,
                   kind='linear', bounds_error=False, fill_value='extrapolate')
    q_i = interp1d(st_t, np.array([s.orientation for s in states]), axis=0,
                   kind='linear', bounds_error=False, fill_value='extrapolate')

    out = []
    for f in np.unique(fi):
        m = (fi == f) & keep
        n = int(m.sum())
        if n < MIN_N:
            continue
        fr = radar[int(f)]
        t_f = fr.timestamp + RADAR_OFF
        v_b = quat_to_rotation_matrix(q_i(t_f)).T @ v_i(t_f)
        if np.linalg.norm(v_b) < 0.2:
            continue
        P = np.asarray(fr.positions, float)[pi[m]]
        u_b = (R_BS @ (P / np.maximum(np.linalg.norm(P, axis=1),
                                      1e-6)[:, None]).T).T
        out.append((r[m], u_b, v_b))
    del data
    return out


def main():
    repo = Path(__file__).resolve().parent.parent
    F = {k: frames(k, repo) for k in BAG_KEYS}

    print('=' * 86)
    print('(1) HOW MUCH OF A REFERENCE VELOCITY ERROR SURVIVES THE FRAME-MEAN')
    print('    Removing 1 DoF (the frame mean, what the notebook does) vs the')
    print('    full 3 DoF of a per-frame velocity error.  The gap is the part of')
    print('    a reference velocity error that the "reference-immune" statistic')
    print('    does NOT remove.')
    print('=' * 86)
    print(f"{'bag':<13} {'frames':>7} {'1 DoF removed':>15} {'3 DoF removed':>15} "
          f"{'MAD drop':>10}")
    for key in BAG_KEYS:
        d1, d3 = [], []
        for r, u, v in F[key]:
            n = len(r)
            d1.append((r - r.mean()) / np.sqrt(1 - 1 / n))
            # least-squares removal of a full 3-vector velocity error
            sol, *_ = np.linalg.lstsq(u, r, rcond=None)
            res = r - u @ sol
            if n > 3:
                d3.append(res / np.sqrt(1 - 3 / n))
        d1, d3 = np.concatenate(d1), np.concatenate(d3)
        s1, s3 = robust_sigma(d1), robust_sigma(d3)
        print(f'{SHORT[key]:<13} {len(F[key]):>7} {s1:>15.4f} {s3:>15.4f} '
              f'{100*(1-s3/s1):>9.1f}%')
    print('  Reading: the statistic is immune to a SCALAR offset common to every')
    print('  ray, not to a 3-vector velocity error.  The drop bounds the part of')
    print('  sigma_i that a frame-level velocity error could still be supplying.')

    print()
    print('=' * 86)
    print('(2) IS THERE A COHERENT MOUNTING-ROTATION COMPONENT INSIDE sigma_i?')
    print('    A constant extrinsic rotation error eps gives r_j = -eps.(u_j x v),')
    print('    the SAME |v| sin(theta) law as a random bearing error but coherent,')
    print('    hence removable by calibration rather than by down-weighting.')
    print('=' * 86)
    print(f"{'bag':<13} {'eps roll/pitch/yaw (deg)':>28} {'|eps|':>7} "
          f"{'var share':>10} {'sigma_i before/after':>22}")
    rng = np.random.default_rng(0)
    for key in BAG_KEYS:
        rows, Ms = [], []
        for r, u, v in F[key]:
            n = len(r)
            # within-frame deviations of both the residual and the regressor,
            # so the fit sees only what the reference-immune statistic sees
            X = -np.cross(u, v)                       # (n,3), d r / d eps
            Xc = X - X.mean(axis=0)
            rc = r - r.mean()
            rows.append(rc)
            Ms.append(Xc)
        Rv = np.concatenate(rows)
        M = np.vstack(Ms)
        eps, *_ = np.linalg.lstsq(M, Rv, rcond=None)
        resid = Rv - M @ eps
        s_before, s_after = robust_sigma(Rv), robust_sigma(resid)
        share = 1 - np.var(resid) / np.var(Rv)
        print(f'{SHORT[key]:<13} '
              f'{np.degrees(eps[0]):>+8.2f}/{np.degrees(eps[1]):>+8.2f}/'
              f'{np.degrees(eps[2]):>+8.2f} {np.degrees(np.linalg.norm(eps)):>7.2f} '
              f'{100*share:>9.1f}% {s_before:>10.4f}/{s_after:<11.4f}')
        # bootstrap over frames on the yaw component
        boots = []
        for _ in range(200):
            pick = rng.integers(0, len(Ms), len(Ms))
            Mb = np.vstack([Ms[i] for i in pick])
            rb = np.concatenate([rows[i] for i in pick])
            e, *_ = np.linalg.lstsq(Mb, rb, rcond=None)
            boots.append(np.degrees(e[2]))
        lo, hi = np.percentile(boots, [16, 84])
        print(f'{"":<13} yaw 68% CI [{lo:+.2f}, {hi:+.2f}] deg')

    print()
    print('  If the bags disagree, this is not one clean mounting error -- but a')
    print('  coherent component of the same order as the quoted rms bearing error')
    print('  is not separated from it, which makes chapter 8b\'s dphi an UPPER')
    print('  bound.  Note the solver never estimates radar roll or yaw')
    print('  (`optimize_pitch_only: true`), so any real part of this is an')
    print('  uncalibrated, and therefore free, accuracy lever.')


if __name__ == '__main__':
    main()
