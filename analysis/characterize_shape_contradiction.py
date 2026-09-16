"""Resolve the frame-level vs per-point shape contradiction in the radar noise law.

Two reference-immune statistics disagree about the shape of the per-return
Doppler noise:

  frame level  (per-frame within-frame variance s_f^2, characterize_speed_law.py
               + notebook cell 51):  sigma ~ v^2   beats  sigma ~ a v rms_f(sin)
               by dBIC ~ 9;
  per point    (within-frame deviations, characterize_pointwise_noise.py):
               sigma ~ v sin(theta)  (a BEARING error) wins.

A v-linear law cannot average into a v-quadratic one, so as stated the two
results are inconsistent.  This script decides whether the inconsistency is in
the data or in the two tests, WITHOUT touching either published script: it
rebuilds one per-return table under the identical protocol, reproduces both
published ladders off it (gate 0), and then re-runs them under corrected
statistics and extra rungs.

RESOLVED (2026-08-06, see worklog/reports/MECHANISM_VERDICT_2026-08-06.md): in the two tests.
Neither ladder offered a rung with an off-boresight geometry factor, and that
is the direction the whole discrimination lives in -- both published laws leave
a monotone factor-2 trend in phi that a boresight-aware law removes (block 1g).
The noise is a bearing error, LINEAR in speed.  What is NOT resolved is which
member of the boresight family it is; three of them score within ~600 of each
other against a 4000-7000 margin over anything without one.

Blocks
  0  unified per-return table; reproduce both published ladders  (gate)
  1  are the two selection statistics valid?
       1a pairwise composite likelihood on a COMMON pair set / COMMON trim
       1b exact chi^2 likelihood for the per-frame variance (dof-weighted ML)
       1c block bootstrap -> effective sample size behind dBIC
       1d matched-estimator 2x2 (robust vs second moment) + tail share vs speed
       1e parametric-bootstrap POWER CALIBRATION of both ladders
  2  the missing rung: v^2 sin(theta), and free exponents sigma ~ v^p sin^q
  3  confounders: coherent extrinsic, 3-DoF reference, acceleration, selection
  4  transfer (backflips / fast2 held-out / ICINS 1-4) and the verdict inputs

Nothing here changes a deployed default.  All residuals are formed against
MoCap through the exact forward model; the solver's output is never used.

Run from analysis/:
  ../.venv/bin/python3 characterize_shape_contradiction.py --block 0
  ../.venv/bin/python3 characterize_shape_contradiction.py --block all
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, minimize
from scipy.special import gammaln

from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (compute_doppler_residuals,
                                  rotation_matrix_from_euler, unwrap_doppler,
                                  quat_to_rotation_matrix)

REPO = Path(__file__).resolve().parent.parent

CFG = load_config()
BAGS = CFG['bags']['bags']
TIMING = CFG['bags']['timing']
FLIPPED = set(CFG['bags']['flipped'])
RC = CFG['bags'].get('radar_config', {})
EXT_OVER = CFG['bags'].get('extrinsics_overrides', {})
EXT = CFG['extrinsics']

# Deployed own-platform extrinsics: pitch locked at the MEASURED 27.5 deg, not
# the stale 25.5 self-cal seed in extrinsics.yaml.  Same convention as every
# other characterize_*.py.
R_DEPLOYED = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
T_DEPLOYED = np.array(EXT['translation_body_m'])
R_ICINS = rotation_matrix_from_euler(*np.radians([-178.501, -0.099, 46.997]))
T_ICINS = np.array([0.01, 0.1, 0.06])
R_YAWFLIP = rotation_matrix_from_euler(0.0, 0.0, np.pi)

T_CPI = 128 * 3 * 130e-6            # 49.92 ms, from 6843AOP_best_velocity.cfg
MIN_RANGE_CLAMP = 0.3               # m, matches characterize_pointwise_noise

RACING = ['slow_racing_best_velocity', 'fast_racing_best_velocity']
BACKFLIPS = 'backflips_best_velocity'
ALL_BAGS = RACING + [BACKFLIPS]
TRANSFER = ['fast_racing_best_velocity_no_clustering',
            'icins_flight_1', 'icins_flight_2', 'icins_flight_3',
            'icins_flight_4']
SHORT = {'slow_racing_best_velocity': 'slow racing',
         'fast_racing_best_velocity': 'fast racing',
         'backflips_best_velocity': 'backflips',
         'fast_racing_best_velocity_no_clustering': 'fast2 (held-out)',
         'icins_flight_1': 'icins 1', 'icins_flight_2': 'icins 2',
         'icins_flight_3': 'icins 3', 'icins_flight_4': 'icins 4'}

# Default is repo-local and gitignored.  It used to be a hardcoded session
# scratchpad under /tmp, which meant every fresh session silently paid a ~6 min
# rebuild (or, worse, found a stale copy).  Override with $SHAPE_CACHE.
CACHE = Path(os.environ.get('SHAPE_CACHE', REPO / '.cache' / 'shape'))


def robust_sigma(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


# ===========================================================================
# BLOCK 0 -- one per-return table, built under the published protocol
# ===========================================================================
def bag_offsets(key):
    ov = EXT_OVER.get(key, {})
    return (ov.get('imu_mocap_offset_sec', EXT['imu_mocap_offset_sec'])
            - ov.get('radar_imu_offset_sec', EXT['radar_imu_offset_sec']))


def table_inputs(key):
    """Everything build_table's result depends on besides the bag's contents.

    Returned as one dict so that `table()` can hash exactly what the table was
    built from.  Audit round 3 (2026-08-10, finding 1.2) found the cache keyed
    on bag name + TABLE_VERSION alone, so the 2026-08-09 fast2 timing fix left
    every warm cache silently serving the pre-fix table.  Anything added here
    that build_table reads must be read THROUGH here, or the hash lies.
    """
    is_icins = key.startswith('icins')
    R_BS = R_ICINS if is_icins else R_DEPLOYED
    T_BS = T_ICINS if is_icins else T_DEPLOYED
    if key in FLIPPED:
        R_BS = R_YAWFLIP @ R_BS
        T_BS = R_YAWFLIP @ T_BS
    return dict(
        bag=str(BAGS[key]),
        timing=[float(x) for x in TIMING[key]],
        radar_off=float(bag_offsets(key)),
        v_max=float(RC.get('best_velocity' if 'best_velocity' in key
                           else 'default', {}).get('v_max', 4.99)),
        R_BS=np.asarray(R_BS, float), T_BS=np.asarray(T_BS, float))


def build_table(key):
    """Per-RETURN table under the exact protocol of the published scripts.

    Nothing is cut here except compute_doppler_residuals' min_range=0.2: the
    6-sigma clip flag, the alias flag and the frame speed are all carried so
    that both published populations can be reproduced downstream.

    per-return arrays
      r        residual  unw - pred   (MoCap forward model, GT-resolved unwrap)
      fid      frame slot (index into the per-frame arrays)
      pid      point index into the frame's raw positions array
      keep6    bag-level 6-sigma MAD clip on r (computed over ALL returns)
      alias    the unwrap moved this measurement
      inten    reported intensity (on-chip SNR in dB)
      rho      per-point range |P| (m), UNCLAMPED
      sin2     sin^2(theta) between the body-frame ray and the body velocity
      sin2a    the same against the ANTENNA velocity v + omega x t_bs
      ux,uy,uz body-frame unit ray
    per-frame arrays (indexed by fid)
      f_t      MoCap-clock frame time
      f_v      |v_b|            f_vs  |v_ant|
      f_om     |omega|          f_acc |a| (MoCap accel; nan if unavailable)
      f_rho    median range over ALL points of the frame (frame_table col 3)
      f_vb     (F,3) body-frame velocity      f_ab (F,3) body-frame accel
      f_vs3    (F,3) SENSOR-frame velocity    f_va3 (F,3) sensor-frame v_ant
               (needed by the direction-cosine rung, which is written in the
               array's own coordinates rather than in angles)
    """
    cfg = table_inputs(key)
    R_BS, T_BS = cfg['R_BS'], cfg['T_BS']
    v_max = cfg['v_max']
    radar_off = cfg['radar_off']

    data = load_bag_topics(str(REPO / cfg['bag']), verbose=False)
    t0 = data.start_time + cfg['timing'][0]
    t1 = t0 + cfg['timing'][1]
    radar = [f for f in data.radar_velocity
             if t0 <= f.timestamp <= t1 and f.positions is not None]
    states = data.agiros_state
    dres = compute_doppler_residuals(states, radar, T_BS, R_BS,
                                     time_offset=radar_off, min_range=0.2)
    meas, pred = dres['measurements'], dres['predictions']
    unw = unwrap_doppler(meas, pred, v_max)
    r = unw - pred
    fi = np.asarray(dres['frame_indices'])
    pi = np.asarray(dres['point_indices'])

    st_t = np.array([s.timestamp for s in states])
    v_i = interp1d(st_t, np.array([s.velocity for s in states]), axis=0,
                   kind='linear', bounds_error=False, fill_value='extrapolate')
    w_i = interp1d(st_t, np.array([s.angular_velocity for s in states]), axis=0,
                   kind='linear', bounds_error=False, fill_value='extrapolate')
    q_i = interp1d(st_t, np.array([s.orientation for s in states]), axis=0,
                   kind='linear', bounds_error=False, fill_value='extrapolate')
    has_acc = states[0].acceleration is not None
    a_i = (interp1d(st_t, np.array([s.acceleration for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
           if has_acc else None)

    F = len(radar)
    f_t = np.full(F, np.nan)
    f_v = np.full(F, np.nan)
    f_vs = np.full(F, np.nan)
    f_om = np.full(F, np.nan)
    f_acc = np.full(F, np.nan)
    f_rho = np.full(F, np.nan)
    f_vb = np.full((F, 3), np.nan)
    f_ab = np.full((F, 3), np.nan)
    f_vs3 = np.full((F, 3), np.nan)
    f_va3 = np.full((F, 3), np.nan)
    f_n = np.zeros(F, int)

    sin2 = np.zeros(len(r))
    sin2a = np.zeros(len(r))
    rho = np.zeros(len(r))
    U = np.zeros((len(r), 3))
    US = np.zeros((len(r), 3))          # SENSOR-frame unit ray (boresight = +x)

    for f in range(F):
        fr = radar[f]
        t_f = fr.timestamp + radar_off
        R_wb = quat_to_rotation_matrix(q_i(t_f))
        v_b = R_wb.T @ v_i(t_f)
        w_b = np.asarray(w_i(t_f), float)
        v_ant = v_b + np.cross(w_b, T_BS)
        P_all = np.asarray(fr.positions, float)
        f_t[f] = t_f
        f_v[f] = np.linalg.norm(v_b)
        f_vs[f] = np.linalg.norm(v_ant)
        f_om[f] = np.linalg.norm(w_b)
        f_rho[f] = np.median(np.linalg.norm(P_all, axis=1))
        f_vb[f] = v_b
        f_vs3[f] = R_BS.T @ v_b            # sensor frame: R_BS maps sensor->body
        f_va3[f] = R_BS.T @ v_ant
        f_n[f] = len(P_all)
        if a_i is not None:
            a_w = np.asarray(a_i(t_f), float)
            f_acc[f] = np.linalg.norm(a_w)
            f_ab[f] = R_wb.T @ a_w

        m = fi == f
        if not np.any(m):
            continue
        P = P_all[pi[m]]
        rr = np.linalg.norm(P, axis=1)
        u_b = (R_BS @ (P / np.maximum(rr, 1e-6)[:, None]).T).T
        rho[m] = rr
        U[m] = u_b
        US[m] = P / np.maximum(rr, 1e-6)[:, None]
        nv = max(np.linalg.norm(v_b), 1e-9)
        nva = max(np.linalg.norm(v_ant), 1e-9)
        sin2[m] = np.clip(1.0 - (u_b @ v_b / nv) ** 2, 0.0, 1.0)
        sin2a[m] = np.clip(1.0 - (u_b @ v_ant / nva) ** 2, 0.0, 1.0)

    del data
    return dict(
        r=r, unw=unw, fid=fi.astype(np.int64), pid=pi.astype(np.int64),
        keep6=(np.abs(r - np.median(r)) < 6 * robust_sigma(r)),
        alias=(unw != meas),
        inten=np.asarray(dres['intensities'], float),
        rho=rho, sin2=sin2, sin2a=sin2a,
        ux=U[:, 0], uy=U[:, 1], uz=U[:, 2],
        sx=US[:, 0], sy=US[:, 1], sz=US[:, 2],
        f_t=f_t, f_v=f_v, f_vs=f_vs, f_om=f_om, f_acc=f_acc, f_rho=f_rho,
        f_vb=f_vb, f_ab=f_ab, f_vs3=f_vs3, f_va3=f_va3, f_n=f_n)


# bump whenever build_table gains a column, so a stale cache cannot silently
# feed a KeyError (or worse, a missing predictor) into a later block
TABLE_VERSION = 2


def cache_digest(key):
    """Short hash of table_inputs(key): the cache filename's config half.

    A config change (timing window, extrinsics, offsets, bag path) changes the
    digest, so the old table becomes unreachable instead of being served.  It
    does NOT expire when build_table's *code* changes -- that is what
    TABLE_VERSION is for, and it still has to be bumped by hand.
    """
    cfg = table_inputs(key)
    payload = json.dumps(
        {k: (np.asarray(v).round(12).tolist() if isinstance(v, np.ndarray)
             else v) for k, v in sorted(cfg.items())}, sort_keys=True)
    return hashlib.blake2s(payload.encode(), digest_size=4).hexdigest()


def table(key, refresh=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f'{key}_v{TABLE_VERSION}_{cache_digest(key)}.npz'
    if p.exists() and not refresh:
        z = np.load(p)
        return {k: z[k] for k in z.files}
    T = build_table(key)
    np.savez_compressed(p, **T)
    return T


# ---------------------------------------------------------------- views ----
def point_view(T, alias_cut=True, speed_cut=True, min_n=6, clamp_rho=True,
               extra_keep=None):
    """Per-point view == characterize_pointwise_noise.build_point_table.

    Returns per-KEPT-return arrays with the within-frame deviation d, plus the
    covariates.  alias_cut / speed_cut / min_n reproduce that script's hygiene:
    6-sigma clip AND alias cut, N_f >= 6, frames with |v_b| < 0.2 dropped.
    """
    keep0 = T['keep6'].copy()
    if alias_cut:
        keep0 &= ~T['alias']
    if extra_keep is not None:
        keep0 &= extra_keep
    fid = T['fid']
    out = {k: [] for k in ('d', 'r', 'unw', 'fid', 'nf', 'v', 'vs', 'om', 'acc',
                           'sin2', 'sin2a', 'rho', 'inten', 't',
                           'ux', 'uy', 'uz', 'sx', 'sy', 'sz',
                           'vbx', 'vby', 'vbz', 'vsx', 'vsy', 'vsz')}
    for f in np.unique(fid):
        m = (fid == f) & keep0
        n = int(m.sum())
        if n < min_n:
            continue
        if speed_cut and T['f_v'][f] < 0.2:
            continue
        rf = T['r'][m]
        out['d'].append((rf - rf.mean()) / np.sqrt(1.0 - 1.0 / n))
        out['r'].append(rf)
        out['unw'].append(T['unw'][m])
        for c, nm in zip(range(3), ('vbx', 'vby', 'vbz')):
            out[nm].append(np.full(n, T['f_vb'][f, c]))
        for c, nm in zip(range(3), ('vsx', 'vsy', 'vsz')):
            out[nm].append(np.full(n, T['f_vs3'][f, c]))
        out['fid'].append(np.full(n, int(f)))
        out['nf'].append(np.full(n, n))
        for nm, src in (('v', 'f_v'), ('vs', 'f_vs'), ('om', 'f_om'),
                        ('acc', 'f_acc'), ('t', 'f_t')):
            out[nm].append(np.full(n, T[src][f]))
        out['sin2'].append(T['sin2'][m])
        out['sin2a'].append(T['sin2a'][m])
        out['rho'].append(np.maximum(T['rho'][m], MIN_RANGE_CLAMP) if clamp_rho
                          else T['rho'][m])
        out['inten'].append(T['inten'][m])
        for nm in ('ux', 'uy', 'uz', 'sx', 'sy', 'sz'):
            out[nm].append(T[nm][m])
    if not out['d']:
        return {k: np.array([]) for k in out}
    return {k: np.concatenate(v) for k, v in out.items()}


def frame_view(T, alias_cut=True, speed_cut=False, min_n=6, extra_keep=None):
    """Per-frame view == characterize_speed_law.frame_table (+ extra columns).

    s2   plain sample variance of the kept residuals (ddof=1) -- the published
         statistic; srob is the robust counterpart added here.
    """
    keep0 = T['keep6'].copy()
    if alias_cut:
        keep0 &= ~T['alias']
    if extra_keep is not None:
        keep0 &= extra_keep
    fid = T['fid']
    cols = {k: [] for k in ('s2', 'srob', 'mean', 'n', 'v', 'vs', 'om', 'acc',
                            'rho', 'rms_sin', 'rms_sin2', 'rms_sin_a', 't',
                            'fid', 'rms_arr')}
    for f in np.unique(fid):
        m = (fid == f) & keep0
        n = int(m.sum())
        if n < min_n:
            continue
        if speed_cut and T['f_v'][f] < 0.2:
            continue
        rf = T['r'][m]
        s2 = float(np.var(rf, ddof=1))
        cols['s2'].append(s2)
        cols['srob'].append(robust_sigma(rf) / np.sqrt(1.0 - 1.0 / n))
        cols['mean'].append(float(rf.mean()))
        cols['n'].append(n)
        cols['v'].append(T['f_v'][f])
        cols['vs'].append(T['f_vs'][f])
        cols['om'].append(T['f_om'][f])
        cols['acc'].append(T['f_acc'][f])
        cols['rho'].append(T['f_rho'][f])
        cols['t'].append(T['f_t'][f])
        cols['fid'].append(int(f))
        s2j = T['sin2'][m]
        cols['rms_sin'].append(float(np.sqrt(np.mean(s2j))))
        cols['rms_sin2'].append(float(np.sqrt(np.mean(s2j ** 2))))
        cols['rms_sin_a'].append(float(np.sqrt(np.mean(T['sin2a'][m]))))
        # geometry factor of the ARRAY law projected onto the frame statistic:
        # E[s_f^2] = s0^2 + a^2 v^2 mean_j(sin_j^2 / cos^2 phi_j)
        cols['rms_arr'].append(float(np.sqrt(np.mean(
            s2j / np.maximum(T['sx'][m], 0.15) ** 2))))
    return {k: np.asarray(v, float) for k, v in cols.items()}


# ===================================================== published ladders ====
def _phi(P):
    """Angle of the ray off the SENSOR boresight (+x), degrees.  A sensor-fixed
    quantity: it knows nothing about where the platform is going.

    +x is the boresight by DRIVER convention, not by how we mounted the module:
    DataHandlerClass.cpp:633-635 publishes ROS x = mmWave sensor Y = forward.
    So this predictor is well defined for any mounting of this driver; for the
    ICINS bags it rests on their converter following the same convention, which
    was not independently verified."""
    return np.degrees(np.arccos(np.clip(P['sx'], -1.0, 1.0)))


def _dubar(P):
    """|u_j - mean_f(u)| : the geometric signature of a frame-shared velocity
    error surviving the frame-mean subtraction."""
    u = np.column_stack([P['ux'], P['uy'], P['uz']])
    out = np.zeros(len(u))
    for f in np.unique(P['fid']):
        m = P['fid'] == f
        out[m] = np.linalg.norm(u[m] - u[m].mean(axis=0), axis=1)
    return out


def _az(P):
    """|azimuth| off boresight in the sensor frame, degrees."""
    return np.abs(np.degrees(np.arctan2(P['sy'], P['sx'])))


def _el(P):
    """|elevation| off boresight in the sensor frame, degrees.

    Kept separate from azimuth as a MEASUREMENT, not as a prediction: the
    IWR6843AOP's virtual-array layout has not been checked against its
    datasheet here, and the fitted anisotropy comes out the opposite way round
    from an aperture-limited expectation (see block 2b), so nothing in this
    script should be read as 'azimuth dominates, as expected'."""
    return np.abs(np.degrees(np.arcsin(np.clip(P['sz'], -1.0, 1.0))))


def _dircos(P, clamp=0.15):
    """The SAME physics as the ARRAY rung, written in the array's own
    coordinates instead of in angles -- and with nothing left to choose.

    An FFT beamformer resolves direction COSINES, not angles: its resolution is
    constant in (u, w) = (s_y, s_z) and it is the mapping back to angle that
    broadens off boresight.  So the primitive error is a constant sigma_u on
    (s_y, s_z), with the longitudinal component fixed by |s| = 1:
        ds_x = -(s_y ds_y + s_z ds_z) / s_x
    and the Doppler residual r = -ds^T v_s becomes, exactly,
        r = -ds_y (v_y - s_y v_x/s_x) - ds_z (v_z - s_z v_x/s_x).
    One parameter, no clamp needed inside the FOV, no chosen power.  Returns
    (total, az channel, el channel); the total is
    sqrt(|v|^2 sin^2(theta) - ((s x v)_x)^2) / s_x, i.e. the ARRAY rung minus
    the ray-parallel component that carries no information.
    """
    sxc = np.maximum(P['sx'], clamp)
    ay = np.abs(P['vsy'] - P['sy'] * P['vsx'] / sxc)
    az = np.abs(P['vsz'] - P['sz'] * P['vsx'] / sxc)
    return np.sqrt(ay ** 2 + az ** 2), ay, az


def pt_predictors(P):
    """The published per-point ladder (characterize_pointwise_noise.predictors)
    plus the rungs it is missing.  name -> (list of X_j, list of kinds)."""
    sin1 = np.sqrt(P['sin2'])
    cos1 = np.sqrt(np.clip(1.0 - P['sin2'], 0.0, 1.0))
    sin1a = np.sqrt(P['sin2a'])
    v, vs, rho = P['v'], P['vs'], P['rho']
    return {
        'floors only               ': ([np.zeros(len(v))], [None]),
        'sin only     (geometry)   ': ([sin1], [None]),
        'v            (speed only) ': ([v], [None]),
        'v cos = |v_r| (SCALE err) ': ([v * cos1], [None]),
        'v sin        (BEARING)    ': ([v * sin1], ['bearing']),
        'vs sin       (BEARING,ant)': ([vs * sin1a], ['bearing']),
        'v^2          (5b deployed)': ([v ** 2], [None]),
        'v^2 sin^2                 ': ([v ** 2 * P['sin2']], [None]),
        'v^2 sin^2/rho  (SMEAR)    ': ([v ** 2 * P['sin2'] / rho], ['smear']),
        'vs^2 sin^2/rho (SMEAR,ant)': ([vs ** 2 * P['sin2a'] / rho], ['smear']),
        'v sin + v^2   (joint)     ': ([v * sin1, v ** 2], ['bearing', None]),
        'v sin + smear (joint)     ': ([v * sin1, v ** 2 * P['sin2'] / rho],
                                       ['bearing', 'smear']),
        # ---- rungs the published ladder does not offer (block 2) ----------
        'v^2 sin      (NEW)        ': ([v ** 2 * sin1], [None]),
        'vs^2 sin     (NEW,ant)    ': ([vs ** 2 * sin1a], [None]),
        'v^1.5 sin    (NEW)        ': ([v ** 1.5 * sin1], [None]),
        'v sin + v^2 sin (NEW joint)': ([v * sin1, v ** 2 * sin1],
                                        ['bearing', None]),
        # ---- does the ray error grow with the SENSOR-FIXED angle instead? ---
        'v sin^1.5    (NEW q=1.5) ': ([v * sin1 ** 1.5], [None]),
        'v sin^2      (NEW q=2)   ': ([v * P['sin2']], [None]),
        'v sin * phi  (antenna)   ': ([v * sin1 * _phi(P) / 50.0], [None]),
        'v sin * phi^2(antenna)   ': ([v * sin1 * (_phi(P) / 50.0) ** 2], [None]),
        'v sin^2 + v sin*phi      ': ([v * P['sin2'], v * sin1 * _phi(P) / 50.0],
                                      [None, None]),
        # ---- is the boresight angle causal, or is SNR the mediator? --------
        'v sin * SNR^-1/2 (CRLB)  ': ([v * sin1 * 10 ** (-P['inten'] / 20.0)
                                       * 10 ** (18.0 / 20.0)], [None]),
        'v sin * SNR^-1/4         ': ([v * sin1 * 10 ** (-P['inten'] / 40.0)
                                       * 10 ** (18.0 / 40.0)], [None]),
        'v sin * az               ': ([v * sin1 * _az(P) / 30.0], [None]),
        'v sin * el               ': ([v * sin1 * _el(P) / 20.0], [None]),
        'v sin * (az^2+el^2)      ': ([v * sin1 * ((_az(P) / 30.0) ** 2
                                                   + (_el(P) / 20.0) ** 2)],
                                      [None]),
        'v sin*phi^2 + SNR^-1/2   ': ([v * sin1 * (_phi(P) / 50.0) ** 2,
                                       v * sin1 * 10 ** (-P['inten'] / 20.0)
                                       * 10 ** (18.0 / 20.0)], [None, None]),
        'v sin*az^2 + v sin*el^2  ': ([v * sin1 * (_az(P) / 30.0) ** 2,
                                       v * sin1 * (_el(P) / 20.0) ** 2],
                                      [None, None]),
        'v sin * rho              ': ([v * sin1 * P['rho'] / 4.0], [None]),
        # ---- DERIVED: a uniform array's beamwidth broadens as 1/cos(phi)
        # (projected aperture shrinks by cos phi), so the DOA error should too.
        # Shape fixed before fitting; the single coefficient is the BORESIGHT
        # bearing error dphi_0 / sqrt2.
        'v sin / cos(phi) (ARRAY) ': ([v * sin1 / np.maximum(P['sx'], 0.15)],
                                      ['bearing']),
        # ---- the reference confound: a frame-SHARED velocity error dv gives
        # d_j = -(u_j - ubar)^T dv, whose scale grows with the ray's distance
        # from the frame's own mean ray -- which correlates with phi.  If these
        # beat the array law, the effect is reference error, not the sensor.
        'ref |u-ubar|             ': ([_dubar(P)], [None]),
        'ref v|u-ubar|            ': ([v * _dubar(P)], [None]),
        'ref v|u-ubar| + ARRAY    ': ([v * _dubar(P),
                                       v * sin1 / np.maximum(P['sx'], 0.15)],
                                      [None, 'bearing']),
        'v sin / cos(az)  (ARRAY) ': ([v * sin1 / np.maximum(
            np.cos(np.radians(_az(P))), 0.15)], ['bearing']),
        'v sin /(cos az cos el)   ': ([v * sin1 / np.maximum(
            np.cos(np.radians(_az(P))) * np.cos(np.radians(_el(P))), 0.15)],
            ['bearing']),
        # ---- the same physics with NOTHING left to choose (see _dircos) -----
        'DIRCOS exact (derived)   ': ([_dircos(P)[0]], ['bearing']),
        'DIRCOS aniso (az,el)     ': (list(_dircos(P)[1:]), ['bearing',
                                                             'bearing']),
    }


def binned_fit(views, Xs, n_bins=10, min_n=120, robust=True):
    """(A) binned-scale fit -- identical to characterize_pointwise_noise.binned_fit
    when robust=True.  robust=False swaps the bin statistic to the plain sd,
    which is the estimator the FRAME-level ladder uses."""
    nx = len(next(iter(Xs.values())))
    cen, sig, bagidx = [], [], []
    for bi, key in enumerate(views):
        P, XL = views[key], Xs[key]
        X0 = XL[0]
        edges = np.unique(np.percentile(X0, np.linspace(0, 100, n_bins + 1)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (X0 >= lo) & (X0 < hi)
            if m.sum() >= min_n:
                cen.append([X[m].mean() for X in XL])
                sig.append(robust_sigma(P['d'][m]) if robust
                           else float(np.std(P['d'][m], ddof=1)))
                bagidx.append(bi)
    c, s = np.array(cen), np.array(sig)
    b = np.array(bagidx, dtype=int)
    nb = len(views)

    def model(th):
        acc = np.abs(th[b]) ** 2
        for m in range(nx):
            acc = acc + (np.abs(th[nb + m]) * c[:, m]) ** 2
        return np.sqrt(acc)

    th0 = np.r_[[np.median(s)] * nb, [1e-3] * nx]
    fsc = max(1.4826 * np.median(np.abs(s - np.median(s))), 1e-4)
    sol = least_squares(lambda th: model(th) - s, th0, loss='soft_l1',
                        f_scale=fsc, max_nfev=40000)
    res = model(sol.x) - s
    rss = float(res @ res)
    return dict(theta=np.abs(sol.x), n_bins=len(s),
                r2=1 - rss / float(((s - s.mean()) ** 2).sum()))


def fr_specs():
    """The frame-level ladder: characterize_speed_law.SPECS plus the two rungs
    that live only in notebook cell 51, plus the ones neither offers.

    Signature: fn(p, om, v, rho, rs, rs2, acc) -> extra variance.
    """
    return [
        ('floors only             ', lambda p, om, v, rho, rs, rs2, ac: 0.0, []),
        ('rate-lin  (c om)^2      ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * om) ** 2, [0.03]),
        ('speed-lin (a v)^2       ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v) ** 2, [0.05]),
        ('speed-quad (q v^2)^2    ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v ** 2) ** 2, [0.02]),
        ('trans (b v^2/rho)^2     ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v ** 2 / rho) ** 2, [0.05]),
        ('speed-lin+quad          ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v) ** 2 + (p[1] * v ** 2) ** 2, [0.05, 0.01]),
        ('speed+rate              ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v) ** 2 + (p[1] * om) ** 2, [0.05, 0.02]),
        ('speed-quad+rate         ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v ** 2) ** 2 + (p[1] * om) ** 2, [0.02, 0.02]),
        ('BEARING (a v rms_sin)^2 ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v * rs) ** 2, [0.05]),
        ('bearing+speed-quad      ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v * rs) ** 2 + (p[1] * v ** 2) ** 2, [0.05, 0.01]),
        # ---- rungs neither ladder offers (block 2/3) ----------------------
        ('NEW (q v^2 rms_sin)^2   ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v ** 2 * rs) ** 2, [0.01]),
        ('NEW bearing+v^2 rms_sin ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v * rs) ** 2 + (p[1] * v ** 2 * rs) ** 2, [0.05, 0.01]),
        ('NEW accel (c |a|)^2     ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * ac) ** 2, [0.01]),
        ('NEW ARRAY (a v rms_arr)^2', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v * rs2) ** 2, [0.03]),
        ('NEW array+speed-quad    ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v * rs2) ** 2 + (p[1] * v ** 2) ** 2, [0.03, 0.01]),
        ('NEW bearing+accel       ', lambda p, om, v, rho, rs, rs2, ac: (p[0] * v * rs) ** 2 + (p[1] * ac) ** 2, [0.05, 0.01]),
    ]


def fit_frames(tabs, fn, p0, stat='s2', loss='soft_l1'):
    """Frame-level joint fit: per-bag floor + one shared response, on the sigma
    scale, soft-L1 -- the published protocol (characterize_speed_law.fit_joint,
    notebook cell 51).  stat='srob' swaps in the robust per-frame scale."""
    keys = list(tabs)
    if stat == 's2':
        sig = np.concatenate([np.sqrt(tabs[k]['s2']) for k in keys])
    else:
        sig = np.concatenate([tabs[k]['srob'] for k in keys])
    om = np.concatenate([tabs[k]['om'] for k in keys])
    v = np.concatenate([tabs[k]['v'] for k in keys])
    rho = np.concatenate([tabs[k]['rho'] for k in keys])
    rs = np.concatenate([tabs[k]['rms_sin'] for k in keys])
    rs2 = np.concatenate([tabs[k]['rms_arr'] for k in keys])
    ac = np.concatenate([tabs[k]['acc'] for k in keys])
    bag = np.concatenate([np.full(len(tabs[k]['s2']), i)
                          for i, k in enumerate(keys)]).astype(int)

    def model(th):
        s0 = np.abs(th[bag])
        return np.sqrt(s0 ** 2 + fn(np.abs(th[len(keys):]), om, v, rho, rs,
                                    rs2, ac))

    th0 = np.r_[[np.sqrt(np.median(tabs[k]['s2'])) for k in keys], p0]
    fsc = max(1.4826 * np.median(np.abs(sig - np.median(sig))), 1e-3)
    sol = least_squares(lambda th: model(th) - sig, th0, loss=loss,
                        f_scale=fsc, max_nfev=40000)
    res = model(sol.x) - sig
    rss = float(np.sum(res ** 2))
    n, k = len(sig), len(th0)
    tot = float(np.sum((sig - sig.mean()) ** 2))
    return dict(theta=np.abs(sol.x), rss=rss, r2=1 - rss / tot,
                bic=n * np.log(rss / n) + k * np.log(n), n=n, k=k)


# ===========================================================================
def block0(keys, refresh=False):
    print('=' * 78)
    print('BLOCK 0 -- one per-return table, both published ladders reproduced')
    print('=' * 78)
    print('protocol: R_BS = euler(180, 27.5, 0) (measured pitch, not the 25.5')
    print('  self-cal seed); T_BS from extrinsics.yaml; time offset =')
    print('  imu_mocap - radar_imu; min_range 0.2 m; GT unwrap at v_max;')
    print('  residual r = unwrapped measurement - MoCap forward-model prediction.')
    print('  Frame-common scalar removed by the within-frame mean (per point) or')
    print('  by the variance operator (per frame).  No RANSAC, matching both')
    print('  published ladders.')
    print()
    tabs = {}
    for k in keys:
        T = table(k, refresh=refresh)
        tabs[k] = T
        P = point_view(T)
        Fv = frame_view(T)
        print(f'  {SHORT.get(k, k):<16} {len(T["r"]):6d} returns, '
              f'{len(T["f_t"]):4d} frames  ->  point view {len(P["d"]):6d} in '
              f'{len(np.unique(P["fid"])):4d} frames, frame view '
              f'{len(Fv["s2"]):4d} frames   alias {100*T["alias"].mean():4.1f}%')
    print()
    print('  off-boresight coverage, and how often the cos(phi) floor of 0.15')
    print('  (= 81.4 deg) actually binds.  An unreported clamp reads like a')
    print('  cheat parameter, so it is reported:')
    for k in keys:
        P = point_view(tabs[k])
        if not len(P['d']):
            continue
        ph = _phi(P)
        q = np.percentile(ph, [50, 90, 99, 100])
        print(f'  {SHORT.get(k, k):<16} phi median {q[0]:5.1f}  p90 {q[1]:5.1f}'
              f'  p99 {q[2]:5.1f}  max {q[3]:5.1f} deg   past 60 deg '
              f'{100*np.mean(ph > 60):4.1f}%   clamped '
              f'{100*np.mean(P["sx"] < 0.15):4.2f}%')
    return tabs


def gate0(tabs):
    print()
    print('-' * 78)
    print('GATE 0a: frame-level ladder (published = characterize_speed_law.py')
    print('  --quick + notebook cell 51).  Fitted on the two racing bags,')
    print('  alias-cut, per-bag floors + one shared response, soft-L1 on sigma.')
    print('-' * 78)
    FV = {k: frame_view(tabs[k]) for k in RACING}
    print(f"{'form':<26} {'R^2':>6} {'BIC':>10}   floors slow/fast | shared")
    fits = {}
    for name, fn, p0 in fr_specs():
        f = fit_frames(FV, fn, p0)
        fits[name] = f
        th = f['theta']
        print(f"{name:<26} {f['r2']:>6.3f} {f['bic']:>10.1f}   "
              + '/'.join(f'{x:.3f}' for x in th[:2]) + ' | '
              + '/'.join(f'{x:.4f}' for x in th[2:]))
    print()
    print('  published (notebook cell 51):  floors 0.133/-1822.3   rate 0.141/'
          '-1819.7   v-lin 0.276/-1880.6')
    print('                                 v-quad 0.331/-1908.9   trans 0.250/'
          '-1868.2  BEARING 0.315/-1900.4')

    print()
    print('-' * 78)
    print('GATE 0b: per-point ladder (published = characterize_pointwise_noise.py)')
    print('  (A) binned robust sigma, 10 quantile bins/bag, min 120 pts/bin.')
    print('-' * 78)
    PV = {k: point_view(tabs[k]) for k in RACING}
    print(f"{'per-point predictor X':<28} {'R^2':>7} {'coef(s)':>18} "
          f"{'floors sl/fa':>14}")
    for name in pt_predictors(PV[RACING[0]]):
        if name.startswith('floors'):
            continue
        Xs = {k: pt_predictors(PV[k])[name][0] for k in RACING}
        fb = binned_fit(PV, Xs)
        th = fb['theta']
        extra = ''
        if 'BEARING' in name:
            extra = f'   -> {np.degrees(th[2]*np.sqrt(2)):.1f} deg'
        print(f'{name:<28} {fb["r2"]:>7.3f} '
              f'{"/".join(f"{c:.5f}" for c in th[2:]):>18} '
              f'{th[0]:>6.3f}/{th[1]:<7.3f}{extra}')
    print()
    print('  published: v cos 0.414 | v 0.818 | v^2 0.910 | v^2 sin^2 0.841 |')
    print('             v sin 0.935 (coef 0.04292, 3.5 deg) | smear 0.603')
    return FV, PV


# ===========================================================================
# BLOCK 1 -- do the two selection statistics mean what they claim?
# ===========================================================================
def pair_index(fid, rng, max_pairs_per_frame=60):
    """All within-frame pairs (subsampled) -- identical to the published one."""
    ii, jj = [], []
    order = np.argsort(fid, kind='stable')
    fs = fid[order]
    bounds = np.flatnonzero(np.diff(fs)) + 1
    for grp in np.split(order, bounds):
        n = len(grp)
        a, b = np.triu_indices(n, k=1)
        if len(a) > max_pairs_per_frame:
            sel = rng.choice(len(a), max_pairs_per_frame, replace=False)
            a, b = a[sel], b[sel]
        ii.append(grp[a])
        jj.append(grp[b])
    return np.concatenate(ii), np.concatenate(jj)


def build_pairs(views, seed=1):
    """One pair set, shared by every model (the published code also rebuilds an
    identical set per model, so this only makes the sharing explicit)."""
    D, XIdx, bag = [], [], []
    rng = np.random.default_rng(seed)   # ONE rng across bags, as published
    for bi, key in enumerate(views):
        P = views[key]
        i, j = pair_index(P['fid'], rng)
        D.append(P['d'][i] - P['d'][j])
        XIdx.append((bi, i, j))
        bag.append(np.full(len(i), bi))
    return dict(D=np.concatenate(D), idx=XIdx,
                bag=np.concatenate(bag).astype(int), nb=len(views))


def _pair_X(pairs, Xs, keys):
    XI, XJ = [], []
    for (bi, i, j) in pairs['idx']:
        XL = Xs[keys[bi]]
        XI.append(np.column_stack([X[i] for X in XL]))
        XJ.append(np.column_stack([X[j] for X in XL]))
    return np.concatenate(XI), np.concatenate(XJ)


def pair_fit2(pairs, Xs, keys, trim_z=4.0, fixed_mask=None):
    """(B) pairwise composite likelihood, with the trim made explicit.

    trim_z=None or fixed_mask given -> no adaptive re-trim, so every model is
    scored on the SAME number of pairs.  The published version re-derives the
    mask per model and never reports the count.
    """
    D, bag, nb = pairs['D'], pairs['bag'], pairs['nb']
    XI, XJ = _pair_X(pairs, Xs, keys)
    nx = XI.shape[1]

    def variance(th):
        var = 2 * np.abs(th[bag]) ** 2
        for m in range(nx):
            var = var + np.abs(th[nb + m]) ** 2 * (XI[:, m] ** 2 + XJ[:, m] ** 2)
        return np.maximum(var, 1e-9)

    def nll(th, mask):
        var = variance(th)[mask]
        return float(np.sum(D[mask] ** 2 / var + np.log(var)))

    mask = np.ones(len(D), bool) if fixed_mask is None else fixed_mask.copy()
    th = np.r_[[robust_sigma(D) / np.sqrt(2)] * nb, [1e-3] * nx]
    n_it = 3 if (trim_z is not None and fixed_mask is None) else 1
    for _ in range(n_it):
        sol = minimize(nll, th, args=(mask,), method='Nelder-Mead',
                       options=dict(maxiter=40000, maxfev=40000,
                                    xatol=1e-9, fatol=1e-9))
        th = sol.x
        if trim_z is not None and fixed_mask is None:
            mask = np.abs(D) / np.sqrt(variance(th)) < trim_z
    return dict(theta=np.abs(th), nll=nll(th, mask), mask=mask,
                n_pairs=int(mask.sum()), variance=variance)


def block1a(PV):
    print()
    print('=' * 78)
    print('1a -- PER-POINT primary ranking: is the pairwise composite likelihood')
    print('      a fair comparison?  The published fit re-derives its 4-sigma')
    print('      trim mask PER MODEL (characterize_pointwise_noise.py:304) and')
    print('      never prints n_pairs (:305).  Each pair contributes')
    print('      D^2/var + log var, which is net NEGATIVE, so a model that keeps')
    print('      more pairs scores lower for free.')
    print('=' * 78)
    keys = list(PV)
    pairs = build_pairs(PV)
    names = [n for n in pt_predictors(PV[keys[0]])]
    res = {}
    for name in names:
        Xs = {k: pt_predictors(PV[k])[name][0] for k in keys}
        res[name] = pair_fit2(pairs, Xs, keys)
    common = np.ones(len(pairs['D']), bool)
    for name in names:
        common &= res[name]['mask']
    print(f'  pairs built: {len(pairs["D"])}; common (all models keep): '
          f'{int(common.sum())}')
    print()
    print(f"{'predictor':<28} {'published 2NLL':>14} {'n_pairs':>8} "
          f"{'2NLL common n':>14} {'no-trim 2NLL':>13} {'coef(s)':>18}")
    rows = {}
    for name in names:
        r = res[name]
        Xs = {k: pt_predictors(PV[k])[name][0] for k in keys}
        # same theta, scored on the common pair set -> count confound removed
        var = r['variance'](r['theta'])
        nll_c = float(np.sum(pairs['D'][common] ** 2 / var[common]
                             + np.log(var[common])))
        nt = pair_fit2(pairs, Xs, keys, trim_z=None)
        rows[name] = dict(pub=2 * r['nll'], n=r['n_pairs'], common=2 * nll_c,
                          notrim=2 * nt['nll'], theta=r['theta'],
                          theta_nt=nt['theta'])
        print(f'{name:<28} {2*r["nll"]:>14.1f} {r["n_pairs"]:>8d} '
              f'{2*nll_c:>14.1f} {2*nt["nll"]:>13.1f} '
              f'{"/".join(f"{c:.5f}" for c in r["theta"][2:]):>18}')
    b = 'v sin        (BEARING)    '
    q = 'v^2          (5b deployed)'
    print()
    print('  bearing - v^2 (negative = bearing wins):')
    print(f'    as published        {rows[b]["pub"]-rows[q]["pub"]:>10.1f}   '
          f'(bearing keeps {rows[b]["n"]-rows[q]["n"]:+d} more pairs)')
    print(f'    common pair set     {rows[b]["common"]-rows[q]["common"]:>10.1f}')
    print(f'    no trim at all      {rows[b]["notrim"]-rows[q]["notrim"]:>10.1f}')
    return rows


def chi2_nll(sig_model, s2, n):
    """-log L of the per-frame sample variance under Gaussian within-frame
    noise: (n-1) s2 / sigma^2 ~ chi^2(n-1).  Exact, dof-weighted."""
    k = n - 1.0
    v = np.maximum(sig_model ** 2, 1e-12)
    ll = ((k / 2 - 1) * np.log(np.maximum(s2, 1e-12))
          + (k / 2) * np.log(k / 2) - (k / 2) * np.log(v)
          - k * s2 / (2 * v) - gammaln(k / 2))
    return -float(np.sum(ll))


def fit_frames_ml(tabs, fn, p0):
    """Frame-level fit by EXACT chi^2 maximum likelihood (dof-weighted), so BIC
    is a real BIC instead of a Gaussian BIC computed off a soft-L1 RSS."""
    keys = list(tabs)
    s2 = np.concatenate([tabs[k]['s2'] for k in keys])
    n = np.concatenate([tabs[k]['n'] for k in keys])
    om = np.concatenate([tabs[k]['om'] for k in keys])
    v = np.concatenate([tabs[k]['v'] for k in keys])
    rho = np.concatenate([tabs[k]['rho'] for k in keys])
    rs = np.concatenate([tabs[k]['rms_sin'] for k in keys])
    rs2 = np.concatenate([tabs[k]['rms_arr'] for k in keys])
    ac = np.concatenate([tabs[k]['acc'] for k in keys])
    bag = np.concatenate([np.full(len(tabs[k]['s2']), i)
                          for i, k in enumerate(keys)]).astype(int)

    def sig(th):
        s0 = np.abs(th[bag])
        return np.sqrt(s0 ** 2 + fn(np.abs(th[len(keys):]), om, v, rho, rs,
                                    rs2, ac))

    th0 = np.r_[[np.sqrt(np.median(tabs[k]['s2'])) for k in keys], p0]
    sol = minimize(lambda th: chi2_nll(sig(th), s2, n), th0,
                   method='Nelder-Mead',
                   options=dict(maxiter=40000, maxfev=40000, xatol=1e-10,
                                fatol=1e-8))
    nll = chi2_nll(sig(sol.x), s2, n)
    nf, kk = len(s2), len(th0)
    return dict(theta=np.abs(sol.x), nll=nll, bic=2 * nll + kk * np.log(nf),
                n=nf, k=kk)


def fit_frames_w(tabs, fn, p0):
    """Same soft-L1 sigma-scale fit as published, but with the dof weight the
    published fit omits: sd(s_f) / s_f ~ 1/sqrt(2(n-1))."""
    keys = list(tabs)
    sig = np.concatenate([np.sqrt(tabs[k]['s2']) for k in keys])
    n = np.concatenate([tabs[k]['n'] for k in keys])
    w = np.sqrt(2 * (n - 1.0))
    om = np.concatenate([tabs[k]['om'] for k in keys])
    v = np.concatenate([tabs[k]['v'] for k in keys])
    rho = np.concatenate([tabs[k]['rho'] for k in keys])
    rs = np.concatenate([tabs[k]['rms_sin'] for k in keys])
    rs2 = np.concatenate([tabs[k]['rms_arr'] for k in keys])
    ac = np.concatenate([tabs[k]['acc'] for k in keys])
    bag = np.concatenate([np.full(len(tabs[k]['s2']), i)
                          for i, k in enumerate(keys)]).astype(int)

    def model(th):
        s0 = np.abs(th[bag])
        return np.sqrt(s0 ** 2 + fn(np.abs(th[len(keys):]), om, v, rho, rs,
                                    rs2, ac))

    th0 = np.r_[[np.sqrt(np.median(tabs[k]['s2'])) for k in keys], p0]
    resid = lambda th: (model(th) - sig) * w / model(th)
    fsc = max(1.4826 * np.median(np.abs(resid(th0))), 1e-3)
    sol = least_squares(resid, th0, loss='soft_l1', f_scale=fsc, max_nfev=40000)
    rss = float(np.sum(resid(sol.x) ** 2))
    nf, kk = len(sig), len(th0)
    return dict(theta=np.abs(sol.x), rss=rss, bic=nf * np.log(rss / nf)
                + kk * np.log(nf), n=nf)


def block1b(FV):
    print()
    print('=' * 78)
    print('1b -- FRAME-LEVEL selection statistic.  The published BIC is a')
    print('      Gaussian BIC computed from the RSS of a soft-L1 fit')
    print('      (characterize_speed_law.py:175) on an unweighted sigma scale,')
    print('      although s_f^2 is chi^2 with n_f-1 dof and n_f varies 6..40.')
    print('      Re-scored by exact chi^2 ML and by a dof-weighted robust fit.')
    print('=' * 78)
    print(f"{'form':<26} {'pub BIC':>10} {'chi2-ML BIC':>12} "
          f"{'dof-w BIC':>10}   shared params (ML)")
    out = {}
    for name, fn, p0 in fr_specs():
        a = fit_frames(FV, fn, p0)
        b = fit_frames_ml(FV, fn, p0)
        c = fit_frames_w(FV, fn, p0)
        out[name] = dict(pub=a['bic'], ml=b['bic'], w=c['bic'],
                         theta_ml=b['theta'])
        print(f'{name:<26} {a["bic"]:>10.1f} {b["bic"]:>12.1f} '
              f'{c["bic"]:>10.1f}   '
              + '/'.join(f'{x:.4f}' for x in b['theta'][2:]))
    bb = 'BEARING (a v rms_sin)^2 '
    qq = 'speed-quad (q v^2)^2    '
    nn = 'NEW (q v^2 rms_sin)^2   '
    print()
    print('  dBIC vs speed-quad (negative = better than v^2):')
    for nm in (bb, nn):
        print(f'    {nm}  pub {out[nm]["pub"]-out[qq]["pub"]:+7.1f}   '
              f'chi2-ML {out[nm]["ml"]-out[qq]["ml"]:+8.1f}   '
              f'dof-w {out[nm]["w"]-out[qq]["w"]:+7.1f}')
    return out


def block1c(FV, PV, n_boot=200, seed=0):
    print()
    print('=' * 78)
    print('1c -- EFFECTIVE SAMPLE SIZE.  Both ladders treat frames as')
    print('      independent, but speed and scene geometry are smooth in time.')
    print('      Moving-block bootstrap over contiguous frames; a preference')
    print('      that flips sign under resampling is not evidence.')
    print('=' * 78)
    qq = 'speed-quad (q v^2)^2    '
    bb = 'BEARING (a v rms_sin)^2 '
    nn = 'NEW (q v^2 rms_sin)^2   '
    specs = {n: (f, p) for n, f, p in fr_specs()}
    keys = list(FV)
    # residual autocorrelation of the winner, per bag
    for k in keys:
        f = fit_frames({k: FV[k]}, *specs[qq])
        v, rho = FV[k]['v'], FV[k]['rho']
        rs, rs2, ac, om = (FV[k]['rms_sin'], FV[k]['rms_sin2'], FV[k]['acc'],
                           FV[k]['om'])
        mdl = np.sqrt(f['theta'][0] ** 2
                      + specs[qq][0](f['theta'][1:], om, v, rho, rs, rs2, ac))
        e = np.sqrt(FV[k]['s2']) - mdl
        e = e - e.mean()
        ac1 = [float(np.corrcoef(e[:-L], e[L:])[0, 1]) for L in (1, 2, 5, 10)]
        print(f'  {SHORT[k]:<12} fit-residual autocorr lag 1/2/5/10 frames: '
              + '  '.join(f'{a:+.2f}' for a in ac1))
    print()
    print(f"{'block':<10} {'dBIC bearing-quad':>20} {'bearing wins':>14} "
          f"{'dBIC NEW-quad':>15} {'NEW wins':>10}")
    rng = np.random.default_rng(seed)
    for L in (1, 10, 20, 50):
        db_b, db_n = [], []
        for _ in range(n_boot):
            tb = {}
            for k in keys:
                N = len(FV[k]['s2'])
                nblk = int(np.ceil(N / L))
                starts = rng.integers(0, max(N - L + 1, 1), nblk)
                idx = np.concatenate([np.arange(s, min(s + L, N))
                                      for s in starts])[:N]
                tb[k] = {c: FV[k][c][idx] for c in FV[k]}
            try:
                fq = fit_frames(tb, *specs[qq])
                fb = fit_frames(tb, *specs[bb])
                fn_ = fit_frames(tb, *specs[nn])
                db_b.append(fb['bic'] - fq['bic'])
                db_n.append(fn_['bic'] - fq['bic'])
            except Exception:
                continue
        db_b, db_n = np.array(db_b), np.array(db_n)
        lab = 'iid frames' if L == 1 else f'{L} frames'
        print(f'  {lab:<10} {np.median(db_b):>8.1f} '
              f'[{np.percentile(db_b,16):.1f},{np.percentile(db_b,84):.1f}]'
              f' {100*np.mean(db_b<0):>13.0f}%'
              f' {np.median(db_n):>10.1f} {100*np.mean(db_n<0):>9.0f}%')
    print('  (block length in frames; the radar runs at 10 Hz, so 10 frames = 1 s)')


def block1d(tabs, FV, PV):
    print()
    print('=' * 78)
    print('1d -- MATCHED ESTIMATORS.  The frame ladder scores a plain sample')
    print('      variance; the per-point ladder scores a MAD.  With excess')
    print('      kurtosis ~11 those are not the same functional of the noise.')
    print('      Both statistics, both scale estimators, everything else fixed.')
    print('=' * 78)
    specs = {n: (f, p) for n, f, p in fr_specs()}
    qq = 'speed-quad (q v^2)^2    '
    bb = 'BEARING (a v rms_sin)^2 '
    nn = 'NEW (q v^2 rms_sin)^2   '
    for stat, lab in (('s2', 'second moment (published)'), ('srob', 'robust MAD')):
        fq = fit_frames(FV, *specs[qq], stat=stat)
        fb = fit_frames(FV, *specs[bb], stat=stat)
        fn_ = fit_frames(FV, *specs[nn], stat=stat)
        print(f'  FRAME  / {lab:<26} R^2 quad {fq["r2"]:.3f}  bearing '
              f'{fb["r2"]:.3f}  NEW {fn_["r2"]:.3f} | dBIC bear-quad '
              f'{fb["bic"]-fq["bic"]:+.1f}, NEW-quad {fn_["bic"]-fq["bic"]:+.1f}')
    keys = list(PV)
    for rob, lab in ((True, 'robust MAD (published)'), (False, 'second moment')):
        line = []
        for nm in ('v sin        (BEARING)    ', 'v^2          (5b deployed)',
                   'v^2 sin      (NEW)        '):
            Xs = {k: pt_predictors(PV[k])[nm][0] for k in keys}
            fb = binned_fit(PV, Xs, robust=rob)
            line.append(f'{nm.split("(")[0].strip():<12} R^2 {fb["r2"]:.3f}')
        print(f'  POINT  / {lab:<26} ' + ' | '.join(line))
    print('  (per-point R^2 is within each model\'s own binning -- not a valid')
    print('   cross-model ranking; shown only to compare the two estimators.)')

    print()
    print('  heavy tail vs speed (the mechanism by which a second moment and a')
    print('  MAD can disagree):')
    print(f"    {'bag':<12} {'v bin':>10} {'n':>6} {'kurtosis':>9} "
          f"{'sd/MAD':>7} {'top1% share of sum d^2':>23}")
    for k in keys:
        P = PV[k]
        edges = [0, 2, 3, 4, 6]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (P['v'] >= lo) & (P['v'] < hi)
            if m.sum() < 100:
                continue
            d = P['d'][m]
            z = d / robust_sigma(d)
            kurt = float(np.mean(z ** 4) / np.mean(z ** 2) ** 2 - 3)
            sd = float(np.std(d, ddof=1))
            srt = np.sort(d ** 2)[::-1]
            top = int(max(1, round(0.01 * len(d))))
            share = float(srt[:top].sum() / srt.sum())
            print(f'    {SHORT[k]:<12} {lo}-{hi:<8} {m.sum():>6d} '
                  f'{kurt:>9.1f} {sd/robust_sigma(d):>7.2f} {100*share:>22.1f}%')


def block1d3(tabs, n_boot=200, seed=0):
    """Where does the disagreement live?  Winsorize the per-point deviations at
    a fixed number of core sigmas, rebuild the per-frame second moment, refit.
    Trim = inf is the published statistic; small trim -> the robust one."""
    print()
    print('=' * 78)
    print('1d(iii) -- WINSORIZE SWEEP.  The frame statistic is rebuilt from the')
    print('      same returns with the per-point deviations clipped at z core')
    print('      sigmas (core sigma = per-frame MAD).  z=inf is the published')
    print('      second moment; the sweep shows how much tail mass has to be')
    print('      removed before the shape preference flips.')
    print('=' * 78)
    specs = {n: (f, p) for n, f, p in fr_specs()}
    qq = 'speed-quad (q v^2)^2    '
    bb = 'BEARING (a v rms_sin)^2 '
    nn = 'NEW (q v^2 rms_sin)^2   '
    PV = {k: point_view(tabs[k]) for k in RACING}
    FV0 = {k: frame_view(tabs[k]) for k in RACING}
    print(f"{'winsor z':>9} {'% points clipped':>17} {'R^2 quad':>9} "
          f"{'R^2 bear':>9} {'dBIC bear-quad':>15} {'dBIC NEW-quad':>14}")
    for z in (np.inf, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0):
        VV, clipped, tot = {}, 0, 0
        for k in RACING:
            P, F = PV[k], FV0[k]
            s2, srob = [], []
            keep_f = []
            for row, f in enumerate(F['fid']):
                m = P['fid'] == int(f)
                if not np.any(m):
                    continue
                d = P['d'][m]
                n = len(d)
                # rebuild the raw within-frame residual deviations, clip, refit
                sc = robust_sigma(d)
                lim = z * sc if np.isfinite(z) else np.inf
                dc = np.clip(d, -lim, lim)
                clipped += int(np.sum(np.abs(d) > lim))
                tot += n
                # d already carries the 1/sqrt(1-1/n) unshrink; undo it so the
                # statistic is the same functional as the published one
                s2.append(float(np.var(dc * np.sqrt(1 - 1 / n), ddof=1)))
                srob.append(robust_sigma(dc))
                keep_f.append(row)
            rows = np.array(keep_f, int)
            base = {c: F[c][rows] for c in F}
            VV[k] = dict(base, s2=np.array(s2), srob=np.array(srob))
        fq = fit_frames(VV, *specs[qq])
        fb = fit_frames(VV, *specs[bb])
        fn_ = fit_frames(VV, *specs[nn])
        zl = 'inf' if not np.isfinite(z) else f'{z:.1f}'
        print(f'{zl:>9} {100*clipped/max(tot,1):>16.2f}% {fq["r2"]:>9.3f} '
              f'{fb["r2"]:>9.3f} {fb["bic"]-fq["bic"]:>15.1f} '
              f'{fn_["bic"]-fq["bic"]:>14.1f}')

    print()
    print('  binned per-frame scale vs speed, both estimators (m/s):')
    for k in RACING:
        F = FV0[k]
        edges = [0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0]
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (F['v'] >= lo) & (F['v'] < hi)
            if m.sum() < 8:
                continue
            sd = float(np.sqrt(np.mean(F['s2'][m])))
            md = float(np.median(F['srob'][m]))
            rows.append(f'{0.5*(lo+hi):.2f}: {sd:.3f}/{md:.3f}={sd/md:.2f}')
        print(f'    {SHORT[k]:<12} ' + '  '.join(rows))
    print('    (rms second moment / median robust scale = ratio; a ratio that')
    print('     FALLS with speed flattens the low-speed end of the second-')
    print('     moment curve and pushes its fitted shape toward v^2.)')


def _scale_calib(nmax=200, ndraw=40000, seed=7):
    """Finite-sample calibration of the two per-frame scale estimators under
    N(0,1).  BOTH are biased at small n and n_f runs 6..40 here:
      E[sqrt(s^2)] = c4(n) sigma  (c4(6)=0.952)  -- the PUBLISHED statistic
      E[1.4826 MAD_n] != sigma either (worse, and non-monotone at small n)
    If n_f correlates with speed, either bias masquerades as a speed law."""
    rng = np.random.default_rng(seed)
    c_sd, c_mad = np.ones(nmax + 1), np.ones(nmax + 1)
    for n in range(4, nmax + 1):
        x = rng.standard_normal((ndraw, n))
        c_sd[n] = np.mean(np.std(x, axis=1, ddof=1))
        c_mad[n] = np.mean(1.4826 * np.median(
            np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1))
    return c_sd, c_mad


def block1d2(tabs, FV, n_boot=200, seed=0):
    print()
    print('=' * 78)
    print('1d(ii) -- is the reversal itself an estimator artifact?  Both scale')
    print('      estimators are biased at small n, and n_f runs 6..40.  Exact')
    print('      finite-sample calibration under N(0,1), then refit.')
    print('=' * 78)
    c_sd, c_mad = _scale_calib()
    print(f'  calibration: c_sd(6)={c_sd[6]:.4f} c_sd(10)={c_sd[10]:.4f} '
          f'c_sd(20)={c_sd[20]:.4f} | c_mad(6)={c_mad[6]:.4f} '
          f'c_mad(10)={c_mad[10]:.4f} c_mad(20)={c_mad[20]:.4f}')
    for k in FV:
        n, v = FV[k]['n'], FV[k]['v']
        print(f'  {SHORT[k]:<12} n_f med {np.median(n):.0f} '
              f'[{n.min():.0f},{n.max():.0f}]  corr(n_f, v) '
              f'{np.corrcoef(n, v)[0, 1]:+.2f}  corr(n_f, v^2) '
              f'{np.corrcoef(n, v**2)[0, 1]:+.2f}')

    specs = {n: (f, p) for n, f, p in fr_specs()}
    qq = 'speed-quad (q v^2)^2    '
    bb = 'BEARING (a v rms_sin)^2 '
    nn = 'NEW (q v^2 rms_sin)^2   '
    variants = {}
    for k in FV:
        n = FV[k]['n'].astype(int)
        base = dict(FV[k])
        variants.setdefault('sd raw (published)', {})[k] = dict(
            base, srob=np.sqrt(base['s2']))
        variants.setdefault('sd unbiased', {})[k] = dict(
            base, srob=np.sqrt(base['s2']) / c_sd[n])
        variants.setdefault('MAD raw', {})[k] = dict(
            base, srob=base['srob'] * np.sqrt(1 - 1 / n))
        variants.setdefault('MAD unbiased', {})[k] = dict(
            base, srob=base['srob'] * np.sqrt(1 - 1 / n) / c_mad[n])
        # 20% symmetric trimmed sd, an intermediate breakdown point
        tr = []
        for f in base['fid']:
            tr.append(np.nan)
        variants.setdefault('n_f >= 12 (sd)', {})[k] = None

    print()
    print(f"{'per-frame scale':<20} {'R^2 quad':>9} {'R^2 bear':>9} "
          f"{'dBIC bear-quad':>15} {'dBIC NEW-quad':>14} {'bear wins boot':>15}")
    rng = np.random.default_rng(seed)
    for vn in ('sd raw (published)', 'sd unbiased', 'MAD raw', 'MAD unbiased'):
        VV = variants[vn]
        fq = fit_frames(VV, *specs[qq], stat='srob')
        fb = fit_frames(VV, *specs[bb], stat='srob')
        fn_ = fit_frames(VV, *specs[nn], stat='srob')
        wins = []
        for _ in range(n_boot):
            tb = {}
            for k in VV:
                N = len(VV[k]['s2'])
                idx = rng.integers(0, N, N)
                tb[k] = {c: VV[k][c][idx] for c in VV[k]}
            try:
                wins.append(fit_frames(tb, *specs[bb], stat='srob')['bic']
                            - fit_frames(tb, *specs[qq], stat='srob')['bic'])
            except Exception:
                continue
        wins = np.array(wins)
        print(f'  {vn:<18} {fq["r2"]:>9.3f} {fb["r2"]:>9.3f} '
              f'{fb["bic"]-fq["bic"]:>15.1f} {fn_["bic"]-fq["bic"]:>14.1f} '
              f'{100*np.mean(wins<0):>14.0f}%')

    # n_f-matched subsample: kills any n-dependence outright
    print()
    print('  n_f-matched control (frames with n_f in [9,14] only, both scales):')
    sub = {k: {c: FV[k][c][(FV[k]['n'] >= 9) & (FV[k]['n'] <= 14)]
               for c in FV[k]} for k in FV}
    for stat, lab in (('s2', 'second moment'), ('srob', 'robust MAD')):
        fq = fit_frames(sub, *specs[qq], stat=stat)
        fb = fit_frames(sub, *specs[bb], stat=stat)
        print(f'    {lab:<16} n={fq["n"]:4d}  dBIC bear-quad '
              f'{fb["bic"]-fq["bic"]:+7.1f}')


def block1f(tabs):
    """Decompose the per-point deviation scale into a CORE and a TAIL excess
    and give each its own speed law.  If the core follows the bearing shape and
    the excess follows something steeper, the two statistics were measuring two
    different things and both were right about their own functional."""
    print()
    print('=' * 78)
    print('1f -- CORE vs TAIL, each with its own speed law.  Pooled per-point')
    print('      within-frame deviations, binned by frame speed:')
    print('        core  = 1.4826 MAD           (what the per-point ladder fits)')
    print('        total = sd                   (what the frame ladder fits)')
    print('        excess= sqrt(total^2-core^2)  (the part only the sd sees)')
    print('=' * 78)
    edges = np.array([0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.5])
    for k in RACING:
        P = point_view(tabs[k])
        print(f'  {SHORT[k]}')
        print(f"    {'v':>10} {'n':>6} {'core':>7} {'total':>7} {'excess':>7} "
              f"{'rms sin':>8} {'core/(v sin)':>13} {'>3 core %':>10}")
        vc, core, tot, exc = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (P['v'] >= lo) & (P['v'] < hi)
            if m.sum() < 80:
                continue
            d = P['d'][m]
            c = robust_sigma(d)
            t = float(np.std(d, ddof=1))
            e = float(np.sqrt(max(t ** 2 - c ** 2, 0.0)))
            rs = float(np.sqrt(np.mean(P['sin2'][m])))
            vm = float(P['v'][m].mean())
            vc.append(vm); core.append(c); tot.append(t); exc.append(e)
            print(f'    {lo:.1f}-{hi:<5.1f} {m.sum():>6d} {c:>7.3f} {t:>7.3f} '
                  f'{e:>7.3f} {rs:>8.3f} {c/(vm*rs):>13.4f} '
                  f'{100*np.mean(np.abs(d) > 3*c):>9.2f}%')
        vc, core, tot, exc = map(np.array, (vc, core, tot, exc))
        for nm, y in (('core', core), ('total', tot), ('excess', exc)):
            good = y > 1e-6
            if good.sum() >= 3:
                p = np.polyfit(np.log(vc[good]), np.log(y[good]), 1)
                print(f'    power-law slope of {nm:<7} vs v: '
                      f'{p[0]:+.2f}   (bearing predicts +1, deployed +2)')


def _empirical_z(P, edges):
    """Standardized within-frame deviations, kept SEPARATELY per speed bin, so a
    resample reproduces the measured kurtosis-vs-speed structure instead of a
    single pooled tail."""
    out = {}
    for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (P['v'] >= lo) & (P['v'] < hi)
        if m.sum() >= 60:
            d = P['d'][m]
            out[bi] = d / robust_sigma(d)
    return out


def block1e(tabs, n_draw=200, seed=0):
    print()
    print('=' * 78)
    print('1e -- POWER CALIBRATION.  Synthesise returns on the REAL geometry')
    print('      (same frames, same sin_j, same N_f, same v_f) from a KNOWN law,')
    print('      then run both ladders unchanged.  The decisive case is the')
    print('      third: a pure bearing core plus the measured heavy tail.')
    print('=' * 78)
    keys = RACING
    PV = {k: point_view(tabs[k]) for k in keys}
    FVm = {}
    for k in keys:
        fids = np.unique(PV[k]['fid'])
        F = frame_view(tabs[k], speed_cut=True)
        FVm[k] = {c: F[c][np.isin(F['fid'], fids)] for c in F}
    specs = {n: (f, p) for n, f, p in fr_specs()}
    qq = 'speed-quad (q v^2)^2    '
    bb = 'BEARING (a v rms_sin)^2 '
    edges = np.array([0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.5])
    EZ = {k: _empirical_z(PV[k], edges) for k in keys}
    BIN = {k: np.clip(np.digitize(PV[k]['v'], edges) - 1, 0, len(edges) - 2)
           for k in keys}

    def sigma_of(kind, k, P):
        if kind == 'bear':
            s0b = 0.041 if k == keys[0] else 0.064
            return np.sqrt(s0b ** 2
                           + (0.04292 * P['v'] * np.sqrt(P['sin2'])) ** 2)
        if kind == 'bear_nofloor':
            return 0.047 * P['v'] * np.sqrt(P['sin2'])
        s0 = 0.072 if k == keys[0] else 0.094
        return np.sqrt(s0 ** 2 + (0.0104 * P['v'] ** 2) ** 2)

    cases = [('bearing (fitted floors)', 'bear', 'gaussian'),
             ('quadratic (fitted floors)', 'quad', 'gaussian'),
             ('bearing core, NO floor', 'bear_nofloor', 'gaussian'),
             ('bearing core + MEASURED tail', 'bear_nofloor', 'empirical')]
    print(f"{'generating law':<30} {'noise':<10} {'FRAME sd':>12} "
          f"{'FRAME MAD':>12} {'POINT':>22}")
    rng = np.random.default_rng(seed)
    for cname, kind, zname in cases:
        d_sd, d_mad, d_pt = [], [], []
        for _ in range(n_draw):
            pb, fb = {}, {}
            for k in keys:
                P = PV[k]
                s = sigma_of(kind, k, P)
                if zname == 'gaussian':
                    z = rng.standard_normal(len(s))
                else:
                    z = np.empty(len(s))
                    for bi, zz in EZ[k].items():
                        m = BIN[k] == bi
                        if m.any():
                            z[m] = rng.choice(zz, int(m.sum()), replace=True)
                    m = ~np.isfinite(z)
                    if m.any():
                        z[m] = rng.standard_normal(int(m.sum()))
                e = s * z
                d = np.empty_like(e)
                s2, srob, nn_, order = [], [], [], []
                for f in np.unique(P['fid']):
                    m = P['fid'] == f
                    n = int(m.sum())
                    ef = e[m]
                    d[m] = (ef - ef.mean()) / np.sqrt(1 - 1 / n)
                    s2.append(np.var(ef, ddof=1))
                    srob.append(robust_sigma(ef) / np.sqrt(1 - 1 / n))
                    nn_.append(n)
                    order.append(f)
                pb[k] = dict(P, d=d)
                base = FVm[k]
                ix = np.array([np.flatnonzero(base['fid'] == f)[0]
                               for f in order])
                fb[k] = {c: base[c][ix] for c in base}
                fb[k]['s2'] = np.array(s2)
                fb[k]['srob'] = np.array(srob)
                fb[k]['n'] = np.array(nn_, float)
            d_sd.append(fit_frames(fb, *specs[bb])['bic']
                        - fit_frames(fb, *specs[qq])['bic'])
            d_mad.append(fit_frames(fb, *specs[bb], stat='srob')['bic']
                         - fit_frames(fb, *specs[qq], stat='srob')['bic'])
            Xq = {k: pt_predictors(pb[k])['v^2          (5b deployed)'][0]
                  for k in keys}
            Xb = {k: pt_predictors(pb[k])['v sin        (BEARING)    '][0]
                  for k in keys}
            d_pt.append(binned_fit(pb, Xb)['r2'] - binned_fit(pb, Xq)['r2'])
        a, b, c = map(np.array, (d_sd, d_mad, d_pt))
        print(f'{cname:<30} {zname:<10} '
              f'{np.median(a):>+7.1f} {100*np.mean(a<0):>3.0f}% '
              f'{np.median(b):>+7.1f} {100*np.mean(b<0):>3.0f}% '
              f'  bear R2 higher {100*np.mean(c>0):>3.0f}% (med {np.median(c):+.3f})')
    print('  FRAME columns: median dBIC(bearing - quad) and % of draws where the')
    print('  bearing rung wins.  Negative/high% = the test recovers a bearing truth.')
    print()
    print('  REAL DATA for comparison:')
    FVr = {k: frame_view(tabs[k]) for k in keys}
    for stat, lab in (('s2', 'FRAME sd '), ('srob', 'FRAME MAD')):
        fq = fit_frames(FVr, *specs[qq], stat=stat)
        fbb = fit_frames(FVr, *specs[bb], stat=stat)
        print(f'    {lab}  dBIC(bearing - quad) = {fbb["bic"]-fq["bic"]:+.1f}')


def _model_sigma(P, model, k):
    """sigma_j under each candidate, with the published fitted constants."""
    s0 = {'slow_racing_best_velocity': 0.041,
          'fast_racing_best_velocity': 0.064}.get(k, 0.05)
    s0q = {'slow_racing_best_velocity': 0.055,
           'fast_racing_best_velocity': 0.064}.get(k, 0.06)
    if model == 'bearing':
        return np.sqrt(s0 ** 2 + (0.04292 * P['v'] * np.sqrt(P['sin2'])) ** 2)
    if model == 'bearing_nf':
        return 0.047 * P['v'] * np.sqrt(P['sin2'])
    if model == 'quad':
        return np.sqrt(s0q ** 2 + (0.00966 * P['v'] ** 2) ** 2)
    if model == 'array':
        # coefficient from the per-point pairwise fit with no adaptive trim
        # (block 1a), i.e. dphi_0 = 2.12 deg at boresight
        return np.sqrt(s0 ** 2 + (0.02618 * P['v'] * np.sqrt(P['sin2'])
                                  / np.maximum(P['sx'], 0.15)) ** 2)
    if model == 'dircos':
        return np.sqrt(s0 ** 2 + (0.03291 * _dircos(P)[0]) ** 2)
    if model == 'const':
        return np.full(len(P['v']), 1.0)
    raise ValueError(model)


def block1h(tabs):
    """Model ADEQUACY, which neither ladder tests: whiten the per-point
    deviations by each candidate and ask whether what is left is structureless.
    A correct sigma_j model leaves unit scale in EVERY speed bin AND every
    sin(theta) bin simultaneously.  Selection statistics can disagree; a
    whitening residual cannot be argued with."""
    print()
    print('=' * 78)
    print('1h -- WHITENING ADEQUACY.  w_j = d_j / sigma_j(model).  A correct')
    print('      model gives robust scale 1.00 in every stratum and no residual')
    print('      trend.  Strata are speed (across frames), sin(theta) (WITHIN')
    print('      frames) -- the two directions the ladders each see only one of')
    print('      -- and the off-boresight angle phi, which NEITHER ladder')
    print('      offered a rung for.  This is not a selection statistic: a')
    print('      model that leaves a monotone trend in a stratum is wrong in')
    print('      that stratum whatever its BIC says.')
    print('=' * 78)
    v_edges = np.array([0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.5])
    s_edges = np.array([0.0, 0.4, 0.6, 0.75, 0.87, 0.95, 1.01])
    p_edges = np.array([0.0, 25.0, 40.0, 50.0, 60.0, 90.0])
    for model in ('const', 'quad', 'bearing', 'bearing_nf', 'array', 'dircos'):
        print(f'  model = {model}')
        for k in RACING:
            P = point_view(tabs[k])
            w = P['d'] / _model_sigma(P, model, k)
            if model == 'const':
                w = w / robust_sigma(w)
            sin1 = np.sqrt(P['sin2'])

            def strat(x, edges, fmt):
                out = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = (x >= lo) & (x < hi)
                    if m.sum() >= 80:
                        out.append(f'{0.5*(lo+hi):{fmt}}:'
                                   f'{robust_sigma(w[m]):.2f}')
                return out

            def spread(rows):
                vals = [float(x.split(':')[1]) for x in rows]
                return max(vals) / max(min(vals), 1e-6)

            rows_v = strat(P['v'], v_edges, '.1f')
            rows_s = strat(sin1, s_edges, '.2f')
            rows_p = strat(_phi(P), p_edges, '.0f')
            z = w / robust_sigma(w)
            kurt = float(np.mean(z ** 4) / np.mean(z ** 2) ** 2 - 3)
            print(f'    {SHORT[k]:<12} vs v   ' + ' '.join(rows_v)
                  + f'   spread {spread(rows_v):.2f}x')
            print(f'    {"":<12} vs sin ' + ' '.join(rows_s)
                  + f'   spread {spread(rows_s):.2f}x   excess kurt {kurt:.1f}')
            print(f'    {"":<12} vs phi ' + ' '.join(rows_p)
                  + f'   spread {spread(rows_p):.2f}x')


def block1g(tabs):
    """What IS the tail?  If the excess the second moment sees is a distinct
    population, it should be identifiable by range / intensity / elevation /
    frame occupancy rather than smeared uniformly over the core."""
    print()
    print('=' * 78)
    print('1g -- IDENTIFYING THE TAIL.  Points with |d| > 3 core sigma, against')
    print('      the rest, on covariates that were never used to define them.')
    print('=' * 78)
    print(f"    {'bag':<12} {'group':<8} {'n':>6} {'rho m':>7} {'intens dB':>10} "
          f"{'|u_z|':>7} {'sin':>6} {'N_f':>6} {'|d| med':>8}")
    for k in RACING:
        P = point_view(tabs[k])
        sc = np.zeros(len(P['d']))
        for f in np.unique(P['fid']):
            m = P['fid'] == f
            sc[m] = robust_sigma(P['d'][m])
        # threshold scaled by the BEARING model, not by a per-frame constant:
        # under sigma_j = a v sin_j a constant threshold would flag high-sin
        # points by construction and manufacture the very correlation reported.
        sig = _model_sigma(P, 'bearing', k)
        tail = np.abs(P['d']) > 3 * sig
        for lab, m in (('core', ~tail), ('tail', tail)):
            if m.sum() < 5:
                continue
            print(f'    {SHORT[k]:<12} {lab:<8} {m.sum():>6d} '
                  f'{np.median(P["rho"][m]):>7.2f} '
                  f'{np.median(P["inten"][m]):>10.1f} '
                  f'{np.median(np.abs(P["uz"][m])):>7.3f} '
                  f'{np.median(np.sqrt(P["sin2"][m])):>6.3f} '
                  f'{np.median(P["nf"][m]):>6.0f} '
                  f'{np.median(np.abs(P["d"][m])):>8.3f}')
        # how concentrated is the tail in frames?
        nfr = len(np.unique(P['fid']))
        ftail = len(np.unique(P['fid'][tail]))
        print(f'    {SHORT[k]:<12} tail occupies {ftail}/{nfr} frames '
              f'({100*ftail/nfr:.0f}%); tail rate {100*tail.mean():.2f}% of '
              f'returns (Gaussian would give 0.27%)')



# ===========================================================================
# BLOCK 2 -- free exponents: sigma_j^2 = s0^2 + a^2 v^(2p) sin^(2q)
# ===========================================================================
def free_fit_point(PV, seed=1, p0=(1.0, 1.0), fix=None):
    """Per-point free-exponent fit by the pairwise composite likelihood.

    fix=(p,q) pins the exponents (used for the nested comparisons).
    Returns theta = [s0_slow, s0_fast, a, p, q] and the 2*NLL on the full pair
    set (no adaptive trim, so every (p,q) is scored on identical pairs).
    """
    keys = list(PV)
    pairs = build_pairs(PV, seed=seed)
    VI, VJ, SI, SJ = [], [], [], []
    for (bi, i, j) in pairs['idx']:
        P = PV[keys[bi]]
        VI.append(P['v'][i]); VJ.append(P['v'][j])
        SI.append(np.sqrt(P['sin2'][i])); SJ.append(np.sqrt(P['sin2'][j]))
    VI, VJ = np.concatenate(VI), np.concatenate(VJ)
    SI, SJ = np.concatenate(SI), np.concatenate(SJ)
    D, bag, nb = pairs['D'], pairs['bag'], pairs['nb']
    eps = 1e-6

    def nll(th):
        s0 = np.abs(th[bag]) ** 2
        a2 = th[2] ** 2
        p, q = th[3], th[4]
        xi = (np.maximum(VI, eps) ** (2 * p)) * (np.maximum(SI, eps) ** (2 * q))
        xj = (np.maximum(VJ, eps) ** (2 * p)) * (np.maximum(SJ, eps) ** (2 * q))
        var = np.maximum(2 * s0 + a2 * (xi + xj), 1e-9)
        return float(np.sum(D ** 2 / var + np.log(var)))

    if fix is not None:
        def nll3(t3):
            return nll(np.r_[t3, fix])
        th0 = np.r_[[robust_sigma(D) / np.sqrt(2)] * nb, 0.05]
        sol = minimize(nll3, th0, method='Nelder-Mead',
                       options=dict(maxiter=40000, maxfev=40000, xatol=1e-10,
                                    fatol=1e-9))
        return dict(theta=np.r_[np.abs(sol.x), fix], nll=nll3(sol.x),
                    n=len(D))
    th0 = np.r_[[robust_sigma(D) / np.sqrt(2)] * nb, 0.05, p0[0], p0[1]]
    sol = minimize(nll, th0, method='Nelder-Mead',
                   options=dict(maxiter=80000, maxfev=80000, xatol=1e-10,
                                fatol=1e-9))
    return dict(theta=np.r_[np.abs(sol.x[:3]), sol.x[3:]], nll=nll(sol.x),
                n=len(D))


def _frame_sin_matrix(PV, FV):
    """Padded per-frame sin(theta) matrix aligned to the frame-view rows."""
    out = {}
    for k in PV:
        P, F = PV[k], FV[k]
        rows, keep = [], []
        for row, f in enumerate(F['fid']):
            m = P['fid'] == int(f)
            if not np.any(m):
                continue
            rows.append(np.sqrt(P['sin2'][m]))
            keep.append(row)
        nmax = max(len(x) for x in rows)
        M = np.full((len(rows), nmax), np.nan)
        for i, x in enumerate(rows):
            M[i, :len(x)] = x
        out[k] = (M, np.array(keep, int))
    return out


def free_fit_frame(FV, SINM, stat='s2', fix=None, p0=(1.0, 1.0)):
    """Frame-level free-exponent fit.  The projection of a per-point law
    sigma_j = a v^p sin_j^q onto E[s_f^2] is exact:
        E[s_f^2] = s0^2 + a^2 v^(2p) mean_j(sin_j^(2q)).
    """
    keys = list(FV)
    sig, vv, MM, bag = [], [], [], []
    for i, k in enumerate(keys):
        M, rows = SINM[k]
        F = {c: FV[k][c][rows] for c in FV[k]}
        sig.append(np.sqrt(F['s2']) if stat == 's2' else F['srob'])
        vv.append(F['v'])
        MM.append(M)
        bag.append(np.full(len(rows), i))
    sig = np.concatenate(sig)
    vv = np.concatenate(vv)
    bag = np.concatenate(bag).astype(int)
    nmax = max(M.shape[1] for M in MM)
    MP = np.full((len(sig), nmax), np.nan)
    o = 0
    for M in MM:
        MP[o:o + M.shape[0], :M.shape[1]] = M
        o += M.shape[0]
    MASK = np.isfinite(MP)
    MPc = np.where(MASK, MP, 1.0)
    cnt = MASK.sum(1)

    def model(th, p, q):
        s0 = np.abs(th[bag]) ** 2
        mom = np.where(MASK, np.maximum(MPc, 1e-6) ** (2 * q), 0.0).sum(1) / cnt
        return np.sqrt(s0 + th[2] ** 2 * vv ** (2 * p) * mom)

    def obj(th, p, q):
        r = model(th, p, q) - sig
        fs = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-3)
        return float(np.sum(2 * fs ** 2 * (np.sqrt(1 + (r / fs) ** 2) - 1)))

    if fix is not None:
        f = lambda t: obj(t, fix[0], fix[1])
        th0 = np.r_[[np.median(sig)] * len(keys), 0.05]
        sol = minimize(f, th0, method='Nelder-Mead',
                       options=dict(maxiter=40000, maxfev=40000, xatol=1e-10,
                                    fatol=1e-10))
        return dict(theta=np.r_[np.abs(sol.x), fix], cost=f(sol.x), n=len(sig))
    g = lambda t: obj(t[:3], t[3], t[4])
    th0 = np.r_[[np.median(sig)] * len(keys), 0.05, p0[0], p0[1]]
    sol = minimize(g, th0, method='Nelder-Mead',
                   options=dict(maxiter=80000, maxfev=80000, xatol=1e-10,
                                fatol=1e-10))
    return dict(theta=np.r_[np.abs(sol.x[:3]), sol.x[3:]], cost=g(sol.x),
                n=len(sig))


def block2(tabs, n_boot=100, seed=0):
    print()
    print('=' * 78)
    print('BLOCK 2 -- FREE EXPONENTS.  sigma_j^2 = s0_bag^2 + a^2 v^(2p) sin^(2q).')
    print('      (p,q) = (1,1) is the bearing law, (2,0) the deployed quartic')
    print('      weight\'s law.  Neither ladder can express anything in between,')
    print('      and 1h showed neither whitens both directions.  The frame-level')
    print('      projection E[s_f^2] = s0^2 + a^2 v^2p mean_j(sin_j^2q) is exact,')
    print('      so the SAME (p,q) is estimable from both statistics.')
    print('=' * 78)
    PV = {k: point_view(tabs[k]) for k in RACING}
    FV = {k: frame_view(tabs[k]) for k in RACING}
    SINM = _frame_sin_matrix(PV, FV)

    fp = free_fit_point(PV)
    print(f'  PER-POINT (pairwise, no trim)   p = {fp["theta"][3]:.2f}  '
          f'q = {fp["theta"][4]:.2f}   a = {fp["theta"][2]:.4f}  '
          f'floors {fp["theta"][0]:.3f}/{fp["theta"][1]:.3f}   '
          f'2NLL {2*fp["nll"]:.1f}')
    for lab, fx in (('bearing (1,1)', (1.0, 1.0)), ('deployed (2,0)', (2.0, 0.0)),
                    ('v^2 sin  (2,1)', (2.0, 1.0)), ('sin only (0,1)', (0.0, 1.0))):
        f = free_fit_point(PV, fix=fx)
        print(f'      pinned {lab:<16} 2NLL {2*f["nll"]:>10.1f}  '
              f'(free - pinned = {2*fp["nll"]-2*f["nll"]:+.1f})')

    for stat, lab in (('s2', 'second moment'), ('srob', 'robust MAD')):
        ff = free_fit_frame(FV, SINM, stat=stat)
        print(f'  FRAME / {lab:<15}      p = {ff["theta"][3]:.2f}  '
              f'q = {ff["theta"][4]:.2f}   a = {ff["theta"][2]:.4f}  '
              f'floors {ff["theta"][0]:.3f}/{ff["theta"][1]:.3f}')
        for flab, fx in (('bearing (1,1)', (1.0, 1.0)),
                         ('deployed (2,0)', (2.0, 0.0))):
            f = free_fit_frame(FV, SINM, stat=stat, fix=fx)
            print(f'      pinned {flab:<16} cost {f["cost"]:>10.4f}  '
                  f'(free {ff["cost"]:.4f})')

    print()
    print(f'  bootstrap over frames ({n_boot} draws), 68% intervals on (p,q):')
    rng = np.random.default_rng(seed)
    for which in ('point', 'frame_s2', 'frame_srob'):
        P_, Q_ = [], []
        for _ in range(n_boot):
            pb, fb = {}, {}
            for k in RACING:
                fr = np.unique(PV[k]['fid'])
                pick = rng.choice(fr, len(fr), replace=True)
                idx = np.concatenate([np.flatnonzero(PV[k]['fid'] == f)
                                      for f in pick])
                pb[k] = {c: PV[k][c][idx] for c in PV[k]}
                # relabel frames so repeats stay distinct
                _, inv = np.unique(np.concatenate(
                    [np.full(int((PV[k]['fid'] == f).sum()), i)
                     for i, f in enumerate(pick)]), return_inverse=True)
                pb[k]['fid'] = np.concatenate(
                    [np.full(int((PV[k]['fid'] == f).sum()), i)
                     for i, f in enumerate(pick)]).astype(float)
                rows = np.array([np.flatnonzero(FV[k]['fid'] == f)[0]
                                 for f in pick])
                fb[k] = {c: FV[k][c][rows] for c in FV[k]}
                fb[k]['fid'] = np.arange(len(rows), dtype=float)
            try:
                if which == 'point':
                    t = free_fit_point(pb)['theta']
                else:
                    sm = _frame_sin_matrix(pb, fb)
                    t = free_fit_frame(
                        fb, sm, stat='s2' if which == 'frame_s2' else 'srob'
                    )['theta']
                P_.append(t[3]); Q_.append(t[4])
            except Exception:
                continue
        P_, Q_ = np.array(P_), np.array(Q_)
        if len(P_) < 5:
            print(f'    {which:<12} bootstrap failed')
            continue
        print(f'    {which:<12} p {np.median(P_):.2f} '
              f'[{np.percentile(P_,16):.2f},{np.percentile(P_,84):.2f}]   '
              f'q {np.median(Q_):.2f} '
              f'[{np.percentile(Q_,16):.2f},{np.percentile(Q_,84):.2f}]   '
              f'(n={len(P_)})')


def block2b(tabs):
    """WHY q > 1?  The residual is -du^T v, so a sin(theta) factor is forced by
    geometry no matter what the ray error is.  q > 1 therefore means the ray
    error dphi_j ITSELF grows with angle.  Two candidates, and they are
    confounded whenever the velocity points near boresight:

      (i)  dphi grows with the VELOCITY-RELATIVE angle theta  -> physically odd:
           the sensor cannot know where the platform is going;
      (ii) dphi grows with the SENSOR-FIXED off-boresight angle phi -> ordinary
           antenna/DOA behaviour, and calibratable.

    Separated here by stratifying on both at once.
    """
    print()
    print('=' * 78)
    print('2b -- IS THE EXTRA ANGLE DEPENDENCE VELOCITY-RELATIVE OR SENSOR-FIXED?')
    print('      sigma_j = (dphi_j/sqrt2) |v| sin(theta_j) is forced by geometry;')
    print('      q > 1 says dphi_j itself grows with angle.  theta = angle(ray,')
    print('      velocity); phi = angle(ray, sensor boresight +x).  If dphi')
    print('      tracks phi, this is an antenna-pattern effect and calibratable.')
    print('=' * 78)
    PVa = {kk: point_view(tabs[kk]) for kk in RACING}
    _fit = score_shortlist(PVa)[SHORTLIST[-1]]['theta']
    FLOOR = {RACING[0]: _fit[0], RACING[1]: _fit[1]}
    print(f'  floors from the ARRAY fit itself: '
          f'{FLOOR[RACING[0]]:.3f} / {FLOOR[RACING[1]]:.3f} m/s')
    for k in RACING:
        P = PVa[k]
        sin1 = np.sqrt(np.clip(P['sin2'], 1e-9, 1))
        phi = np.degrees(np.arccos(np.clip(P['sx'], -1, 1)))
        # dphi_j implied by the geometry-forced form, per point
        print(f'  {SHORT[k]}: boresight angle phi median {np.median(phi):.0f} deg '
              f'[{np.percentile(phi,5):.0f},{np.percentile(phi,95):.0f}], '
              f'corr(phi, theta) = '
              f'{np.corrcoef(phi, np.degrees(np.arcsin(sin1)))[0,1]:+.2f}')
        s_edges = [0.0, 0.7, 0.85, 0.95, 1.01]
        p_edges = [0, 35, 50, 62, 90]
        # floor-subtracted: at small sin the floor dominates sigma and an
        # unsubtracted ratio reports the FLOOR, not the ray error.
        s0 = FLOOR[k]
        print(f"    implied dphi (deg), floor {s0:.3f} m/s removed in quadrature;"
              f" rows sin(theta), cols phi")
        print('      sin \\ phi   ' + ''.join(
            f'{str(p_edges[i]) + "-" + str(p_edges[i + 1]):>14}'
            for i in range(len(p_edges) - 1)))
        for lo, hi in zip(s_edges[:-1], s_edges[1:]):
            cells = []
            for pl, ph in zip(p_edges[:-1], p_edges[1:]):
                m = ((sin1 >= lo) & (sin1 < hi) & (phi >= pl) & (phi < ph))
                if m.sum() < 60:
                    cells.append(f'{"-":>14}')
                    continue
                sg = robust_sigma(P['d'][m])
                ex = np.sqrt(max(sg ** 2 - s0 ** 2, 0.0))
                vs_ = np.median(P['v'][m] * sin1[m])
                dphi = np.degrees(np.sqrt(2) * ex / max(vs_, 1e-6))
                cells.append(f'{dphi:6.1f} (n={m.sum():4d})')
            print(f'      {lo:.2f}-{hi:<5.2f}' + ''.join(cells))
    print('    Read across a row: does dphi grow with phi at fixed theta?')
    print('    Read down a column: does dphi grow with theta at fixed phi?')


# ===========================================================================
# BLOCK 3 -- confounders.  Every candidate is re-scored under each data
# transformation on a COMMON pair set with no adaptive trim.
# ===========================================================================
# ARRAY stays LAST: blocks 2b/4 index SHORTLIST[-1] as "the" model.  DIRCOS is
# the same physics with nothing left to choose, carried through every control
# and every held-out flight so the two can be compared on equal footing.
SHORTLIST = ['floors only               ',
             'v^2          (5b deployed)',
             'v sin        (BEARING)    ',
             'v sin * phi^2(antenna)   ',
             'DIRCOS exact (derived)   ',
             'v sin / cos(phi) (ARRAY) ']


def score_shortlist(PV, seed=1):
    keys = list(PV)
    pairs = build_pairs(PV, seed=seed)
    out = {}
    for name in SHORTLIST:
        Xs = {k: pt_predictors(PV[k])[name][0] for k in keys}
        f = pair_fit2(pairs, Xs, keys, trim_z=None)
        out[name] = dict(nll2=2 * f['nll'], theta=f['theta'])
    return out


def _ransac_mask_frame(P3, v, rng, thresh=0.15, iters=150):
    """Deployed reve-style 3D-LSQ RANSAC prefilter (verbatim protocol)."""
    n = len(v)
    if n < 5:
        return np.ones(n, bool)
    Hn = P3 / np.maximum(np.linalg.norm(P3, axis=1, keepdims=True), 1e-6)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            vc = np.linalg.solve(Hn[idx], v[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(Hn @ vc - v) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    return best if best is not None and best.sum() >= 5 else np.ones(n, bool)


def _remove_coherent_eps(P, weights=None):
    """Fit ONE constant mounting rotation eps per bag on the within-frame
    deviations and subtract it.  A constant eps produces r_j = -eps.(u_j x v),
    i.e. exactly the |v| sin(theta) law but COHERENT -- so it is the direct
    calibration confound for the bearing family.  It carries no phi structure."""
    u = np.column_stack([P['ux'], P['uy'], P['uz']])
    vb = np.column_stack([P['vbx'], P['vby'], P['vbz']])
    X = -np.cross(u, vb)                      # d r / d eps, per return
    for f in np.unique(P['fid']):             # centre within frame, so the fit
        m = P['fid'] == f                     # sees only what the statistic sees
        X[m] -= X[m].mean(axis=0)
    w = np.ones(len(P['d'])) if weights is None else 1.0 / np.maximum(weights, 1e-6)
    eps, *_ = np.linalg.lstsq(X * w[:, None], P['d'] * w, rcond=None)
    d_new = P['d'] - X @ eps
    return dict(P, d=d_new), eps


def _remove_3dof(P, rng, split_half=True):
    """Remove a full per-frame 3-DoF velocity error by least squares.  A
    reference VELOCITY error enters as r_j = -u_j^T dv, so the frame-mean
    subtraction removes only 1 of its 3 DoF.  Fitted on a random half of each
    frame and applied to the other half, so the drop is not self-fitted."""
    u = np.column_stack([P['ux'], P['uy'], P['uz']])
    keep = np.zeros(len(P['d']), bool)
    d_new = P['d'].copy()
    for f in np.unique(P['fid']):
        m = np.flatnonzero(P['fid'] == f)
        n = len(m)
        if n < 10:
            continue
        perm = rng.permutation(n)
        a, b = m[perm[:n // 2]], m[perm[n // 2:]]
        if len(a) < 4 or len(b) < 4:
            continue
        if not split_half:
            a = b = m
        sol, *_ = np.linalg.lstsq(u[a], P['d'][a], rcond=None)
        d_new[b] = P['d'][b] - u[b] @ sol
        keep[b] = True
    sub = {c: (d_new[keep] if c == 'd' else P[c][keep]) for c in P}
    return sub


def block3(tabs, seed=0):
    print()
    print('=' * 78)
    print('BLOCK 3 -- CONFOUNDERS.  Each row re-derives the per-point statistic')
    print('      under one transformation and re-scores every candidate on a')
    print('      common pair set with NO adaptive trim (lower 2NLL = better).')
    print('      The question is only whether the ARRAY law keeps its margin.')
    print('=' * 78)
    rng = np.random.default_rng(seed)
    base = {k: point_view(tabs[k]) for k in RACING}

    variants = {}
    variants['baseline (published)'] = base

    eps_txt = []
    ve = {}
    for k in RACING:
        Pe, eps = _remove_coherent_eps(base[k])
        ve[k] = Pe
        eps_txt.append(f'{SHORT[k]} eps=({np.degrees(eps[0]):+.2f},'
                       f'{np.degrees(eps[1]):+.2f},{np.degrees(eps[2]):+.2f}) deg')
    variants['coherent eps removed'] = ve

    variants['3-DoF ref removed (split-half)'] = {
        k: _remove_3dof(base[k], np.random.default_rng(1)) for k in RACING}

    variants['no alias cut'] = {k: point_view(tabs[k], alias_cut=False)
                                for k in RACING}

    rmask = {}
    for k in RACING:
        T = tabs[k]
        r_ = np.random.default_rng(0)
        keep = np.ones(len(T['r']), bool)
        for f in np.unique(T['fid']):
            m = np.flatnonzero(T['fid'] == f)
            P3 = np.column_stack([T['sx'][m], T['sy'][m], T['sz'][m]]) \
                * T['rho'][m][:, None]
            keep[m] = _ransac_mask_frame(P3, T['unw'][m], r_)
        rmask[k] = keep
    variants['RANSAC prefilter (deployed)'] = {
        k: point_view(tabs[k], extra_keep=rmask[k]) for k in RACING}

    variants['sin(theta) > 0.4'] = {}
    variants['rho > 1 m'] = {}
    for k in RACING:
        P = base[k]
        m = P['sin2'] > 0.16
        variants['sin(theta) > 0.4'][k] = {c: P[c][m] for c in P}
        m = P['rho'] > 1.0
        variants['rho > 1 m'][k] = {c: P[c][m] for c in P}

    print('  ' + '; '.join(eps_txt))
    print()
    hdr = f"{'transformation':<32} {'n pts':>7} "
    for nm in SHORTLIST:
        hdr += f'{nm.split("(")[0].strip()[:11]:>12}'
    print(hdr)
    for vn, VV in variants.items():
        try:
            sc = score_shortlist(VV)
        except Exception as e:
            print(f'  {vn:<32} FAILED {e}')
            continue
        n = sum(len(VV[k]['d']) for k in VV)
        line = f'  {vn:<32} {n:>7d} '
        for nm in SHORTLIST:
            line += f'{sc[nm]["nll2"]:>12.0f}'
        arr = sc['v sin / cos(phi) (ARRAY) ']['theta'][2]
        bea = sc['v sin        (BEARING)    ']['theta'][2]
        print(line + f'   dphi0 {np.degrees(arr*np.sqrt(2)):.2f} deg '
              f'(flat-bearing {np.degrees(bea*np.sqrt(2)):.2f})')
    print('  Lower is better; all rows are internally comparable, rows are not')
    print('  comparable to each other (different n).  What matters is that the')
    print('  ARRAY column stays the minimum in every row.')


# ===========================================================================
# BLOCK 4 -- transfer with frozen parameters
# ===========================================================================
def transfer_score(P, coef, seed=1):
    """Freeze the response coefficient, refit ONLY the floor, score by the
    pairwise likelihood on this bag's own pairs."""
    PV = {'x': P}
    pairs = build_pairs(PV, seed=seed)
    D = pairs['D']
    out = {}
    for name in SHORTLIST:
        X = pt_predictors(P)[name][0]
        XI, XJ = _pair_X(pairs, {'x': X}, ['x'])
        c = coef.get(name, 0.0)

        def nll(t):
            var = np.maximum(2 * t[0] ** 2 + c ** 2 * (XI[:, 0] ** 2
                                                       + XJ[:, 0] ** 2), 1e-9)
            return float(np.sum(D ** 2 / var + np.log(var)))

        sol = minimize(nll, [robust_sigma(D) / np.sqrt(2)],
                       method='Nelder-Mead',
                       options=dict(maxiter=20000, maxfev=20000, xatol=1e-10,
                                    fatol=1e-9))
        # locally refit coefficient, for the "does the constant transfer" read
        def nll2(t):
            var = np.maximum(2 * t[0] ** 2 + t[1] ** 2 * (XI[:, 0] ** 2
                                                          + XJ[:, 0] ** 2), 1e-9)
            return float(np.sum(D ** 2 / var + np.log(var)))
        sol2 = minimize(nll2, [robust_sigma(D) / np.sqrt(2), max(c, 1e-3)],
                        method='Nelder-Mead',
                        options=dict(maxiter=40000, maxfev=40000, xatol=1e-10,
                                     fatol=1e-9))
        out[name] = dict(frozen=2 * nll(sol.x), local=2 * nll2(sol2.x),
                         floor=abs(sol.x[0]), coef_local=abs(sol2.x[1]),
                         n=len(D))
    return out


def block_deploy(tabs):
    """The constants a solver may deploy, and the ONE fit they come from.

    The deployed --speed-weight uses s0 = 0.082, the geometric mean of the
    QUADRATIC fit's per-bag floors.  Pairing that s0 with a coefficient fitted
    under a different noise model is a mixed-provenance constant: s0 is what
    sets the speed at which down-weighting begins, and the floors move a lot
    between models.  So every candidate's (s0, coef) pair is printed from its
    OWN fit, and nothing else may supply them.

    Fit: per-point pairwise composite likelihood, no adaptive trim (so every
    candidate is scored on identical pairs), both racing bags jointly, per-bag
    floor + one shared response.  theta = [s0_slow, s0_fast, coef...].
    """
    print()
    print('=' * 78)
    print('DEPLOY -- constants for the solver, each pair from a SINGLE fit.')
    print('  w_j = 1 / (1 + (coef * X_j / s0)^2),  X_j the predictor below.')
    print('  s0 = geometric mean of the per-bag floors OF THAT SAME FIT.')
    print('=' * 78)
    PV = {k: point_view(tabs[k]) for k in RACING}
    pairs = build_pairs(PV)
    rows = [('v^2          (5b deployed)', 'v^2, the RETRACTED shape'),
            ('v sin        (BEARING)    ', 'flat bearing'),
            ('v sin / cos(phi) (ARRAY) ', 'array, sin(theta)/cos(phi) spelling'),
            ('DIRCOS exact (derived)   ', 'DIRCOS, the derived form'),
            ('DIRCOS aniso (az,el)     ', 'DIRCOS split per array axis')]
    print(f"{'candidate':<38} {'s0 slow/fast':>14} {'s0 geomean':>11} "
          f"{'coef':>18} {'dphi_0 deg':>11}")
    out = {}
    for name, label in rows:
        Xs = {k: pt_predictors(PV[k])[name][0] for k in RACING}
        th = pair_fit2(pairs, Xs, RACING, trim_z=None)['theta']
        s0 = float(np.sqrt(th[0] * th[1]))
        cf = th[2:]
        deg = '/'.join(f'{np.degrees(c*np.sqrt(2)):.2f}' for c in cf)
        out[name] = (s0, cf)
        print(f'{label:<38} {th[0]:>6.4f}/{th[1]:<7.4f} {s0:>11.4f} '
              + '/'.join(f'{c:.5f}' for c in cf).rjust(18)
              + f' {deg:>11}')
    print()
    print('  For reference, the currently deployed pair is s0 = 0.082 with')
    print('  v_0 = 2.81 (i.e. coef = s0/v_0^2 = 0.0104): the s0 comes from the')
    print('  quadratic fit above, so it is at least self-consistent -- but the')
    print('  shape it belongs to is the one this file retracts.')
    print()
    print('  Variant A (no velocity-vector plumbing) uses the ARRAY pair with')
    print('  sin(theta_j) -> 1, i.e. w_j = 1/(1 + (a|v|/(s0 cos phi_j))^2).')
    print('  Variant B uses the DIRCOS pair directly on q_j (see the module')
    print('  docstring of _dircos for the predictor).')

    # ---- what the weight COSTS, which the gate battery measured end to end --
    s0, (su,) = out['DIRCOS exact (derived)   ']
    print()
    print('-' * 78)
    print('  WHAT THE WEIGHT COSTS.  The law down-weights WIDE-ANGLE returns,')
    print('  which thins a frame\'s angular diversity: M = sum_j w_j u_j u_j^T /')
    print('  sum_j w_j is the frame\'s ray-direction scatter, and a larger')
    print('  cond(M) means the frame constrains one direction far worse than')
    print('  another.  The table below measures that thinning.')
    print()
    print('  RETRACTED 2026-08-07 -- this is NOT the mechanism behind the gate')
    print('  battery\'s orientation cost, and an earlier version of this box said')
    print('  it was.  A single radar frame\'s Jacobian has rank 3 of 9 (every')
    print('  parameter enters through one 3-vector), so one frame carries NO')
    print('  attitude information and a per-frame statistic cannot reach the')
    print('  orientation channel even in principle.  Rotation is identified only')
    print('  ACROSS frames.  Three further candidates have since been tested and')
    print('  died; the cost remains unexplained.  See the sec. 11 ledger and')
    print('  worklog/2026-08-06_bearing-weight-battery.md.')
    print('-' * 78)
    print(f"  {'bag':<12} {'cond(M) unwtd':>14} {'weighted':>10} {'ratio':>7} "
          f"{'N_eff/N':>8} {'mean w, phi<40':>15} {'phi>60':>8}")
    for k in tabs:
        T = tabs[k]
        cu, cw, ne, wlo, whi = [], [], [], [], []
        for f in np.unique(T['fid']):
            m = (T['fid'] == f) & T['keep6'] & ~T['alias']
            n = int(m.sum())
            if n < 6:
                continue
            s = np.column_stack([T['sx'][m], T['sy'][m], T['sz'][m]])
            cphi = np.maximum(s[:, 0], 0.15)
            v = T['f_vs3'][f]
            q = np.sqrt((v[1] - s[:, 1] * v[0] / cphi) ** 2
                        + (v[2] - s[:, 2] * v[0] / cphi) ** 2)
            w = 1.0 / (1.0 + (su * q / s0) ** 2)
            cu.append(np.linalg.cond(s.T @ s / n))
            cw.append(np.linalg.cond((s * w[:, None]).T @ s / w.sum()))
            ne.append(w.sum() ** 2 / np.sum(w ** 2) / n)
            ph = np.degrees(np.arccos(np.clip(s[:, 0], -1, 1)))
            if (ph < 40).any():
                wlo.append(w[ph < 40].mean())
            if (ph > 60).any():
                whi.append(w[ph > 60].mean())
        if not cu:
            continue
        print(f'  {SHORT.get(k, k):<12} {np.median(cu):>14.2f} '
              f'{np.median(cw):>10.2f} {np.median(cw)/np.median(cu):>7.2f} '
              f'{np.median(ne):>8.2f} {np.mean(wlo):>15.3f} '
              f'{np.mean(whi):>8.3f}')
    print()
    print('  N_eff/N stays near 0.88, so the weight is not simply discarding')
    print('  returns -- it is discarding DIRECTIONS.  Note backflips starts')
    print('  worst-conditioned by 3x, i.e. it is already geometry-starved,')
    print('  which is where the measured orientation regression is largest.')
    return out


def block_sigmac(tabs):
    """sigma_c as an ABSOLUTE scale, on the stream the solver actually consumes.

    Why this block exists.  The deployed constant `radar_frame_sigma_c = 0.4` is
    a RATIO to sigma_i, and the legacy whitening operator uses it that way
    (gamma = 1 - 1/sqrt(1 + n sigma_c^2), dimensionless, i.e. sigma_i = 1).  Once
    8b measures sigma_j PER POINT that ratio is not well defined: there is no
    single sigma_i for a frame to be a ratio of.  The physical model is

        Cov(r_f) = diag(sigma_j^2) + sigma_c^2 11^T

    with c_f one scalar per frame, so sigma_c is an ABSOLUTE scale.  The
    heteroscedastic operator (radar_frame_hetero=1) assumes exactly that, and
    needs the constant in units of the per-point floor s0: sigma_c / s0.

    Two things therefore have to be measured rather than converted: sigma_c in
    m/s, and on the POST-PREFILTER stream, because the solver consumes the
    RANSAC-kept returns and the published ratio was measured on the raw ones.
    (The ratio moved 0.26-0.53 -> 0.5-1.0 between those streams; the question
    this block answers is how much of that was sigma_c and how much sigma_i.)
    """
    print()
    print('=' * 78)
    print('SIGMA_C -- absolute, per stream.  sigma_i^2 = mean within-frame var;')
    print('  sigma_c^2 = var_f(frame mean) - sigma_i^2 / Nbar  (the estimator of')
    print('  8c, unchanged).  s0 = 0.083 m/s is the adopted law\'s per-point')
    print('  floor (--block deploy, DIRCOS).  The constant the heteroscedastic')
    print('  operator needs is sigma_c / s0, NOT sigma_c / sigma_i.')
    print('=' * 78)
    S0 = 0.0827
    rng = np.random.default_rng(0)
    print(f"  {'bag':<12} {'stream':<16} {'frames':>7} {'Nbar':>6} "
          f"{'sigma_i':>8} {'sigma_c':>8} {'ratio':>7} {'sigma_c/s0':>11}")
    acc = {}
    for k in tabs:
        T = tabs[k]
        base = T['keep6']
        P3 = np.column_stack([T['sx'], T['sy'], T['sz']]) * T['rho'][:, None]
        streams = [('raw (6-sigma only)', base),
                   ('alias-cut', base & ~T['alias'])]
        # the deployed front end: RANSAC on each frame's own returns
        rmask = np.zeros(len(T['r']), bool)
        for f in np.unique(T['fid']):
            m = T['fid'] == f
            if m.sum() < 5:
                rmask[m] = True
                continue
            rmask[m] = _ransac_mask_frame(P3[m], T['unw'][m], rng)
        streams.append(('RANSAC (deployed)', base & rmask))
        for label, keep in streams:
            wv, mu, ns = [], [], []
            for f in np.unique(T['fid']):
                m = (T['fid'] == f) & keep
                n = int(m.sum())
                if n < 6:
                    continue
                wv.append(np.var(T['r'][m], ddof=1))
                mu.append(np.mean(T['r'][m]))
                ns.append(n)
            if len(mu) < 8:
                continue
            s_i2 = float(np.mean(wv))
            s_c2 = max(float(np.var(mu, ddof=1)) - s_i2 / float(np.mean(ns)), 0.0)
            s_i, s_c = np.sqrt(s_i2), np.sqrt(s_c2)
            acc.setdefault(label, []).append(s_c)
            print(f'  {SHORT.get(k, k):<12} {label:<16} {len(mu):>7d} '
                  f'{np.mean(ns):>6.1f} {s_i:>8.3f} {s_c:>8.3f} '
                  f'{s_c/max(s_i,1e-9):>7.2f} {s_c/S0:>11.2f}')
    print()
    for label, v in acc.items():
        gm = float(np.exp(np.mean(np.log(np.maximum(v, 1e-9)))))
        print(f'  geomean sigma_c over bags, {label:<18} {gm:.3f} m/s'
              f'   -> sigma_c/s0 = {gm/S0:.2f}')
    print()
    print('  Read this against the deployed 0.4.  If sigma_c/s0 on the RANSAC')
    print('  stream is far from 0.4, the heteroscedastic operator cannot be')
    print('  adopted at the deployed constant, and its neutral gate battery was')
    print('  run at the wrong value.')
    print()
    print('  CAVEAT (2026-08-08, added by --block sigmacw): dividing by s0 is')
    print('  only right if the solver\'s per-point weights normalise to w = 1 at')
    print('  sigma = s0.  The DEPLOYED weights do not -- w_int is taken relative')
    print('  to each frame\'s MEDIAN intensity, so w = 1 is a median return, not')
    print('  a boresight one.  `--block sigmacw` measures the ratio the operator')
    print('  actually consumes, per weight channel, and supersedes the last')
    print('  column here.')
    return acc


# ---------------------------------------------------------------------------
# The constant the whitening operator actually consumes
# ---------------------------------------------------------------------------
# radar_frame_sigma_c is NOT an absolute scale and NOT a fixed ratio to s0.
# The factor row-scales each residual by sqrt(w_j) BEFORE whitening
# (sliding_window_solver.cpp:453) and then whitens Sigma = I + sigma_c^2 v v^T
# with v_j = sqrt(w_j).  The identity coefficient is 1, so the operator asserts
# Var(sqrt(w_j) r_j) = 1.  Writing the true covariance of the row-scaled stack,
#
#     Cov(r') = sigma_0^2 I + sigma_c^2 v v^T
#             = sigma_0^2 [ I + (sigma_c / sigma_0)^2 v v^T ],
#
# and noting that the leading sigma_0^2 is absorbed by the frame ScaledLoss
# (s_frame = w_omega * radar_weight), the config key is exactly
#
#     radar_frame_sigma_c = sigma_c / sigma_0,     sigma_0^2 = Var(sqrt(w_j) r_j)
#
# in BOTH branches.  `hetero` changes the rank-one term's DIRECTION (1 -> v),
# not its units.  So the right value is a property of the radar AND of the
# weight channel: change the weights and the constant moves, because sigma_0
# is whatever those weights normalise to.  That is what this block measures --
# no normalisation argument, just the ratio itself, per channel.
#
# Estimator (reduces exactly to block_sigmac's at w == 1).  Per frame, with
# x = v .* r, P = I - v v^T / S, S = sum_j w_j:
#     sigma_0,f^2 = ||P x||^2 / (n-1) = (sum w_j r_j^2 - (sum w_j r_j)^2 / S)/(n-1)
#     t_f         = (sum_j w_j r_j) / S            -- weighted frame mean, m/s
#     Var(t_f)    = sigma_c^2 + sigma_0^2 / S      -- hence sigma_c below.
_SIGMACW_S0, _SIGMACW_COEF = 0.0827, 0.03291   # DIRCOS, --block deploy
_SIGMACW_COS_FLOOR = 0.15                      # validate_live_solver _BW_COS_FLOOR
_SIGMACW_AL = (0.095, 4.5, 0.6)                # sigma0, omega0, sigma_al


def _weight_channel(T, name):
    """Per-return weight w_j exactly as the solver forms it, for one channel.

    'none'      w = 1                       (reproduces block_sigmac)
    'intensity' w_int only                  (isolates the intensity factor)
    'alias'     alias rule only            (the shipping default since
                                            2026-08-09, when the SNR weight
                                            was deleted)
    'deployed'  w_int * alias rule          (what shipped BEFORE 2026-08-09;
                                            kept because the published
                                            sigma_c/sigma_0 = 0.81 is on it)
    'bearing'   w_int * (bearing law on clean returns, alias rule on aliased)
                                            (what --bearing-weight b would run)

    w_int is relative to each FRAME's median intensity (sliding_window_solver
    .cpp:413-423), which is the whole point: it makes w = 1 a median return.
    """
    n = len(T['r'])
    if name == 'none':
        return np.ones(n)
    w_int = np.ones(n)
    inten, fid = T['inten'], T['fid']
    for f in np.unique(fid):
        m = fid == f
        pos = inten[m] > 0
        if pos.sum() < 3:
            continue
        med = np.median(inten[m][pos])
        if med > 0:
            w_int[m] = np.where(pos, inten[m] / med, 1.0)
    if name == 'intensity':
        return w_int
    s0_al, om0_al, sal = _SIGMACW_AL
    om = T['f_om'][fid]
    w_al = np.clip(s0_al ** 2 * (1.0 + (om / om0_al) ** 2) / sal ** 2, 0.01, 1.0)
    if name == 'alias':                       # deployed since 2026-08-09
        return np.where(T['alias'], w_al, 1.0)
    w = w_int * np.where(T['alias'], w_al, 1.0)
    if name == 'deployed':
        return w
    if name not in ('bearing', 'bearing_noint'):
        raise ValueError(name)
    s = np.column_stack([T['sx'], T['sy'], T['sz']])
    cphi = np.maximum(s[:, 0], _SIGMACW_COS_FLOOR)
    vv = T['f_va3'][fid]                       # sensor-frame v_ant (MoCap)
    X = np.sqrt((vv[:, 1] - s[:, 1] * vv[:, 0] / cphi) ** 2
                + (vv[:, 2] - s[:, 2] * vv[:, 0] / cphi) ** 2)
    w_bear = 1.0 / (1.0 + (_SIGMACW_COEF * X / _SIGMACW_S0) ** 2)
    if name == 'bearing_noint':
        # what --bearing-weight actually runs since the SNR-weight deletion
        # (2026-08-09): no intensity factor.  The published 1.24 is on the
        # OLD int-included channel above.
        return np.where(T['alias'], w_al, w_bear)
    return w_int * np.where(T['alias'], w_al, w_bear)


def _sigmacw_fit(r, w, fid, huber=None):
    """(sigma_0, sigma_c, ratio, nframes, mean_n, mean_w) for one population."""
    if huber is not None and huber > 0:
        w = w * np.where(np.abs(r) > huber, huber / np.maximum(np.abs(r), 1e-9), 1.0)
    s0f, tf, Sf, ns = [], [], [], []
    for f in np.unique(fid):
        m = fid == f
        n = int(m.sum())
        if n < 6:
            continue
        rw, ww = r[m], w[m]
        S = float(ww.sum())
        if S <= 0:
            continue
        swr = float((ww * rw).sum())
        ss = float((ww * rw * rw).sum()) - swr * swr / S
        s0f.append(max(ss, 0.0) / (n - 1))
        tf.append(swr / S)
        Sf.append(S)
        ns.append(n)
    if len(tf) < 8:
        return None
    s0_2 = float(np.mean(s0f))
    s_c2 = max(float(np.var(tf, ddof=1)) - float(np.mean(np.array(s0f) / np.array(Sf))), 0.0)
    s_0, s_c = np.sqrt(s0_2), np.sqrt(s_c2)
    return s_0, s_c, s_c / max(s_0, 1e-9), len(tf), float(np.mean(ns)), float(np.mean(w))


def block_sigmacw(tabs):
    """radar_frame_sigma_c measured per weight channel, on the deployed stream.

    The prediction being tested (from reading the solver, so it is an inference
    until this block runs): the constant tracks whatever the weight channel
    normalises to.  Deployed weights normalise to a frame-MEDIAN return, so the
    ratio should land near sigma_c/sigma_i ~ 0.5; the bearing weight normalises
    to the floor s0 (w = 1/(1 + (coef X / s0)^2) is 1 at X = 0), so its ratio
    should land near sigma_c/s0 ~ 1.2.  If instead both channels give the same
    number, the reading is wrong and the constant IS channel-independent.
    """
    print()
    print('=' * 78)
    print('SIGMA_C AS THE OPERATOR CONSUMES IT.  r\'_j = sqrt(w_j) r_j, then')
    print('  Cov(r\') = sigma_0^2 I + sigma_c^2 v v^T,  v_j = sqrt(w_j).  The')
    print('  config key radar_frame_sigma_c is the RATIO sigma_c/sigma_0, in')
    print('  both hetero branches -- hetero changes the rank-one DIRECTION')
    print('  (1 -> v), not the units.  So it is a property of the radar AND of')
    print('  the weight channel.  Deployed constant: 0.4.')
    print()
    print('  The deployed prefilter is RANSAC, i.e. STOCHASTIC, and the kept set')
    print('  moves the estimate: on fast racing a single seed gives anywhere in')
    print('  0.74-1.15.  So every row is a MEDIAN over NSEED prefilter seeds with')
    print('  the full seed range beside it, and each bag is seeded from 0')
    print('  independently (a single RNG walked across bags would make the answer')
    print('  depend on bag ORDER).')
    print('=' * 78)
    NSEED = 6
    CH = ['none', 'intensity', 'alias', 'deployed', 'bearing', 'bearing_noint']
    # The channel NAMES are historical; what ships today is the alias rule
    # alone, because the SNR weight was deleted 2026-08-09.  Label accordingly
    # so the table cannot be read as "0.81 is the deployed constant".
    CH_LABEL = {'none': 'none', 'intensity': 'intensity',
                'alias': 'alias = DEPLOYED', 'deployed': 'int x alias (old)',
                'bearing': 'int x bearing (old)',
                'bearing_noint': 'bearing = --bearing-weight'}
    # sigma_0 scales exactly with a global weight level (sigma_0^2 -> c sigma_0^2
    # for w -> c w), so comparing raw sigma_0 ACROSS channels measures the level,
    # not the weighting.  `s0/lvl` = sigma_0 / sqrt(mean w) removes it and is the
    # only column that answers "does this weight reduce variance".  The RATIO
    # column is deliberately NOT normalised: it is level-dependent by design,
    # because the operator's sigma_0 is whatever the weights produce.
    print(f"  {'bag':<12} {'channel':<18} {'frames':>7} {'Nbar':>6} {'mean w':>7} "
          f"{'sigma_0':>8} {'s0/lvl':>8} {'sigma_c':>8} {'RATIO':>7} {'seed range':>13}")
    acc = {c: [] for c in CH}
    for k in tabs:
        T = tabs[k]
        P3 = np.column_stack([T['sx'], T['sy'], T['sz']]) * T['rho'][:, None]
        W = {c: _weight_channel(T, c) for c in CH}
        runs = {c: [] for c in CH}
        for seed in range(NSEED):
            rng = np.random.default_rng(seed)
            keep = T['keep6'].copy()
            for f in np.unique(T['fid']):             # deployed front end
                m = T['fid'] == f
                if m.sum() < 5:
                    continue
                keep[m] &= _ransac_mask_frame(P3[m], T['unw'][m], rng)
            r, fid = T['r'][keep], T['fid'][keep]
            for ch in CH:
                out = _sigmacw_fit(r, W[ch][keep], fid, huber=1.0)
                if out is not None:
                    runs[ch].append(out)
        for ch in CH:
            if not runs[ch]:
                continue
            A = np.array(runs[ch], float)   # (seed, [s0 sc ratio nf nbar meanw])
            med = np.median(A, axis=0)
            lo, hi = A[:, 2].min(), A[:, 2].max()
            acc[ch].append(float(med[2]))
            s0_lvl = med[0] / np.sqrt(max(med[5], 1e-9))
            print(f'  {SHORT.get(k, k):<12} {CH_LABEL[ch]:<18} {med[3]:>7.0f} {med[4]:>6.1f} '
                  f'{med[5]:>7.3f} {med[0]:>8.3f} {s0_lvl:>8.3f} {med[1]:>8.3f} '
                  f'{med[2]:>7.2f} {lo:>6.2f}-{hi:<6.2f}')
    print()
    for ch in CH:
        if not acc[ch]:
            continue
        gm = float(np.exp(np.mean(np.log(np.maximum(acc[ch], 1e-9)))))
        print(f'  geomean RATIO over bags, channel {CH_LABEL[ch]:<18} {gm:.2f}')
    print()
    print('  How to read it.  The `none` row must reproduce block_sigmac\'s')
    print('  sigma_c/sigma_i column (same estimator at w = 1) -- that is the')
    print('  self-check.  The `alias = DEPLOYED` row is the constant to gate')
    print('  radar_frame_hetero at TODAY (0.84); `int x alias (old)` is the')
    print('  channel that shipped before the 2026-08-09 SNR deletion, kept')
    print('  because the published 0.81 is on it.  The `bearing` row is the')
    print('  constant the same operator would need if --bearing-weight were')
    print('  adopted: they must be re-derived together, never inherited.')
    return acc


def block_shared_time(tabs):
    """Is the frame-shared error c_f independent FRAME TO FRAME?  The model says yes.

    8c/8d model one frame as  r_fj = c_f + e_fj  with c_f a scalar drawn afresh
    per frame -- that independence is what makes the whitening a per-frame
    operation at all.  It has never been tested, and it is testable directly:
    the frame-mean residual is c_f plus a white term of variance sigma_i^2/N, so
    its lag-k autocorrelation, divided by the attenuation
    sigma_c^2/(sigma_c^2 + sigma_i^2/N), estimates the autocorrelation of c_f.

    Independent c_f => rho_1 = 0.  Anything else is a misspecification, and it
    is one the per-frame operator cannot represent no matter what sigma_c is.

    Note this is NOT independent of 8e: `corr_time_irr` already measures a
    correlation time for the whole radar residual stream (353/78 ms on the two
    racing flights) and folds it into lambda* = 1/(sigma^2 f_s tau).  What that
    scalar cannot say is WHERE the correlation lives.  This block says it lives
    in the frame-shared direction, which is exactly the direction the sigma_c
    whitening assumes is refreshed every frame.
    """
    print()
    print('=' * 78)
    print('IS THE FRAME-SHARED ERROR WHITE ACROSS FRAMES?  Autocorrelation of')
    print('  the frame-mean residual, corrected for the white per-point term')
    print('  that dilutes it.  The sigma_c model assumes rho_1 = 0.')
    print('=' * 78)
    print(f"  {'flight':<12} {'Nbar':>5} {'dt_ms':>6} {'atten':>6} " +
          ''.join(f'{"rho_" + str(l):>7}' for l in (1, 2, 3, 5, 10)))
    for k in tabs:
        T = tabs[k]
        P3 = np.column_stack([T['sx'], T['sy'], T['sz']]) * T['rho'][:, None]
        rng = np.random.default_rng(0)
        keep = T['keep6'].copy()
        for f in np.unique(T['fid']):
            m = T['fid'] == f
            if m.sum() < 5:
                continue
            keep[m] &= _ransac_mask_frame(P3[m], T['unw'][m], rng)
        r, fid, ft = T['r'][keep], T['fid'][keep], T['f_t']
        fs = [f for f in np.unique(fid) if (fid == f).sum() >= 6]
        if len(fs) < 40:
            continue
        mu = np.array([r[fid == f].mean() for f in fs])
        ns = np.array([(fid == f).sum() for f in fs], float)
        tt = np.array([ft[f] for f in fs])
        si2 = float(np.mean([np.var(r[fid == f], ddof=1) for f in fs]))
        sc2 = max(float(np.var(mu, ddof=1)) - si2 / ns.mean(), 0.0)
        att = sc2 / max(sc2 + si2 / ns.mean(), 1e-12)
        dt = float(np.median(np.diff(tt)))
        x = mu - mu.mean()
        v = float(np.mean(x * x))
        out = []
        for lag in (1, 2, 3, 5, 10):
            gap = tt[lag:] - tt[:-lag]          # lag-k gap, NOT np.diff(n=k)
            ok = np.abs(gap - lag * dt) < 0.5 * dt
            a, b = x[:-lag][ok], x[lag:][ok]
            out.append(float(np.mean(a * b)) / v / att if len(a) > 20 else np.nan)
        print(f'  {SHORT.get(k, k)[:12]:<12} {ns.mean():>5.1f} {1000*dt:>6.0f} '
              f'{att:>6.2f} ' + ''.join(f'{q:>7.2f}' for q in out))
    print()
    print('  Read: rho_1 = 0 would mean the per-frame model is right.  Measured')
    print('  0.16-0.87, decaying over ~0.2-0.5 s, i.e. the shared error persists')
    print('  across several frames.  So `one independent scalar per frame` is')
    print('  wrong, and no choice of sigma_c fixes it -- the operator has no')
    print('  state to carry a correlated offset between frames.  Reported as a')
    print('  measured limitation of the adopted model, with no mechanism')
    print('  attached: a slow reference error and a slowly-varying sensor or')
    print('  calibration error both produce it, and this statistic cannot')
    print('  separate them.')


def block_cfbias(tabs):
    """Is the frame-shared error c_f a toward-boresight bearing BIAS?

    Literature-named candidate (Babgei 2026, arXiv:2607.26980, measured on an
    AWR6843AOP: "under sidelobe leakage the per-cell angle-FFT direction
    estimate is displaced toward boresight, so the unit vectors under-represent
    the lateral projection").  A bearing BIAS -- unlike the zero-mean bearing
    noise of 8b -- is common to the frame's geometry, speed-scaled and
    temporally smooth: exactly c_f's signature (--block ctime).

    Test, derived from the mechanism: a bias delta toward boresight shifts each
    ray's azimuth alpha_j by -delta*sign(alpha_j) (elevation analogously), so
        r_j ~ delta_az * b_az,j + delta_el * b_el,j
        b_az,j = -sign(alpha_j) * (du/dalpha_j) . v_ant
        b_el,j = -sign(eps_j)   * (du/deps_j)   . v_ant
    in the sensor frame with MoCap v_ant.  Frame-averaging the bases gives two
    regressors for c_f.  Pre-registered predictions (worklog
    2026-08-08_literature-review.md, committed before this ran): consistent-sign
    delta_az across bags; R^2 well above the dead time-offset candidate's
    0.00-0.02; |delta| ~ 0.1-2 deg.
    """
    print()
    print('=' * 78)
    print('IS c_f A TOWARD-BORESIGHT BEARING BIAS?  Regress the frame-mean')
    print('  residual on the frame-mean bias basis (azimuth + elevation),')
    print('  deployed stream, same frame protocol as --block ctime.')
    print('=' * 78)
    print(f"  {'flight':<12} {'frames':>7} {'d_az(deg)':>10} {'d_el(deg)':>10} "
          f"{'R2':>6} {'R2_az':>7}   verdict vs pre-registration")
    for k in tabs:
        T = tabs[k]
        P3 = np.column_stack([T['sx'], T['sy'], T['sz']]) * T['rho'][:, None]
        rng = np.random.default_rng(0)
        keep = T['keep6'].copy()
        for f in np.unique(T['fid']):
            m = T['fid'] == f
            if m.sum() < 5:
                continue
            keep[m] &= _ransac_mask_frame(P3[m], T['unw'][m], rng)
        r, fid = T['r'][keep], T['fid'][keep]
        s = np.column_stack([T['sx'], T['sy'], T['sz']])[keep]   # sensor-frame ray
        al = np.arctan2(s[:, 1], s[:, 0])                        # azimuth
        ep = np.arctan2(s[:, 2], np.hypot(s[:, 0], s[:, 1]))     # elevation
        ca, sa, ce, se = np.cos(al), np.sin(al), np.cos(ep), np.sin(ep)
        du_dal = np.column_stack([-ce * sa,  ce * ca, np.zeros_like(al)])
        du_dep = np.column_stack([-se * ca, -se * sa, ce])
        vv = T['f_va3'][fid]                                     # sensor-frame v_ant
        b_az = -np.sign(al) * np.einsum('ij,ij->i', du_dal, vv)
        b_el = -np.sign(ep) * np.einsum('ij,ij->i', du_dep, vv)
        fs = [f for f in np.unique(fid) if (fid == f).sum() >= 6]
        if len(fs) < 40 or not np.isfinite(T['f_va3'][fs]).all():
            continue
        cf = np.array([r[fid == f].mean() for f in fs])
        X = np.column_stack([[b_az[fid == f].mean() for f in fs],
                             [b_el[fid == f].mean() for f in fs]])
        # two-regressor least squares + azimuth-only variant
        beta, *_ = np.linalg.lstsq(X, cf, rcond=None)
        R2 = 1.0 - np.var(cf - X @ beta) / np.var(cf)
        bz = float(np.dot(X[:, 0], cf) / max(np.dot(X[:, 0], X[:, 0]), 1e-12))
        R2a = 1.0 - np.var(cf - bz * X[:, 0]) / np.var(cf)
        d_az, d_el = np.degrees(beta[0]), np.degrees(beta[1])
        verdict = ('SUPPORTS' if (R2 > 0.15 and abs(d_az) < 3.0)
                   else 'weak' if R2 > 0.05 else 'NO')
        print(f'  {SHORT.get(k, k)[:12]:<12} {len(fs):>7d} {d_az:>10.2f} '
              f'{d_el:>10.2f} {R2:>6.2f} {R2a:>7.2f}   {verdict}')
    print()
    print('  Pre-registered bar: consistent-sign d_az across bags, R^2 >> the')
    print('  time-offset candidate (0.00-0.02), |d| ~ 0.1-2 deg.  Anything')
    print('  short of all three kills the candidate; report either way.')


def block4(tabs, extra_keys=None, refresh=False):
    print()
    print('=' * 78)
    print('BLOCK 4 -- TRANSFER.  Coefficients frozen on the two racing bags,')
    print('      only the floor refit, scored on each held-out flight\'s own')
    print('      within-frame pairs.  "gap" = frozen 2NLL - locally refit 2NLL')
    print('      (0 = the frozen constant is as good as a local fit).')
    print('=' * 78)
    PV = {k: point_view(tabs[k]) for k in RACING}
    fit = score_shortlist(PV)
    coef = {nm: fit[nm]['theta'][2] for nm in SHORTLIST}
    print('  frozen on racing: '
          + '  '.join(f'{nm.split("(")[0].strip()[:11]}={coef[nm]:.5f}'
                      for nm in SHORTLIST if not nm.startswith('floors')))
    print(f'    ARRAY dphi_0 = {np.degrees(coef[SHORTLIST[-1]]*np.sqrt(2)):.2f} '
          f'deg at boresight')
    print()
    hdr = f"{'held-out flight':<18} {'pairs':>6} "
    for nm in SHORTLIST[1:]:
        hdr += f'{nm.split("(")[0].strip()[:11]:>13}'
    print(hdr + '   (frozen 2NLL; gap to local in brackets)')
    keys = extra_keys if extra_keys is not None else [BACKFLIPS] + TRANSFER
    for k in keys:
        try:
            T = tabs[k] if k in tabs else table(k, refresh=refresh)
        except Exception as e:
            print(f'  {SHORT.get(k, k):<18} load failed: {e}')
            continue
        P = point_view(T)
        if len(P['d']) < 200:
            print(f'  {SHORT.get(k, k):<18} too few returns ({len(P["d"])})')
            continue
        sc = transfer_score(P, coef)
        line = f'  {SHORT.get(k, k):<18} {sc[SHORTLIST[1]]["n"]:>6d} '
        for nm in SHORTLIST[1:]:
            line += f'{sc[nm]["frozen"]:>8.0f}[{sc[nm]["frozen"]-sc[nm]["local"]:>3.0f}]'
        best = min(SHORTLIST[1:], key=lambda n: sc[n]['frozen'])
        print(line + f'  best={best.split("(")[0].strip()[:11]}')
        print(f'    {"":<18} local dphi_0 (ARRAY) = '
              f'{np.degrees(sc[SHORTLIST[-1]]["coef_local"]*np.sqrt(2)):.2f} deg'
              f'   vs frozen {np.degrees(coef[SHORTLIST[-1]]*np.sqrt(2)):.2f} deg')


def main():
    ap = argparse.ArgumentParser()
    # free-string --block used to mean that a typo ('sigma_c' for 'sigmac')
    # ran block 0 and exited 0, i.e. looked like a successful run of the wrong
    # thing.  Enumerated instead.
    ap.add_argument('--block', default='0', choices=[
        '0', 'all', '1', '1a', '1b', '1c', '1d', '1e', '1g', '1h',
        '2', '2b', '3', '4', 'deploy', 'sigmac', 'sigmacw', 'ctime', 'cfbias'])
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--bags', default=None)
    ap.add_argument('--nboot', type=int, default=200)
    args = ap.parse_args()

    keys = args.bags.split(',') if args.bags else ALL_BAGS
    tabs = block0(keys, refresh=args.refresh)
    want = args.block
    if want in ('0', 'all'):
        gate0(tabs)
    if want in ('1', 'all', '1a', '1b', '1c', '1d', '1e', '1g', '1h'):
        FV = {k: frame_view(tabs[k]) for k in RACING}
        PV = {k: point_view(tabs[k]) for k in RACING}
        if want in ('1', 'all', '1a'):
            block1a(PV)
        if want in ('1', 'all', '1b'):
            block1b(FV)
        if want in ('1', 'all', '1c'):
            block1c(FV, PV, n_boot=args.nboot)
        if want in ('1', 'all', '1d'):
            block1d(tabs, FV, PV)
            block1d2(tabs, FV, n_boot=args.nboot)
            block1d3(tabs, n_boot=args.nboot)
            block1f(tabs)
        if want in ('1', 'all', '1g', '1h'):
            block1h(tabs)      # whitening adequacy (its header prints as 1h)
            block1g(tabs)      # tail identification
        if want in ('1', 'all', '1e'):
            block1e(tabs, n_draw=max(20, args.nboot // 4))
    if want in ('3', 'all'):
        block3(tabs)
    if want in ('4', 'all'):
        block4(tabs, refresh=args.refresh)
    if want in ('2', 'all', '2b'):
        if want in ('2', 'all'):
            block2(tabs, n_boot=max(20, args.nboot // 2))
        block2b(tabs)
    if want in ('deploy', 'all'):
        block_deploy(tabs)
    if want in ('sigmac', 'all'):
        block_sigmac(tabs)
    if want in ('sigmacw', 'all'):
        block_sigmacw(tabs)
    if want in ('ctime', 'all'):
        block_shared_time(tabs)
    if want in ('cfbias', 'all'):
        block_cfbias(tabs)


if __name__ == '__main__':
    main()
