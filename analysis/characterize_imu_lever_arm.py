#!/usr/bin/env python3
"""Estimator-free fit of the IMU lever arm (BACKFLIPS_PLAN.md experiment 1,
2026-09-16).

Both accelerometer models (spline `AccelAnalyticFactor`, GP `accel_dv`) place
the IMU at the trajectory point, i.e. the MoCap body origin.  An offset r
adds the unmodelled term  omega x (omega x r) + omegadot x r  to the specific
force: 100 r m/s^2 at 10 rad/s (2 cm -> 2 m/s^2), invisible at racing rates.

Model, linear in the unknowns (per IMU sample, body frame):
    z_a - R^T (a_w - g)  =  b_a  +  Omega r  [+ nuisance terms]
    Omega = [omegadot]_x + [omega]_x [omega]_x       (omega from the gyro)
Nuisance terms (fitted jointly so the lever arm is not a sink for them):
    IMU-vs-MoCap-body misalignment delta (3):  -[f_pred]_x delta
    per-axis scale k (3):                       diag(f_pred) k
    rotor drag c (2, x/y):                      -c_ax v_body,ax
Reference: the pose-derived kinematics of the loader (`derive_pose_reference`,
zero-phase Butterworth) at a LADDER of acceleration cut-offs, because the
loader's 5 Hz was set for racing and a 10 rad/s flip has harmonics above it.
Measurement and prediction are low-passed with the SAME in-band filter before
the fit (systematics are low-frequency; kills vibration), as in
`characterize_accel_model.py` (2026-07-16: racing-only, lever ~0 cm at the
old reference and clock).  Timing: IMU stamps + imu_mocap_offset_sec.

Per bag, then pooled (per-bag bias, one lever arm), then held out.  The
decision rule (plan): |r| > 1 cm AND consistent across bags -> mount
constant, port to both models and gate; otherwise record the negative.

Run from analysis/:  ../.venv/bin/python3 characterize_imu_lever_arm.py [bag ...]
"""
import sys
from pathlib import Path
sys.path.insert(0, 'lib'); sys.path.insert(0, '.')
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
from config_loader import load_config
import rosbag_loader.loader as L
from rosbag_loader.loader import load_bag_topics, derive_pose_reference

_cfg = load_config()
BAGS = _cfg['bags']['bags']; TIMING = _cfg['bags']['timing']
IMU_OFF = _cfg['extrinsics']['imu_mocap_offset_sec']
G = np.array([0.0, 0.0, -9.81])
FLIGHTS = sys.argv[1:] or ['backflips_best_velocity', 'slow_racing_best_velocity', 'fast_racing_best_velocity']
FC_LP = 10.0           # in-band filter applied to both sides
FC_ACC_LADDER = (5.0, 10.0, 15.0)
RATE_BINS = [(0, 1), (1, 2), (2, 4), (4, 7), (7, 12)]
rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def lowpass(x, fs, fc=FC_LP):
    b, a = butter(2, fc / (fs / 2), btype='low')
    return filtfilt(b, a, x, axis=0)


def load(key, fc_acc):
    """Per-sample arrays at the IMU stamps (+ offset): residual, regressors."""
    d = load_bag_topics(str(Path('..') / BAGS[key]), verbose=False)
    t0 = d.start_time + TIMING[key][0]; t1 = t0 + TIMING[key][1]
    L.REFERENCE_FC_ACC = fc_acc                       # ladder rung
    S = derive_pose_reference(d.agiros_state)         # re-derive from the pose
    S = [s for s in S if t0 - 1 <= s.timestamp <= t1 + 1]
    ts = np.array([s.timestamp for s in S]); keep = np.ones(len(ts), bool); keep[1:] = np.diff(ts) > 1e-3
    ts = ts[keep]; S = [s for s, k in zip(S, keep) if k]
    ai = interp1d(ts, np.array([s.acceleration for s in S]), axis=0, bounds_error=False, fill_value=np.nan)
    vi = interp1d(ts, np.array([s.velocity for s in S]), axis=0, bounds_error=False, fill_value=np.nan)
    wi = interp1d(ts, np.array([s.angular_velocity for s in S]), axis=0, bounds_error=False, fill_value=np.nan)
    Q = np.array([s.orientation for s in S]); okq = np.ones(len(ts), bool); okq[1:] = np.abs(np.diff(Q, axis=0)).max(axis=1) > 1e-12
    sl = Slerp(ts[okq], Rotation.from_quat(Q[okq]))
    imu = [m for m in d.imu_data if t0 <= m.timestamp <= t1]
    t = np.array([m.timestamp for m in imu]) + IMU_OFF
    fs = 1.0 / np.median(np.diff(t))
    acc = np.array([m.linear_acceleration for m in imu]); gyr = np.array([m.angular_velocity for m in imu])
    a_w = ai(t); v_w = vi(t); w_ref = wi(t)
    ok = np.all(np.isfinite(a_w), axis=1) & np.all(np.isfinite(w_ref), axis=1)
    t, acc, gyr, a_w, v_w, w_ref = t[ok], acc[ok], gyr[ok], a_w[ok], v_w[ok], w_ref[ok]
    Rm = sl(np.clip(t, ts[okq][0], ts[okq][-1])).as_matrix()          # R_wb
    f_pred = np.einsum('nji,nj->ni', Rm, a_w - G)                      # R^T (a_w - g)
    v_b = np.einsum('nji,nj->ni', Rm, v_w)
    # lever-arm regressor from the gyro (full rate), then in-band like everything else
    g_lp = lowpass(gyr, fs); dg = np.gradient(g_lp, t, axis=0)
    Om = np.array([skew(dg[i]) + skew(g_lp[i]) @ skew(g_lp[i]) for i in range(len(t))])
    Om_lp = lowpass(Om.reshape(len(t), 9), fs).reshape(len(t), 3, 3)
    # cross-check regressor from the reference rate instead of the gyro
    dwr = np.gradient(w_ref, t, axis=0)
    Om_ref = np.array([skew(dwr[i]) + skew(w_ref[i]) @ skew(w_ref[i]) for i in range(len(t))])
    Om_ref_lp = lowpass(Om_ref.reshape(len(t), 9), fs).reshape(len(t), 3, 3)
    return dict(key=key, t=t, fs=fs, res_raw=acc - f_pred, res=lowpass(acc, fs) - lowpass(f_pred, fs),
                f=lowpass(f_pred, fs), vb=lowpass(v_b, fs), Om=Om_lp, Om_ref=Om_ref_lp,
                om=np.linalg.norm(g_lp, axis=1), n=len(t))


def design(D, cols, bias_slot=0, n_bias=1, om_key='Om'):
    """Rows: 3 per sample.  cols in {'lever','mis','scale','drag'}; bias always."""
    n = D['n']; ncol = 3 * n_bias + 3 * ('lever' in cols) + 3 * ('mis' in cols) + 3 * ('scale' in cols) + 2 * ('drag' in cols)
    A = np.zeros((3 * n, ncol)); c = 3 * n_bias
    idx = {}
    I3 = np.tile(np.eye(3), (n, 1))
    A[:, 3 * bias_slot:3 * bias_slot + 3] = I3
    if 'lever' in cols:
        A[:, c:c + 3] = D[om_key].reshape(3 * n, 3); idx['lever'] = slice(c, c + 3); c += 3
    if 'mis' in cols:
        A[:, c:c + 3] = np.concatenate([-skew(f) for f in D['f']]); idx['mis'] = slice(c, c + 3); c += 3
    if 'scale' in cols:
        A[:, c:c + 3] = np.concatenate([np.diag(f) for f in D['f']]); idx['scale'] = slice(c, c + 3); c += 3
    if 'drag' in cols:
        B = np.zeros((3 * n, 2)); B[0::3, 0] = -D['vb'][:, 0]; B[1::3, 1] = -D['vb'][:, 1]
        A[:, c:c + 2] = B; idx['drag'] = slice(c, c + 2); c += 2
    return A, idx


def solve(A, y, fs):
    th, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ th
    n_eff = fs / (2 * FC_LP)                       # samples per independent in-band sample
    cov = np.linalg.pinv(A.T @ A) * rms(r) ** 2 * n_eff
    return th, r, np.sqrt(np.diag(cov))


def fmt_r(th, se, sl):
    return f"r = {(100 * th[sl]).round(2)} +- {(100 * se[sl]).round(2)} cm (|r| {100 * np.linalg.norm(th[sl]):.2f})"


def binned(D, r):
    r = r.reshape(-1, 3); out = []
    for lo, hi in RATE_BINS:
        m = (D['om'] >= lo) & (D['om'] < hi)
        if m.sum() > 200: out.append(f"{lo}-{hi}: {rms(r[m]):.3f} (n={m.sum()})")
    return '  '.join(out)


print(f"IMU offset {IMU_OFF*1e3:+.1f} ms; in-band {FC_LP:.0f} Hz; reference: pose-derived, acceleration cut-off ladder {FC_ACC_LADDER} Hz")
print("Deployed accelerometer models put the IMU at the MoCap body origin (r = 0).\n")
data = {}
for fc in FC_ACC_LADDER:
    print(f"================ reference acceleration cut-off {fc:.0f} Hz ================")
    data[fc] = {k: load(k, fc) for k in FLIGHTS}
    for k in FLIGHTS:
        D = data[fc][k]; y = D['res'].reshape(-1)
        print(f"-- {k}  (n={D['n']}, fs={D['fs']:.0f} Hz, |omega| max {D['om'].max():.1f} rad/s)")
        print(f"   raw residual rms {rms(D['res_raw']):.3f}, in-band {rms(y):.3f} m/s^2; in-band by |omega| bin: {binned(D, y)}")
        for cols in ([], ['lever'], ['lever', 'mis', 'scale'], ['lever', 'mis', 'scale', 'drag']):
            A, idx = design(D, cols); th, r, se = solve(A, y, D['fs'])
            line = f"   bias{'+' + '+'.join(cols) if cols else ''}: rms -> {rms(r):.3f}"
            if 'lever' in idx: line += f"   {fmt_r(th, se, idx['lever'])}"
            if 'mis' in idx: line += f"   delta {np.degrees(th[idx['mis']]).round(2)} deg"
            if 'scale' in idx: line += f"   k {(100 * th[idx['scale']]).round(2)} %"
            if 'drag' in idx: line += f"   c_d {th[idx['drag']].round(3)} 1/s"
            print(line)
            if cols == ['lever']:
                print(f"      by |omega| bin after lever: {binned(D, r)}")
        A, idx = design(D, ['lever'], om_key='Om_ref'); th, r, se = solve(A, y, D['fs'])
        print(f"   bias+lever, regressor from the REFERENCE rate: rms -> {rms(r):.3f}   {fmt_r(th, se, idx['lever'])}")
        # flip-only / high-rate-only subset (where the term carries signal)
        m = D['om'] > 4.0
        if m.sum() > 500:
            Dm = dict(D); Dm['n'] = int(m.sum()); Dm['Om'] = D['Om'][m]; Dm['f'] = D['f'][m]; Dm['vb'] = D['vb'][m]
            A, idx = design(Dm, ['lever', 'mis', 'scale']); th, r, se = solve(A, D['res'][m].reshape(-1), D['fs'])
            print(f"   |omega| > 4 only (n={m.sum()}): bias+lever+mis+scale rms {rms(D['res'][m]):.3f} -> {rms(r):.3f}   {fmt_r(th, se, idx['lever'])}")
    # pooled: per-bag bias, one lever arm (+ mis + scale shared: mount constants)
    print(f"-- pooled over {len(FLIGHTS)} bags (per-bag bias; lever, misalignment, scale shared)")
    for cols in (['lever'], ['lever', 'mis', 'scale']):
        blocks, ys, fss = [], [], []
        for i, k in enumerate(FLIGHTS):
            D = data[fc][k]; A, idx = design(D, cols, bias_slot=i, n_bias=len(FLIGHTS)); blocks.append(A); ys.append(D['res'].reshape(-1)); fss.append(D['fs'])
        A = np.vstack(blocks); y = np.concatenate(ys); th, r, se = solve(A, y, np.mean(fss))
        line = f"   bias+{'+'.join(cols)}: rms {rms(y):.3f} -> {rms(r):.3f}   {fmt_r(th, se, idx['lever'])}"
        if 'mis' in idx: line += f"   delta {np.degrees(th[idx['mis']]).round(2)} deg   k {(100 * th[idx['scale']]).round(2)} %"
        print(line)
        # held out: fit on all but one, score that one (its own bias refitted)
        for k in FLIGHTS:
            others = [o for o in FLIGHTS if o != k]
            bl, yy = [], []
            for i, o in enumerate(others):
                D = data[fc][o]; A2, idx2 = design(D, cols, bias_slot=i, n_bias=len(others)); bl.append(A2); yy.append(D['res'].reshape(-1))
            th2, _, _ = solve(np.vstack(bl), np.concatenate(yy), np.mean(fss))
            D = data[fc][k]; A3, idx3 = design(D, cols); y3 = D['res'].reshape(-1)
            th3 = np.zeros(A3.shape[1]); th3[3:] = th2[3 * len(others):]
            # refit only the held-out bag's bias with the transferred mount constants
            y_c = y3 - A3[:, 3:] @ th3[3:]; b = y_c.reshape(-1, 3).mean(axis=0)
            r_before = y3 - np.tile(y3.reshape(-1, 3).mean(axis=0), D['n']); r_after = y_c - np.tile(b, D['n'])
            print(f"      held out {k[:9]}: rms {rms(r_before):.3f} (bias only) -> {rms(r_after):.3f} with the others' r = {(100 * th2[3 * len(others):3 * len(others) + 3]).round(2)} cm;"
                  f" by bin -> {binned(D, r_after)}")
    print()
