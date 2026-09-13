#!/usr/bin/env python3
"""Derive the static stream weights from the explicit model
    lambda* = 1 / (sigma^2 * rate * tau_corr)
(one formula, all streams): sigma = in-band/core robust residual width vs the
MoCap forward model, rate = samples per second, tau_corr = integrated
autocorrelation time of the residual series. Racing bags only (backflips GT
glitches). Run from analysis/.

RADAR SIGMA: the deployed radar_weight uses the Huber INFLUENCE-CAPPED second
moment, sigma_cap^2 = E[min(|r|, delta)^2], not the Gaussian-core MAD width.
That is the M-estimator sandwich numerator (Huber 1964): the asymptotic
information per residual of an M-estimator with influence psi is
E[psi']^2 / E[psi^2], and for the Huber loss E[psi(r)^2] is exactly
sigma_cap^2.  The core width is the wrong moment for a Huberized stream, and
empirically it fails: it gives per-flight weights that disagree by ~3.6x and
loses the end-to-end battery, while the capped moment reconciles them.
Both are computed and printed below so the difference stays visible.

NOTE (2026-08-05 audit): before this revision the script computed ONLY the core
version, so the deployed radar_weight had no runnable derivation in the repo --
the capped protocol lived in a scratchpad script that no longer exists.  The
'deployed' annotations in the printout were also stale (gyro 4.0 / accel 0.01 /
radar 1.0, i.e. the pre-2026-07-17 tuned set)."""
import sys
from pathlib import Path
sys.path.insert(0, 'lib'); sys.path.insert(0, '.')
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (compute_doppler_residuals, rotation_matrix_from_euler,
                                  unwrap_doppler, quat_to_rotation_matrix)

_cfg = load_config()
BAGS = _cfg['bags']['bags']; TIMING = _cfg['bags']['timing']; EXT = _cfg['extrinsics']
R_BS = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
T_BS = np.array(EXT['translation_body_m'])
OFF = EXT['imu_mocap_offset_sec'] - EXT['radar_imu_offset_sec']
IMU_OFF = EXT['imu_mocap_offset_sec']
G = np.array([0.0, 0.0, -9.81])
rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))


def corr_time_regular(x, dt):
    """integrated autocorrelation time of a regularly sampled series"""
    x = x - x.mean()
    ac = np.correlate(x, x, 'full')[len(x) - 1:]
    ac = ac / ac[0]
    tau = dt
    for k in range(1, min(len(ac), 8000)):
        if ac[k] <= 0:
            break
        tau += 2 * ac[k] * dt
    return tau


def corr_time_irregular(t, x, max_lag=3.0, bin_w=0.05):
    """integrated autocorrelation time for an irregular series (radar points).
    rho(0+) pairs (same frame) land in the first bin."""
    x = x - x.mean()
    var = np.mean(x ** 2)
    order = np.argsort(t)
    t, x = t[order], x[order]
    n = len(t)
    nbins = int(max_lag / bin_w)
    num = np.zeros(nbins); cnt = np.zeros(nbins)
    j_hi = 0
    for i in range(n):
        while j_hi < n and t[j_hi] - t[i] < max_lag:
            j_hi += 1
        for j in range(i + 1, j_hi):
            b = int((t[j] - t[i]) / bin_w)
            num[b] += x[i] * x[j]; cnt[b] += 1
    rho = np.where(cnt > 50, num / np.maximum(cnt, 1) / var, 0.0)
    rho = np.clip(rho, 0.0, None)          # integrate the positive part
    # stop at the first sustained zero
    tau = 0.0
    for b in range(nbins):
        if rho[b] <= 0 and b > 2:
            break
        tau += 2 * rho[b] * bin_w
    rate = n / (t[-1] - t[0])
    return 1.0 / rate + tau, rho, rate


rows = {}
for key in ['slow_racing_best_velocity', 'fast_racing_best_velocity']:
    data = load_bag_topics(str(Path('..') / BAGS[key]), verbose=False)
    t0 = data.start_time + TIMING[key][0]; t1 = t0 + TIMING[key][1]
    radar = [f for f in data.radar_velocity
             if t0 <= f.timestamp <= t1 and f.positions is not None]
    imu = [m for m in data.imu_data if t0 <= m.timestamp <= t1]
    states = data.agiros_state
    st_t = np.array([s.timestamp for s in states])
    t_imu = np.array([m.timestamp for m in imu]) + IMU_OFF
    fs = 1.0 / np.median(np.diff(t_imu))

    # ---------- radar: RANSAC-kept residuals with timestamps ----------
    d = compute_doppler_residuals(states, radar, T_BS, R_BS,
                                  time_offset=OFF, min_range=0.2)
    unw = unwrap_doppler(d['measurements'], d['predictions'], 3.136)
    r = unw - d['predictions']; r = r - np.median(r)
    fi = np.array(d['frame_indices']); pi = np.array(d['point_indices'])
    rng = np.random.default_rng(0)

    def mask_fn(P, v, thresh=0.15, iters=150):
        n = len(v)
        if n < 5:
            return np.ones(n, bool)
        Hn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
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

    keep = np.zeros(len(r), bool)
    for i, f in enumerate(radar):
        if f.num_points() < 5:
            continue
        v = np.asarray(f.velocities, float).copy()
        sel = fi == i
        v[pi[sel]] = unw[sel]
        m = mask_fn(np.asarray(f.positions, float), v)
        for j in np.where(sel)[0]:
            if pi[j] < len(m):
                keep[j] = m[pi[j]]
    t_pts = np.array([radar[i].timestamp for i in fi]) + OFF
    rk, tk = r[keep], t_pts[keep]
    DELTA = 1.0                       # deployed radar Huber delta (m/s)
    sig_r = rs(rk)                    # Gaussian-core width (the WRONG moment)
    tau_r_all, rho_r, rate_r = corr_time_irregular(tk, rk)
    # cross-frame-only variant: exclude the same-frame (first) bin
    tau_r_x = tau_r_all - 2 * rho_r[0] * 0.05
    lam_r = 1.0 / (sig_r ** 2 * rate_r * tau_r_all)
    lam_r_x = 1.0 / (sig_r ** 2 * rate_r * max(tau_r_x, 1.0 / rate_r))

    # --- deployed protocol: Huber influence-capped moment, on the capped series
    psi = np.sign(rk) * np.minimum(np.abs(rk), DELTA)     # Huber influence
    sig_cap = float(np.sqrt(np.mean(psi ** 2)))           # sqrt(E[psi^2])
    e_psi = float(np.mean(np.abs(rk) < DELTA))            # E[psi'] (sandwich)
    tau_cap, _, _ = corr_time_irregular(tk, psi)
    lam_cap = 1.0 / (sig_cap ** 2 * rate_r * tau_cap)
    # variant kept for provenance: capped sigma but the UNCAPPED tau
    lam_cap_tau_raw = 1.0 / (sig_cap ** 2 * rate_r * tau_r_all)

    # --- 2026-09-07 (kernel removal): the Huber loss is dropped from the
    # solver, so the sandwich numerator reduces to the PLAIN second moment.
    # Aliased returns enter the solver at w_al = 0.019, i.e. they are absent
    # from the moment that sets the scale, so the plain moment is taken on
    # the NON-ALIASED kept stream (alias flag = the unwrap changed the value).
    aliased = np.zeros(len(r), bool)
    aliased[pi >= 0] = np.abs(unw - np.asarray([radar[i].velocities[p] for i, p in zip(fi, pi)])) > 1e-9
    ka = keep & ~aliased
    rk_na, tk_na = r[ka], t_pts[ka]
    sig_plain = float(np.sqrt(np.mean(rk_na ** 2)))          # RMS, no cap
    tau_plain, _, rate_na = corr_time_irregular(tk_na, rk_na)
    lam_plain = 1.0 / (sig_plain ** 2 * rate_na * tau_plain)
    psi_na = np.sign(rk_na) * np.minimum(np.abs(rk_na), DELTA)
    sig_cap_na = float(np.sqrt(np.mean(psi_na ** 2)))
    tau_cap_na, _, _ = corr_time_irregular(tk_na, psi_na)
    lam_cap_na = 1.0 / (sig_cap_na ** 2 * rate_na * tau_cap_na)
    frac_al = float(np.mean(aliased[keep]))

    # ---------- gyro / accel: sigma_ib + tau at B = 5/10/20 Hz ----------
    om_i = interp1d(st_t, np.array([s.angular_velocity for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    g_res = np.array([m.angular_velocity for m in imu]) - om_i(t_imu)
    g_res -= np.median(g_res, axis=0)
    ac_i = interp1d(st_t, np.array([s.acceleration for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    qu_i = interp1d(st_t, np.array([s.orientation for s in states]), axis=0,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    acc = np.array([m.linear_acceleration for m in imu])
    a_w = ac_i(t_imu); q = qu_i(t_imu)
    a_res = np.empty_like(acc)
    for i in range(len(t_imu)):
        a_res[i] = acc[i] - quat_to_rotation_matrix(q[i]).T @ (a_w[i] - G)
    a_res -= np.median(a_res, axis=0)

    out = {'radar': dict(sig=sig_r, rate=rate_r, tau=tau_r_all, tau_x=tau_r_x,
                         lam=lam_r, lam_x=lam_r_x, rho0=rho_r[0],
                         sig_cap=sig_cap, tau_cap=tau_cap, lam_cap=lam_cap,
                         e_psi=e_psi, lam_cap_tau_raw=lam_cap_tau_raw,
                         sig_plain=sig_plain, tau_plain=tau_plain, rate_na=rate_na,
                         lam_plain=lam_plain, sig_cap_na=sig_cap_na,
                         lam_cap_na=lam_cap_na, frac_al=frac_al)}
    for name, res in [('gyro', g_res), ('accel', a_res)]:
        for B in (5.0, 10.0, 20.0):
            b, a = butter(2, B / (fs / 2), btype='low')
            ib = filtfilt(b, a, res, axis=0)
            sig = rs(ib.ravel())
            tau = np.mean([corr_time_regular(ib[:, k], 1 / fs) for k in range(3)])
            lam = 1.0 / (sig ** 2 * fs * tau)
            out[f'{name}_B{B:.0f}'] = dict(sig=sig, tau=tau, lam=lam)
    rows[key] = out
    print(f'\n===== {key} (IMU {fs:.0f} Hz)')
    o = out['radar']
    print(f"radar : sigma_core {o['sig']:.3f}  rate {o['rate']:.0f}/s  "
          f"tau {1e3*o['tau']:.0f} ms (cross-frame-only {1e3*o['tau_x']:.0f} ms, "
          f"rho_sameframe {o['rho0']:.2f})")
    print(f"        core-sigma lambda* = {o['lam']:.2f}  "
          f"(cross-frame-only {o['lam_x']:.2f})   <- WRONG moment, kept for "
          f"contrast")
    print(f"        CAPPED sigma_cap {o['sig_cap']:.3f}  tau_cap "
          f"{1e3*o['tau_cap']:.0f} ms  E[psi'] {o['e_psi']:.4f}  "
          f"(sandwich correction E[psi']^2 = {o['e_psi']**2:.3f})")
    print(f"        DEPLOYED-PROTOCOL lambda* = {o['lam_cap']:.2f}   "
          f"(capped sigma with uncapped tau: {o['lam_cap_tau_raw']:.2f})")
    print(f"        NO-KERNEL (2026-09-07) non-aliased kept stream ({100*o['frac_al']:.1f}% of kept were aliased): "
          f"RMS sigma_plain {o['sig_plain']:.3f}  rate {o['rate_na']:.0f}/s  tau {1e3*o['tau_plain']:.0f} ms  "
          f"lambda* = {o['lam_plain']:.2f}   [capped on the same stream: sigma {o['sig_cap_na']:.3f}, lambda* {o['lam_cap_na']:.2f}]")
    for name, dep in [('gyro', 3.13), ('accel', 0.00392)]:
        line = f'{name:<6}:'
        for B in (5, 10, 20):
            oo = out[f'{name}_B{B}']
            line += (f"  B={B}: sig {oo['sig']:.3f} tau {1e3*oo['tau']:.0f}ms "
                     f"lam* {oo['lam']:.4g}")
        print(line + f'   deployed {dep}')

print('\n===== DERIVED SET (geometric mean over racing bags, B=10) =====')
gm = lambda a, b: float(np.sqrt(a * b))
lam_g = gm(rows['slow_racing_best_velocity']['gyro_B10']['lam'],
           rows['fast_racing_best_velocity']['gyro_B10']['lam'])
lam_a = gm(rows['slow_racing_best_velocity']['accel_B10']['lam'],
           rows['fast_racing_best_velocity']['accel_B10']['lam'])
lam_r = gm(rows['slow_racing_best_velocity']['radar']['lam'],
           rows['fast_racing_best_velocity']['radar']['lam'])
lam_r_cap = gm(rows['slow_racing_best_velocity']['radar']['lam_cap'],
               rows['fast_racing_best_velocity']['radar']['lam_cap'])
lam_r_cap2 = gm(rows['slow_racing_best_velocity']['radar']['lam_cap_tau_raw'],
                rows['fast_racing_best_velocity']['radar']['lam_cap_tau_raw'])
print(f'lambda_gyro*  = {lam_g:.3g}      (deployed 3.13)')
print(f'lambda_accel* = {lam_a:.3g}   (deployed 0.00392)')
print(f'radar_weight* = {lam_r_cap:.3g}      (deployed 2.11)   '
      f'<- capped moment, the deployed protocol')
print(f'  variants: core-sigma {lam_r:.3g} (per-bag spread '
      f"{rows['fast_racing_best_velocity']['radar']['lam']/rows['slow_racing_best_velocity']['radar']['lam']:.1f}x), "
      f'capped-sigma-with-uncapped-tau {lam_r_cap2:.3g}')
print(f'  per-bag capped lambda*: '
      + ', '.join(f"{k.split('_')[0]} {rows[k]['radar']['lam_cap']:.2f}"
                  for k in rows))
lam_r_plain = gm(rows['slow_racing_best_velocity']['radar']['lam_plain'],
                 rows['fast_racing_best_velocity']['radar']['lam_plain'])
print(f'radar_weight* NO-KERNEL (plain RMS on the non-aliased kept stream) = {lam_r_plain:.3g}   '
      + 'per bag: ' + ', '.join(f"{k.split('_')[0]} {rows[k]['radar']['lam_plain']:.2f}" for k in rows))
print(f'relative gyro/radar: derived {lam_g/lam_r_cap:.2f}')
print(f'relative accel/radar: derived {lam_a/lam_r_cap:.4g}')
