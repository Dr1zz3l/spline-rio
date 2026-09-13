#!/usr/bin/env python3
"""Radar measurement characterization against MoCap ground truth.

Reproducible successor to the exploratory notebook
``notebooks/02_radar_time_sync.ipynb`` (see worklog/FEEDBACK_PLAN.md §D for the audit
of that notebook).  For each bag this script measures, from the raw data and
the MoCap reference only (no solver in the loop):

  1. Temporal structure: radar frame rate, points per frame, fraction of
     frames below the 5-return floor, CPU-cycle vs ROS-time jitter.
  2. Doppler noise: per-point residuals against the ground-truth forward
     model (GT-unwrapped), core/robust sigma, bias, skewness, kurtosis,
     Gaussianity (Q-Q R^2).
  3. Outlier structure: residual sigma vs intensity (SNR) and vs range;
     sigma and BIAS vs elevation angle (sensor-frame u_z) -> the
     elevation-correlated Doppler systematic the RANSAC prefilter targets.
  4. Rate dependence: residual sigma vs |omega| for radar, gyroscope, and
     accelerometer -> measured noise-inflation curves that the
     dynamics-adaptive weighting law should reproduce.
  5. Aliasing: fraction of returns wrapped beyond v_max.
  6. Doppler quantization: measured bin size.
  7. Radar<->MoCap time offset: residual-sigma sweep (cross-check of the
     configured offset).

Usage (from analysis/):
    ../.venv/bin/python3 characterize_radar.py slow_racing_best_velocity \
        fast_racing_best_velocity backflips_best_velocity
Outputs: ../plots/characterization/<bag>.png + <bag>.json + summary table.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from scipy.interpolate import interp1d
from scipy.stats import linregress

from config_loader import load_config
from rosbag_loader.loader import load_bag_topics, stitch_cpu_counter_resets_improved
from radar_velocity_utils import (
    compute_doppler_residuals,
    rotation_matrix_from_euler,
    unwrap_doppler,
)

_cfg = load_config()
BAGS = _cfg['bags']['bags']
TIMING = _cfg['bags']['timing']
RADAR_CFG = _cfg['bags'].get('radar_config', {})
EXT = _cfg['extrinsics']

# Deployed extrinsics (paper values): pitch locked at the measured 27.5 deg,
# NOT the 25.5 deg self-cal seed that extrinsics.yaml carries.
ROTATION_EULER_DEG = np.array([180.0, 27.5, 0.0])
T_BS = np.array(EXT['translation_body_m'])
R_BS = rotation_matrix_from_euler(*np.radians(ROTATION_EULER_DEG))

# Radar timestamps -> MoCap axis (see extrinsics.yaml for sign conventions)
RADAR_OFFSET = EXT['imu_mocap_offset_sec'] - EXT['radar_imu_offset_sec']
IMU_OFFSET = EXT['imu_mocap_offset_sec']

GRAVITY_W = np.array([0.0, 0.0, -9.81])

OMEGA_BIN_EDGES = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0])


def robust_sigma(x):
    """1.4826 * MAD: sigma estimate insensitive to heavy tails."""
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def core_stats(residuals):
    """Distribution stats on the 1-99% core plus tail measures on all data.

    The Q-Q R^2 is computed on the |r - median| < 6*sigma_MAD core (same
    definition as the plot panel).
    """
    q1, q99 = np.percentile(residuals, [1, 99])
    core = residuals[(residuals >= q1) & (residuals <= q99)]
    s_mad = robust_sigma(residuals)
    qq_core = residuals[np.abs(residuals - np.median(residuals)) < 6 * s_mad]
    (_, _), (slope, intercept, r) = sp_stats.probplot(qq_core, dist='norm',
                                                      plot=None)
    return {
        'n': int(len(residuals)),
        'mean': float(np.mean(core)),
        'sigma_core': float(np.std(core)),
        'sigma_mad': float(s_mad),
        'skew': float(sp_stats.skew(core)),
        'excess_kurtosis': float(sp_stats.kurtosis(residuals)),
        'qq_r2': float(r ** 2),
        'frac_beyond_3sigma': float(np.mean(np.abs(residuals - np.median(residuals))
                                            > 3 * s_mad)),
    }


def binned_sigma(x, residuals, edges):
    """Robust sigma of residuals in bins of x."""
    centers, sigmas, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if np.sum(m) >= 30:
            centers.append(0.5 * (lo + hi))
            sigmas.append(robust_sigma(residuals[m]))
            counts.append(int(np.sum(m)))
    return np.array(centers), np.array(sigmas), np.array(counts)


def binned_sigma_bias(x, residuals, edges):
    """Robust sigma AND median bias of residuals in bins of x."""
    centers, sigmas, biases, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if np.sum(m) >= 30:
            centers.append(0.5 * (lo + hi))
            sigmas.append(robust_sigma(residuals[m]))
            biases.append(float(np.median(residuals[m])))
            counts.append(int(np.sum(m)))
    return (np.array(centers), np.array(sigmas), np.array(biases),
            np.array(counts))


def quat_to_R(q):
    from radar_velocity_utils import quat_to_rotation_matrix
    return quat_to_rotation_matrix(q)


def characterize_bag(bag_key, out_dir, offset_sweep=True):
    bag_path = Path(__file__).parent.parent / BAGS[bag_key]
    start_offset, duration = TIMING.get(bag_key, [0.0, None])
    rc = RADAR_CFG.get('best_velocity' if 'best_velocity' in bag_key else 'default', {})
    v_max = rc.get('v_max', 4.99)

    print(f"\n{'=' * 70}\n{bag_key}  (v_max={v_max} m/s)\n{'=' * 70}")
    data = load_bag_topics(str(bag_path), verbose=False)
    t0 = data.start_time + start_offset
    t1 = t0 + duration if duration else data.end_time

    radar = [f for f in data.radar_velocity
             if t0 <= f.timestamp <= t1 and f.positions is not None]
    imu = [m for m in data.imu_data if t0 <= m.timestamp <= t1]
    states = data.agiros_state  # keep full range for interpolation
    res = {'bag': bag_key, 'v_max': v_max, 'window': [float(t0), float(t1)]}

    # ------------------------------------------------------------------ 1
    # Temporal structure
    ppf = np.array([f.num_points() for f in radar])
    frame_times = np.array([f.timestamp for f in radar])
    dt_frames = np.diff(frame_times)
    imu_rate = (len(imu) - 1) / (imu[-1].timestamp - imu[0].timestamp)
    res['temporal'] = {
        'n_frames': int(len(radar)),
        'frame_rate_hz': float(1.0 / np.median(dt_frames)),
        'frame_dt_max_s': float(np.max(dt_frames)),
        'points_per_frame_median': float(np.median(ppf)),
        'points_per_frame_mean': float(np.mean(ppf)),
        'points_per_frame_min': int(np.min(ppf)),
        'points_per_frame_max': int(np.max(ppf)),
        'frac_frames_lt5': float(np.mean(ppf < 5)),
        'imu_rate_hz': float(imu_rate),
    }

    # CPU-cycle timing jitter (all frames of the bag, stitched)
    jitter = None
    try:
        stitched, _ = stitch_cpu_counter_resets_improved(
            list(data.radar_velocity), verbose=False)
        tr = np.array([f.timestamp for f in stitched
                       if f.time_cpu_cycles is not None and len(f.time_cpu_cycles)
                       and f.time_cpu_cycles[0] > 0])
        cc = np.array([f.time_cpu_cycles[0] for f in stitched
                       if f.time_cpu_cycles is not None and len(f.time_cpu_cycles)
                       and f.time_cpu_cycles[0] > 0])
        if len(cc) > 50:
            sl, ic, r, _, _ = linregress(cc, tr)
            jit = tr - (sl * cc + ic)
            jitter = {'sigma_ms': float(np.std(jit) * 1e3),
                      'p95_ms': float(np.percentile(np.abs(jit), 95) * 1e3),
                      'max_ms': float(np.max(np.abs(jit)) * 1e3),
                      'clock_mhz': float(1.0 / sl / 1e6),
                      'r2': float(r ** 2)}
            res['timing_jitter'] = jitter
    except Exception as e:  # old bags without cycles etc.
        print(f"  [jitter analysis skipped: {e}]")

    # ------------------------------------------------------------------ 2
    # Doppler residuals vs ground truth (GT-unwrapped)
    dres = compute_doppler_residuals(states, radar, T_BS, R_BS,
                                     time_offset=RADAR_OFFSET, min_range=0.2)
    meas, pred = dres['measurements'], dres['predictions']
    unwrapped = unwrap_doppler(meas, pred, v_max)
    aliased = unwrapped != meas
    r_unw = unwrapped - pred
    res['doppler_noise'] = core_stats(r_unw)
    res['aliasing'] = {'frac_points_aliased': float(np.mean(aliased)),
                       'frac_pred_beyond_vmax': float(np.mean(np.abs(pred) > v_max))}
    # histogram dump so paper figure scripts can run from the JSON alone
    s_mad = res['doppler_noise']['sigma_mad']
    med = float(np.median(r_unw))
    dens, edges = np.histogram(
        r_unw[np.abs(r_unw - med) < 6 * s_mad], bins=90, density=True)
    res['residual_hist'] = {'edges': edges.tolist(), 'density': dens.tolist()}
    # standardized Q-Q dump (r - median)/sigma_rob vs normal quantiles,
    # downsampled but keeping the extreme tails
    (osm, osr), _ = sp_stats.probplot((r_unw - med) / s_mad, dist='norm',
                                      plot=None)
    n = len(osm)
    idx = np.unique(np.concatenate([
        np.arange(0, 20), np.arange(n - 20, n),
        np.linspace(0, n - 1, 260).astype(int)]))
    res['qq'] = {'theoretical': osm[idx].tolist(),
                 'sample': osr[idx].tolist()}

    # per-point |omega| / |v| tags via frame lookup
    st_t = np.array([s.timestamp for s in states])
    om_i = interp1d(st_t, np.array([s.angular_velocity for s in states]),
                    axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    ve_i = interp1d(st_t, np.array([s.velocity for s in states]),
                    axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    t_pts = np.array([radar[i].timestamp for i in dres['frame_indices']]) + RADAR_OFFSET
    om_pts = np.linalg.norm(om_i(t_pts), axis=1)
    v_pts = np.linalg.norm(ve_i(t_pts), axis=1)

    # ------------------------------------------------------------------ 3
    # sigma vs intensity (deciles) and vs range
    inten = dres['intensities']
    i_edges = np.unique(np.percentile(inten, np.linspace(0, 100, 11)))
    res['sigma_vs_intensity'] = [
        {'center': float(c), 'sigma': float(s), 'n': n}
        for c, s, n in zip(*binned_sigma(inten, r_unw, i_edges))]
    rng = dres['ranges']
    r_edges = np.unique(np.percentile(rng, np.linspace(0, 100, 9)))
    res['sigma_vs_range'] = [
        {'center': float(c), 'sigma': float(s), 'n': n}
        for c, s, n in zip(*binned_sigma(rng, r_unw, r_edges))]

    # sigma and bias vs elevation angle (sensor-frame u_z): the
    # elevation-correlated Doppler systematic (small 12-channel aperture)
    u_z = np.array([
        radar[fi].positions[pi][2]
        / np.linalg.norm(radar[fi].positions[pi])
        for fi, pi in zip(dres['frame_indices'], dres['point_indices'])])
    uz_edges = np.linspace(-1.0, 1.0, 11)
    c_z, s_z, b_z, n_z = binned_sigma_bias(u_z, r_unw, uz_edges)
    res['sigma_bias_vs_uz'] = [
        {'u_z': float(c), 'sigma': float(s), 'bias': float(b), 'n': int(n)}
        for c, s, b, n in zip(c_z, s_z, b_z, n_z)]

    # ------------------------------------------------------------------ 4
    # sigma vs |omega|: radar
    c_r, s_r, n_r = binned_sigma(om_pts, r_unw, OMEGA_BIN_EDGES)
    res['radar_sigma_vs_omega'] = [
        {'omega': float(c), 'sigma': float(s), 'n': n}
        for c, s, n in zip(c_r, s_r, n_r)]

    # gyro residuals vs MoCap omega (constant bias removed)
    t_imu = np.array([m.timestamp for m in imu]) + IMU_OFFSET
    gyro = np.array([m.angular_velocity for m in imu])
    accel = np.array([m.linear_acceleration for m in imu])
    om_ref = om_i(t_imu)
    g_res = gyro - om_ref
    g_res -= np.median(g_res, axis=0)
    om_imu = np.linalg.norm(om_ref, axis=1)
    g_res_flat = g_res.ravel()
    om_flat = np.repeat(om_imu, 3)
    c_g, s_g, n_g = binned_sigma(om_flat, g_res_flat, OMEGA_BIN_EDGES)
    res['gyro_sigma_vs_omega'] = [
        {'omega': float(c), 'sigma': float(s), 'n': n}
        for c, s, n in zip(c_g, s_g, n_g)]
    res['gyro_noise'] = core_stats(g_res_flat)

    # accel residuals vs specific-force prediction from MoCap
    acc_stats = None
    if states[0].acceleration is not None:
        ac_i = interp1d(st_t, np.array([s.acceleration for s in states]),
                        axis=0, kind='linear', bounds_error=False,
                        fill_value='extrapolate')
        qu_i = interp1d(st_t, np.array([s.orientation for s in states]),
                        axis=0, kind='linear', bounds_error=False,
                        fill_value='extrapolate')
        a_w = ac_i(t_imu)
        a_res = np.empty_like(accel)
        for i in range(len(t_imu)):
            R_wb = quat_to_R(qu_i(t_imu[i]))
            a_res[i] = accel[i] - R_wb.T @ (a_w[i] - GRAVITY_W)
        a_res -= np.median(a_res, axis=0)
        # sanity: specific-force prediction should be in the right ballpark
        if robust_sigma(a_res.ravel()) < 5.0:
            c_a, s_a, n_a = binned_sigma(np.repeat(om_imu, 3), a_res.ravel(),
                                         OMEGA_BIN_EDGES)
            res['accel_sigma_vs_omega'] = [
                {'omega': float(c), 'sigma': float(s), 'n': n}
                for c, s, n in zip(c_a, s_a, n_a)]
            acc_stats = core_stats(a_res.ravel())
            res['accel_noise'] = acc_stats
        else:
            print("  [accel residuals implausible; skipped]")

    # ------------------------------------------------------------------ 5
    # Doppler quantization: spacing of unique raw values
    vals = np.sort(np.unique(np.concatenate([np.asarray(f.velocities)
                                             for f in radar])))
    dv = np.diff(vals)
    dv = dv[dv > 1e-6]
    res['quantization_bin'] = float(np.median(dv)) if len(dv) else None

    # ------------------------------------------------------------------ 7
    # time-offset sweep (residual sigma vs offset)
    sweep = None
    if offset_sweep:
        offs = RADAR_OFFSET + np.linspace(-0.06, 0.06, 25)
        sig = []
        for o in offs:
            d = compute_doppler_residuals(states, radar, T_BS, R_BS,
                                          time_offset=float(o), min_range=0.2)
            ru = unwrap_doppler(d['measurements'], d['predictions'], v_max) \
                - d['predictions']
            sig.append(robust_sigma(ru))
        sweep = {'offsets': offs.tolist(), 'sigma': [float(s) for s in sig],
                 'best_offset': float(offs[int(np.argmin(sig))]),
                 'configured_offset': float(RADAR_OFFSET)}
        res['offset_sweep'] = sweep

    # ------------------------------------------------------------------ plots
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle(f'Radar measurement characterization: {bag_key} '
                 f'(GT-unwrapped, extrinsics locked)', fontweight='bold')

    ax = axes[0, 0]
    ax.hist(ppf, bins=np.arange(0, ppf.max() + 2) - 0.5, alpha=0.75,
            edgecolor='black')
    ax.axvline(5, color='r', ls='--', label='5-return floor')
    ax.set_xlabel('returns per frame')
    ax.set_ylabel('frames')
    ax.set_title(f"median {np.median(ppf):.0f}, "
                 f"{100 * np.mean(ppf < 5):.0f}% frames <5")
    ax.legend()

    ax = axes[0, 1]
    st = res['doppler_noise']
    lim = 6 * st['sigma_mad']
    core = r_unw[np.abs(r_unw - st['mean']) < lim]
    ax.hist(core, bins=120, density=True, alpha=0.75, edgecolor='none')
    x = np.linspace(-lim, lim, 300)
    ax.plot(x, sp_stats.norm.pdf(x, st['mean'], st['sigma_mad']), 'r-',
            label=f"N(0, {st['sigma_mad']:.3f}²) robust")
    ax.plot(x, sp_stats.norm.pdf(x, st['mean'], st['sigma_core']), 'g--',
            label=f"N(0, {st['sigma_core']:.3f}²) core")
    ax.set_xlabel('Doppler residual (m/s)')
    ax.set_title(f"n={st['n']}, kurt={st['excess_kurtosis']:.1f}, "
                 f"aliased {100 * res['aliasing']['frac_points_aliased']:.1f}%")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    (q, v), (sl, ic, r) = sp_stats.probplot(core, dist='norm', plot=None)
    ax.scatter(q, v, s=4, alpha=0.3)
    ax.plot(q, sl * q + ic, 'r-', label=f'R²={r**2:.4f}')
    ax.set_xlabel('theoretical quantiles')
    ax.set_ylabel('residual quantiles')
    ax.set_title('Q-Q (core, |r|<6σ)')
    ax.legend()

    ax = axes[0, 3]
    if len(c_z):
        ax.plot(c_z, s_z, 'b-o', label='robust σ')
        ax.plot(c_z, b_z, 'r-s', label='median bias')
        ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('elevation $u_z$ (sensor frame)')
    ax.set_ylabel('m/s')
    ax.set_title('noise / bias vs elevation angle')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    si = res['sigma_vs_intensity']
    ax.plot([b['center'] for b in si], [b['sigma'] for b in si], 'b-o')
    ax.set_xlabel('intensity (SNR proxy)')
    ax.set_ylabel('robust σ (m/s)')
    ax.set_title('noise vs intensity')

    ax = axes[1, 3]
    sr = res['sigma_vs_range']
    ax.plot([b['center'] for b in sr], [b['sigma'] for b in sr], 'b-o')
    ax.set_xlabel('range (m)')
    ax.set_ylabel('robust σ (m/s)')
    ax.set_title('noise vs range')

    ax = axes[1, 1]
    if len(c_r):
        ax.plot(c_r, s_r / s_r[0], 'b-o', label='radar (Doppler)')
    if len(c_g):
        ax.plot(c_g, s_g / s_g[0], 'g-s', label='gyro')
    if acc_stats is not None:
        aso = res['accel_sigma_vs_omega']
        ax.plot([b['omega'] for b in aso],
                np.array([b['sigma'] for b in aso])
                / aso[0]['sigma'], 'm-^', label='accel')
    xw = np.linspace(0.01, min(13, OMEGA_BIN_EDGES[-1]), 100)
    ax.plot(xw, np.sqrt(1 + (xw / 4) ** 2), 'k--', alpha=0.6,
            label='deployed radar law $\\sqrt{1+(\\omega/4)^2}$')
    ax.set_xlabel('|ω| (rad/s)')
    ax.set_ylabel('σ(ω) / σ(low rate)')
    ax.set_title('noise inflation vs body rate')
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    if sweep is not None:
        ax.plot(1e3 * np.array(sweep['offsets']), sweep['sigma'], 'b-o', ms=3)
        ax.axvline(1e3 * sweep['configured_offset'], color='r', ls='--',
                   label=f"configured {1e3 * sweep['configured_offset']:.0f} ms")
        ax.axvline(1e3 * sweep['best_offset'], color='g', ls=':',
                   label=f"best {1e3 * sweep['best_offset']:.0f} ms")
        ax.set_xlabel('radar→MoCap offset (ms)')
        ax.set_ylabel('robust σ (m/s)')
        ax.set_title('time-offset sweep')
        ax.legend(fontsize=8)
    elif jitter is not None:
        ax.set_title('timing jitter')
    fig.tight_layout()
    out_png = out_dir / f'{bag_key}.png'
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    with open(out_dir / f'{bag_key}.json', 'w') as f:
        json.dump(res, f, indent=1,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f"  wrote {out_png}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('bags', nargs='*', default=[
        'slow_racing_best_velocity', 'fast_racing_best_velocity',
        'backflips_best_velocity'])
    ap.add_argument('--out', default=str(Path(__file__).parent.parent
                                         / 'plots' / 'characterization'))
    ap.add_argument('--no-sweep', action='store_true')
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_res = [characterize_bag(b, out_dir, offset_sweep=not args.no_sweep)
               for b in args.bags]

    hdr = (f"{'bag':<28} {'pts/frm':>7} {'<5':>5} {'σ_mad':>7} {'kurt':>6} "
           f"{'alias%':>7} {'σ_gyro':>7} {'quant':>6} {'jit_ms':>6}")
    print(f"\n{'=' * len(hdr)}\nSUMMARY\n{hdr}\n{'-' * len(hdr)}")
    for r in all_res:
        t, d = r['temporal'], r['doppler_noise']
        jit = r.get('timing_jitter', {}).get('sigma_ms', float('nan'))
        print(f"{r['bag']:<28} {t['points_per_frame_median']:>7.0f} "
              f"{100 * t['frac_frames_lt5']:>4.0f}% {d['sigma_mad']:>7.3f} "
              f"{d['excess_kurtosis']:>6.1f} "
              f"{100 * r['aliasing']['frac_points_aliased']:>6.1f}% "
              f"{r['gyro_noise']['sigma_mad']:>7.4f} "
              f"{r['quantization_bin'] if r['quantization_bin'] else float('nan'):>6.3f} "
              f"{jit:>6.2f}")


if __name__ == '__main__':
    main()
