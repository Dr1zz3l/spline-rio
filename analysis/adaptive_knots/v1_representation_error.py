"""V1 — go/no-go: is the backflips orientation error a spline-bandwidth problem?

Fits the cumulative SO(3) B-spline DIRECTLY to backflips MoCap orientation (no
odometry, no optimizer) and measures representation error for:
  (a) uniform dt_ori = 8 ms       (current SW/batch operating point)
  (b) adaptive, EQUAL knot budget (|omega|-driven inverse-CDF placement)
  (c) uniform dt_ori = 4 ms       (2x budget upper bound)

Gate (from the plan):
  - If (a) already represents MoCap to << 8.3 deg during flips, the backflips ori
    gap is NOT bandwidth -> stop, rethink.
  - If (b) materially beats (a) during flips (target >= 2x) at equal budget,
    the adaptive-knot hypothesis is validated -> proceed to Phase 1.

Also cross-checks that |omega|-driven placement covers the measured gyro-residual
hotspots (plots/<bag>/residual_stats_*.npz).

Run:  cd analysis && ../.venv/bin/python3 adaptive_knots/v1_representation_error.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from nonuniform_bspline import (DEG, NonUniformSO3Spline, extend_knots,
                                so3_exp, so3_log)
from config_loader import load_config
from rosbag_loader.loader import load_bag_topics
from scipy.spatial.transform import Rotation, Slerp

BAG_KEY = 'backflips_best_velocity'
OMEGA_FLIP = 4.0          # rad/s: "during-flip" classification threshold
DT_BASE = 0.008
DT_MIN = 0.002            # densest tier allowed for adaptive placement
DT_MAX = 0.032            # coarsest allowed (avoid huge gaps in quiet phases)
N_REFINE = 8              # fixed-point quasi-interpolation refinement iterations

cfg = load_config()
bag_rel = cfg['bags']['bags'][BAG_KEY]
t_off, dur = cfg['bags']['timing'][BAG_KEY]
bag_path = str(Path(__file__).parent.parent.parent / bag_rel)

print(f"Loading {bag_path} (window +{t_off}s, {dur}s) ...")
bag = load_bag_topics(bag_path, verbose=False)
t0 = bag.start_time + t_off
t1 = t0 + dur
poses = [p for p in bag.mocap_pose if t0 - 0.5 <= p.timestamp <= t1 + 0.5]
print(f"  {len(poses)} mocap poses in window "
      f"({len(poses)/ (poses[-1].timestamp - poses[0].timestamp):.0f} Hz)")

tp = np.array([p.timestamp for p in poses]) - t0
quats = np.array([p.orientation for p in poses])  # xyzw
# de-duplicate timestamps (Slerp requires strictly increasing)
keep = np.concatenate([[True], np.diff(tp) > 1e-6])
tp, quats = tp[keep], quats[keep]
rots = Rotation.from_quat(quats)
slerp = Slerp(tp, rots)

T0, T1 = tp[0], tp[-1]


def truth_R(t):
    return slerp(np.clip(t, T0, T1)).as_matrix()


# ---------------------------------------------------------------------------
# |omega|(t) from mocap (FD on 500 Hz grid, then smoothed) — placement signal
# ---------------------------------------------------------------------------
h_fd = 0.002
tg = np.arange(T0 + h_fd, T1 - h_fd, h_fd)
Rg = slerp(tg).as_matrix()
omega_fd = np.zeros((len(tg) - 1, 3))
for i in range(len(tg) - 1):
    omega_fd[i] = so3_log(Rg[i].T @ Rg[i + 1]) / h_fd
t_om = 0.5 * (tg[:-1] + tg[1:])
om_norm = np.linalg.norm(omega_fd, axis=1)
# ~50 ms moving-average smoothing
w = max(1, int(round(0.05 / h_fd)))
om_smooth = np.convolve(om_norm, np.ones(w) / w, mode='same')
print(f"  |omega|: median {np.median(om_norm):.2f}, p95 {np.percentile(om_norm,95):.2f}, "
      f"max {om_norm.max():.2f} rad/s; "
      f"{(om_norm > OMEGA_FLIP).mean()*100:.1f}% of time above {OMEGA_FLIP} rad/s")

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


def uniform_grid(dt):
    return np.arange(T0, T1 + 1e-9, dt)


def adaptive_grid(n_knots, omega_ref):
    """Inverse-CDF placement: knot density ~ clip(|omega_smooth|/omega_ref, 1, dt_base/dt_min),
    floor at 1/DT_MAX relative density. Total count fixed -> equal budget."""
    dens = np.clip(om_smooth / omega_ref, 1.0, DT_BASE / DT_MIN)
    dens = np.maximum(dens, DT_BASE / DT_MAX)
    cdf = np.concatenate([[0.0], np.cumsum(dens[:-1] * np.diff(t_om))])
    cdf /= cdf[-1]
    targets = np.linspace(0.0, 1.0, n_knots)
    kt = np.interp(targets, cdf, t_om)
    kt[0], kt[-1] = T0, T1
    # enforce strict monotonicity
    for i in range(1, len(kt)):
        if kt[i] <= kt[i - 1]:
            kt[i] = kt[i - 1] + 1e-5
    return kt


def greville_sample(kt):
    ext = extend_knots(kt)
    R = np.empty((len(kt), 3, 3))
    for j in range(len(kt)):
        xi = (ext[j + 4] + ext[j + 5] + ext[j + 6]) / 3.0
        R[j] = truth_R(xi)
    return R


def refine(spline, n_iter):
    """Fixed-point quasi-interpolation refinement at Greville abscissae."""
    kt, ext = spline.kt, spline.ext
    for _ in range(n_iter):
        R_new = spline.R.copy()
        for j in range(len(kt)):
            xi = np.clip((ext[j + 4] + ext[j + 5] + ext[j + 6]) / 3.0,
                         spline.t_start, spline.t_end - 1e-9)
            R_sp, _ = spline.evaluate(xi)
            E = R_sp.T @ truth_R(xi)
            R_new[j] = spline.R[j] @ E
        spline = NonUniformSO3Spline(kt, R_new)
    return spline


def evaluate_grid(name, kt):
    sp = NonUniformSO3Spline(kt, greville_sample(kt))
    sp = refine(sp, N_REFINE)
    ts = np.arange(sp.t_start + 0.01, sp.t_end - 0.01, 0.002)
    err = np.empty(len(ts))
    for i, t in enumerate(ts):
        R_s, _ = sp.evaluate(t)
        err[i] = np.degrees(np.linalg.norm(so3_log(R_s.T @ truth_R(t))))
    om_at = np.interp(ts, t_om, om_smooth)
    flip = om_at > OMEGA_FLIP
    dts = np.diff(kt)
    res = dict(name=name, kt=kt, ts=ts, err=err, flip=flip,
               rmse=float(np.sqrt(np.mean(err**2))),
               rmse_flip=float(np.sqrt(np.mean(err[flip]**2))) if flip.any() else 0.0,
               rmse_quiet=float(np.sqrt(np.mean(err[~flip]**2))),
               max=float(err.max()), n=len(kt),
               dt_min=float(dts.min()), dt_max=float(dts.max()))
    print(f"  {name:28s} n={res['n']:5d}  RMSE {res['rmse']:6.3f}  "
          f"flip {res['rmse_flip']:6.3f}  quiet {res['rmse_quiet']:6.3f}  "
          f"max {res['max']:6.3f} deg   dt [{res['dt_min']*1e3:.1f}, {res['dt_max']*1e3:.1f}] ms")
    return res


print("\nRepresentation error vs MoCap (deg, geodesic):")
res_u8 = evaluate_grid('uniform 8ms', uniform_grid(DT_BASE))
budget = res_u8['n']
res_ad = evaluate_grid('adaptive (equal budget)', adaptive_grid(budget, omega_ref=1.5))
res_ad2 = evaluate_grid('adaptive (omega_ref=3)', adaptive_grid(budget, omega_ref=3.0))
res_u4 = evaluate_grid('uniform 4ms (2x budget)', uniform_grid(0.004))
res_u16 = evaluate_grid('uniform 16ms (0.5x budget)', uniform_grid(0.016))

# ---------------------------------------------------------------------------
# Cross-check: does |omega|-placement cover the gyro-residual hotspots?
# ---------------------------------------------------------------------------
npz_path = Path(__file__).parent.parent.parent / 'plots' / BAG_KEY / f'residual_stats_{BAG_KEY}.npz'
if npz_path.exists():
    d = np.load(npz_path)
    tr = d['t'] - (t0 - bag.start_time) - bag.start_time  # residual t is absolute
    gr = np.linalg.norm(d['gyro_res'], axis=1)
    sel = (tr > T0) & (tr < T1)
    if sel.sum() > 100:
        kt = res_ad['kt']
        dens_t = 0.5 * (kt[:-1] + kt[1:])
        dens = 1.0 / np.diff(kt)
        dens_at = np.interp(tr[sel], dens_t, dens)
        c = np.corrcoef(dens_at, gr[sel])[0, 1]
        print(f"\ncorr(adaptive knot density, |gyro residual|) = {c:+.3f} "
              f"(placement covers residual hotspots if strongly positive)")
else:
    print(f"\n[warn] {npz_path} not found — skipping residual cross-check")

# ---------------------------------------------------------------------------
# Verdict + plot
# ---------------------------------------------------------------------------
ratio = res_u8['rmse_flip'] / max(res_ad['rmse_flip'], 1e-9)
print(f"\n=== V1 VERDICT ===")
print(f"uniform-8ms during-flip RMSE: {res_u8['rmse_flip']:.3f} deg "
      f"(vs 8.3 deg batch ori ceiling)")
print(f"adaptive equal-budget improvement during flips: {ratio:.2f}x")
if res_u8['rmse_flip'] < 0.5:
    print("-> uniform 8ms ALREADY represents MoCap well during flips: the ori gap is")
    print("   NOT raw spline bandwidth at the representation level. Check the gain of")
    print("   uniform-4ms full-solve instead; adaptive knots may still help the")
    print("   OPTIMIZATION (conditioning/regularization), but the simple bandwidth")
    print("   story is dead. STOP and reassess before Phase 1.")
elif ratio >= 2.0:
    print("-> GO: adaptive placement materially increases representation capacity")
    print("   at equal budget. Proceed to Phase 1 (C++ generalization).")
else:
    print("-> MARGINAL: adaptive helps but < 2x. Inspect error-vs-time plot before")
    print("   committing to Phase 1.")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for r, color in ((res_u8, 'tab:red'), (res_ad, 'tab:green'), (res_u4, 'tab:blue')):
    axes[0].plot(r['ts'], r['err'], label=f"{r['name']} (flip RMSE {r['rmse_flip']:.2f}°)",
                 lw=0.8, color=color)
axes[0].set_ylabel('geodesic error [deg]')
axes[0].legend()
axes[0].set_title(f'{BAG_KEY}: SO(3) spline representation error vs MoCap')
ax2 = axes[1]
ax2.plot(t_om, om_smooth, 'k-', lw=0.8, label='|omega| smoothed [rad/s]')
ax2.axhline(OMEGA_FLIP, color='gray', ls='--', lw=0.6)
kt = res_ad['kt']
ax2b = ax2.twinx()
ax2b.plot(0.5 * (kt[:-1] + kt[1:]), 1e3 * np.diff(kt), 'g.', ms=2, label='adaptive dt [ms]')
ax2b.set_ylabel('knot dt [ms]', color='g')
ax2.set_ylabel('|omega| [rad/s]')
ax2.set_xlabel('t [s]')
ax2.legend(loc='upper left')
out = Path(__file__).parent.parent.parent / 'plots' / 'adaptive_knots'
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'v1_representation_error.png', dpi=130, bbox_inches='tight')
print(f"\nplot: {out / 'v1_representation_error.png'}")
