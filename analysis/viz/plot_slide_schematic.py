#!/usr/bin/env python3
"""Schematic single-slide figure: radar + continuous time + odometry.

Side view, fully synthetic (no bag data): ground truth is a sine spanning
the full width; the estimate (a coarse B-spline with visible control
points, slightly off) and the drone reach only the middle -- the live edge
of odometry. The drone is descending; ahead of it, radar returns sit on the
floor line, each ray measuring the projection of the velocity onto it.

Usage (from analysis/):  ../.venv/bin/python3 viz/plot_slide_schematic.py
"""
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Ellipse, FancyArrow, FancyBboxPatch
from scipy.interpolate import make_lsq_spline

# ---------------- tunables ----------------
X_MAX   = 10.0
X_DRONE = 5.0
MEAN_Z, AMP, PERIOD = 1.9, 0.85, 6.0
CREST_X = 3.4        # sine crest location -> descending at X_DRONE
LAG, DRIFT = 0.22, 0.055   # estimate: phase lag (m) + linear drift (m/m)
KNOT_DT = 0.62       # didactic CP spacing along x (m)
RET_X = np.array([6.1, 6.8, 7.6, 8.4, 9.1, 9.7])   # floor returns
SPEED = 1.4          # |v| for the schematic Doppler coloring
# ------------------------------------------


def gt_z(x):
    return MEAN_Z + AMP * np.sin(2 * np.pi * (np.asarray(x) - CREST_X) / PERIOD
                                 + np.pi / 2)


def est_z(x):
    x = np.asarray(x)
    return gt_z(x - LAG) + 0.04 + DRIFT * x


xx_gt = np.linspace(0, X_MAX, 400)
xx_est = np.linspace(0, X_DRONE, 200)

# didactic spline through the estimate segment (curve IS spline(CPs))
k = 3
inner = np.arange(KNOT_DT, X_DRONE - KNOT_DT / 2, KNOT_DT)
knots = np.r_[[0] * (k + 1), inner, [X_DRONE] * (k + 1)]
spl = make_lsq_spline(xx_est, est_z(xx_est), knots, k)
cps_x = np.array([knots[i + 1:i + k + 1].mean() for i in range(len(spl.c))])
cps = np.column_stack([cps_x, spl.c])   # Greville abscissae x coefficient z

p_d = np.array([X_DRONE, float(spl(X_DRONE))])
slope = float(spl.derivative()(X_DRONE))
v_dir = np.array([1.0, slope]) / np.hypot(1.0, slope)
v = SPEED * v_dir
tilt = np.arctan2(slope, 1.0) * 0.7          # nose-down, slightly damped

# ---- figure ----
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(11, 5.6))

# floor with ground hatching
ax.plot([-0.2, X_MAX + 0.2], [0, 0], color='0.25', lw=2.5, zorder=2)
for hx in np.arange(0.1, X_MAX + 0.2, 0.45):
    ax.plot([hx, hx - 0.18], [0, -0.16], color='0.55', lw=1.1, zorder=1)

# curves
ax.plot(xx_gt, gt_z(xx_gt), color='0.55', lw=2.6, zorder=3)
ax.plot(xx_est, spl(xx_est), color='#1565c0', lw=3.0, zorder=4)
ax.plot(cps[:, 0], cps[:, 1], '--', color='#1565c0', lw=1.0, alpha=0.45,
        zorder=3)
ax.plot(cps[:, 0], cps[:, 1], 'o', ms=9, mfc='white', mec='#1565c0', mew=2.0,
        zorder=5)

# radar: rays + returns colored by radial velocity (projection on the ray)
rets = np.column_stack([RET_X, np.zeros_like(RET_X)])
rad_v = []
for r in rets:
    u = (r - p_d) / np.linalg.norm(r - p_d)
    rad_v.append(float(-np.dot(u, v)))       # TI: positive = receding
    ax.plot([p_d[0], r[0]], [p_d[1], r[1]], color='0.78', lw=1.1, zorder=2)
ax.scatter(rets[:, 0], rets[:, 1], c=rad_v, cmap='coolwarm', vmin=-1.6,
           vmax=1.6, s=110, zorder=6, edgecolors='0.3', linewidths=0.8)

# velocity arrow (tangent, descending)
ax.add_patch(FancyArrow(p_d[0], p_d[1], v[0] * 0.85, v[1] * 0.85,
                        width=0.028, head_width=0.13, head_length=0.16,
                        color='#2e7d32', zorder=7, length_includes_head=True))

# ---- drone pictogram (side view, recognizable quad) ----
loc = (transforms.Affine2D().rotate(tilt).translate(*p_d) + ax.transData)
body = FancyBboxPatch((-0.21, -0.075), 0.42, 0.15,
                      boxstyle='round,pad=0.02,rounding_size=0.06',
                      fc='0.15', ec='k', transform=loc, zorder=8)
ax.add_patch(body)
for sx in (-1, 1):
    ax.plot([0, sx * 0.34], [0.03, 0.14], color='k', lw=2.6, transform=loc,
            zorder=8, solid_capstyle='round')                    # arm
    ax.add_patch(Ellipse((sx * 0.34, 0.185), 0.40, 0.05, fc='0.45', ec='k',
                         lw=1.2, transform=loc, zorder=9))       # rotor disc
    ax.plot([sx * 0.34, sx * 0.34], [0.14, 0.165], color='k', lw=2.2,
            transform=loc, zorder=8)                             # motor
    ax.plot([sx * 0.10, sx * 0.17], [-0.075, -0.20], color='k', lw=2.0,
            transform=loc, zorder=8, solid_capstyle='round')     # leg
ax.plot([0.21, 0.30], [0.0, -0.03], color='#2e7d32', lw=0, transform=loc)

# start marker
ax.plot(0, est_z(0), 's', ms=11, mfc='0.3', mec='k', zorder=6)

# ---- direct labels ----
ax.text(9.05, 3.05, 'ground truth', color='0.4', fontsize=16,
        ha='center', va='bottom')
ax.text(-0.15, 2.92, 'estimated trajectory\n(continuous-time B-spline)',
        color='#1565c0', fontsize=16, ha='left', va='bottom',
        fontweight='bold')
i_cp = 5
ax.annotate('control points', xy=(cps[i_cp, 0], cps[i_cp, 1]),
            xytext=(2.9, 0.55), textcoords='data', color='#1565c0',
            fontsize=15, ha='left',
            arrowprops=dict(arrowstyle='-', color='#1565c0', lw=1.0,
                            shrinkB=8))
ax.annotate('velocity', xy=(p_d[0] + v[0] * 0.85, p_d[1] + v[1] * 0.85),
            xytext=(10, 4), textcoords='offset points', color='#2e7d32',
            fontsize=15, fontweight='bold', bbox=dict(fc='white', ec='none', alpha=0.9, pad=2.5))
ax.annotate('radar returns\n(measure radial velocity)', xy=(8.0, 0.05),
            xytext=(0, 30), textcoords='offset points', color='0.25',
            fontsize=15, ha='center', bbox=dict(fc='white', ec='none', alpha=0.9, pad=2.5))
ax.annotate('start', xy=(0, est_z(0)), xytext=(-4, -34),
            textcoords='offset points', fontsize=14, color='0.3', ha='left',
            bbox=dict(fc='white', ec='none', alpha=0.9, pad=2.5))

ax.set_title('Continuous-Time Radar-Inertial Odometry',
             color='#0065BD', fontsize=21, fontweight='bold', pad=18)
ax.set_xlim(-0.4, X_MAX + 0.4)
ax.set_ylim(-0.35, 3.35)
ax.set_aspect('equal')
ax.axis('off')
fig.tight_layout()

out = Path(__file__).resolve().parents[2] / 'plots' / 'slide'
out.mkdir(parents=True, exist_ok=True)
for fmt in ('pdf', 'svg', 'png'):
    fig.savefig(out / f'slide_schematic.{fmt}', dpi=300,
                bbox_inches='tight', pad_inches=0.15)
print(f'saved plots/slide/slide_schematic.[pdf|svg|png] '
      f'({len(cps)} CPs, radial velocities '
      + ' '.join(f'{r:+.2f}' for r in rad_v) + ')')
