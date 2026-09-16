#!/usr/bin/env python3
"""Single-slide explainer figure: radar + continuous time + odometry.

Top-down (xy) view of one slow-racing lap with, in one frame:
  - the estimated trajectory drawn as a coarse DIDACTIC B-spline (0.5 s
    knots, LSQ-fit to the deployed estimate) with its control points and
    dashed control polygon -- the drawn curve IS the spline of the drawn
    control points (real shape, schematic density; the deployed grid is
    40 ms and would be an unreadable bead chain on a slide),
  - the MoCap ground-truth trajectory,
  - a drone glyph at one time T_STAR with its velocity arrow,
  - one radar frame transformed to world, colored by measured Doppler,
  - thin rays from the drone to a few returns: each return measures the
    projection of the velocity arrow onto its ray -- the whole sensor
    principle in one glance.

Inputs: the traj-arrays npz of a --save-arrays run (regenerated via
subprocess if missing) for the two curves; the bag itself for the pose and
the radar frame. Outputs PDF/SVG/PNG to plots/slide/.

Usage (from analysis/):  ../.venv/bin/python3 viz/plot_slide_figure.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from scipy.interpolate import make_lsq_spline
from scipy.spatial.transform import Rotation, Slerp

from rosbag_loader import load_bag_topics                    # noqa: E402
from radar_velocity_utils import rotation_matrix_from_euler  # noqa: E402

# ---------------- tunables ----------------
BAG_KEY   = 'slow_racing_best_velocity'
T_STAR    = 9.5     # seconds after eval start: where the drone sits
KNOT_DT   = 0.5     # didactic spline knot spacing (s)
N_RAYS    = 7       # rays drawn from drone to returns
SPAN      = 12.0    # seconds of trajectory drawn (one lap, keeps it clean)
ARROW_S   = 0.9     # velocity arrow scale (s of travel)
# ------------------------------------------

ANALYSIS = Path(__file__).resolve().parent.parent
REPO = ANALYSIS.parent
NPZ = REPO / 'plots' / BAG_KEY / 'live_solver' / \
    f'traj_arrays_{BAG_KEY}_mocap-init_mocap-heading_sw.npz'

if not NPZ.exists():
    print('npz missing; running the deployed command with --save-arrays ...')
    subprocess.run(
        [sys.executable, 'validate_live_solver.py', BAG_KEY, '--mocap-yaw',
         '--cpp', '--sliding-window', '--no-plot', '--save-arrays',
         '--set', 'marg_prior_scale=1.0', '--set', 'accel_soft_sigma=8.0',
         '--set', 'dt_pos=0.04', '--set', 'dt_ori=0.016',
         '--set', 'lambda_heading=0.6', '--set', 'lock_extrinsics=1',
         '--set-ext', 'rotation_euler_deg=[180.0,27.5,0.0]'],
        cwd=str(ANALYSIS), check=True)

d = np.load(NPZ)
t_rel = d['t_rel']
est3 = d['settled']
gt3 = d['mocap']
m = t_rel <= SPAN
t_rel, est3, gt3 = t_rel[m], est3[m], gt3[m]
est, gt = est3[:, :2], gt3[:, :2]

# ---- didactic spline: LSQ cubic B-spline with KNOT_DT knots on the estimate
k = 3
t0, t1 = t_rel[0], t_rel[-1]
inner = np.arange(t0 + KNOT_DT, t1 - KNOT_DT / 2, KNOT_DT)
knots = np.r_[[t0] * (k + 1), inner, [t1] * (k + 1)]
spl_x = make_lsq_spline(t_rel, est3[:, 0], knots, k)
spl_y = make_lsq_spline(t_rel, est3[:, 1], knots, k)
spl_z = make_lsq_spline(t_rel, est3[:, 2], knots, k)
tt = np.linspace(t0, t1, 800)
curve3 = np.column_stack([spl_x(tt), spl_y(tt), spl_z(tt)])
cps3 = np.column_stack([spl_x.c, spl_y.c, spl_z.c])
curve = curve3[:, :2]
cps = cps3[:, :2]      # B-spline coefficients = CPs

# ---- bag: pose + radar frame at T_STAR ----
cfg = yaml.safe_load((ANALYSIS / 'config' / 'bags.yaml').read_text())
ext = yaml.safe_load((ANALYSIS / 'config' / 'extrinsics.yaml').read_text())
bag_rel = cfg['bags'][BAG_KEY]
bag_path = (REPO / bag_rel)
if not bag_path.exists():
    bag_path = (REPO / '..' / bag_rel).resolve()
start_off, dur = cfg['timing'][BAG_KEY]
bd = load_bag_topics(str(bag_path), verbose=False)
t_abs0 = bd.start_time + start_off
t_star_abs = t_abs0 + T_STAR

R_bs = rotation_matrix_from_euler(*np.radians([180.0, 27.5, 0.0]))
t_bs = np.asarray(ext['translation_body_m'], float)
off = ext['imu_mocap_offset_sec'] - ext['radar_imu_offset_sec']

gt_states = [s for s in bd.agiros_state
             if t_abs0 - 2 <= s.timestamp <= t_abs0 + dur]
st_t = np.array([s.timestamp for s in gt_states])
st_p = np.array([s.position for s in gt_states])
st_v = np.array([s.velocity for s in gt_states])
st_R = Rotation.from_quat(np.array([s.orientation for s in gt_states]))
slerp = Slerp(st_t, st_R)

frames = [f for f in bd.radar_velocity if f.velocities is not None
          and len(f.velocities) >= 12]
f_star = min(frames, key=lambda f: abs(f.timestamp + off - t_star_abs))
tf = np.clip(f_star.timestamp + off, st_t[0], st_t[-1])
R_wb = slerp(tf).as_matrix()
p_wb = np.array([np.interp(tf, st_t, st_p[:, i]) for i in range(3)])
v_w = np.array([np.interp(tf, st_t, st_v[:, i]) for i in range(3)])

P_s = np.asarray(f_star.positions, float)
V_m = np.asarray(f_star.velocities, float)
keep = np.linalg.norm(P_s, axis=1) >= 0.2
P_s, V_m = P_s[keep], V_m[keep]
P_w = (R_wb @ (R_bs @ P_s.T + t_bs[:, None])).T + p_wb

# The npz curves are SE3-aligned to mocap; the bag pose lives in the same
# Vicon frame up to that small alignment, so drawing both together is fine
# at slide scale. Place the drone exactly on the GT curve point nearest tf.
i_gt = np.argmin(np.abs(t_rel - (tf - t_abs0)))
p_draw3 = gt3[i_gt].copy()
p_draw = p_draw3[:2]
P_w3 = P_w + (p_draw3 - p_wb)       # cloud in the aligned frame, full 3D
P_w = P_w3.copy()
P_w[:, :2] = P_w3[:, :2]            # 2D path uses the same aligned xy

# ---- figure ----
plt.rcParams.update({'font.size': 15, 'axes.linewidth': 0.8})
fig, ax = plt.subplots(figsize=(9.5, 7))

ax.plot(gt[:, 0], gt[:, 1], color='0.55', lw=2.2, label='ground truth (MoCap)',
        zorder=2)
ax.plot(curve[:, 0], curve[:, 1], color='#1565c0', lw=2.6,
        label='estimated trajectory (B-spline)', zorder=3)
ax.plot(cps[:, 0], cps[:, 1], '--', color='#1565c0', lw=0.9, alpha=0.45,
        zorder=2)
ax.plot(cps[:, 0], cps[:, 1], 'o', ms=7, mfc='white', mec='#1565c0', mew=1.8,
        label='control points', zorder=4)

sc = ax.scatter(P_w[:, 0], P_w[:, 1], c=V_m, cmap='coolwarm', vmin=-1.5,
                vmax=1.5, s=26, alpha=0.9, edgecolors='none',
                label='radar returns (color = radial velocity)', zorder=3)

# rays: N_RAYS returns spread in azimuth
az = np.arctan2(P_w[:, 1] - p_draw[1], P_w[:, 0] - p_draw[0])
order = np.argsort(az)
picks = order[np.linspace(0, len(order) - 1, N_RAYS).astype(int)]
for j in picks:
    ax.plot([p_draw[0], P_w[j, 0]], [p_draw[1], P_w[j, 1]], color='0.75',
            lw=0.9, zorder=1)

# velocity arrow
ax.add_patch(FancyArrow(p_draw[0], p_draw[1], v_w[0] * ARROW_S,
                        v_w[1] * ARROW_S, width=0.035, head_width=0.16,
                        head_length=0.18, color='#2e7d32', zorder=6,
                        length_includes_head=True))
ax.annotate('velocity', xy=(p_draw[0] + v_w[0] * ARROW_S,
                            p_draw[1] + v_w[1] * ARROW_S),
            xytext=(6, 6), textcoords='offset points', color='#2e7d32',
            fontsize=13, fontweight='bold')

# drone glyph: X-quad, yaw from GT attitude
yaw = np.arctan2(R_wb[1, 0], R_wb[0, 0])
arm = 0.22
for a in (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4):
    dx, dy = arm * np.cos(yaw + a), arm * np.sin(yaw + a)
    ax.plot([p_draw[0], p_draw[0] + dx], [p_draw[1], p_draw[1] + dy],
            color='k', lw=2.0, zorder=7, solid_capstyle='round')
    ax.add_patch(Circle((p_draw[0] + dx, p_draw[1] + dy), 0.09, fc='white',
                        ec='k', lw=1.6, zorder=8))
ax.add_patch(Circle((p_draw[0], p_draw[1]), 0.055, fc='k', zorder=8))

# start marker
ax.plot(gt[0, 0], gt[0, 1], 's', ms=10, mfc='0.3', mec='k', zorder=5)
ax.annotate('start', xy=(gt[0, 0], gt[0, 1]), xytext=(8, -14),
            textcoords='offset points', fontsize=13, color='0.25')

ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend(loc='upper left', fontsize=12.5, framealpha=0.95,
          handletextpad=0.6, borderpad=0.7)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()

out = REPO / 'plots' / 'slide'
out.mkdir(parents=True, exist_ok=True)
for fmt in ('pdf', 'svg', 'png'):
    fig.savefig(out / f'slide_figure.{fmt}', dpi=300, bbox_inches='tight', pad_inches=0.25)
print(f'saved plots/slide/slide_figure.[pdf|svg|png]  '
      f'({len(cps)} CPs, frame at t={tf - t_abs0:.1f}s, '
      f'{len(P_w)} returns, |v|={np.linalg.norm(v_w):.2f} m/s)')

# ======================= 3D variant =======================
ELEV, AZIM = 28, -55        # camera; iterate visually

fig3 = plt.figure(figsize=(10.5, 8))
ax3 = fig3.add_subplot(projection='3d')

# ground-plane shadow (depth cue)
ax3.plot(curve3[:, 0], curve3[:, 1], 0, color='0.85', lw=1.6, zorder=1)
ax3.plot([p_draw3[0]], [p_draw3[1]], [0], 'o', ms=5, color='0.8', zorder=1)

ax3.plot(gt3[:, 0], gt3[:, 1], gt3[:, 2], color='0.55', lw=2.0,
         label='ground truth (MoCap)')
ax3.plot(curve3[:, 0], curve3[:, 1], curve3[:, 2], color='#1565c0', lw=2.4,
         label='estimated trajectory (B-spline)')
ax3.plot(cps3[:, 0], cps3[:, 1], cps3[:, 2], '--', color='#1565c0', lw=0.8,
         alpha=0.45)
ax3.plot(cps3[:, 0], cps3[:, 1], cps3[:, 2], 'o', ms=6, mfc='white',
         mec='#1565c0', mew=1.6, label='control points')

ax3.scatter(P_w3[:, 0], P_w3[:, 1], P_w3[:, 2], c=V_m, cmap='coolwarm',
            vmin=-1.5, vmax=1.5, s=30, alpha=0.95, edgecolors='none',
            label='radar returns (color = radial velocity)')

az3 = np.arctan2(P_w3[:, 1] - p_draw3[1], P_w3[:, 0] - p_draw3[0])
order3 = np.argsort(az3)
picks3 = order3[np.linspace(0, len(order3) - 1, N_RAYS).astype(int)]
for j in picks3:
    ax3.plot([p_draw3[0], P_w3[j, 0]], [p_draw3[1], P_w3[j, 1]],
             [p_draw3[2], P_w3[j, 2]], color='0.75', lw=0.8)

ax3.quiver(p_draw3[0], p_draw3[1], p_draw3[2],
           v_w[0] * ARROW_S, v_w[1] * ARROW_S, v_w[2] * ARROW_S,
           color='#2e7d32', lw=2.6, arrow_length_ratio=0.15)
ax3.text(p_draw3[0] + v_w[0] * ARROW_S, p_draw3[1] + v_w[1] * ARROW_S,
         p_draw3[2] + v_w[2] * ARROW_S + 0.12, 'velocity', color='#2e7d32',
         fontsize=13, fontweight='bold')

# drone glyph
for a in (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4):
    dx, dy = 0.26 * np.cos(yaw + a), 0.26 * np.sin(yaw + a)
    ax3.plot([p_draw3[0], p_draw3[0] + dx], [p_draw3[1], p_draw3[1] + dy],
             [p_draw3[2], p_draw3[2]], color='k', lw=2.0,
             solid_capstyle='round')
    ax3.scatter([p_draw3[0] + dx], [p_draw3[1] + dy], [p_draw3[2]], s=60,
                facecolors='white', edgecolors='k', linewidths=1.6,
                depthshade=False)
ax3.scatter([p_draw3[0]], [p_draw3[1]], [p_draw3[2]], s=25, color='k',
            depthshade=False)

ax3.plot([gt3[0, 0]], [gt3[0, 1]], [gt3[0, 2]], 's', ms=9, mfc='0.3',
         mec='k')
ax3.text(gt3[0, 0] + 0.15, gt3[0, 1], gt3[0, 2] + 0.15, 'start',
         fontsize=13, color='0.25')

# clean panes, honest proportions
ax3.view_init(elev=ELEV, azim=AZIM)
allp = np.vstack([gt3, curve3, P_w3, [[*p_draw3]]])
rng = allp.max(axis=0) - allp.min(axis=0)
ax3.set_box_aspect(rng)
for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
    pane.pane.set_visible(False)
    pane._axinfo['grid'].update(color='0.9', linewidth=0.5)
ax3.set_zlim(bottom=0)
ax3.set_xlabel('x (m)', labelpad=8)
ax3.set_ylabel('y (m)', labelpad=8)
ax3.set_zlabel('z (m)', labelpad=10)
ax3.legend(loc='upper left', fontsize=12, framealpha=0.95,
           handletextpad=0.6, borderpad=0.7)
fig3.tight_layout()
for fmt in ('pdf', 'svg', 'png'):
    fig3.savefig(out / f'slide_figure_3d.{fmt}', dpi=300)
print('saved plots/slide/slide_figure_3d.[pdf|svg|png]')
