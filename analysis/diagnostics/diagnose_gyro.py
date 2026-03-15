import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""Diagnose: compare IMU gyro against MoCap-derived angular velocity.
Check for per-axis sign/scale/offset errors and frame misalignment."""
import numpy as np
from scipy.interpolate import interp1d
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import quat_to_rotation_matrix
from config_loader import load_config

_cfg = load_config()
BAGS = _cfg['bags']['bags']
FLIPPED_BAGS = set(_cfg['bags']['flipped'])
_TIMING = _cfg['timing']
_EXT = _cfg['extrinsics']

bag_key = sys.argv[1] if len(sys.argv) > 1 else "circle"
BAG_PATH = BAGS[bag_key]
START_OFFSET, DURATION = _TIMING.get(bag_key, [0.0, 5.0])
FLIP = bag_key in FLIPPED_BAGS
IMU_MOCAP_OFFSET = _EXT['imu_mocap_offset_sec']

print(f"Bag: {bag_key}, flip={FLIP}")
bag_data = load_bag_topics(BAG_PATH, verbose=False)
t_start = bag_data.start_time + START_OFFSET
t_end = t_start + DURATION

agiros = [s for s in bag_data.agiros_state if t_start <= s.timestamp <= t_end]
imu = [d for d in bag_data.imu_data if t_start <= d.timestamp <= t_end]

# Apply time offset to IMU
for d in imu:
    d.timestamp += IMU_MOCAP_OFFSET

agiros_times = np.array([s.timestamp for s in agiros])
agiros_omega = np.array([s.angular_velocity for s in agiros])
agiros_quat = np.array([s.orientation for s in agiros])

imu_times = np.array([d.timestamp for d in imu])
imu_gyro = np.array([d.angular_velocity for d in imu])
imu_accel = np.array([d.linear_acceleration for d in imu])

print(f"MoCap samples: {len(agiros)}, IMU samples: {len(imu)}")

# ===== 1. MoCap angular velocity (from agiros state) =====
# Agiros already provides omega_body. Let's also numerically differentiate quaternions as a cross-check.
print("\n" + "="*60)
print("1. MoCap angular velocity (agiros state)")
print("="*60)
print(f"  omega range: x=[{agiros_omega[:,0].min():.3f}, {agiros_omega[:,0].max():.3f}]")
print(f"               y=[{agiros_omega[:,1].min():.3f}, {agiros_omega[:,1].max():.3f}]")
print(f"               z=[{agiros_omega[:,2].min():.3f}, {agiros_omega[:,2].max():.3f}]")
print(f"  omega mean:  [{agiros_omega[:,0].mean():.4f}, {agiros_omega[:,1].mean():.4f}, {agiros_omega[:,2].mean():.4f}]")

# ===== 2. Raw IMU gyro =====
print("\n" + "="*60)
print("2. Raw IMU gyro")
print("="*60)
print(f"  gyro range:  x=[{imu_gyro[:,0].min():.3f}, {imu_gyro[:,0].max():.3f}]")
print(f"               y=[{imu_gyro[:,1].min():.3f}, {imu_gyro[:,1].max():.3f}]")
print(f"               z=[{imu_gyro[:,2].min():.3f}, {imu_gyro[:,2].max():.3f}]")
print(f"  gyro mean:   [{imu_gyro[:,0].mean():.4f}, {imu_gyro[:,1].mean():.4f}, {imu_gyro[:,2].mean():.4f}]")

# ===== 3. Interpolate MoCap omega at IMU timestamps =====
omega_interp = interp1d(agiros_times, agiros_omega, axis=0, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
mocap_omega_at_imu = omega_interp(imu_times)

# ===== 4. Per-axis correlation and statistics =====
print("\n" + "="*60)
print("4. Per-axis comparison: IMU gyro vs MoCap omega")
print("="*60)
for ax, label in enumerate(['x', 'y', 'z']):
    corr = np.corrcoef(imu_gyro[:, ax], mocap_omega_at_imu[:, ax])[0, 1]
    diff = imu_gyro[:, ax] - mocap_omega_at_imu[:, ax]
    print(f"  {label}: corr={corr:+.4f}, diff_mean={diff.mean():+.4f} rad/s ({np.degrees(diff.mean()):+.2f} deg/s), diff_std={diff.std():.4f}")

# ===== 5. Check all sign permutations =====
print("\n" + "="*60)
print("5. Sign permutation check (which sign combo gives best correlation?)")
print("="*60)
print(f"  {'sx':>3s} {'sy':>3s} {'sz':>3s} {'corr_x':>8s} {'corr_y':>8s} {'corr_z':>8s} {'total_r':>8s} {'mean_diff':>30s}")
best_signs = None
best_total = -1e10
for sx in [+1, -1]:
    for sy in [+1, -1]:
        for sz in [+1, -1]:
            test_gyro = imu_gyro * np.array([sx, sy, sz])
            corrs = []
            diffs = []
            for ax in range(3):
                c = np.corrcoef(test_gyro[:, ax], mocap_omega_at_imu[:, ax])[0, 1]
                d = test_gyro[:, ax] - mocap_omega_at_imu[:, ax]
                corrs.append(c)
                diffs.append(d.mean())
            total = sum(corrs)
            if total > best_total:
                best_total = total
                best_signs = (sx, sy, sz)
            print(f"  {sx:+3d} {sy:+3d} {sz:+3d} {corrs[0]:+8.4f} {corrs[1]:+8.4f} {corrs[2]:+8.4f} {total:+8.4f} [{diffs[0]:+.4f}, {diffs[1]:+.4f}, {diffs[2]:+.4f}]")

print(f"\n  Best signs: {best_signs}, total correlation: {best_total:+.4f}")

# ===== 6. Check axis permutations (what if x,y,z are swapped?) =====
print("\n" + "="*60)
print("6. Axis permutation check (are axes swapped?)")
print("="*60)
import itertools
print(f"  {'perm':>10s} {'corr_x':>8s} {'corr_y':>8s} {'corr_z':>8s} {'total':>8s}")
best_perm = None
best_perm_total = -1e10
for perm in itertools.permutations([0, 1, 2]):
    for sx in [+1, -1]:
        for sy in [+1, -1]:
            for sz in [+1, -1]:
                signs = np.array([sx, sy, sz])
                test_gyro = imu_gyro[:, list(perm)] * signs
                corrs = []
                for ax in range(3):
                    c = np.corrcoef(test_gyro[:, ax], mocap_omega_at_imu[:, ax])[0, 1]
                    corrs.append(c)
                total = sum(corrs)
                if total > best_perm_total:
                    best_perm_total = total
                    best_perm = (perm, tuple(signs.tolist()))

perm, signs = best_perm
print(f"  Best: axes={perm}, signs={signs}, total corr={best_perm_total:+.4f}")
# Show details for best
test_gyro = imu_gyro[:, list(perm)] * np.array(signs)
for ax, label in enumerate(['x', 'y', 'z']):
    corr = np.corrcoef(test_gyro[:, ax], mocap_omega_at_imu[:, ax])[0, 1]
    diff = test_gyro[:, ax] - mocap_omega_at_imu[:, ax]
    print(f"    {label}: corr={corr:+.4f}, diff_mean={diff.mean():+.4f} rad/s ({np.degrees(diff.mean()):+.2f} deg/s)")

# ===== 7. Time offset sweep for gyro =====
print("\n" + "="*60)
print("7. Time offset sweep (is the 20ms offset correct for gyro?)")
print("="*60)
best_offset = 0
best_offset_corr = -1e10
for dt_ms in range(-50, 52, 2):
    dt = dt_ms / 1000.0
    shifted_mocap = omega_interp(imu_times - dt)  # shift MoCap by dt
    total_corr = 0
    for ax in range(3):
        c = np.corrcoef(imu_gyro[:, ax], shifted_mocap[:, ax])[0, 1]
        total_corr += c
    if total_corr > best_offset_corr:
        best_offset_corr = total_corr
        best_offset = dt_ms
    if dt_ms % 10 == 0:
        print(f"  dt={dt_ms:+4d}ms: total_corr={total_corr:+.4f}")

print(f"\n  Best offset: {best_offset}ms (total corr={best_offset_corr:+.4f})")
print(f"  Current offset: {int(IMU_MOCAP_OFFSET*1000)}ms")

# ===== 8. Detailed z-axis comparison =====
print("\n" + "="*60)
print("8. Z-axis gyro detailed comparison")
print("="*60)
z_diff = imu_gyro[:, 2] - mocap_omega_at_imu[:, 2]
print(f"  IMU gyro_z:   mean={imu_gyro[:,2].mean():+.4f}, std={imu_gyro[:,2].std():.4f}")
print(f"  MoCap omega_z: mean={mocap_omega_at_imu[:,2].mean():+.4f}, std={mocap_omega_at_imu[:,2].std():.4f}")
print(f"  Difference:    mean={z_diff.mean():+.4f} rad/s = {np.degrees(z_diff.mean()):+.2f} deg/s")
print(f"                 std={z_diff.std():.4f} rad/s = {np.degrees(z_diff.std()):.2f} deg/s")
print(f"  Scale factor:  {imu_gyro[:,2].std()/mocap_omega_at_imu[:,2].std():.4f}")

# ===== 9. Time-windowed z-axis residual (is it constant or drifting?) =====
print("\n" + "="*60)
print("9. Z-axis gyro residual over time (is bias constant?)")
print("="*60)
n_windows = 8
window_size = len(imu) // n_windows
for w in range(n_windows):
    start = w * window_size
    end = (w + 1) * window_size if w < n_windows - 1 else len(imu)
    t_w = imu_times[start:end]
    z_d = z_diff[start:end]
    t_rel = t_w.mean() - agiros_times[0]
    print(f"  t={t_rel:5.1f}s: z_diff_mean={z_d.mean():+.4f} rad/s ({np.degrees(z_d.mean()):+.2f} deg/s)")

# ===== 10. Solver-style residual: what the solver actually sees =====
print("\n" + "="*60)
print("10. Solver-style gyro residual: r = z_gyro - omega_body - b_g")
print("    (omega_body from MoCap, b_g = 0)")
print("="*60)
# r = imu_gyro - mocap_omega_at_imu
solver_resid = imu_gyro - mocap_omega_at_imu
for ax, label in enumerate(['x', 'y', 'z']):
    rms = np.sqrt(np.mean(solver_resid[:, ax]**2))
    print(f"  {label}: mean={solver_resid[:,ax].mean():+.4f}, std={solver_resid[:,ax].std():.4f}, RMSE={rms:.4f} rad/s")

# Total cost contribution with lambda=0.5
lambda_gyro = 0.5
cost = 0.5 * lambda_gyro * np.sum(solver_resid**2)
print(f"\n  Total gyro cost (lambda=0.5): {cost:.1f}")
print(f"  Per-axis cost: x={0.5*lambda_gyro*np.sum(solver_resid[:,0]**2):.1f}, "
      f"y={0.5*lambda_gyro*np.sum(solver_resid[:,1]**2):.1f}, "
      f"z={0.5*lambda_gyro*np.sum(solver_resid[:,2]**2):.1f}")

# What if we subtract the mean bias from each axis?
bias_est = solver_resid.mean(axis=0)
corrected = solver_resid - bias_est
cost_corrected = 0.5 * lambda_gyro * np.sum(corrected**2)
print(f"\n  If b_g = {bias_est} were subtracted:")
print(f"  Total cost would be: {cost_corrected:.1f} (saved {cost - cost_corrected:.1f})")
print(f"  ({np.degrees(bias_est[0]):+.2f}, {np.degrees(bias_est[1]):+.2f}, {np.degrees(bias_est[2]):+.2f}) deg/s")

# ===== 11. Integrate IMU gyro → orientation and compare to MoCap =====
print("\n" + "="*60)
print("11. Gyro integration → orientation drift visualization")
print("="*60)
from scipy.spatial.transform import Rotation

# MoCap Euler angles (roll, pitch, yaw) from quaternions [qx, qy, qz, qw]
mocap_rots = Rotation.from_quat(agiros_quat)  # scipy expects [x, y, z, w]
mocap_euler = mocap_rots.as_euler('xyz', degrees=True)  # roll, pitch, yaw
ag_t_rel = agiros_times - agiros_times[0]

# Integrate IMU gyro to get orientation
# Initialize with MoCap orientation at first IMU timestamp
imu_t_rel = imu_times - agiros_times[0]
# Find MoCap orientation closest to first IMU time
idx0 = np.argmin(np.abs(agiros_times - imu_times[0]))
R_current = Rotation.from_quat(agiros_quat[idx0])

imu_euler_raw = np.zeros((len(imu_times), 3))  # integrated WITHOUT bias correction
imu_euler_corrected = np.zeros((len(imu_times), 3))  # integrated WITH bias correction

# Raw integration (no bias subtracted)
R_raw = Rotation.from_quat(agiros_quat[idx0])
for i in range(len(imu_times)):
    imu_euler_raw[i] = R_raw.as_euler('xyz', degrees=True)
    if i < len(imu_times) - 1:
        dt = imu_times[i + 1] - imu_times[i]
        omega = imu_gyro[i]
        angle = np.linalg.norm(omega) * dt
        if angle > 1e-10:
            axis = omega / np.linalg.norm(omega)
            R_raw = R_raw * Rotation.from_rotvec(omega * dt)

# Corrected integration (bias subtracted)
R_corr = Rotation.from_quat(agiros_quat[idx0])
for i in range(len(imu_times)):
    imu_euler_corrected[i] = R_corr.as_euler('xyz', degrees=True)
    if i < len(imu_times) - 1:
        dt = imu_times[i + 1] - imu_times[i]
        omega = imu_gyro[i] - bias_est  # subtract estimated bias
        R_corr = R_corr * Rotation.from_rotvec(omega * dt)

# Print final drift
drift_raw = imu_euler_raw[-1] - mocap_euler[-1]
drift_corr = imu_euler_corrected[-1] - mocap_euler[-1]
print(f"  Duration: {imu_t_rel[-1]:.1f}s")
print(f"  Final drift (raw):       roll={drift_raw[0]:+.1f}°  pitch={drift_raw[1]:+.1f}°  yaw={drift_raw[2]:+.1f}°")
print(f"  Final drift (corrected): roll={drift_corr[0]:+.1f}°  pitch={drift_corr[1]:+.1f}°  yaw={drift_corr[2]:+.1f}°")
print(f"  Bias used for correction: ({np.degrees(bias_est[0]):+.2f}, {np.degrees(bias_est[1]):+.2f}, {np.degrees(bias_est[2]):+.2f}) deg/s")

# ===== Plot =====
import matplotlib.pyplot as plt

# Interpolate MoCap euler onto IMU timestamps for RMSE computation
mocap_euler_at_imu = np.zeros((len(imu_times), 3))
for ax_i in range(3):
    mocap_euler_at_imu[:, ax_i] = np.interp(imu_times, agiros_times, mocap_euler[:, ax_i])

# Per-axis RMSE
rmse_raw = np.sqrt(np.mean((imu_euler_raw - mocap_euler_at_imu)**2, axis=0))
rmse_corr = np.sqrt(np.mean((imu_euler_corrected - mocap_euler_at_imu)**2, axis=0))
rmse_raw_mean = np.mean(rmse_raw)
rmse_corr_mean = np.mean(rmse_corr)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(f'IMU Gyro Integration vs MoCap Orientation — {bag_key}', fontsize=14, fontweight='bold')
labels = ['Roll', 'Pitch', 'Yaw']

for ax_i, (ax, label) in enumerate(zip(axes, labels)):
    ax.plot(ag_t_rel, mocap_euler[:, ax_i], 'k-', linewidth=1.5, alpha=0.8, label='MoCap (ground truth)')
    ax.plot(imu_t_rel, imu_euler_raw[:, ax_i], 'r-', linewidth=1, alpha=0.7, label='IMU integrated (no bias corr.)')
    ax.plot(imu_t_rel, imu_euler_corrected[:, ax_i], 'b--', linewidth=1, alpha=0.7, label='IMU integrated (bias corrected)')
    ax.set_ylabel(f'{label} (°)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    # RMSE text box per subplot
    rmse_text = f'RMSE raw: {rmse_raw[ax_i]:.1f}°\nRMSE corrected: {rmse_corr[ax_i]:.1f}°'
    ax.text(0.02, 0.95, rmse_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

axes[-1].set_xlabel('Time (s)')

# Add drift annotation
drift_text = (
    f"Bias: ({np.degrees(bias_est[0]):+.1f}, {np.degrees(bias_est[1]):+.1f}, {np.degrees(bias_est[2]):+.1f}) °/s\n"
    f"Raw drift after {imu_t_rel[-1]:.1f}s: ({drift_raw[0]:+.1f}, {drift_raw[1]:+.1f}, {drift_raw[2]:+.1f})°  |  Mean RMSE: {rmse_raw_mean:.1f}°\n"
    f"Corrected drift: ({drift_corr[0]:+.1f}, {drift_corr[1]:+.1f}, {drift_corr[2]:+.1f})°  |  Mean RMSE: {rmse_corr_mean:.1f}°"
)
fig.text(0.02, 0.01, drift_text, fontsize=9, fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
outname = f'gyro_drift_{bag_key}.png'
plt.savefig(f'analysis/{outname}', dpi=150, bbox_inches='tight')
print(f"\n  Saved: analysis/{outname}")
plt.show()
