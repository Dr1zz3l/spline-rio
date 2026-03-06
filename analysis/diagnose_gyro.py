"""Diagnose: compare IMU gyro against MoCap-derived angular velocity.
Check for per-axis sign/scale/offset errors and frame misalignment."""
import sys
import numpy as np
from scipy.interpolate import interp1d
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import quat_to_rotation_matrix

BAGS = {
    "circle": ("rosbags/circle_2025-12-17-17-21-37.bag", 25.5, 8.5, False),
    "circle_fwd": ("rosbags/circle_forward_2025-12-17-17-37-38.bag", 28.0, 8.0, True),
}

bag_key = sys.argv[1] if len(sys.argv) > 1 else "circle"
BAG_PATH, START_OFFSET, DURATION, FLIP = BAGS[bag_key]
IMU_MOCAP_OFFSET = 0.020

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
