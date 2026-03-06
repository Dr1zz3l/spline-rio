"""Diagnostic: Check IMU biases and frame alignment."""
import sys
import numpy as np
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import quat_to_rotation_matrix

BAGS = {
    "circle": ("rosbags/circle_2025-12-17-17-21-37.bag", 25.5, 8.5),
    "circle_fwd": ("rosbags/circle_forward_2025-12-17-17-37-38.bag", 28.0, 8.0),
}

bag_key = sys.argv[1] if len(sys.argv) > 1 else "circle"
BAG_PATH, START_OFFSET, DURATION = BAGS[bag_key]
g_world = np.array([0, 0, -9.81])

print("Loading bag data...")
bag_data = load_bag_topics(BAG_PATH, verbose=False)
t_start = bag_data.start_time + START_OFFSET
t_end = t_start + DURATION

agiros = [s for s in bag_data.agiros_state if t_start <= s.timestamp <= t_end]
imu = [d for d in bag_data.imu_data if t_start <= d.timestamp <= t_end]

print(f"\nFlight window: {START_OFFSET}s + {DURATION}s")
print(f"  AgirosState samples: {len(agiros)}")
print(f"  IMU samples: {len(imu)}")

# ===== 1. Check Agiros bias estimates =====
print("\n" + "="*60)
print("1. AGIROS KALMAN FILTER BIAS ESTIMATES")
print("="*60)
acc_biases = [s.acc_bias for s in agiros if s.acc_bias is not None]
gyr_biases = [s.gyr_bias for s in agiros if s.gyr_bias is not None]
if acc_biases:
    acc_biases = np.array(acc_biases)
    print(f"  Accel bias samples: {len(acc_biases)}")
    print(f"  Mean: [{acc_biases[:,0].mean():.4f}, {acc_biases[:,1].mean():.4f}, {acc_biases[:,2].mean():.4f}] m/s²")
    print(f"  Std:  [{acc_biases[:,0].std():.4f}, {acc_biases[:,1].std():.4f}, {acc_biases[:,2].std():.4f}] m/s²")
else:
    print("  No accel bias estimates available!")

if gyr_biases:
    gyr_biases = np.array(gyr_biases)
    print(f"  Gyro bias samples: {len(gyr_biases)}")
    print(f"  Mean: [{gyr_biases[:,0].mean():.6f}, {gyr_biases[:,1].mean():.6f}, {gyr_biases[:,2].mean():.6f}] rad/s")
    print(f"  Std:  [{gyr_biases[:,0].std():.6f}, {gyr_biases[:,1].std():.6f}, {gyr_biases[:,2].std():.6f}] rad/s")
else:
    print("  No gyro bias estimates available!")

# ===== 2. Raw IMU statistics =====
print("\n" + "="*60)
print("2. RAW IMU ACCELEROMETER STATISTICS (body frame)")
print("="*60)
imu_accel = np.array([d.linear_acceleration for d in imu])
imu_gyro = np.array([d.angular_velocity for d in imu])
print(f"  Mean accel: [{imu_accel[:,0].mean():.3f}, {imu_accel[:,1].mean():.3f}, {imu_accel[:,2].mean():.3f}] m/s²")
print(f"  Std  accel: [{imu_accel[:,0].std():.3f}, {imu_accel[:,1].std():.3f}, {imu_accel[:,2].std():.3f}] m/s²")
print(f"  Mean |accel|: {np.linalg.norm(imu_accel, axis=1).mean():.3f} m/s² (expected ~9.81 for hover)")
print(f"  Mean gyro:  [{imu_gyro[:,0].mean():.4f}, {imu_gyro[:,1].mean():.4f}, {imu_gyro[:,2].mean():.4f}] rad/s")

# ===== 3. Expected vs measured IMU acceleration =====
print("\n" + "="*60)
print("3. IMU RESIDUAL ANALYSIS (using MoCap as ground truth)")
print("="*60)
# For each IMU measurement, compute what the accelerometer SHOULD read
# given MoCap orientation and MoCap-derived acceleration
# Model: z_acc = R_bw @ (a_world - g) + b_a
# So predicted: z_acc_pred = R_bw @ (a_world - g) where a_world = d²p/dt²

# Get MoCap acceleration from AgirosState (which has Kalman-smoothed values)
agiros_times = np.array([s.timestamp for s in agiros])
agiros_accel = np.array([s.acceleration for s in agiros if s.acceleration is not None])
agiros_vel = np.array([s.velocity for s in agiros])

if len(agiros_accel) > 0 and len(agiros_accel) == len(agiros):
    print(f"  AgirosState provides acceleration data ({len(agiros_accel)} samples)")
    print(f"  Mean Agiros accel (world): [{agiros_accel[:,0].mean():.3f}, {agiros_accel[:,1].mean():.3f}, {agiros_accel[:,2].mean():.3f}] m/s²")
    print(f"  (Expected ~[0,0,0] for level flight, since gravity is separate)")
    
    # Compute expected body-frame specific force for each Agiros sample
    residuals_with_bias = []
    residuals_no_bias = []
    predicted_list = []
    
    for s in agiros:
        if s.acceleration is None:
            continue
        R_wb = quat_to_rotation_matrix(s.orientation)  # world_from_body
        R_bw = R_wb.T  # body_from_world
        
        # Predicted specific force (what IMU should read)
        a_world = s.acceleration
        z_pred = R_bw @ (a_world - g_world)  # body frame, includes gravity reaction
        predicted_list.append(z_pred)
    
    predicted = np.array(predicted_list)
    print(f"\n  Predicted IMU accel (from MoCap a_world, no bias):")
    print(f"    Mean: [{predicted[:,0].mean():.3f}, {predicted[:,1].mean():.3f}, {predicted[:,2].mean():.3f}] m/s²")
    
    # Find closest IMU measurement for each Agiros sample
    imu_times = np.array([d.timestamp for d in imu])
    matched_residuals = []
    matched_imu_list = []
    matched_pred_list = []
    pred_idx = 0
    for i, s in enumerate(agiros):
        if s.acceleration is None:
            continue
        pred = predicted_list[pred_idx]
        pred_idx += 1
        # Find nearest IMU sample (within 5ms)
        dt = np.abs(imu_times - s.timestamp)
        j = np.argmin(dt)
        if dt[j] < 0.005:
            r = imu[j].linear_acceleration - pred
            matched_residuals.append(r)
            matched_imu_list.append(imu[j].linear_acceleration)
            matched_pred_list.append(pred)
    
    matched_residuals = np.array(matched_residuals)
    matched_imu = np.array(matched_imu_list)
    matched_pred = np.array(matched_pred_list)
    if len(matched_residuals) > 0:
        print(f"\n  IMU - Predicted (= unmodeled bias + noise), {len(matched_residuals)} matched pairs:")
        print(f"    Mean: [{matched_residuals[:,0].mean():.3f}, {matched_residuals[:,1].mean():.3f}, {matched_residuals[:,2].mean():.3f}] m/s²")
        print(f"    Std:  [{matched_residuals[:,0].std():.3f}, {matched_residuals[:,1].std():.3f}, {matched_residuals[:,2].std():.3f}] m/s²")
        print(f"    (This is what the solver's accel residual looks like with b_a=0)")
        
        # Per-axis correlation
        print(f"\n  PER-AXIS CORRELATION (IMU vs Predicted):")
        for ax, label in enumerate(['x', 'y', 'z']):
            c = np.corrcoef(matched_imu[:, ax], matched_pred[:, ax])[0, 1]
            print(f"    {label}: {c:+.4f}")
        
        # Check with 180° yaw rotation (R_z(180°) = diag(-1,-1,1))
        imu_rotated = matched_imu.copy()
        imu_rotated[:, 0] *= -1
        imu_rotated[:, 1] *= -1
        print(f"\n  PER-AXIS CORRELATION after R_z(180°) applied to IMU:")
        for ax, label in enumerate(['x', 'y', 'z']):
            c = np.corrcoef(imu_rotated[:, ax], matched_pred[:, ax])[0, 1]
            print(f"    {label}: {c:+.4f}")
        residuals_rot = imu_rotated - matched_pred
        print(f"    Residual mean: [{residuals_rot[:,0].mean():.3f}, {residuals_rot[:,1].mean():.3f}, {residuals_rot[:,2].mean():.3f}]")
        print(f"    Residual std:  [{residuals_rot[:,0].std():.3f}, {residuals_rot[:,1].std():.3f}, {residuals_rot[:,2].std():.3f}]")
        
        # Now with Agiros bias subtracted
        if acc_biases is not None and len(acc_biases) > 0:
            mean_bias = acc_biases.mean(axis=0)
            corrected = matched_residuals - mean_bias
            print(f"\n  IMU - Predicted - Agiros_bias:")
            print(f"    Mean: [{corrected[:,0].mean():.3f}, {corrected[:,1].mean():.3f}, {corrected[:,2].mean():.3f}] m/s²")
            print(f"    Std:  [{corrected[:,0].std():.3f}, {corrected[:,1].std():.3f}, {corrected[:,2].std():.3f}] m/s²")
            print(f"    (Residual after applying Agiros bias correction)")
else:
    print("  AgirosState does NOT provide acceleration data - computing from velocity")
    # Numerically differentiate velocity
    dt_agiros = np.diff(agiros_times)
    a_numerical = np.diff(agiros_vel, axis=0) / dt_agiros[:, None]
    print(f"  Numerical accel (world): mean=[{a_numerical[:,0].mean():.3f}, {a_numerical[:,1].mean():.3f}, {a_numerical[:,2].mean():.3f}]")

# ===== 4. Check IMU frame vs body frame =====
print("\n" + "="*60)
print("4. IMU FRAME CHECK (gravity direction during first 0.5s)")
print("="*60)
# During initial hover, the accelerometer should read ~[0,0,+9.81] in body z-up (FLU)
# If it reads [0,0,-9.81], the frame is z-down (FRD)
hover_start = t_start
hover_end = t_start + 0.5
hover_imu = [d for d in imu if hover_start <= d.timestamp <= hover_end]
if hover_imu:
    hover_accel = np.array([d.linear_acceleration for d in hover_imu])
    print(f"  Samples in first 0.5s: {len(hover_imu)}")
    print(f"  Mean accel: [{hover_accel[:,0].mean():.3f}, {hover_accel[:,1].mean():.3f}, {hover_accel[:,2].mean():.3f}]")
    print(f"  |accel|: {np.linalg.norm(hover_accel, axis=1).mean():.3f} m/s²")
    if hover_accel[:,2].mean() > 5:
        print(f"  -> z > 0: Consistent with FLU (z-up) convention ✓")
    elif hover_accel[:,2].mean() < -5:
        print(f"  -> z < 0: Consistent with FRD (z-down) convention ✗ FRAME MISMATCH!")
    else:
        print(f"  -> z ≈ 0: Drone is NOT hovering, or accelerometer x/y have gravity component")
        print(f"  -> This suggests the IMU is tilted relative to body frame!")

# ===== 5. Check Agiros angular_velocity vs raw IMU gyro =====
print("\n" + "="*60)
print("5. GYRO COMPARISON: Raw IMU vs Agiros")
print("="*60)
agiros_omega = np.array([s.angular_velocity for s in agiros])
print(f"  Agiros omega mean: [{agiros_omega[:,0].mean():.4f}, {agiros_omega[:,1].mean():.4f}, {agiros_omega[:,2].mean():.4f}] rad/s")
print(f"  Raw IMU omega mean: [{imu_gyro[:,0].mean():.4f}, {imu_gyro[:,1].mean():.4f}, {imu_gyro[:,2].mean():.4f}] rad/s")

# Per-axis correlation for gyro
agiros_times = np.array([s.timestamp for s in agiros])
imu_times = np.array([d.timestamp for d in imu])
gyro_imu_matched = []
gyro_agiros_matched = []
for s in agiros:
    dt = np.abs(imu_times - s.timestamp)
    j = np.argmin(dt)
    if dt[j] < 0.005:
        gyro_imu_matched.append(imu[j].angular_velocity)
        gyro_agiros_matched.append(s.angular_velocity)
gyro_imu_matched = np.array(gyro_imu_matched)
gyro_agiros_matched = np.array(gyro_agiros_matched)

if len(gyro_imu_matched) > 0:
    print(f"\n  PER-AXIS CORRELATION (raw IMU gyro vs Agiros gyro), {len(gyro_imu_matched)} pairs:")
    for ax, label in enumerate(['x', 'y', 'z']):
        c = np.corrcoef(gyro_imu_matched[:, ax], gyro_agiros_matched[:, ax])[0, 1]
        print(f"    {label}: {c:+.4f}")
    
    # With 180° yaw rotation
    gyro_rotated = gyro_imu_matched.copy()
    gyro_rotated[:, 0] *= -1
    gyro_rotated[:, 1] *= -1
    print(f"\n  PER-AXIS CORRELATION after R_z(180°) on IMU gyro:")
    for ax, label in enumerate(['x', 'y', 'z']):
        c = np.corrcoef(gyro_rotated[:, ax], gyro_agiros_matched[:, ax])[0, 1]
        print(f"    {label}: {c:+.4f}")

# ===== 6. MoCap z-velocity during flight =====
print("\n" + "="*60)
print("6. MOCAP Z-VELOCITY DURING FLIGHT")
print("="*60)
vz = agiros_vel[:, 2]
print(f"  Mean vz: {vz.mean():.4f} m/s")
print(f"  Std vz:  {vz.std():.4f} m/s")
print(f"  Min/Max: [{vz.min():.3f}, {vz.max():.3f}] m/s")
print(f"  (Should be ~0 for level circle flight)")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)

# ===== 7. Time offset sweep for accel correlation =====
print("\n" + "="*60)
print("7. TIME OFFSET SWEEP (accel x-axis correlation vs offset)")
print("="*60)
# The IMU_MOCAP_OFFSET is +20ms. Let's sweep -50ms to +50ms
offsets_ms = np.arange(-50, 51, 5)
best_corr_x = -2
best_offset = 0
for offset_ms in offsets_ms:
    offset_s = offset_ms / 1000.0
    pairs = []
    for i, s in enumerate(agiros):
        if s.acceleration is None:
            continue
        R_wb = quat_to_rotation_matrix(s.orientation)
        z_pred = R_wb.T @ (s.acceleration - g_world)
        # Find IMU sample at s.timestamp + offset (i.e., IMU is behind by offset)
        target_time = s.timestamp + offset_s
        dt = np.abs(imu_times - target_time)
        j = np.argmin(dt)
        if dt[j] < 0.005:
            pairs.append((imu[j].linear_acceleration, z_pred))
    if len(pairs) > 100:
        arr_imu = np.array([p[0] for p in pairs])
        arr_pred = np.array([p[1] for p in pairs])
        cx = np.corrcoef(arr_imu[:, 0], arr_pred[:, 0])[0, 1]
        cy = np.corrcoef(arr_imu[:, 1], arr_pred[:, 1])[0, 1]
        if cx > best_corr_x:
            best_corr_x = cx
            best_offset = offset_ms
        if offset_ms % 10 == 0:
            print(f"  offset={offset_ms:+4d} ms: corr_x={cx:+.3f}  corr_y={cy:+.3f}  n={len(pairs)}")
print(f"  Best x-corr: {best_corr_x:+.3f} at offset={best_offset:+d} ms")
