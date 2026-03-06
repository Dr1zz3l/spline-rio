"""Diagnose: compare MoCap-predicted Doppler against measured Doppler.
Goal: find the systematic error causing -2.7 m/s z-velocity bias."""
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (
    quat_to_rotation_matrix,
    rotation_matrix_from_euler,
    predict_doppler_velocity,
)

BAGS = {
    "circle": ("rosbags/circle_2025-12-17-17-21-37.bag", 25.5, 8.5, False),
    "circle_fwd": ("rosbags/circle_forward_2025-12-17-17-37-38.bag", 28.0, 8.0, True),
}

bag_key = sys.argv[1] if len(sys.argv) > 1 else "circle"
BAG_PATH, START_OFFSET, DURATION, FLIP = BAGS[bag_key]
IMU_MOCAP_OFFSET = 0.020

# Extrinsics
ROTATION_EULER_DEG = np.array([180.0, 30.0, 0.0])
R_base = rotation_matrix_from_euler(
    np.radians(ROTATION_EULER_DEG[0]),
    np.radians(ROTATION_EULER_DEG[1]),
    np.radians(ROTATION_EULER_DEG[2]),
)
if FLIP:
    TRANSLATION = np.array([-0.07, 0.0, 0.0])
    R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)
    SENSOR_ROTATION = R_yaw_flip @ R_base
else:
    TRANSLATION = np.array([0.07, 0.0, 0.0])
    SENSOR_ROTATION = R_base

print(f"Bag: {bag_key}, flip={FLIP}")
print(f"Sensor translation (body): {TRANSLATION}")
print(f"Sensor rotation R_body_from_sensor:")
print(SENSOR_ROTATION)

# Load data
bag_data = load_bag_topics(BAG_PATH, verbose=False)
t_start = bag_data.start_time + START_OFFSET
t_end = t_start + DURATION

agiros = [s for s in bag_data.agiros_state if t_start <= s.timestamp <= t_end]
radar_frames = [f for f in bag_data.radar_velocity if t_start <= f.timestamp <= t_end]

# Apply time offset to radar
for f in radar_frames:
    f.timestamp += IMU_MOCAP_OFFSET

# Build MoCap interpolators
agiros_times = np.array([s.timestamp for s in agiros])
agiros_vel = np.array([s.velocity for s in agiros])
agiros_quat = np.array([s.orientation for s in agiros])
agiros_omega = np.array([s.angular_velocity for s in agiros])

vel_interp = interp1d(agiros_times, agiros_vel, axis=0, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
omega_interp = interp1d(agiros_times, agiros_omega, axis=0, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
quat_interp = interp1d(agiros_times, agiros_quat, axis=0, kind='linear',
                         bounds_error=False, fill_value='extrapolate')

print(f"\nRadar frames: {len(radar_frames)}")
print(f"MoCap samples: {len(agiros)}")

# ===== 1. Compute MoCap-predicted Doppler for each radar point =====
all_meas = []
all_pred = []
all_residuals = []
all_u_body = []  # unit direction vectors in body frame
all_u_world = []  # unit direction vectors in world frame
all_v_world = []  # MoCap velocities
all_elevations = []  # elevation angles of points

for frame in radar_frames:
    t = frame.timestamp
    if t < agiros_times[0] or t > agiros_times[-1]:
        continue
    
    v_world = vel_interp(t)
    omega = omega_interp(t)
    q = quat_interp(t)
    q = q / np.linalg.norm(q)
    R_wb = quat_to_rotation_matrix(q)
    
    positions = np.array(frame.positions)
    measured = np.array(frame.velocities)
    ranges = np.linalg.norm(positions, axis=1)
    valid = ranges >= 0.2
    
    if not np.any(valid):
        continue
    
    positions = positions[valid]
    measured = measured[valid]
    ranges = ranges[valid]
    
    # Predict Doppler
    predicted = predict_doppler_velocity(
        v_world, omega, R_wb, positions, TRANSLATION, SENSOR_ROTATION
    )
    
    # Unit direction vectors in sensor frame
    u_sensor = positions / ranges[:, None]
    # Transform to body and world frame
    u_body = (SENSOR_ROTATION @ u_sensor.T).T
    u_world = (R_wb @ u_body.T).T
    
    # Elevation angle of each point in sensor frame (angle from x-y plane)
    elev = np.degrees(np.arcsin(u_sensor[:, 2]))  # z component = sin(elevation)
    
    all_meas.extend(measured)
    all_pred.extend(predicted)
    all_residuals.extend(measured - predicted)
    all_u_body.extend(u_body)
    all_u_world.extend(u_world)
    all_v_world.extend([v_world] * len(measured))
    all_elevations.extend(elev)

all_meas = np.array(all_meas)
all_pred = np.array(all_pred)
all_residuals = np.array(all_residuals)
all_u_body = np.array(all_u_body)
all_u_world = np.array(all_u_world)
all_v_world = np.array(all_v_world)
all_elevations = np.array(all_elevations)

print(f"\nTotal radar points: {len(all_meas)}")

# ===== 1b. Measured velocity distribution =====
print("\n" + "="*60)
print("MEASURED VELOCITY DISTRIBUTION")
print("="*60)
print(f"  Mean: {all_meas.mean():+.4f} m/s")
print(f"  Std:  {all_meas.std():.4f} m/s")
print(f"  Min/Max: [{all_meas.min():+.3f}, {all_meas.max():+.3f}] m/s")
unique_v = np.unique(np.round(all_meas, 3))
print(f"  Unique values (rounded to 0.001): {len(unique_v)}")
if len(unique_v) <= 30:
    print(f"  Values: {unique_v}")
# Histogram
bins = np.arange(all_meas.min() - 0.5, all_meas.max() + 1.0, 0.6)
counts, edges = np.histogram(all_meas, bins=bins)
print(f"  Histogram (bin width ~0.6 m/s):")
for c, e in zip(counts, edges[:-1]):
    if c > 0:
        print(f"    [{e:+6.2f}, {e+0.6:+6.2f}): {c:4d} {'#' * min(c, 60)}")

# ===== 1c. Predicted velocity distribution =====
print("\n" + "="*60)
print("PREDICTED VELOCITY DISTRIBUTION")
print("="*60)
print(f"  Mean: {all_pred.mean():+.4f} m/s")
print(f"  Std:  {all_pred.std():.4f} m/s")
print(f"  Min/Max: [{all_pred.min():+.3f}, {all_pred.max():+.3f}] m/s")

# ===== 2. Global residual statistics =====
print("\n" + "="*60)
print("DOPPLER RESIDUAL: measured - MoCap_predicted")
print("="*60)
print(f"  Mean:   {all_residuals.mean():+.4f} m/s")
print(f"  Median: {np.median(all_residuals):+.4f} m/s")
print(f"  Std:    {all_residuals.std():.4f} m/s")
print(f"  RMSE:   {np.sqrt(np.mean(all_residuals**2)):.4f} m/s")
p5, p25, p75, p95 = np.percentile(all_residuals, [5, 25, 75, 95])
print(f"  [5th, 25th, 75th, 95th] percentiles: [{p5:.3f}, {p25:.3f}, {p75:.3f}, {p95:.3f}]")

# ===== 3. What vz offset would minimize residuals? =====
print("\n" + "="*60)
print("OPTIMAL VELOCITY OFFSET (what v_offset minimizes ||meas - pred(v+offset)||²?)")
print("="*60)
# residual = meas - u_body · (R_bw(v_world + v_offset) + lever_arm)
# For a given offset dv in world frame, the change in predicted Doppler is:
# dpred = u_world · dv (since u_body · R_bw · dv = u_world · dv)
# So minimizing sum (r - u_world · dv)² => normal equations:
# (u_world^T u_world) dv = u_world^T r
A = all_u_world.T @ all_u_world
b = all_u_world.T @ all_residuals
dv_optimal = np.linalg.solve(A, b)
print(f"  Optimal world-frame velocity offset: [{dv_optimal[0]:+.4f}, {dv_optimal[1]:+.4f}, {dv_optimal[2]:+.4f}] m/s")
print(f"  (This is the v_world correction the radar data 'wants')")
new_residuals = all_residuals - all_u_world @ dv_optimal
print(f"  Residual RMSE before: {np.sqrt(np.mean(all_residuals**2)):.4f} m/s")
print(f"  Residual RMSE after:  {np.sqrt(np.mean(new_residuals**2)):.4f} m/s")

# ===== 4. Elevation angle distribution =====
print("\n" + "="*60)
print("ELEVATION ANGLE DISTRIBUTION (sensor frame)")
print("="*60)
print(f"  Mean: {all_elevations.mean():.2f}°")
print(f"  Std:  {all_elevations.std():.2f}°")
print(f"  Min/Max: [{all_elevations.min():.2f}°, {all_elevations.max():.2f}°]")
print(f"  |elev| < 5°: {np.sum(np.abs(all_elevations) < 5) / len(all_elevations) * 100:.1f}%")
print(f"  |elev| < 10°: {np.sum(np.abs(all_elevations) < 10) / len(all_elevations) * 100:.1f}%")

# ===== 5. Direction coverage in body frame =====
print("\n" + "="*60)
print("RAY DIRECTION COVERAGE (body frame)")
print("="*60)
print(f"  u_body mean: [{all_u_body[:,0].mean():.4f}, {all_u_body[:,1].mean():.4f}, {all_u_body[:,2].mean():.4f}]")
print(f"  u_body std:  [{all_u_body[:,0].std():.4f}, {all_u_body[:,1].std():.4f}, {all_u_body[:,2].std():.4f}]")

# ===== 6. Direction coverage in world frame =====
print("\n" + "="*60)
print("RAY DIRECTION COVERAGE (world frame)")
print("="*60)
print(f"  u_world mean: [{all_u_world[:,0].mean():.4f}, {all_u_world[:,1].mean():.4f}, {all_u_world[:,2].mean():.4f}]")
print(f"  u_world std:  [{all_u_world[:,0].std():.4f}, {all_u_world[:,1].std():.4f}, {all_u_world[:,2].std():.4f}]")
# Condition of the u_world^T u_world matrix (observability)
eigvals = np.linalg.eigvalsh(A)
print(f"  Eigenvalues of (U^T U): [{eigvals[0]:.1f}, {eigvals[1]:.1f}, {eigvals[2]:.1f}]")
print(f"  Condition number: {eigvals[-1]/eigvals[0]:.1f}")

# ===== 7. Per-axis breakdown of residual projection =====
print("\n" + "="*60)
print("RESIDUAL PROJECTION ONTO WORLD AXES")
print("="*60)
# For each axis, what fraction of the residual projects onto that axis?
for ax, label in enumerate(['x', 'y', 'z']):
    # Weighted projection: how much does the optimal dv explain the residuals along this axis?
    projection = all_u_world[:, ax] * dv_optimal[ax]
    print(f"  {label}: dv={dv_optimal[ax]:+.3f} m/s, mean u_{label} component={all_u_world[:,ax].mean():+.4f}")

# ===== 8. Doppler sign convention check =====
print("\n" + "="*60)
print("SIGN CONVENTION CHECK")
print("="*60)
# If there's a sign error, r = meas + pred instead of meas - pred
r_sign_flip = all_meas + all_pred  # what if pred sign is wrong?
print(f"  Normal residual (meas - pred): mean={all_residuals.mean():+.4f}, RMSE={np.sqrt(np.mean(all_residuals**2)):.4f}")
print(f"  Flipped residual (meas + pred): mean={r_sign_flip.mean():+.4f}, RMSE={np.sqrt(np.mean(r_sign_flip**2)):.4f}")
# Correlation
corr_normal = np.corrcoef(all_meas, all_pred)[0, 1]
corr_neg = np.corrcoef(all_meas, -all_pred)[0, 1]
print(f"  Correlation(meas, pred):  {corr_normal:+.4f}")
print(f"  Correlation(meas, -pred): {corr_neg:+.4f}")

# ===== 9. Scatter: measured vs predicted =====
print("\n" + "="*60)
print("MEASURED vs PREDICTED (sample points)")
print("="*60)
# Show a few representative points
idx = np.linspace(0, len(all_meas)-1, 20, dtype=int)
print(f"  {'meas':>8s} {'pred':>8s} {'resid':>8s} {'u_bx':>7s} {'u_by':>7s} {'u_bz':>7s} {'elev':>6s}")
for i in idx:
    print(f"  {all_meas[i]:+8.3f} {all_pred[i]:+8.3f} {all_residuals[i]:+8.3f} "
          f"{all_u_body[i,0]:+7.3f} {all_u_body[i,1]:+7.3f} {all_u_body[i,2]:+7.3f} "
          f"{all_elevations[i]:+6.1f}°")

# ===== 10. What if we sweep extrinsic pitch? =====
print("\n" + "="*60)
print("EXTRINSIC PITCH SWEEP (finding optimal tilt angle)")
print("="*60)
best_rmse = 1e10
best_pitch = 0
for pitch_deg in np.arange(0, 60, 2):
    R_test = rotation_matrix_from_euler(0, np.radians(pitch_deg), 0)
    if FLIP:
        R_test = rotation_matrix_from_euler(0, 0, np.pi) @ R_test
    
    # Recompute predictions with this rotation
    test_pred = []
    for frame in radar_frames:
        t = frame.timestamp
        if t < agiros_times[0] or t > agiros_times[-1]:
            continue
        v_world = vel_interp(t)
        omega = omega_interp(t)
        q = quat_interp(t)
        q = q / np.linalg.norm(q)
        R_wb = quat_to_rotation_matrix(q)
        positions = np.array(frame.positions)
        ranges = np.linalg.norm(positions, axis=1)
        valid = ranges >= 0.2
        if not np.any(valid):
            continue
        positions = positions[valid]
        pred = predict_doppler_velocity(
            v_world, omega, R_wb, positions, TRANSLATION, R_test
        )
        test_pred.extend(pred)
    
    test_pred = np.array(test_pred)
    test_resid = all_meas - test_pred
    rmse = np.sqrt(np.mean(test_resid**2))
    mean_r = test_resid.mean()
    if rmse < best_rmse:
        best_rmse = rmse
        best_pitch = pitch_deg
    if pitch_deg % 10 == 0:
        print(f"  pitch={pitch_deg:5.1f}°: RMSE={rmse:.4f} m/s, mean={mean_r:+.4f} m/s")

print(f"\n  Best pitch: {best_pitch}° (RMSE={best_rmse:.4f} m/s)")
print(f"  Current pitch: {ROTATION_EULER_DEG[1]}°")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)

# ===== 11. Decompose: v_body vs lever arm vs ray direction =====
print("\n" + "="*60)
print("11. FORWARD MODEL DECOMPOSITION (first few frames)")
print("="*60)
# Show detailed breakdown for a few frames to verify correctness
for frame_i, frame in enumerate(radar_frames[:3]):
    t = frame.timestamp
    if t < agiros_times[0] or t > agiros_times[-1]:
        continue
    v_world = vel_interp(t)
    omega = omega_interp(t)
    q = quat_interp(t)
    q = q / np.linalg.norm(q)
    R_wb = quat_to_rotation_matrix(q)
    R_bw = R_wb.T
    
    v_body = R_bw @ v_world
    lever = np.cross(omega, TRANSLATION)
    v_ant = v_body + lever
    
    positions = np.array(frame.positions)
    measured = np.array(frame.velocities)
    ranges = np.linalg.norm(positions, axis=1)
    
    print(f"\n  Frame {frame_i}: t={t-agiros_times[0]:.3f}s")
    print(f"    v_world = [{v_world[0]:+.3f}, {v_world[1]:+.3f}, {v_world[2]:+.3f}]")
    print(f"    omega   = [{omega[0]:+.3f}, {omega[1]:+.3f}, {omega[2]:+.3f}]")
    print(f"    v_body  = [{v_body[0]:+.3f}, {v_body[1]:+.3f}, {v_body[2]:+.3f}]")
    print(f"    lever   = [{lever[0]:+.3f}, {lever[1]:+.3f}, {lever[2]:+.3f}]")
    print(f"    v_ant   = [{v_ant[0]:+.3f}, {v_ant[1]:+.3f}, {v_ant[2]:+.3f}]")
    
    for j in range(min(3, len(positions))):
        u_s = positions[j] / ranges[j]
        u_b = SENSOR_ROTATION @ u_s
        v_pred = np.dot(u_b, v_ant)
        print(f"    pt[{j}]: pos_s=[{positions[j,0]:+.2f},{positions[j,1]:+.2f},{positions[j,2]:+.2f}] "
              f"u_s=[{u_s[0]:+.3f},{u_s[1]:+.3f},{u_s[2]:+.3f}] "
              f"u_b=[{u_b[0]:+.3f},{u_b[1]:+.3f},{u_b[2]:+.3f}] "
              f"pred={v_pred:+.3f} meas={measured[j]:+.3f} r={measured[j]-v_pred:+.3f}")

# ===== 12. Doppler sign check: negate measured and recompute =====
print("\n" + "="*60)
print("12. NEGATE MEASURED DOPPLER (sign convention test)")
print("="*60)
neg_residuals = -all_meas - all_pred
print(f"  Residual with -measured: mean={neg_residuals.mean():+.4f}, RMSE={np.sqrt(np.mean(neg_residuals**2)):.4f}")
corr_neg_meas = np.corrcoef(-all_meas, all_pred)[0, 1]
print(f"  Correlation(-meas, pred): {corr_neg_meas:+.4f}")

# Optimal offset with negated measurements
b_neg = all_u_world.T @ (-all_meas - all_pred)
dv_neg = np.linalg.solve(A, b_neg)
new_neg_r = (-all_meas - all_pred) - all_u_world @ dv_neg
print(f"  Optimal offset with -meas: [{dv_neg[0]:+.4f}, {dv_neg[1]:+.4f}, {dv_neg[2]:+.4f}]")
print(f"  RMSE after offset: {np.sqrt(np.mean(new_neg_r**2)):.4f}")

# ===== 13. Check: what if velocity field is actually range rate (not radial velocity)? =====
# Range rate = d/dt ||p_target - p_sensor|| = -u · v_sensor (note the negative!)
print("\n" + "="*60)
print("13. RANGE RATE CONVENTION: v_D = -u_b · v_ant (flip prediction sign)")
print("="*60)
neg_pred_residuals = all_meas - (-all_pred)
print(f"  Using v_pred = -u_b·v_ant: mean={neg_pred_residuals.mean():+.4f}, RMSE={np.sqrt(np.mean(neg_pred_residuals**2)):.4f}")
corr_range_rate = np.corrcoef(all_meas, -all_pred)[0, 1]
print(f"  Correlation(meas, -pred): {corr_range_rate:+.4f}")
A2 = all_u_world.T @ all_u_world
b2 = all_u_world.T @ (all_meas + all_pred)  # residual = meas - (-pred) = meas + pred
dv2 = np.linalg.solve(A2, b2)
new_r2 = (all_meas + all_pred) - all_u_world @ dv2
print(f"  Optimal offset: [{dv2[0]:+.4f}, {dv2[1]:+.4f}, {dv2[2]:+.4f}]")
print(f"  RMSE after offset: {np.sqrt(np.mean(new_r2**2)):.4f}")

# ===== 14. Combined sweep: roll × pitch (roll around boresight) =====
print("\n" + "="*60)
print("14. COMBINED SWEEP: roll (around boresight) × pitch (downtilt)")
print("    R_body_from_sensor = [yaw_flip @] Ry(pitch) @ Rx(roll)")
print("    roll: sensor y/z orientation around boresight")
print("    pitch: known 30° downtilt")
print("="*60)

def compute_predictions(R_test):
    """Recompute all predictions with a given rotation."""
    test_pred = []
    for frame in radar_frames:
        t = frame.timestamp
        if t < agiros_times[0] or t > agiros_times[-1]:
            continue
        v_world = vel_interp(t)
        omega = omega_interp(t)
        q = quat_interp(t)
        q = q / np.linalg.norm(q)
        R_wb = quat_to_rotation_matrix(q)
        positions = np.array(frame.positions)
        ranges = np.linalg.norm(positions, axis=1)
        valid = ranges >= 0.2
        if not np.any(valid):
            continue
        pred = predict_doppler_velocity(
            v_world, omega, R_wb, positions[valid], TRANSLATION, R_test
        )
        test_pred.extend(pred)
    return np.array(test_pred)

print(f"  {'roll':>5s} {'pitch':>6s} {'corr':>8s} {'RMSE':>8s} {'mean_r':>8s} {'RMSE_ofs':>9s} {'dv_x':>8s} {'dv_y':>8s} {'dv_z':>8s}")
best_combo = None
best_combo_rmse = 1e10
results = []
for roll_deg in np.arange(0, 360, 10):
    for pitch_deg in [25, 30, 35]:
        # Order: first roll around boresight (x), then pitch down (y)
        R_test = rotation_matrix_from_euler(np.radians(roll_deg), np.radians(pitch_deg), 0)
        if FLIP:
            R_test = rotation_matrix_from_euler(0, 0, np.pi) @ R_test
        
        test_pred = compute_predictions(R_test)
        test_resid = all_meas - test_pred
        rmse = np.sqrt(np.mean(test_resid**2))
        mean_r = test_resid.mean()
        corr = np.corrcoef(all_meas, test_pred)[0, 1]
        
        # Optimal offset
        A_t = all_u_world.T @ all_u_world
        b_t = all_u_world.T @ test_resid
        dv_t = np.linalg.solve(A_t, b_t)
        new_r_t = test_resid - all_u_world @ dv_t
        rmse_ofs = np.sqrt(np.mean(new_r_t**2))
        
        results.append((roll_deg, pitch_deg, corr, rmse, rmse_ofs, dv_t))
        
        if rmse_ofs < best_combo_rmse:
            best_combo_rmse = rmse_ofs
            best_combo = results[-1]

# Print all results at pitch=30 and the top candidates
print("  --- All at pitch=30° ---")
for r in results:
    if r[1] == 30:
        print(f"  {r[0]:5.0f}° {r[1]:6.0f}° {r[2]:+8.4f} {r[3]:8.4f} {r[4]:+8.4f} {r[5][0]:9.4f} {r[5][0]:+8.4f} {r[5][1]:+8.4f} {r[5][2]:+8.4f}")

# Sort by RMSE_offset and show top 10
results.sort(key=lambda x: x[4])
print("\n  --- Top 10 by RMSE_offset ---")
for r in results[:10]:
    print(f"  {r[0]:5.0f}° {r[1]:6.0f}° {r[2]:+8.4f} {r[3]:8.4f} {r[4]:+8.4f} {r[5][0]:9.4f} {r[5][0]:+8.4f} {r[5][1]:+8.4f} {r[5][2]:+8.4f}")

print(f"\n  BEST: roll={best_combo[0]:.0f}°, pitch={best_combo[1]:.0f}°, corr={best_combo[2]:+.4f}, "
      f"RMSE={best_combo[3]:.4f}, RMSE_offset={best_combo[4]:.4f}")
print(f"  BEST dv_offset: [{best_combo[5][0]:+.4f}, {best_combo[5][1]:+.4f}, {best_combo[5][2]:+.4f}]")

# ===== 15. Fine sweep around best roll at pitch=30° =====
best_roll_coarse = best_combo[0]
print("\n" + "="*60)
print(f"15. FINE ROLL SWEEP around {best_roll_coarse:.0f}° at multiple pitches")
print("="*60)
print(f"  {'roll':>5s} {'pitch':>6s} {'corr':>8s} {'RMSE':>8s} {'RMSE_ofs':>9s} {'dv_x':>8s} {'dv_y':>8s} {'dv_z':>8s}")
fine_results = []
for roll_deg in np.arange(best_roll_coarse - 20, best_roll_coarse + 22, 2):
    for pitch_deg in np.arange(20, 42, 2):
        R_test = rotation_matrix_from_euler(np.radians(roll_deg), np.radians(pitch_deg), 0)
        if FLIP:
            R_test = rotation_matrix_from_euler(0, 0, np.pi) @ R_test
        
        test_pred = compute_predictions(R_test)
        test_resid = all_meas - test_pred
        rmse = np.sqrt(np.mean(test_resid**2))
        corr = np.corrcoef(all_meas, test_pred)[0, 1]
        
        A_t = all_u_world.T @ all_u_world
        b_t = all_u_world.T @ test_resid
        dv_t = np.linalg.solve(A_t, b_t)
        new_r_t = test_resid - all_u_world @ dv_t
        rmse_ofs = np.sqrt(np.mean(new_r_t**2))
        
        fine_results.append((roll_deg, pitch_deg, corr, rmse, rmse_ofs, dv_t))

fine_results.sort(key=lambda x: x[4])
for r in fine_results[:15]:
    print(f"  {r[0]:5.0f}° {r[1]:6.0f}° {r[2]:+8.4f} {r[3]:8.4f} {r[4]:9.4f} {r[5][0]:+8.4f} {r[5][1]:+8.4f} {r[5][2]:+8.4f}")

best_fine = fine_results[0]
print(f"\n  BEST FINE: roll={best_fine[0]:.0f}°, pitch={best_fine[1]:.0f}°, corr={best_fine[2]:+.4f}, "
      f"RMSE={best_fine[3]:.4f}, RMSE_offset={best_fine[4]:.4f}")
print(f"  BEST dv_offset: [{best_fine[5][0]:+.4f}, {best_fine[5][1]:+.4f}, {best_fine[5][2]:+.4f}]")

# Print the actual rotation matrix for the best result
R_best = rotation_matrix_from_euler(np.radians(best_fine[0]), np.radians(best_fine[1]), 0)
print(f"\n  R_body_from_sensor (roll={best_fine[0]:.0f}°, pitch={best_fine[1]:.0f}°):")
print(f"    [{R_best[0,0]:+.4f} {R_best[0,1]:+.4f} {R_best[0,2]:+.4f}]")
print(f"    [{R_best[1,0]:+.4f} {R_best[1,1]:+.4f} {R_best[1,2]:+.4f}]")
print(f"    [{R_best[2,0]:+.4f} {R_best[2,1]:+.4f} {R_best[2,2]:+.4f}]")
print(f"  Sensor x-axis (boresight) in body frame: [{R_best[0,0]:+.4f}, {R_best[1,0]:+.4f}, {R_best[2,0]:+.4f}]")
print(f"  Sensor y-axis in body frame:             [{R_best[0,1]:+.4f}, {R_best[1,1]:+.4f}, {R_best[2,1]:+.4f}]")
print(f"  Sensor z-axis in body frame:             [{R_best[0,2]:+.4f}, {R_best[1,2]:+.4f}, {R_best[2,2]:+.4f}]")
