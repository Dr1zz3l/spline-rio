"""Check drone body frame convention empirically."""
import numpy as np
import sys
sys.path.insert(0, 'analysis')
from rosbag_loader.loader import load_bag_topics
from scipy.spatial.transform import Rotation

bag = load_bag_topics('rosbags/2025-12-17-16-02-22.bag', verbose=False)
states = bag.agiros_state
imu = bag.imu_data
t0 = states[0].timestamp

# 1. Check quaternion during hover (first 5 seconds)
print("=== Orientation during first 5s hover ===")
hover_quats = [s.orientation for s in states if s.timestamp - t0 < 5]
hover_quats = np.array(hover_quats)
mean_q = hover_quats.mean(axis=0)
mean_q /= np.linalg.norm(mean_q)
R_hover = Rotation.from_quat(mean_q)
print(f"Mean quaternion [qx,qy,qz,qw]: {mean_q}")
euler = R_hover.as_euler('ZYX', degrees=True)
print(f"As Euler ZYX (deg): yaw={euler[0]:.2f} pitch={euler[1]:.2f} roll={euler[2]:.2f}")
print(f"Rotation matrix R_world_from_body:")
print(R_hover.as_matrix().round(4))

# 2. Body axes in world frame during hover
R_wb = R_hover.as_matrix()
x_body_in_world = R_wb @ [1, 0, 0]
y_body_in_world = R_wb @ [0, 1, 0]
z_body_in_world = R_wb @ [0, 0, 1]
print(f"\nBody x in world: {x_body_in_world.round(4)}  (expect ~[1,0,0] if x=forward)")
print(f"Body y in world: {y_body_in_world.round(4)}  (expect ~[0,1,0] if y=left)")
print(f"Body z in world: {z_body_in_world.round(4)}  (expect ~[0,0,1] if z=up)")

# 3. IMU accelerometer during hover
print("\n=== IMU accel during first 5s hover ===")
hover_imu = [i for i in imu if i.timestamp - t0 < 5]
accels = np.array([i.linear_acceleration for i in hover_imu])
print(f"Mean accel: {accels.mean(axis=0).round(3)}")
print(f"  If body z=up:   expect [~0, ~0, +9.81]")
print(f"  If body z=down: expect [~0, ~0, -9.81]")

# 4. Predicted accel from forward model during hover
# z_imu = R_bw @ (a_world - g)  with a_world ≈ 0 during hover
# z_imu = R_bw @ (-g) = R_bw @ [0, 0, 9.81]
R_bw = R_wb.T
accel_pred_hover = R_bw @ np.array([0, 0, 9.81])
print(f"Forward model prediction R_bw @ [0,0,9.81]: {accel_pred_hover.round(3)}")
accel_pred_hover_neg = R_bw @ np.array([0, 0, -9.81])
print(f"Forward model prediction R_bw @ [0,0,-9.81]: {accel_pred_hover_neg.round(3)}")

# 5. Velocity during motion
print("\n=== Velocity samples during t=5-25s ===")
for s in states:
    t = s.timestamp - t0
    if t < 5 or t > 25:
        continue
    if abs(t - round(t)) > 0.01:
        continue
    speed = np.linalg.norm(s.velocity)
    if speed > 0.05:
        R_wb_t = Rotation.from_quat(s.orientation).as_matrix()
        v_body = R_wb_t.T @ s.velocity
        print(f"t={t:5.1f}s  v_world=[{s.velocity[0]:+.3f},{s.velocity[1]:+.3f},{s.velocity[2]:+.3f}]  "
              f"v_body=[{v_body[0]:+.3f},{v_body[1]:+.3f},{v_body[2]:+.3f}]  speed={speed:.3f}")

# 6. Check the backflips bag - during flips, body z should oscillate wildly
print("\n=== Backflips: body z-axis in world during t=30-35s ===")
bag2 = load_bag_topics('rosbags/backflips_2025-12-17-17-41-24.bag', verbose=False)
states2 = bag2.agiros_state
t02 = states2[0].timestamp
for s in states2:
    t = s.timestamp - t02
    if t < 30 or t > 32:
        continue
    if abs(t * 10 - round(t * 10)) > 0.01:
        continue
    R_wb_t = Rotation.from_quat(s.orientation).as_matrix()
    z_body_in_world = R_wb_t @ [0, 0, 1]
    euler_t = Rotation.from_quat(s.orientation).as_euler('ZYX', degrees=True)
    print(f"t={t:5.1f}s  z_body_world=[{z_body_in_world[0]:+.3f},{z_body_in_world[1]:+.3f},{z_body_in_world[2]:+.3f}]  "
          f"euler=[{euler_t[0]:+.1f},{euler_t[1]:+.1f},{euler_t[2]:+.1f}]")

# 7. Cross-check: what euler convention gives R_y(30) for sensor tilt?
print("\n=== Euler convention check ===")
R_y30 = Rotation.from_euler('ZYX', [0, 30, 0], degrees=True).as_matrix()
R_y_neg30 = Rotation.from_euler('ZYX', [0, -30, 0], degrees=True).as_matrix()
print(f"from_euler('ZYX', [0,30,0]) @ [1,0,0] = {(R_y30 @ [1,0,0]).round(4)}")
print(f"from_euler('ZYX', [0,-30,0]) @ [1,0,0] = {(R_y_neg30 @ [1,0,0]).round(4)}")
print(f"Sensor tilted 30deg DOWN: boresight in body = [cos30, 0, -sin30] = [{np.cos(np.radians(30)):.4f}, 0, {-np.sin(np.radians(30)):.4f}]")
print(f"  → R_body_from_sensor should map [1,0,0] to [0.866, 0, -0.5]")
print(f"  → This matches from_euler('ZYX', [0,+30,0]) = {(R_y30 @ [1,0,0]).round(4)}")
print(f"  → OR from_euler('ZYX', [0,-30,0]) = {(R_y_neg30 @ [1,0,0]).round(4)}")
