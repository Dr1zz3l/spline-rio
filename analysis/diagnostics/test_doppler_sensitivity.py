import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""
Quick test to check if Doppler predictions are sensitive to state perturbations.
"""
import numpy as np
import sys
sys.path.append('.')

from validate_nonlinear_solver import TrajectoryState, compute_radar_residuals_nonlinear
from rosbag_loader import load_bag_topics
from bspline_utils import UniformBSpline, create_uniform_bspline_from_times
from radar_velocity_utils import rotation_matrix_from_euler, quat_to_rotation_matrix
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

# Load minimal data
BAG_PATH = "rosbags/2025-12-17-16-02-22.bag"
START_TIME_OFFSET = 31.5
DURATION = 15.0

print("Loading data...")
bag_data = load_bag_topics(BAG_PATH, verbose=False)

t_start = bag_data.start_time + START_TIME_OFFSET
t_end = t_start + DURATION

agiros_states = [s for s in bag_data.agiros_state if t_start <= s.timestamp <= t_end]
radar_frames_window = [f for f in bag_data.radar_velocity if t_start <= f.timestamp <= t_end]
radar_frames = [radar_frames_window[len(radar_frames_window)//2]]  # Middle frame

print(f"Using {radar_frames[0].num_points()} radar points from one frame")
print(f"Radar frame timestamp: {radar_frames[0].timestamp:.3f}")

# Create simple state
mocap_times_abs = np.array([s.timestamp for s in agiros_states])
t_ref = mocap_times_abs[0]
mocap_times_rel = mocap_times_abs - t_ref
mocap_positions = np.array([s.position for s in agiros_states])
mocap_orientations = np.array([s.orientation for s in agiros_states])

pos_bspline, n_pos = create_uniform_bspline_from_times(mocap_times_rel, 5)
pos_bspline.t_ref = t_ref
pos_interp = interp1d(mocap_times_rel, mocap_positions, axis=0, kind='cubic', fill_value='extrapolate')
init_times = np.linspace(max(pos_bspline.t_start, mocap_times_rel[0]), 
                         min(pos_bspline.t_end, mocap_times_rel[-1]), n_pos)
pos_bspline.control_points = pos_interp(init_times)

ori_bspline = UniformBSpline(np.zeros((35, 3)), 3, pos_bspline.dt * 3.0)
ori_bspline.t_ref = t_ref

nominal_rots = np.array([quat_to_rotation_matrix(q) for q in mocap_orientations[::3][:35]])

state = TrajectoryState(pos_bspline, ori_bspline, nominal_rots)

# Sensor extrinsics
sensor_translation = np.array([0.07, 0.0, 0.0])
sensor_rotation = rotation_matrix_from_euler(0, np.radians(-30), 0)

# Find control point nearest to radar measurement
radar_t_rel = radar_frames[0].timestamp - t_ref
pos_ctrl_times = np.linspace(pos_bspline.t_start, pos_bspline.t_end, n_pos)
ori_ctrl_times = np.linspace(ori_bspline.t_start, ori_bspline.t_end, 35)
pos_idx = np.argmin(np.abs(pos_ctrl_times - radar_t_rel))
ori_idx = np.argmin(np.abs(ori_ctrl_times - radar_t_rel))

print(f"Radar t_rel: {radar_t_rel:.3f}, nearest pos ctrl {pos_idx}/{n_pos}, ori ctrl {ori_idx}/35")

# Compute baseline residuals
print("\n=== BASELINE ===")
t_radar = radar_frames[0].timestamp
print(f"Radar timestamp (abs): {t_radar:.3f}")
print(f"B-spline position: {state.get_position(t_radar, derivative=0)}")
print(f"B-spline velocity: {state.get_position(t_radar, derivative=1)}")
print(f"B-spline omega: {state.get_angular_velocity(t_radar)}")
r0, w0, t0 = compute_radar_residuals_nonlinear(
    state, radar_frames, sensor_translation, sensor_rotation, huber_delta=0.5
)
print(f"Residuals: mean={np.mean(r0):.6f}, std={np.std(r0):.6f}")
print(f"First 5 residuals: {r0[:5]}")

# Perturb position control point
print(f"\n=== PERTURB POSITION [{pos_idx}] by 1.0m ===")
state.pos_bspline.control_points[pos_idx, 0] += 1.0
print(f"B-spline position AFTER: {state.get_position(t_radar, derivative=0)}")
print(f"B-spline velocity AFTER: {state.get_position(t_radar, derivative=1)}")
r1, _, _ = compute_radar_residuals_nonlinear(
    state, radar_frames, sensor_translation, sensor_rotation, huber_delta=0.5
)
print(f"Residuals: mean={np.mean(r1):.6f}, std={np.std(r1):.6f}")
print(f"Change: mean={np.mean(r1-r0):.6f}, max={np.max(np.abs(r1-r0)):.6f}")
state.pos_bspline.control_points[pos_idx, 0] -= 1.0  # restore

# Perturb orientation control point
print(f"\n=== PERTURB ORIENTATION [{ori_idx}] by 0.1 rad ===")
state.ori_bspline.control_points[ori_idx, 0] += 0.1
r2, _, _ = compute_radar_residuals_nonlinear(
    state, radar_frames, sensor_translation, sensor_rotation, huber_delta=0.5
)
print(f"Residuals: mean={np.mean(r2):.6f}, std={np.std(r2):.6f}")
print(f"Change: mean={np.mean(r2-r0):.6f}, max={np.max(np.abs(r2-r0)):.6f}")
state.ori_bspline.control_points[ori_idx, 0] -= 0.1  # restore

# Perturb acc bias
print("\n=== PERTURB ACC_BIAS[0] by 1.0 m/s² ===")
state.acc_bias[0] += 1.0
r3, _, _ = compute_radar_residuals_nonlinear(
    state, radar_frames, sensor_translation, sensor_rotation, huber_delta=0.5
)
print(f"Residuals: mean={np.mean(r3):.6f}, std={np.std(r3):.6f}")
print(f"Change: mean={np.mean(r3-r0):.6f}, max={np.max(np.abs(r3-r0)):.6f}")

print("\n=== CONCLUSION ===")
print("If changes are ~0, Jacobian will be zero and radar won't affect optimization!")
