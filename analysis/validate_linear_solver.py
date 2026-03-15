"""
Phase 2: Linear Solver Validation for B-Spline Trajectory Estimation

This script validates the sparse linear least squares construction by:
1. Using KNOWN orientation and angular velocity from MoCap
2. Solving ONLY for position control points (closed-form optimal solution)
3. Using standard L2 loss (no robustification)
4. Including minimum snap regularization
5. Solving with sparse Cholesky decomposition

This validates:
- Linear least squares construction correctness
- Observability of the problem
- Regularization effectiveness
- Sparse matrix handling

Once this works, we can add nonlinear optimization, bias estimation, etc.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Dict, Any, Tuple
from scipy.interpolate import interp1d
import time

# Add lib/ to path for shared libraries
sys.path.insert(0, str(Path(__file__).parent / 'lib'))

from rosbag_loader import load_bag_topics
from radar_velocity_utils import (
    quat_to_rotation_matrix,
    rotation_matrix_from_euler
)
from bspline_utils import (
    UniformBSpline,
    create_uniform_bspline_from_times,
    build_minimum_snap_regularization
)


def build_radar_jacobian(
    bspline: UniformBSpline,
    radar_frames,
    agiros_states,
    t_ref: float,
    R_body_from_sensor: np.ndarray,
    T_body_from_sensor: np.ndarray,
    time_offset: float,
    min_range: float = 0.2
) -> Tuple[sparse.csr_matrix, np.ndarray, int]:
    """
    Build sparse Jacobian matrix for radar measurements.
    
    For each radar point measuring Doppler v_d:
    residual = v_d_measured - (u_b · v_ant)
    where v_ant = R_b<-w * v_w + omega_b × T_b<-s
    
    The Jacobian is w.r.t. control points, so:
    dr/dc_i = -(u_b · R_b<-w) * dv_w/dc_i
    
    Since v_w(t) = sum(N'_i(t) * c_i), we have:
    dv_w/dc_i = N'_i(t) * I_3x3
    
    Args:
        bspline: B-spline object (with dummy control points)
        radar_frames: List of RadarVelocity objects  
        agiros_states: List of AgirosState objects (for orientation and omega)
        t_ref: Reference time for conversion to relative time
        R_body_from_sensor: Rotation from sensor to body frame
        T_body_from_sensor: Translation from body to sensor in body frame
        time_offset: Time offset for radar measurements
        min_range: Minimum range filter
        
    Returns:
        (J, residuals, n_measurements)
    """
    from scipy.interpolate import interp1d
    
    # Create interpolators for MoCap orientation and angular velocity (using relative time)
    agiros_times = np.array([s.timestamp - t_ref for s in agiros_states])
    agiros_quats = np.array([s.orientation for s in agiros_states])
    agiros_omegas = np.array([s.angular_velocity for s in agiros_states])
    agiros_vels = np.array([s.velocity for s in agiros_states])
    
    quat_interp = interp1d(agiros_times, agiros_quats, axis=0, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    omega_interp = interp1d(agiros_times, agiros_omegas, axis=0, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    vel_interp = interp1d(agiros_times, agiros_vels, axis=0, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
    
    rows = []
    cols = []
    vals = []
    residuals = []
    
    measurement_idx = 0
    
    for frame in radar_frames:
        t_corrected = (frame.timestamp - t_ref) + time_offset  # Convert to relative time
        
        # Skip if outside MoCap range
        if t_corrected < agiros_times[0] or t_corrected > agiros_times[-1]:
            continue
        
        # Get MoCap state at this time
        try:
            quat = quat_interp(t_corrected)
            omega_body = omega_interp(t_corrected)
            v_world_gt = vel_interp(t_corrected)  # For initialization check only
        except (ValueError, RuntimeError):
            continue
        
        R_world_from_body = quat_to_rotation_matrix(quat)
        R_body_from_world = R_world_from_body.T
        
        # Get radar measurements
        positions = np.array(frame.positions)
        measured_dopplers = np.array(frame.velocities)
        
        # Filter by range
        ranges = np.linalg.norm(positions, axis=1)
        valid_mask = ranges >= min_range
        
        if np.sum(valid_mask) == 0:
            continue
        
        valid_positions = positions[valid_mask]
        valid_measurements = measured_dopplers[valid_mask]
        valid_ranges = ranges[valid_mask]
        
        # Unit direction vectors in sensor frame
        u_sensor = valid_positions / valid_ranges[:, None]
        
        # Transform to body frame
        u_body = (R_body_from_sensor @ u_sensor.T).T
        
        # Lever arm velocity
        lever_arm_vel = np.cross(omega_body, T_body_from_sensor)
        
        # Get basis coefficients for velocity (1st derivative)
        vel_coeffs, vel_indices = bspline.get_basis_coefficients(t_corrected, derivative=1)
        
        if len(vel_coeffs) == 0:
            continue
        
        # For each radar point
        for point_idx in range(len(valid_measurements)):
            v_d_meas = valid_measurements[point_idx]
            u_b = u_body[point_idx]
            
            # The measurement model is:
            # v_d = u_b · (R_b<-w * v_w + lever_arm_vel)
            #     = u_b · R_b<-w · v_w + u_b · lever_arm_vel
            #     = (R_b<-w^T · u_b)^T · v_w + const
            #     = direction_world^T · v_w + const
            
            direction_world = R_body_from_world.T @ u_b  # Direction in world frame
            const_term = np.dot(u_b, lever_arm_vel)
            
            # Residual = measured - predicted
            # predicted = direction_world · v_w(t) + const_term
            # where v_w(t) = sum_i N'_i(t) * c_i
            
            # So: dr/dc_i = -direction_world * N'_i(t)
            # This gives 1 row with 3*len(vel_indices) non-zero columns
            
            # For each control point affecting this time
            for coeff_idx, cp_idx in enumerate(vel_indices):
                N_prime = vel_coeffs[coeff_idx]
                
                # Add 3 columns (x, y, z components of control point cp_idx)
                for dim in range(3):
                    row_idx = measurement_idx
                    col_idx = cp_idx * 3 + dim
                    jac_val = -direction_world[dim] * N_prime
                    
                    rows.append(row_idx)
                    cols.append(col_idx)
                    vals.append(jac_val)
            
            # Compute residual (we'll update this after solving)
            # For now, use ground truth velocity as initial guess
            v_w_init = vel_interp(t_corrected)
            predicted = np.dot(direction_world, v_w_init) + const_term
            residual = v_d_meas - predicted
            residuals.append(residual)
            
            measurement_idx += 1
    
    n_measurements = measurement_idx
    n_vars = bspline.n_points * 3
    
    J = sparse.csr_matrix((vals, (rows, cols)), shape=(n_measurements, n_vars))
    residuals = np.array(residuals)
    
    return J, residuals, n_measurements


def build_accelerometer_jacobian(
    bspline: UniformBSpline,
    imu_data,
    agiros_states,
    t_ref: float,
    g_world: np.ndarray = np.array([0, 0, -9.81]),
    subsample: int = 10
) -> Tuple[sparse.csr_matrix, np.ndarray, int]:
    """
    Build sparse Jacobian matrix for accelerometer measurements.
    
    For each IMU measurement:
    residual = z_acc - R_b<-w * (a_w - g_w)
    
    The Jacobian w.r.t. control points:
    dr/dc_i = -R_b<-w * da_w/dc_i
    where a_w(t) = sum(N''_i(t) * c_i)
    
    Uses real IMU acceleration data and MoCap orientation.
    
    Args:
        bspline: B-spline object
        imu_data: List of IMUData objects (for real accelerometer measurements)
        agiros_states: List of AgirosState objects (for orientation)
        t_ref: Reference time for conversion to relative time
        g_world: Gravity vector in world frame
        subsample: Use every N-th IMU sample (IMU at ~1kHz is excessive)
        
    Returns:
        (J, residuals, n_measurements)
    """
    # Build orientation interpolator from MoCap
    agiros_times = np.array([s.timestamp - t_ref for s in agiros_states])
    agiros_quats = np.array([s.orientation for s in agiros_states])
    quat_interp = interp1d(agiros_times, agiros_quats, axis=0, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    
    rows = []
    cols = []
    vals = []
    residuals = []
    
    measurement_idx = 0
    
    for i, imu in enumerate(imu_data):
        if i % subsample != 0:
            continue
        
        t = imu.timestamp - t_ref  # Convert to relative time
        
        # Skip if outside spline range or MoCap range
        if t < bspline.t_start or t > bspline.t_end:
            continue
        if t < agiros_times[0] or t > agiros_times[-1]:
            continue
        
        # Get orientation from MoCap
        quat = quat_interp(t)
        R_world_from_body = quat_to_rotation_matrix(quat)
        R_body_from_world = R_world_from_body.T
        
        # Get basis coefficients for acceleration (2nd derivative)
        acc_coeffs, acc_indices = bspline.get_basis_coefficients(t, derivative=2)
        
        if len(acc_coeffs) == 0:
            continue
        
        # Measurement model: z_acc = R_b<-w * (a_w - g_w) + b_a
        # Ignoring bias: z_acc ~= R_b<-w * (a_w - g_w)
        # Residual: r = z_acc - R_b<-w * (a_w - g_w)
        # Jacobian: dr/dc_i = -R_b<-w * N''_i(t) * I_3x3
        
        for residual_dim in range(3):
            row_idx = measurement_idx * 3 + residual_dim
            
            for coeff_idx, cp_idx in enumerate(acc_indices):
                N_double_prime = acc_coeffs[coeff_idx]
                
                for cp_dim in range(3):
                    col_idx = cp_idx * 3 + cp_dim
                    jac_val = -R_body_from_world[residual_dim, cp_dim] * N_double_prime
                    
                    rows.append(row_idx)
                    cols.append(col_idx)
                    vals.append(jac_val)
        
        # Real IMU accelerometer measurement (specific force in body frame)
        measured_specific_force = imu.linear_acceleration
        
        # Predicted specific force from initial guess: R_b<-w * (a_w - g_w)
        # For the initial residual, use zero accel as placeholder
        # (will be corrected by the solver)
        predicted_specific_force = R_body_from_world @ (np.zeros(3) - g_world)
        
        residual_vec = measured_specific_force - predicted_specific_force
        residuals.extend(residual_vec)
        
        measurement_idx += 1
    
    n_measurements = measurement_idx * 3
    n_vars = bspline.n_points * 3
    
    J = sparse.csr_matrix((vals, (rows, cols)), shape=(n_measurements, n_vars))
    residuals = np.array(residuals)
    
    return J, residuals, measurement_idx


def solve_trajectory_linear(
    bspline: UniformBSpline,
    J_radar: sparse.csr_matrix,
    residuals_radar: np.ndarray,
    J_accel: sparse.csr_matrix,
    residuals_accel: np.ndarray,
    lambda_accel: float = 1.0,
    lambda_snap: float = 0.1,
    lambda_position: float = 0.0,
    velocity_priors: list = None,
    lambda_velocity: float = 0.0,
    verbose: bool = True
) -> np.ndarray:
    """
    Solve for control points using sparse linear least squares.
    
    Minimizes:
    ||J_radar * x - r_radar||^2 + lambda_accel * ||J_accel * x - r_accel||^2 
    + lambda_snap * ||R_snap * x||^2 + lambda_position * ||x - x_init||^2
    + lambda_velocity * ||J_vel * x - v_target||^2
    
    Using normal equations: H * x = b where H = J^T*J + regularization
    
    Args:
        bspline: B-spline object (control points will be updated)
        J_radar: Radar Jacobian
        J_accel: Accelerometer Jacobian
        residuals_radar: Radar residuals
        residuals_accel: Accelerometer residuals
        lambda_accel: Weight for accelerometer term
        lambda_snap: Weight for minimum snap regularization
        lambda_position: Weight for position anchoring (keeps near initial guess)
        velocity_priors: List of (t_rel, v_target) tuples for velocity constraints
        lambda_velocity: Weight for velocity boundary priors
        verbose: Print solver info
        
    Returns:
        Optimized control points (flattened)
    """
    if verbose:
        print(f"\n{'Building Normal Equations':-^80}")
        print(f"Radar measurements: {J_radar.shape[0]}")
        print(f"Accel measurements: {J_accel.shape[0] // 3}")
        print(f"Control points: {bspline.n_points}")
        print(f"Variables: {bspline.n_points * 3}")
        if velocity_priors:
            print(f"Velocity priors: {len(velocity_priors)} points, lambda={lambda_velocity}")
    
    # Build minimum snap regularization
    if verbose:
        print(f"Building minimum snap regularization (lambda={lambda_snap})...")
    R_snap = build_minimum_snap_regularization(bspline, n_samples=100)
    
    # Build normal equations: H * x = b
    # H = J_radar^T * J_radar + lambda_accel * J_accel^T * J_accel 
    #     + lambda_snap * R_snap^T * R_snap + lambda_position * I
    # b = J_radar^T * r_radar + lambda_accel * J_accel^T * r_accel 
    #     + lambda_position * x_init
    
    if verbose:
        print(f"Assembling sparse normal equations...")
    
    t_start = time.time()
    
    # Save initial control points for position regularization
    x_init = bspline.control_points.flatten()
    
    # Compute J^T * J terms
    H_radar = J_radar.T @ J_radar
    H_accel = lambda_accel * (J_accel.T @ J_accel)
    H_snap = lambda_snap * (R_snap.T @ R_snap)
    
    # Position anchoring: penalize deviation from initial guess
    n_vars = bspline.n_points * 3
    H_position = lambda_position * sparse.eye(n_vars)
    
    # Add small Tikhonov regularization for numerical stability
    tikhonov_lambda = 1e-6
    H_tikhonov = tikhonov_lambda * sparse.eye(n_vars)
    
    H = H_radar + H_accel + H_snap + H_position + H_tikhonov
    
    # Velocity boundary priors: ||J_vel * x - v_target||^2
    H_vel = sparse.csr_matrix((n_vars, n_vars))
    b_vel = np.zeros(n_vars)
    if velocity_priors and lambda_velocity > 0:
        vel_rows, vel_cols, vel_vals = [], [], []
        vel_rhs = []
        row_idx = 0
        for t_rel, v_target in velocity_priors:
            vel_coeffs, vel_indices = bspline.get_basis_coefficients(t_rel, derivative=1)
            if len(vel_coeffs) == 0:
                continue
            for k in range(3):  # x, y, z components
                for ci, cp_idx in enumerate(vel_indices):
                    vel_rows.append(row_idx)
                    vel_cols.append(cp_idx * 3 + k)
                    vel_vals.append(vel_coeffs[ci])
                vel_rhs.append(v_target[k])
                row_idx += 1
        if row_idx > 0:
            J_vel = sparse.csr_matrix(
                (vel_vals, (vel_rows, vel_cols)), shape=(row_idx, n_vars))
            r_vel = np.array(vel_rhs)
            H_vel = lambda_velocity * (J_vel.T @ J_vel)
            b_vel = lambda_velocity * (J_vel.T @ r_vel)
    
    H = H + H_vel
    
    # Compute J^T * r terms
    b_radar = J_radar.T @ residuals_radar
    b_accel = lambda_accel * (J_accel.T @ residuals_accel)
    b_position = lambda_position * x_init
    
    b = b_radar + b_accel + b_position + b_vel
    
    t_assembly = time.time() - t_start
    
    if verbose:
        print(f"Assembly time: {t_assembly:.3f}s")
        print(f"H matrix: {H.shape}, nnz={H.nnz}, sparsity={100*(1-H.nnz/(H.shape[0]*H.shape[1])):.2f}%")
        print(f"Solving with sparse Cholesky...")
    
    # Solve using sparse Cholesky
    t_start = time.time()
    
    try:
        from sksparse.cholmod import cholesky
        factor = cholesky(H)
        x = factor(b)
        solver = "CHOLMOD"
    except ImportError:
        # Fallback to scipy sparse solver
        x = spsolve(H, b)
        solver = "scipy.sparse.linalg.spsolve"
    
    t_solve = time.time() - t_start
    
    if verbose:
        print(f"Solver: {solver}")
        print(f"Solve time: {t_solve:.3f}s")
    
    return x


# Available bags
BAGS = {
    "original":     "rosbags/2025-12-17-16-02-22.bag",
    "circle":       "rosbags/circle_2025-12-17-17-21-37.bag",
    "circle_fast":  "rosbags/circle_fast_2025-12-17-17-25-34.bag",
    "circle_fwd":   "rosbags/circle_forward_2025-12-17-17-37-38.bag",
    "loopings":     "rosbags/circle_fast_forward_2025-12-17-17-39-49.bag",
    "backflips":    "rosbags/backflips_2025-12-17-17-41-24.bag",
}

# Bags where the agiros body frame is rotated 180 deg in yaw
FLIPPED_BAGS = {"circle_fwd", "backflips", "loopings"}


def main():
    print("=" * 80)
    print("PHASE 2: LINEAR SOLVER VALIDATION")
    print("B-Spline Trajectory Estimation with Sparse Least Squares")
    print("=" * 80)
    
    # ==================== Configuration ====================
    bag_key = sys.argv[1] if len(sys.argv) > 1 else "original"
    if bag_key in BAGS:
        BAG_PATH = BAGS[bag_key]
    else:
        BAG_PATH = bag_key
    
    START_TIME_OFFSET = 5.0   # Skip initial hover
    DURATION = 120.0          # Full bag
    
    # Extrinsics (validated in physics checks)
    ROTATION_EULER_DEG = np.array([0.0, 30.0, 0.0])  # roll, pitch, yaw in degrees
    
    # Body frame flip for certain trajectory profiles
    FLIP_BODY_FRAME = bag_key in FLIPPED_BAGS
    if "--flip" in sys.argv:
        FLIP_BODY_FRAME = True
    if "--no-flip" in sys.argv:
        FLIP_BODY_FRAME = False
    
    if FLIP_BODY_FRAME:
        TRANSLATION = np.array([-0.07, 0.0, 0.0])
        R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)
        R_body_from_sensor = R_yaw_flip @ rotation_matrix_from_euler(
            np.radians(ROTATION_EULER_DEG[0]),
            np.radians(ROTATION_EULER_DEG[1]),
            np.radians(ROTATION_EULER_DEG[2]),
        )
        print(f"  Body frame FLIPPED (R_z(180 deg) applied) for bag '{bag_key}'")
    else:
        TRANSLATION = np.array([0.07, 0.0, 0.0])
        R_body_from_sensor = rotation_matrix_from_euler(
            np.radians(ROTATION_EULER_DEG[0]),
            np.radians(ROTATION_EULER_DEG[1]),
            np.radians(ROTATION_EULER_DEG[2]),
        )
    
    TIME_OFFSET = -0.020  # IMU-MoCap offset: -20ms
    
    # B-spline parameters
    BSPLINE_DEGREE = 7  # Quintic for continuous snap
    
    # Regularization weights
    LAMBDA_ACCEL = 0.001     # Balance between noise and damping
    LAMBDA_SNAP = 0.0       # Disable snap regularization  
    LAMBDA_POSITION = 0.0  # Position anchor to limit drift
    
    MIN_RANGE = 0.2
    
    print(f"\n{'Configuration':-^80}")
    print(f"Bag: {bag_key} -> {BAG_PATH}")
    print(f"Time window: {START_TIME_OFFSET:.1f}s + {DURATION:.1f}s")
    print(f"B-spline degree: {BSPLINE_DEGREE}")
    print(f"Flip body frame: {FLIP_BODY_FRAME}")
    print(f"Lambda accel: {LAMBDA_ACCEL}")
    print(f"Lambda snap: {LAMBDA_SNAP}")
    print(f"Lambda position: {LAMBDA_POSITION}")
    
    # ==================== Load Data ====================
    print(f"\n{'Loading Data':-^80}")
    
    bag_data = load_bag_topics(BAG_PATH, verbose=True)
    
    t_start = bag_data.start_time + START_TIME_OFFSET
    t_end = t_start + DURATION
    
    agiros_states = [s for s in bag_data.agiros_state if t_start <= s.timestamp <= t_end]
    radar_frames = [f for f in bag_data.radar_velocity if t_start <= f.timestamp <= t_end]
    imu_data = [d for d in bag_data.imu_data if t_start <= d.timestamp <= t_end]
    
    print(f"\nFiltered data:")
    print(f"  MoCap states: {len(agiros_states)}")
    print(f"  Radar frames: {len(radar_frames)}")
    print(f"  IMU samples:  {len(imu_data)}")
    
    if len(agiros_states) == 0 or len(radar_frames) == 0:
        print("ERROR: Insufficient data!")
        return
    
    # ==================== Initialize B-Spline ====================
    print(f"\n{'Initializing B-Spline':-^80}")
    
    # Work with relative times for numerical stability
    mocap_times_abs = np.array([s.timestamp for s in agiros_states])
    t_ref = mocap_times_abs[0]  # Reference time (first measurement)
    mocap_times_rel = mocap_times_abs - t_ref
    
    print(f"Reference time: {t_ref:.2f} (absolute)")
    print(f"Relative time range: [{mocap_times_rel[0]:.2f}, {mocap_times_rel[-1]:.2f}]")
    
    # Create time array from measurements
    bspline, n_control_points = create_uniform_bspline_from_times(
        mocap_times_rel, BSPLINE_DEGREE, boundary_order=3
    )
    
    # Store reference time for later use
    bspline.t_ref = t_ref
    
    print(f"Control points: {n_control_points}")
    print(f"Knot spacing (dt): {bspline.dt:.4f}s")
    print(f"Valid time range: [{bspline.t_start:.2f}, {bspline.t_end:.2f}] (relative)")
    print(f"Variables to solve: {n_control_points * 3}")
    
    # Initialize control points with MoCap trajectory
    mocap_positions = np.array([s.position for s in agiros_states])
    
    from scipy.interpolate import interp1d
    pos_interp = interp1d(mocap_times_rel, mocap_positions, axis=0, kind='cubic',
                          fill_value='extrapolate')
    
    # Sample at knot times (rough initialization)
    init_times = np.linspace(bspline.t_start, bspline.t_end, n_control_points)
    bspline.control_points = pos_interp(init_times)
    
    print(f"Initialized control points from MoCap trajectory")
    
    # ==================== Build Measurement Jacobians ====================
    print(f"\n{'Building Measurement Jacobians':-^80}")
    
    J_radar, r_radar, n_radar = build_radar_jacobian(
        bspline, radar_frames, agiros_states, t_ref,
        R_body_from_sensor, TRANSLATION, TIME_OFFSET, MIN_RANGE
    )
    
    print(f"Radar Jacobian: {J_radar.shape}, nnz={J_radar.nnz}")
    print(f"Radar measurements: {n_radar}")
    
    J_accel, r_accel, n_accel = build_accelerometer_jacobian(
        bspline, imu_data, agiros_states, t_ref, subsample=10
    )
    
    print(f"Accel Jacobian: {J_accel.shape}, nnz={J_accel.nnz}")
    print(f"Accel measurements: {n_accel}")
    
    # ==================== Solve ====================
    print(f"\n{'Solving for Optimal Control Points':#^80}")
    
    verbose = True  # Enable verbose output
    
    x_opt = solve_trajectory_linear(
        bspline, J_radar, r_radar, J_accel, r_accel,
        LAMBDA_ACCEL, LAMBDA_SNAP, LAMBDA_POSITION, verbose=verbose
    )
    
    # Update bspline with optimized control points
    bspline.control_points = x_opt.reshape(-1, 3)
    
    if verbose:
        print(f"\n✅ Optimization complete!")
        print(f"Solution check: max(|x|) = {np.abs(x_opt).max():.4f}, contains NaN: {np.isnan(x_opt).any()}")
    
    # ==================== Evaluate and Compare ====================
    print(f"\n{'Evaluating Trajectory':-^80}")
    
    # Sample trajectory at MoCap times (relative)
    eval_times = mocap_times_rel
    estimated_positions = np.array([bspline(t, derivative=0) for t in eval_times])
    estimated_velocities = np.array([bspline(t, derivative=1) for t in eval_times])
    
    mocap_velocities = np.array([s.velocity for s in agiros_states])
    
    # Transform world velocities to body frame for speed check
    mocap_velocities_body = []
    for s in agiros_states:
        R_world_from_body = quat_to_rotation_matrix(s.orientation)
        v_body = R_world_from_body.T @ s.velocity
        mocap_velocities_body.append(v_body)
    mocap_velocities_body = np.array(mocap_velocities_body)
    
    # Compute errors
    pos_errors = np.linalg.norm(estimated_positions - mocap_positions, axis=1)
    pos_diff = estimated_positions - mocap_positions  # Component-wise offset
    vel_errors = np.linalg.norm(estimated_velocities - mocap_velocities, axis=1)
    
    # Compute speed magnitudes for comparison
    mocap_speeds = np.linalg.norm(mocap_velocities, axis=1)
    mocap_speeds_body_fwd = mocap_velocities_body[:, 0]  # Body x-axis (forward)
    estimated_speeds = np.linalg.norm(estimated_velocities, axis=1)
    
    print(f"\nPosition Errors:")
    print(f"  Mean: {pos_errors.mean():.4f} m")
    print(f"  Std:  {pos_errors.std():.4f} m")
    print(f"  Max:  {pos_errors.max():.4f} m")
    print(f"  RMSE: {np.sqrt(np.mean(pos_errors**2)):.4f} m")
    print(f"\nPosition Offset (estimated - mocap):")
    print(f"  X: mean={pos_diff[:, 0].mean():.4f}, std={pos_diff[:, 0].std():.4f} m")
    print(f"  Y: mean={pos_diff[:, 1].mean():.4f}, std={pos_diff[:, 1].std():.4f} m")
    print(f"  Z: mean={pos_diff[:, 2].mean():.4f}, std={pos_diff[:, 2].std():.4f} m")
    
    print(f"\nVelocity Errors:")
    print(f"  Mean: {vel_errors.mean():.4f} m/s")
    print(f"  Std:  {vel_errors.std():.4f} m/s")
    print(f"  Max:  {vel_errors.max():.4f} m/s")
    print(f"  RMSE: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s")
    
    print(f"\nSpeed Profile (MoCap - World Frame):")
    print(f"  Max speed: {mocap_speeds.max():.2f} m/s")
    print(f"  Mean speed: {mocap_speeds.mean():.2f} m/s")
    print(f"  Speed at t=1s: {mocap_speeds[np.argmin(np.abs(eval_times - 1.0))]:.2f} m/s")
    print(f"  Speed at t=8s: {mocap_speeds[np.argmin(np.abs(eval_times - 8.0))]:.2f} m/s")
    
    print(f"\nSpeed Profile (MoCap - Body Forward):")
    print(f"  Max speed: {mocap_speeds_body_fwd.max():.2f} m/s")
    print(f"  Min speed: {mocap_speeds_body_fwd.min():.2f} m/s")
    print(f"  Mean speed: {mocap_speeds_body_fwd.mean():.2f} m/s")
    print(f"  Speed at t=1s: {mocap_speeds_body_fwd[np.argmin(np.abs(eval_times - 1.0))]:.2f} m/s")
    print(f"  Speed at t=8s: {mocap_speeds_body_fwd[np.argmin(np.abs(eval_times - 8.0))]:.2f} m/s")
    print(f"  Speed at t=15s: {mocap_speeds_body_fwd[np.argmin(np.abs(eval_times - 15.0))]:.2f} m/s")
    
    print(f"\nVelocity Components (MoCap - World Frame):")
    print(f"  X: min={mocap_velocities[:, 0].min():.2f}, max={mocap_velocities[:, 0].max():.2f}, mean={mocap_velocities[:, 0].mean():.2f} m/s")
    print(f"  Y: min={mocap_velocities[:, 1].min():.2f}, max={mocap_velocities[:, 1].max():.2f}, mean={mocap_velocities[:, 1].mean():.2f} m/s")
    print(f"  Z: min={mocap_velocities[:, 2].min():.2f}, max={mocap_velocities[:, 2].max():.2f}, mean={mocap_velocities[:, 2].mean():.2f} m/s")
    
    print(f"\nSpeed Profile (Estimated):")
    print(f"  Max speed: {estimated_speeds.max():.2f} m/s")
    print(f"  Mean speed: {estimated_speeds.mean():.2f} m/s")
    print(f"  Speed at t=1s: {estimated_speeds[np.argmin(np.abs(eval_times - 1.0))]:.2f} m/s")
    print(f"  Speed at t=8s: {estimated_speeds[np.argmin(np.abs(eval_times - 8.0))]:.2f} m/s")
    
    # ==================== Plotting ====================
    print(f"\n{'Generating Plots':-^80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Linear Solver Validation Results', fontsize=14, fontweight='bold')
    
    # 1. Trajectory comparison (X-Y plane)
    ax = axes[0, 0]
    ax.plot(mocap_positions[:, 0], mocap_positions[:, 1], 'b-', label='MoCap Ground Truth', linewidth=2)
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'r--', label='B-Spline Estimate', linewidth=2)
    ax.scatter(bspline.control_points[:, 0], bspline.control_points[:, 1], 
               c='orange', marker='x', s=50, label='Control Points', zorder=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory (X-Y Plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 2. Position error over time
    ax = axes[0, 1]
    time_rel = eval_times - eval_times[0]
    ax.plot(time_rel, pos_errors, 'r-', linewidth=2)
    ax.axhline(pos_errors.mean(), color='b', linestyle='--', label=f'Mean: {pos_errors.mean():.4f}m')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Velocity comparison
    ax = axes[1, 0]
    ax.plot(time_rel, np.linalg.norm(mocap_velocities, axis=1), 'b-', label='MoCap Speed', linewidth=2)
    ax.plot(time_rel, np.linalg.norm(estimated_velocities, axis=1), 'r--', label='Estimated Speed', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Error histogram
    ax = axes[1, 1]
    ax.hist(pos_errors, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(pos_errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {pos_errors.mean():.4f}m')
    ax.set_xlabel('Position Error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Position Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_name = f'linear_solver_{bag_key}.png'
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_name}")
    
    # ==================== Summary ====================
    print(f"\n{'VALIDATION SUMMARY':#^80}")
    
    if pos_errors.mean() < 0.1 and vel_errors.mean() < 0.5:
        print("✅ LINEAR SOLVER VALIDATION SUCCESSFUL!")
        print("   - Position errors are low")
        print("   - Velocity errors are acceptable")
        print("   - Sparse matrix construction is correct")
        print("   - Regularization is working properly")
        print("\nReady for Phase 3: Nonlinear optimization with bias estimation")
    else:
        print("⚠️  LINEAR SOLVER NEEDS TUNING")
        print(f"   - Position RMSE: {np.sqrt(np.mean(pos_errors**2)):.4f} m")
        print(f"   - Velocity RMSE: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s")
        print("\nConsider adjusting:")
        print("   - Regularization weights (lambda_accel, lambda_snap)")
        print("   - B-spline degree or knot spacing")
        print("   - Control point initialization")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
