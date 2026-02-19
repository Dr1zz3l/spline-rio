"""
PHASE 3: NONLINEAR SOLVER VALIDATION
Full radar-inertial odometry with orientation and bias estimation.

Based on the Backward Model formulation:
- Position trajectory: B-spline control points
- Orientation trajectory: SO(3) B-spline (using tangent space)
- Biases: Constant accelerometer and gyroscope biases
- Radar: Huber loss for outlier rejection
- Accelerometer: L2 loss
- Optimization: Levenberg-Marquardt
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import time

from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import (
    quat_to_rotation_matrix,
    rotation_matrix_from_euler,
    predict_doppler_velocity
)
from bspline_utils import (
    UniformBSpline,
    create_uniform_bspline_from_times,
    build_minimum_snap_regularization
)
from generated_jacobians import (
    radar_residual_with_jacobians,
    accel_residual_with_jacobians,
    gyro_residual_with_jacobians,
    Rot3
)


# ==================== Orientation Parameterization ====================

def so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map from so(3) to SO(3).
    
    Args:
        omega: 3D tangent vector
        
    Returns:
        3x3 rotation matrix
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        # Small angle approximation
        return np.eye(3) + skew_symmetric(omega)
    
    axis = omega / theta
    return Rotation.from_rotvec(theta * axis).as_matrix()


def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Logarithmic map from SO(3) to so(3).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        3D tangent vector
    """
    rotvec = Rotation.from_matrix(R).as_rotvec()
    return rotvec


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3D vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def compute_omega_and_jacobians(
    omega_nominal: np.ndarray,
    delta: np.ndarray,
    delta_dot: np.ndarray,
    epsilon: float = 1e-10,
) -> tuple:
    """
    Compute body angular velocity and its Jacobians using SymForce-generated code.
    
    omega_body = exp(-[delta]_x) @ omega_nominal + J_r(delta) @ delta_dot
    
    Uses gyro_residual_with_jacobians with z_gyro=0, b_g=0 to extract omega
    and its exact Jacobians w.r.t. delta and delta_dot.
    
    Returns:
        omega: (3,) body angular velocity
        J_omega_delta: (3,3) ∂omega/∂delta
        J_omega_delta_dot: (3,3) ∂omega/∂delta_dot
    """
    _zero3 = np.zeros(3)
    res, J_d, J_dd, _ = gyro_residual_with_jacobians(
        omega_nominal, delta, delta_dot, _zero3, _zero3, epsilon
    )
    # res = z_gyro - omega - b_g = -omega  (since z=0, b=0)
    omega = -res.flatten()
    J_omega_delta = -J_d          # (3,3)
    J_omega_delta_dot = -J_dd     # (3,3)
    return omega, J_omega_delta, J_omega_delta_dot


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions [x, y, z, w].
    
    Args:
        q1, q2: Quaternions in [x, y, z, w] format
        
    Returns:
        Product quaternion
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


# ==================== State Representation ====================

class TrajectoryState:
    """
    Complete trajectory state for non-linear optimization.
    
    State variables:
    - Position control points (N_pos x 3)
    - Orientation control points (N_ori x 3) in tangent space around nominal
    - Accelerometer bias (3,)
    - Gyroscope bias (3,)
    """
    
    def __init__(
        self,
        pos_bspline: UniformBSpline,
        ori_bspline: UniformBSpline,  # Stores tangent vectors
        nominal_rotations: np.ndarray,  # Reference rotations for tangent space
        acc_bias: np.ndarray = None,
        gyr_bias: np.ndarray = None,
        mocap_slerp = None,  # scipy Slerp for dense nominal rotation interpolation
        mocap_omega_interp = None,  # interp1d for nominal angular velocity
    ):
        self.pos_bspline = pos_bspline
        self.ori_bspline = ori_bspline
        self.nominal_rotations = nominal_rotations  # (N_ori, 3, 3)
        
        self.acc_bias = acc_bias if acc_bias is not None else np.zeros(3)
        self.gyr_bias = gyr_bias if gyr_bias is not None else np.zeros(3)
        
        # Dense interpolation objects (from MoCap)
        self.mocap_slerp = mocap_slerp
        self.mocap_omega_interp = mocap_omega_interp
    
    def get_nominal_rotation(self, t_rel: float) -> np.ndarray:
        """Get the nominal rotation at relative time t_rel using SLERP."""
        if self.mocap_slerp is not None:
            # Clamp to valid range
            t_clamped = np.clip(t_rel, self.mocap_slerp.times[0], self.mocap_slerp.times[-1])
            return self.mocap_slerp(t_clamped).as_matrix()
        else:
            # Fallback: nearest-neighbor from control point array
            ori_ctrl_times = np.linspace(self.ori_bspline.t_start, self.ori_bspline.t_end, 
                                         len(self.nominal_rotations))
            idx = np.argmin(np.abs(ori_ctrl_times - t_rel))
            idx = max(0, min(idx, len(self.nominal_rotations) - 1))
            return self.nominal_rotations[idx]
    
    def get_nominal_angular_velocity(self, t_rel: float) -> np.ndarray:
        """Get the nominal (MoCap) angular velocity at relative time t_rel."""
        if self.mocap_omega_interp is not None:
            t_clamped = np.clip(t_rel, 
                               self.mocap_omega_interp.x[0], 
                               self.mocap_omega_interp.x[-1])
            return self.mocap_omega_interp(t_clamped)
        else:
            return np.zeros(3)
    
    def get_position(self, t: float, derivative: int = 0) -> np.ndarray:
        """Get position (or velocity/acceleration) at time t."""
        t_rel = t - self.pos_bspline.t_ref
        return self.pos_bspline(t_rel, derivative=derivative)
    
    def get_rotation(self, t: float) -> np.ndarray:
        """
        Get rotation matrix at time t.
        
        Uses tangent space parameterization:
        R(t) = R_nominal(t) * exp(delta(t))
        """
        t_rel = t - self.ori_bspline.t_ref
        
        # Get tangent vector from spline
        delta_omega = self.ori_bspline(t_rel, derivative=0)
        
        # Interpolate nominal rotation using SLERP
        R_nominal = self.get_nominal_rotation(t_rel)
        
        # Apply perturbation
        R_delta = so3_exp(delta_omega)
        return R_nominal @ R_delta
    
    def get_angular_velocity(self, t: float) -> np.ndarray:
        """
        Get angular velocity at time t using SymForce-generated nonlinear model.
        
        omega_body = exp(-[delta]_x) * omega_nominal + J_r(delta) * delta_dot
        
        Computed via gyro_residual_with_jacobians (exact, no manual math).
        """
        t_rel = t - self.ori_bspline.t_ref
        omega_nominal = self.get_nominal_angular_velocity(t_rel)
        delta = self.ori_bspline(t_rel, derivative=0)
        delta_dot = self.ori_bspline(t_rel, derivative=1)
        omega, _, _ = compute_omega_and_jacobians(omega_nominal, delta, delta_dot)
        return omega
    
    def to_vector(self) -> np.ndarray:
        """
        Flatten state to optimization vector.
        
        Returns:
            [pos_control_points (N_pos*3),
             ori_control_points (N_ori*3),
             acc_bias (3),
             gyr_bias (3)]
        """
        pos_flat = self.pos_bspline.control_points.flatten()
        ori_flat = self.ori_bspline.control_points.flatten()
        
        return np.concatenate([pos_flat, ori_flat, self.acc_bias, self.gyr_bias])
    
    def from_vector(self, x: np.ndarray):
        """Update state from optimization vector."""
        n_pos = self.pos_bspline.n_points * 3
        n_ori = self.ori_bspline.n_points * 3
        
        self.pos_bspline.control_points = x[:n_pos].reshape(-1, 3)
        self.ori_bspline.control_points = x[n_pos:n_pos+n_ori].reshape(-1, 3)
        self.acc_bias = x[n_pos+n_ori:n_pos+n_ori+3]
        self.gyr_bias = x[n_pos+n_ori+3:n_pos+n_ori+6]
    
    def get_state_size(self) -> int:
        """Total number of optimization variables."""
        return self.pos_bspline.n_points * 3 + self.ori_bspline.n_points * 3 + 6
    
    def relinearize(self):
        """
        Absorb current delta perturbations into the nominal rotations and reset delta to zero.
        
        This keeps the tangent-space linearization valid by ensuring delta stays small.
        
        Key: sample R(t) = R_nominal(t) @ exp(delta_spline(t)) at a DENSE grid,
        then rebuild SLERP from those samples. This avoids interpolation mismatch
        between B-spline delta and SLERP nominal.
        
        For angular velocity: compute analytically from the B-spline BEFORE resetting
        delta, using the full nonlinear formula:
            omega = R_delta^T @ omega_nominal + J_r(delta) @ delta_dot
        """
        n_cp = self.ori_bspline.n_points
        
        # Dense sampling to minimize interpolation error between B-spline and SLERP
        n_dense = max(200, n_cp * 20)  # ~200+ samples over the trajectory
        t_start = self.ori_bspline.t_start
        t_end = self.ori_bspline.t_end
        dense_times = np.linspace(t_start, t_end, n_dense)
        
        # Evaluate full rotation AND angular velocity at each dense time point
        # BEFORE resetting delta (using current B-spline state)
        dense_rots = []
        dense_omegas = []
        for t_rel in dense_times:
            R_nom = self.get_nominal_rotation(t_rel)
            delta = self.ori_bspline(t_rel, derivative=0)
            delta_dot = self.ori_bspline(t_rel, derivative=1)
            omega_nominal = self.get_nominal_angular_velocity(t_rel)
            
            # Full rotation
            R_full = R_nom @ so3_exp(delta)
            dense_rots.append(R_full)
            
            # Full angular velocity via SymForce-generated code (exact)
            omega_full, _, _ = compute_omega_and_jacobians(
                omega_nominal, delta, delta_dot
            )
            dense_omegas.append(omega_full)
        
        dense_omegas = np.array(dense_omegas)
        
        # Rebuild SLERP from dense samples
        scipy_rots = Rotation.from_matrix(np.array(dense_rots))
        self.mocap_slerp = Slerp(dense_times, scipy_rots)
        
        # Update nominal_rotations array at CP times for fallback
        cp_times = np.linspace(t_start, t_end, n_cp)
        self.nominal_rotations = np.array([
            self.mocap_slerp(t).as_matrix() for t in cp_times
        ])
        
        # Use analytically computed angular velocity (NOT numerical differentiation)
        self.mocap_omega_interp = interp1d(dense_times, dense_omegas, axis=0,
                                            kind='linear', fill_value='extrapolate')
        
        # Reset delta control points to zero
        self.ori_bspline.control_points = np.zeros((n_cp, 3))


# ==================== Residual Functions ====================

def huber_loss(r: float, delta: float = 1.0) -> float:
    """
    Huber loss function for robust estimation.
    
    Args:
        r: Residual
        delta: Threshold for switching from L2 to L1
        
    Returns:
        rho(r)
    """
    abs_r = abs(r)
    if abs_r <= delta:
        return 0.5 * r**2
    else:
        return delta * (abs_r - 0.5 * delta)


def huber_weight(r: float, delta: float = 1.0) -> float:
    """
    Weight function for iteratively reweighted least squares.
    
    w(r) = rho'(r) / r
    """
    abs_r = abs(r)
    if abs_r <= delta:
        return 1.0
    else:
        return delta / abs_r


def compute_radar_residuals_nonlinear(
    state: TrajectoryState,
    radar_frames,
    sensor_translation: np.ndarray,
    sensor_rotation: np.ndarray,
    min_range: float = 0.2,
    huber_delta: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute radar Doppler residuals with Huber weights.
    
    Returns:
        residuals: (N,) array of weighted residuals
        weights: (N,) array of Huber weights
        times: (N,) array of measurement times
    """
    residuals = []
    weights = []
    times_list = []
    
    for frame in radar_frames:
        t = frame.timestamp
        
        # Get state at measurement time
        pos_world = state.get_position(t, derivative=0)
        vel_world = state.get_position(t, derivative=1)
        R_world_from_body = state.get_rotation(t)
        omega_body = state.get_angular_velocity(t)
        
        n_points = frame.num_points()
        for i in range(n_points):
            p_s = frame.positions[i]  # position in sensor frame
            range_val = frame.ranges[i] if frame.ranges is not None else np.linalg.norm(p_s)
            
            if range_val < min_range:
                continue
            
            # Unit vector in sensor frame
            unit_vector = p_s / np.linalg.norm(p_s)
            
            # Measured Doppler
            v_doppler_meas = frame.velocities[i]
            
            # Predict Doppler (predict_doppler_velocity expects Nx3 array)
            v_doppler_pred = predict_doppler_velocity(
                vel_world,
                omega_body,
                R_world_from_body,
                p_s.reshape(1, 3),  # Reshape to (1, 3)
                sensor_translation,
                sensor_rotation
            )[0]  # Extract scalar from 1D array
            
            # Residual
            r = v_doppler_meas - v_doppler_pred
            
            # Huber weight
            w = huber_weight(r, delta=huber_delta)
            
            residuals.append(r * np.sqrt(w))  # Apply sqrt for IRLS
            weights.append(w)
            times_list.append(t)
    
    return np.array(residuals), np.array(weights), np.array(times_list)


def compute_accelerometer_residuals_nonlinear(
    state: TrajectoryState,
    imu_data,
    g_world: np.ndarray = np.array([0, 0, -9.81])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute accelerometer residuals using raw IMU data.
    
    Raw IMU model: z_imu = R_bw @ (a_world - g) + b_a + noise
    Residual: r = z_imu - R_bw @ (a_world - g) - b_a
    
    Args:
        state: Current trajectory state
        imu_data: Raw IMU measurements (body frame, includes gravity)
        g_world: Gravity vector in world frame [0, 0, -9.81]
    
    Returns:
        residuals: (N*3,) flattened residuals
        times: (N,) measurement times
    """
    residuals = []
    times_list = []
    
    for imu_msg in imu_data:
        t = imu_msg.timestamp
        
        # Raw IMU accelerometer reading (body frame, includes gravity)
        z_acc = imu_msg.linear_acceleration
        
        # Predicted specific force
        acc_world = state.get_position(t, derivative=2)
        R_world_from_body = state.get_rotation(t)
        
        # Transform to body frame: R_bw @ (a_world - g)
        acc_body_pred = R_world_from_body.T @ (acc_world - g_world)
        
        # Residual (3D vector)
        r = z_acc - acc_body_pred - state.acc_bias
        
        residuals.append(r)
        times_list.append(t)
    
    residuals_array = np.array(residuals).flatten()  # (N*3,)
    times_array = np.array(times_list)
    
    return residuals_array, times_array


def compute_gyroscope_residuals_nonlinear(
    state: TrajectoryState,
    imu_data,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gyroscope residuals using raw IMU data.
    
    Raw gyro model: z_gyro = omega_body + b_g + noise
    Residual: r = z_gyro - omega(t) - b_g
    
    Args:
        state: Current trajectory state
        imu_data: Raw IMU measurements
    
    Returns:
        residuals: (N*3,) flattened residuals
        times: (N,) measurement times
    """
    residuals = []
    times_list = []
    
    for imu_msg in imu_data:
        t = imu_msg.timestamp
        
        # Raw gyroscope reading (body frame)
        z_gyro = imu_msg.angular_velocity
        
        # Predicted angular velocity from spline
        omega_pred = state.get_angular_velocity(t)
        
        # Residual (3D vector)
        r = z_gyro - omega_pred - state.gyr_bias
        
        residuals.append(r)
        times_list.append(t)
    
    residuals_array = np.array(residuals).flatten()  # (N*3,)
    times_array = np.array(times_list)
    
    return residuals_array, times_array


# ==================== Analytical Jacobian ====================

def compute_jacobian_analytical(
    state: TrajectoryState,
    radar_frames,
    imu_data,
    sensor_translation: np.ndarray,
    sensor_rotation: np.ndarray,
    lambda_accel: float,
    lambda_gyro: float,
    huber_delta: float,
    min_range: float = 0.2,
    boundary_vel_priors: list = None,
    boundary_pos_priors: list = None,
    lambda_boundary_vel: float = 0.0,
    lambda_boundary_pos: float = 0.0,
    huber_delta_accel: float = 0.0,
    boundary_ori_priors: list = None,
    boundary_accel_priors: list = None,
    boundary_gyro_priors: list = None,
    lambda_boundary_ori: float = 0.0,
    lambda_boundary_accel: float = 0.0,
    lambda_boundary_gyro: float = 0.0,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Compute Jacobian and residual vector analytically using SymForce-generated functions.
    
    Chain rule (all Jacobians from SymForce):
    - Position CPs affect v_world via N_i'(t) and a_world via N_i''(t)
    - Orientation CPs affect delta via M_j(t) and delta_dot via M_j'(t)
    - omega depends on BOTH delta and delta_dot:
        ∂omega/∂cp = ∂omega/∂delta * M_j(t) + ∂omega/∂delta_dot * M_j'(t)
    - Biases affect residuals directly
    
    Returns:
        J: Sparse Jacobian matrix (N_residuals, N_variables)
        r: Residual vector (N_residuals,)
    """
    n_pos = state.pos_bspline.n_points * 3
    n_ori = state.ori_bspline.n_points * 3
    n_total = state.get_state_size()
    
    # Pre-compute sensor rotation quaternion
    R_bs_quat = Rot3.from_rotation_matrix(sensor_rotation)
    
    # Collect Jacobian entries
    rows = []
    cols = []
    vals = []
    all_residuals = []
    row_idx = 0
    
    # ==================== Radar residuals ====================
    for frame in radar_frames:
        t = frame.timestamp
        t_rel_pos = t - state.pos_bspline.t_ref
        t_rel_ori = t - state.ori_bspline.t_ref
        
        # Get basis functions for position spline (velocity = 1st derivative)
        pos_vel_coeffs, pos_vel_indices = state.pos_bspline.get_basis_coefficients(t_rel_pos, derivative=1)
        
        # Get basis functions for orientation spline (value and 1st derivative)
        ori_val_coeffs, ori_val_indices = state.ori_bspline.get_basis_coefficients(t_rel_ori, derivative=0)
        ori_vel_coeffs, ori_vel_indices = state.ori_bspline.get_basis_coefficients(t_rel_ori, derivative=1)
        
        # Get current state values
        v_world = state.get_position(t, derivative=1)
        delta = state.ori_bspline(t_rel_ori, derivative=0)
        delta_dot = state.ori_bspline(t_rel_ori, derivative=1)
        omega_nominal = state.get_nominal_angular_velocity(t_rel_ori)
        
        # Compute omega and its Jacobians w.r.t. delta, delta_dot (SymForce)
        omega, J_omega_wrt_delta, J_omega_wrt_delta_dot = compute_omega_and_jacobians(
            omega_nominal, delta, delta_dot
        )
        
        # Get nominal rotation at this time (SLERP interpolated)
        R_nominal = state.get_nominal_rotation(t_rel_ori)
        R_nom_quat = Rot3.from_rotation_matrix(R_nominal)
        
        n_points = frame.num_points()
        for i in range(n_points):
            p_s = frame.positions[i]
            range_val = frame.ranges[i] if frame.ranges is not None else np.linalg.norm(p_s)
            if range_val < min_range:
                continue
            
            u_sensor = p_s / np.linalg.norm(p_s)
            v_meas = frame.velocities[i]
            
            # Call generated function
            res, J_v, J_delta_radar, J_omega_radar = radar_residual_with_jacobians(
                v_world, R_nom_quat, delta, omega,
                u_sensor, sensor_translation, R_bs_quat,
                v_meas, 1e-10
            )
            
            r = res[0]  # scalar residual
            w = huber_weight(r, delta=huber_delta)
            sqrt_w = np.sqrt(w)
            
            # Weighted residual
            all_residuals.append(r * sqrt_w)
            
            # Chain rule: ∂r/∂pos_cp_i = J_v * N_i'(t) (scaled by Huber weight)
            for ci, cp_idx in enumerate(pos_vel_indices):
                basis_val = pos_vel_coeffs[ci]
                for dim in range(3):
                    col = cp_idx * 3 + dim
                    val = sqrt_w * J_v[dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            # Full chain rule: ∂r/∂ori_cp_j = (∂r/∂delta + ∂r/∂omega @ ∂omega/∂delta) * M_j(t)
            #                               + (∂r/∂omega @ ∂omega/∂delta_dot) * M_j'(t)
            # J_omega_radar is (3,) row, J_omega_wrt_delta is (3,3)
            J_eff_val = J_delta_radar + J_omega_radar @ J_omega_wrt_delta      # (3,)
            J_eff_dot = J_omega_radar @ J_omega_wrt_delta_dot                  # (3,)
            
            # Value (delta) contribution
            for ci, cp_idx in enumerate(ori_val_indices):
                basis_val = ori_val_coeffs[ci]
                for dim in range(3):
                    col = n_pos + cp_idx * 3 + dim
                    val = sqrt_w * J_eff_val[dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            # Derivative (delta_dot) contribution
            for ci, cp_idx in enumerate(ori_vel_indices):
                basis_val = ori_vel_coeffs[ci]
                for dim in range(3):
                    col = n_pos + cp_idx * 3 + dim
                    val = sqrt_w * J_eff_dot[dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            row_idx += 1
    
    n_radar = row_idx
    
    # ==================== Accelerometer residuals ====================
    sqrt_lambda_accel = np.sqrt(lambda_accel)
    use_huber_accel = huber_delta_accel > 0
    
    for imu_msg in imu_data:
        t = imu_msg.timestamp
        
        t_rel_pos = t - state.pos_bspline.t_ref
        t_rel_ori = t - state.ori_bspline.t_ref
        
        # Basis functions
        pos_acc_coeffs, pos_acc_indices = state.pos_bspline.get_basis_coefficients(t_rel_pos, derivative=2)
        ori_val_coeffs, ori_val_indices = state.ori_bspline.get_basis_coefficients(t_rel_ori, derivative=0)
        
        # Current state
        a_world = state.get_position(t, derivative=2)
        delta = state.ori_bspline(t_rel_ori, derivative=0)
        
        # Nominal rotation (SLERP interpolated)
        R_nominal = state.get_nominal_rotation(t_rel_ori)
        R_nom_quat = Rot3.from_rotation_matrix(R_nominal)
        
        # Raw IMU accelerometer reading (body frame, includes gravity)
        z_acc = imu_msg.linear_acceleration
        g_world = np.array([0, 0, -9.81])
        
        # Call generated function
        res_3, J_a, J_delta_3x3, J_ba = accel_residual_with_jacobians(
            a_world, R_nom_quat, delta, g_world, z_acc, state.acc_bias, 1e-10
        )
        
        # Huber weight based on norm of the 3D residual (before lambda scaling)
        if use_huber_accel:
            accel_res_norm = np.linalg.norm(res_3)
            w_accel = huber_weight(accel_res_norm, delta=huber_delta_accel)
            sqrt_w_accel = np.sqrt(w_accel)
        else:
            sqrt_w_accel = 1.0
        
        scale = sqrt_lambda_accel * sqrt_w_accel
        
        # 3 rows for this measurement
        for k in range(3):
            all_residuals.append(scale * res_3[k])
            
            # ∂r_accel/∂pos_cp_i = J_a[k,:] * N_i''(t)
            for ci, cp_idx in enumerate(pos_acc_indices):
                basis_val = pos_acc_coeffs[ci]
                for dim in range(3):
                    col = cp_idx * 3 + dim
                    val = scale * J_a[k, dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            # ∂r_accel/∂ori_cp_j = J_delta_3x3[k,:] * M_j(t)
            for ci, cp_idx in enumerate(ori_val_indices):
                basis_val = ori_val_coeffs[ci]
                for dim in range(3):
                    col = n_pos + cp_idx * 3 + dim
                    val = scale * J_delta_3x3[k, dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            # ∂r_accel/∂b_a = J_ba[k,:] = -I[k,:]
            for dim in range(3):
                col = n_pos + n_ori + dim  # acc bias index
                val = scale * J_ba[k, dim]
                if abs(val) > 1e-15:
                    rows.append(row_idx)
                    cols.append(col)
                    vals.append(val)
            
            row_idx += 1
    
    n_accel_rows = row_idx - n_radar
    
    # ==================== Gyroscope residuals ====================
    sqrt_lambda_gyro = np.sqrt(lambda_gyro)
    
    for imu_msg in imu_data:
        t = imu_msg.timestamp
        
        t_rel_ori = t - state.ori_bspline.t_ref
        
        # Basis functions for orientation (value AND derivative)
        ori_val_coeffs, ori_val_indices = state.ori_bspline.get_basis_coefficients(t_rel_ori, derivative=0)
        ori_vel_coeffs, ori_vel_indices = state.ori_bspline.get_basis_coefficients(t_rel_ori, derivative=1)
        
        # Current state values
        delta = state.ori_bspline(t_rel_ori, derivative=0)
        delta_dot = state.ori_bspline(t_rel_ori, derivative=1)
        omega_nominal = state.get_nominal_angular_velocity(t_rel_ori)
        
        # Raw gyroscope reading (body frame)
        z_gyro = imu_msg.angular_velocity
        
        # Call SymForce-generated function (returns residual + Jacobians)
        res_gyro_3, J_delta_gyro, J_delta_dot_gyro, J_bg = gyro_residual_with_jacobians(
            omega_nominal, delta, delta_dot, z_gyro, state.gyr_bias, 1e-10
        )
        
        for k in range(3):
            all_residuals.append(sqrt_lambda_gyro * res_gyro_3[k])
            
            # ∂r_gyro/∂ori_cp_j = J_delta[k,:] * M_j(t) + J_delta_dot[k,:] * M_j'(t)
            # Delta contribution (from orientation value)
            for ci, cp_idx in enumerate(ori_val_indices):
                basis_val = ori_val_coeffs[ci]
                for dim in range(3):
                    col = n_pos + cp_idx * 3 + dim
                    val = sqrt_lambda_gyro * J_delta_gyro[k, dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            # Delta_dot contribution (from orientation derivative)
            for ci, cp_idx in enumerate(ori_vel_indices):
                basis_val = ori_vel_coeffs[ci]
                for dim in range(3):
                    col = n_pos + cp_idx * 3 + dim
                    val = sqrt_lambda_gyro * J_delta_dot_gyro[k, dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            # ∂r_gyro/∂b_g
            for dim in range(3):
                col = n_pos + n_ori + 3 + dim  # gyro bias index
                val = sqrt_lambda_gyro * J_bg[k, dim]
                if abs(val) > 1e-15:
                    rows.append(row_idx)
                    cols.append(col)
                    vals.append(val)
            
            row_idx += 1
    
    # ==================== Boundary velocity priors ====================
    n_boundary_vel_rows = 0
    if boundary_vel_priors and lambda_boundary_vel > 0:
        sqrt_lbv = np.sqrt(lambda_boundary_vel)
        for t_abs, v_target in boundary_vel_priors:
            t_rel_pos = t_abs - state.pos_bspline.t_ref
            # Get basis functions for velocity (1st derivative)
            pos_vel_coeffs, pos_vel_indices = state.pos_bspline.get_basis_coefficients(
                t_rel_pos, derivative=1)
            # Current velocity
            v_est = state.get_position(t_abs, derivative=1)
            r_bv = v_est - v_target
            
            for k in range(3):
                all_residuals.append(sqrt_lbv * r_bv[k])
                # Jacobian: ∂v_k/∂cp_i = N_i'(t) for dimension k
                for ci, cp_idx in enumerate(pos_vel_indices):
                    basis_val = pos_vel_coeffs[ci]
                    col = cp_idx * 3 + k
                    val = sqrt_lbv * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
                row_idx += 1
                n_boundary_vel_rows += 1
    
    # ==================== Boundary position priors ====================
    n_boundary_pos_rows = 0
    if boundary_pos_priors and lambda_boundary_pos > 0:
        sqrt_lbp = np.sqrt(lambda_boundary_pos)
        for t_abs, p_target in boundary_pos_priors:
            t_rel_pos = t_abs - state.pos_bspline.t_ref
            # Get basis functions for position (0th derivative)
            pos_val_coeffs, pos_val_indices = state.pos_bspline.get_basis_coefficients(
                t_rel_pos, derivative=0)
            # Current position
            p_est = state.get_position(t_abs, derivative=0)
            r_bp = p_est - p_target
            
            for k in range(3):
                all_residuals.append(sqrt_lbp * r_bp[k])
                # Jacobian: ∂p_k/∂cp_i = N_i(t) for dimension k
                for ci, cp_idx in enumerate(pos_val_indices):
                    basis_val = pos_val_coeffs[ci]
                    col = cp_idx * 3 + k
                    val = sqrt_lbp * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
                row_idx += 1
                n_boundary_pos_rows += 1
    
    # ==================== Boundary orientation priors (delta=0) ====================
    n_boundary_ori_rows = 0
    if boundary_ori_priors and lambda_boundary_ori > 0:
        sqrt_lbo = np.sqrt(lambda_boundary_ori)
        for t_abs in boundary_ori_priors:
            t_rel_ori = t_abs - state.ori_bspline.t_ref
            # Get basis functions for orientation (0th derivative)
            ori_val_coeffs, ori_val_indices = state.ori_bspline.get_basis_coefficients(
                t_rel_ori, derivative=0)
            # Residual: delta(t) should be zero (nominal = MoCap)
            delta_est = state.ori_bspline(t_rel_ori, derivative=0)
            
            for k in range(3):
                all_residuals.append(sqrt_lbo * delta_est[k])
                for ci, cp_idx in enumerate(ori_val_indices):
                    basis_val = ori_val_coeffs[ci]
                    col = n_pos + cp_idx * 3 + k
                    val = sqrt_lbo * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
                row_idx += 1
                n_boundary_ori_rows += 1
    
    # ==================== Boundary acceleration priors ====================
    n_boundary_accel_rows = 0
    if boundary_accel_priors and lambda_boundary_accel > 0:
        sqrt_lba = np.sqrt(lambda_boundary_accel)
        for t_abs, a_target in boundary_accel_priors:
            t_rel_pos = t_abs - state.pos_bspline.t_ref
            pos_acc_coeffs, pos_acc_indices = state.pos_bspline.get_basis_coefficients(
                t_rel_pos, derivative=2)
            a_est = state.get_position(t_abs, derivative=2)
            r_ba = a_est - a_target
            
            for k in range(3):
                all_residuals.append(sqrt_lba * r_ba[k])
                for ci, cp_idx in enumerate(pos_acc_indices):
                    basis_val = pos_acc_coeffs[ci]
                    col = cp_idx * 3 + k
                    val = sqrt_lba * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
                row_idx += 1
                n_boundary_accel_rows += 1
    
    # ==================== Boundary angular velocity priors (delta_dot=0) ====================
    n_boundary_gyro_rows = 0
    if boundary_gyro_priors and lambda_boundary_gyro > 0:
        sqrt_lbg = np.sqrt(lambda_boundary_gyro)
        for t_abs in boundary_gyro_priors:
            t_rel_ori = t_abs - state.ori_bspline.t_ref
            ori_vel_coeffs, ori_vel_indices = state.ori_bspline.get_basis_coefficients(
                t_rel_ori, derivative=1)
            # Residual: delta_dot(t) should be zero (omega = omega_nominal)
            delta_dot_est = state.ori_bspline(t_rel_ori, derivative=1)
            
            for k in range(3):
                all_residuals.append(sqrt_lbg * delta_dot_est[k])
                for ci, cp_idx in enumerate(ori_vel_indices):
                    basis_val = ori_vel_coeffs[ci]
                    col = n_pos + cp_idx * 3 + k
                    val = sqrt_lbg * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
                row_idx += 1
                n_boundary_gyro_rows += 1
    
    # Build sparse Jacobian
    n_residuals = row_idx
    J = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_residuals, n_total)
    )
    r = np.array(all_residuals)
    
    # Cost decomposition: (n_radar, n_accel_rows) stored as attributes
    J.n_radar = n_radar
    J.n_accel = n_accel_rows
    J.n_boundary_vel = n_boundary_vel_rows
    J.n_boundary_pos = n_boundary_pos_rows
    J.n_boundary_ori = n_boundary_ori_rows
    J.n_boundary_accel = n_boundary_accel_rows
    J.n_boundary_gyro = n_boundary_gyro_rows
    
    return J, r


def compute_orientation_rmse(state: TrajectoryState, mocap_times_abs, mocap_rotations):
    """Compute orientation RMSE against MoCap ground truth."""
    errors = []
    for i, t in enumerate(mocap_times_abs):
        try:
            R_est = state.get_rotation(t)
            R_gt = mocap_rotations[i]
            R_err = R_gt.T @ R_est
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            errors.append(np.degrees(angle))
        except Exception:
            pass
    return np.sqrt(np.mean(np.array(errors)**2)) if errors else float('inf')


# ==================== Levenberg-Marquardt Solver ==

def solve_trajectory_nonlinear(
    initial_state: TrajectoryState,
    radar_frames,
    imu_data,
    sensor_translation: np.ndarray,
    sensor_rotation: np.ndarray,
    lambda_accel: float = 1.0,
    lambda_gyro: float = 1.0,
    lambda_snap_pos: float = 0.01,
    lambda_snap_ori: float = 0.01,
    huber_delta: float = 0.5,
    huber_delta_accel: float = 0.0,
    max_iterations: int = 20,
    lock_biases: bool = False,
    use_jacobi_precond: bool = False,
    verbose: bool = True,
    mocap_times_abs: np.ndarray = None,
    mocap_rotations: np.ndarray = None,
    boundary_vel_priors: list = None,
    boundary_pos_priors: list = None,
    lambda_boundary_vel: float = 0.0,
    lambda_boundary_pos: float = 0.0,
    boundary_ori_priors: list = None,
    boundary_accel_priors: list = None,
    boundary_gyro_priors: list = None,
    lambda_boundary_ori: float = 0.0,
    lambda_boundary_accel: float = 0.0,
    lambda_boundary_gyro: float = 0.0,
) -> TrajectoryState:
    """
    Solve for optimal trajectory using Levenberg-Marquardt.
    
    Uses analytical Jacobians derived by SymForce for ~50x speedup.
    The gyroscope model uses the exact SO(3) right Jacobian J_r(delta)
    via SymForce codegen, so re-linearization after every accepted step
    is safe and keeps delta small for better numerical conditioning.
    
    Minimizes:
    E = sum(huber(r_radar)) + lambda_accel * huber_accel(||r_accel||)
        + lambda_gyro * ||r_gyro||^2
        + lambda_snap_pos * ||snap_pos||^2 + lambda_snap_ori * ||snap_ori||^2
        + boundary_priors (vel, pos, ori, accel, gyro)
    """
    if verbose:
        print(f"\n{'Levenberg-Marquardt Optimization':#^80}")
        print(f"Max iterations: {max_iterations} (with re-linearization after each accepted step)")
        print(f"Huber delta (radar): {huber_delta}")
        print(f"Huber delta (accel): {huber_delta_accel if huber_delta_accel > 0 else 'OFF (L2)'}")
        print(f"Lambda accel: {lambda_accel}")
        print(f"Lambda gyro: {lambda_gyro}")
        print(f"Lambda snap pos: {lambda_snap_pos}")
        print(f"Lambda snap ori: {lambda_snap_ori}")
        if lock_biases:
            print(f"*** BIASES LOCKED TO ZERO ***")
        if use_jacobi_precond:
            print(f"*** JACOBI PRECONDITIONING ENABLED ***")
        if boundary_vel_priors:
            print(f"Boundary velocity priors: {len(boundary_vel_priors)} points, lambda={lambda_boundary_vel}")
        if boundary_pos_priors:
            print(f"Boundary position priors: {len(boundary_pos_priors)} points, lambda={lambda_boundary_pos}")
        if boundary_ori_priors:
            print(f"Boundary orientation priors: {len(boundary_ori_priors)} points, lambda={lambda_boundary_ori}")
        if boundary_accel_priors:
            print(f"Boundary acceleration priors: {len(boundary_accel_priors)} points, lambda={lambda_boundary_accel}")
        if boundary_gyro_priors:
            print(f"Boundary ang. vel. priors: {len(boundary_gyro_priors)} points, lambda={lambda_boundary_gyro}")
    
    state = initial_state
    
    # Build regularization matrices once
    if verbose:
        print("\nBuilding regularization matrices...")
    R_snap_pos = build_minimum_snap_regularization(state.pos_bspline, n_samples=100)
    R_snap_ori = build_minimum_snap_regularization(state.ori_bspline, n_samples=50)
    
    # Separate regularization matrices for position and orientation blocks
    n_pos = state.pos_bspline.n_points * 3
    n_ori = state.ori_bspline.n_points * 3
    n_total = state.get_state_size()
    
    # Create block-diagonal regularization matrix
    R_snap = sparse.lil_matrix((n_total, n_total))
    R_snap[:n_pos, :n_pos] = lambda_snap_pos * (R_snap_pos.T @ R_snap_pos)
    R_snap[n_pos:n_pos+n_ori, n_pos:n_pos+n_ori] = lambda_snap_ori * (R_snap_ori.T @ R_snap_ori)
    R_snap = R_snap.tocsr()
    
    # Helper: build Jacobian with all params
    def _build_jacobian(st):
        return compute_jacobian_analytical(
            st, radar_frames, imu_data,
            sensor_translation, sensor_rotation,
            lambda_accel, lambda_gyro, huber_delta,
            boundary_vel_priors=boundary_vel_priors,
            boundary_pos_priors=boundary_pos_priors,
            lambda_boundary_vel=lambda_boundary_vel,
            lambda_boundary_pos=lambda_boundary_pos,
            huber_delta_accel=huber_delta_accel,
            boundary_ori_priors=boundary_ori_priors,
            boundary_accel_priors=boundary_accel_priors,
            boundary_gyro_priors=boundary_gyro_priors,
            lambda_boundary_ori=lambda_boundary_ori,
            lambda_boundary_accel=lambda_boundary_accel,
            lambda_boundary_gyro=lambda_boundary_gyro,
        )
    
    lambda_lm = 1e-3
    prev_cost = None
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'Iteration ' + str(iteration + 1):-^80}")
            # Track orientation RMSE per iteration
            if mocap_times_abs is not None and mocap_rotations is not None:
                ori_rmse = compute_orientation_rmse(state, mocap_times_abs, mocap_rotations)
                delta_norms = np.linalg.norm(state.ori_bspline.control_points, axis=1)
                print(f"Orientation RMSE: {ori_rmse:.1f} deg | delta max: {np.degrees(delta_norms.max()):.2f}° mean: {np.degrees(delta_norms.mean()):.2f}°")
                print(f"Acc bias: [{state.acc_bias[0]:.3f}, {state.acc_bias[1]:.3f}, {state.acc_bias[2]:.3f}]"
                      f"  Gyr bias: [{state.gyr_bias[0]:.3f}, {state.gyr_bias[1]:.3f}, {state.gyr_bias[2]:.3f}]")
        
        t_start = time.time()
        
        # Build Jacobian + residual vector analytically (includes all costs)
        if verbose:
            print("Computing analytical Jacobian...")
        
        J, r_total = _build_jacobian(state)
        
        cost_total = np.sum(r_total**2)
        
        if verbose:
            # Cost decomposition: residuals ordered as radar|accel|gyro|bnd_vel|bnd_pos|bnd_ori|bnd_accel|bnd_gyro
            n_r = getattr(J, 'n_radar', 0)
            n_a = getattr(J, 'n_accel', 0)
            n_bv = getattr(J, 'n_boundary_vel', 0)
            n_bp = getattr(J, 'n_boundary_pos', 0)
            n_bo = getattr(J, 'n_boundary_ori', 0)
            n_bac = getattr(J, 'n_boundary_accel', 0)
            n_bg = getattr(J, 'n_boundary_gyro', 0)
            n_g = len(r_total) - n_r - n_a - n_bv - n_bp - n_bo - n_bac - n_bg
            idx = 0
            cost_radar = np.sum(r_total[idx:idx+n_r]**2); idx += n_r
            cost_accel = np.sum(r_total[idx:idx+n_a]**2); idx += n_a
            cost_gyro = np.sum(r_total[idx:idx+n_g]**2); idx += n_g
            cost_bv = np.sum(r_total[idx:idx+n_bv]**2); idx += n_bv
            cost_bp = np.sum(r_total[idx:idx+n_bp]**2); idx += n_bp
            cost_bo = np.sum(r_total[idx:idx+n_bo]**2); idx += n_bo
            cost_bac = np.sum(r_total[idx:idx+n_bac]**2); idx += n_bac
            cost_bg = np.sum(r_total[idx:idx+n_bg]**2); idx += n_bg
            bnd_parts = []
            if n_bv > 0: bnd_parts.append(f"bnd_vel={cost_bv:.1f}")
            if n_bp > 0: bnd_parts.append(f"bnd_pos={cost_bp:.1f}")
            if n_bo > 0: bnd_parts.append(f"bnd_ori={cost_bo:.1f}")
            if n_bac > 0: bnd_parts.append(f"bnd_acc={cost_bac:.1f}")
            if n_bg > 0: bnd_parts.append(f"bnd_gyr={cost_bg:.1f}")
            bnd_str = (" " + " ".join(bnd_parts)) if bnd_parts else ""
            print(f"Cost: total={cost_total:.2f} | radar={cost_radar:.1f} accel={cost_accel:.1f} gyro={cost_gyro:.1f}{bnd_str}")
            print(f"Jacobian: {J.shape}, nnz={J.nnz}, sparsity={100*(1-J.nnz/(J.shape[0]*J.shape[1])):.2f}%")
        
        # Build normal equations with LM damping
        H = J.T @ J + lambda_lm * sparse.eye(n_total) + R_snap
        b = J.T @ r_total
        
        # Solve (optionally with Jacobi preconditioning)
        try:
            if use_jacobi_precond:
                # Jacobi preconditioning: normalize H to remove scale mismatch
                diag_H = H.diagonal().copy()
                diag_H[diag_H < 1e-10] = 1.0
                M_inv_sqrt = sparse.diags(1.0 / np.sqrt(diag_H))
                H_scaled = M_inv_sqrt @ H @ M_inv_sqrt
                b_scaled = M_inv_sqrt @ b
                delta_x_scaled = spsolve(H_scaled, -b_scaled)
                delta_x = M_inv_sqrt @ delta_x_scaled  # Unscale
            else:
                delta_x = spsolve(H, -b)
        except Exception:
            if verbose:
                print("[WARN] Solver failed, increasing damping...")
            lambda_lm *= 10
            continue
        
        # Try update
        x_current = state.to_vector()
        # Zero out bias updates when locked
        if lock_biases:
            delta_x[-(6):] = 0.0
        x_new = x_current + delta_x
        state.from_vector(x_new)
        
        # Evaluate new cost
        _, r_new = _build_jacobian(state)
        new_cost = np.sum(r_new**2)
        
        # LM acceptance: accept only if cost decreased
        if new_cost < cost_total:
            lambda_lm = max(1e-15, lambda_lm * 0.1)  # Aggressive: snap to Gauss-Newton mode
            prev_cost = new_cost
            if verbose:
                delta_norms = np.linalg.norm(state.ori_bspline.control_points, axis=1)
                print(f"  Accepted: cost {cost_total:.1f} -> {new_cost:.1f} (max |delta|={np.degrees(delta_norms.max()):.1f}°)")
            # Re-linearize: absorb delta into nominal orientation
            max_delta_deg = np.degrees(
                np.linalg.norm(state.ori_bspline.control_points, axis=1).max())
            state.relinearize()
            if verbose:
                print(f"  Re-linearized: absorbed max |delta|={max_delta_deg:.1f}° into nominal")
        else:
            # Reject and increase damping
            state.from_vector(x_current)
            lambda_lm *= 10.0  # Strong punishment for bad step
            if verbose:
                print(f"  [WARN] Cost increased ({new_cost:.2f} > {cost_total:.2f}), rejecting step")
        
        # Convergence check
        delta_norm = np.linalg.norm(delta_x)
        if verbose:
            print(f"Update norm: {delta_norm:.6f}")
            print(f"LM damping: {lambda_lm:.2e}")
            print(f"Iteration time: {time.time() - t_start:.3f}s")
        
        if delta_norm < 1e-4:
            if verbose:
                print("\n[OK] Converged!")
            break
    
    if verbose:
        print(f"\n{'Optimization Complete':#^80}")
    
    return state


# ==================== Bag Catalogue ====================
BAGS = {
    "original":     "rosbags/2025-12-17-16-02-22.bag",
    "circle":       "rosbags/circle_2025-12-17-17-21-37.bag",
    "circle_fast":  "rosbags/circle_fast_2025-12-17-17-25-34.bag",
    "circle_fwd":   "rosbags/circle_forward_2025-12-17-17-37-38.bag",
    "loopings":     "rosbags/circle_fast_forward_2025-12-17-17-39-49.bag",
    "backflips":    "rosbags/backflips_2025-12-17-17-41-24.bag",
}

# Bags where the agiros body frame is rotated 180 deg in yaw
FLIPPED_BAGS = {"circle_fwd", "loopings"}

# ==================== Main Validation ====================

def main():
    import time
    start_time = time.time()
    from datetime import datetime
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("PHASE 3: NONLINEAR SOLVER VALIDATION")
    print("Full Radar-Inertial Odometry Estimation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # ==================== Configuration ====================
    bag_key = sys.argv[1] if len(sys.argv) > 1 else "circle_fwd"
    if bag_key in BAGS:
        BAG_PATH = BAGS[bag_key]
    else:
        BAG_PATH = bag_key

    START_TIME_OFFSET = 30.0   # Skip initial hover
    DURATION = 5.0          # only timeframe of the loopings
        
    # Sensor extrinsics (from Phase 1 calibration)
    ROTATION_EULER_DEG = np.array([0.0, 30.0, 0.0])  # roll, pitch, yaw — +30° pitch = downlooking

    # Body frame flip for certain trajectory profiles
    FLIP_BODY_FRAME = bag_key in FLIPPED_BAGS
    if "--flip" in sys.argv:
        FLIP_BODY_FRAME = True
    if "--no-flip" in sys.argv:
        FLIP_BODY_FRAME = False

    R_base = rotation_matrix_from_euler(
        np.radians(ROTATION_EULER_DEG[0]),
        np.radians(ROTATION_EULER_DEG[1]),
        np.radians(ROTATION_EULER_DEG[2]),
    )
    if FLIP_BODY_FRAME:
        TRANSLATION = np.array([-0.07, 0.0, 0.0])
        R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)
        SENSOR_ROTATION = R_yaw_flip @ R_base
        print(f"  Body frame FLIPPED (R_z(180 deg) applied) for bag '{bag_key}'")
    else:
        TRANSLATION = np.array([0.07, 0.0, 0.0])
        SENSOR_ROTATION = R_base
    
    BSPLINE_DEGREE = 7  # Quintic for continuous snap
    DT_POS = 0.02   # Fixed knot spacing for position spline (seconds)
    DT_ORI = 0.02   # Fixed knot spacing for orientation spline (seconds)
    
    # Regularization weights
    LAMBDA_ACCEL = 0.01      # Accelerometer weight: gravity direction constrains orientation
    LAMBDA_GYRO = 5.0       # Gyroscope weight: omega_nominal now included
    LAMBDA_SNAP_POS = 0.0 # Position smoothness
    LAMBDA_SNAP_ORI = 0.0   # Orientation smoothness
    
    HUBER_DELTA = 0.5  # meters/second (Huber threshold for radar)
    HUBER_DELTA_ACCEL = 2.0  # m/s² (Huber threshold for accelerometer — clips spikes linearly)
    MIN_RANGE = 0.2
    MAX_ITERATIONS = 20  # LM iterations (re-linearization after each accepted step)
    USE_PHASE2_INIT = False  # Initialize position from Phase 2 linear solver
    LOCK_BIASES = True  # Lock biases to zero — force solver to fix orientation instead
    USE_JACOBI_PRECOND = '--precond' in sys.argv  # Toggle Jacobi preconditioning
    
    # Boundary priors: pin spline state at START to MoCap ground truth (no end priors)
    LAMBDA_BOUNDARY_VEL = 100.0   # Weight for boundary velocity priors
    LAMBDA_BOUNDARY_POS = 100.0   # Weight for boundary position priors
    LAMBDA_BOUNDARY_ORI = 100.0   # Weight for boundary orientation priors (delta=0)
    LAMBDA_BOUNDARY_ACCEL = 0.001  # Weight for boundary acceleration priors
    LAMBDA_BOUNDARY_GYRO = 10.0  # Weight for boundary angular velocity priors (delta_dot=0)
    BOUNDARY_WINDOW = 0.3         # Seconds near start boundary to apply priors
    
    print(f"\n{'Configuration':-^80}")
    print(f"Bag: {bag_key} -> {BAG_PATH}")
    print(f"Flip body frame: {FLIP_BODY_FRAME}")
    print(f"Time window: {START_TIME_OFFSET:.1f}s + {DURATION:.1f}s")
    print(f"B-spline degree: {BSPLINE_DEGREE}")
    print(f"Lambda accel: {LAMBDA_ACCEL}")
    print(f"Lambda gyro: {LAMBDA_GYRO}")
    print(f"Lambda snap (pos/ori): {LAMBDA_SNAP_POS}/{LAMBDA_SNAP_ORI}")
    print(f"Huber delta (radar): {HUBER_DELTA} m/s")
    print(f"Huber delta (accel): {HUBER_DELTA_ACCEL} m/s²")
    print(f"Max iterations: {MAX_ITERATIONS} (re-linearize after each accepted step)")
    print(f"Use Phase 2 init: {USE_PHASE2_INIT}")
    print(f"Lock biases: {LOCK_BIASES}")
    print(f"Jacobi preconditioning: {USE_JACOBI_PRECOND}")
    print(f"Boundary priors (START only): window={BOUNDARY_WINDOW}s")
    print(f"  λ_bnd: vel={LAMBDA_BOUNDARY_VEL} pos={LAMBDA_BOUNDARY_POS} ori={LAMBDA_BOUNDARY_ORI} acc={LAMBDA_BOUNDARY_ACCEL} gyr={LAMBDA_BOUNDARY_GYRO}")
    
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
    print(f"  IMU samples: {len(imu_data)}")
    
    if len(agiros_states) == 0 or len(radar_frames) == 0 or len(imu_data) == 0:
        print("ERROR: Insufficient data!")
        return
    
    # Downsample IMU data for computational efficiency (~200 Hz is sufficient)
    IMU_DOWNSAMPLE = max(1, len(imu_data) // (int(DURATION * 200)))
    imu_data = imu_data[::IMU_DOWNSAMPLE]
    print(f"  IMU after downsampling (1/{IMU_DOWNSAMPLE}): {len(imu_data)}")
    
    # IMU diagnostics: verify frame alignment
    imu_accels = np.array([d.linear_acceleration for d in imu_data])
    imu_gyros = np.array([d.angular_velocity for d in imu_data])
    print(f"\n  IMU accel mean: [{imu_accels[:,0].mean():.2f}, {imu_accels[:,1].mean():.2f}, {imu_accels[:,2].mean():.2f}] m/s²")
    print(f"  IMU accel std:  [{imu_accels[:,0].std():.2f}, {imu_accels[:,1].std():.2f}, {imu_accels[:,2].std():.2f}] m/s²")
    print(f"  IMU accel norm: {np.linalg.norm(imu_accels.mean(axis=0)):.2f} m/s² (expect ~9.81)")
    print(f"  IMU gyro mean:  [{imu_gyros[:,0].mean():.3f}, {imu_gyros[:,1].mean():.3f}, {imu_gyros[:,2].mean():.3f}] rad/s")
    
    # Cross-validate IMU vs MoCap at closest timestamps
    print(f"\n  {'IMU-MoCap Cross-Validation':-^60}")
    g_world = np.array([0, 0, -9.81])
    n_diag = min(5, len(imu_data))
    for idx in range(0, len(imu_data), max(1, len(imu_data) // n_diag)):
        imu_msg = imu_data[idx]
        # Find closest agiros state
        i_closest = np.argmin(np.abs(np.array([s.timestamp for s in agiros_states]) - imu_msg.timestamp))
        mocap = agiros_states[i_closest]
        R_wb = quat_to_rotation_matrix(mocap.orientation)
        R_bw = R_wb.T
        
        # Predicted IMU: z_imu_pred = R_bw @ (a_mocap_world - g) where a_mocap_world from derivative
        # Approximate world accel from MoCap velocity (finite diff)
        dt_mocap = 1.0/300  # ~300 Hz
        if i_closest > 0 and i_closest < len(agiros_states)-1:
            a_world_est = (agiros_states[i_closest+1].velocity - agiros_states[i_closest-1].velocity) / (
                agiros_states[i_closest+1].timestamp - agiros_states[i_closest-1].timestamp)
        else:
            a_world_est = np.zeros(3)
        
        z_imu_pred = R_bw @ (a_world_est - g_world)
        z_imu_actual = imu_msg.linear_acceleration
        z_gyro_pred = mocap.angular_velocity
        z_gyro_actual = imu_msg.angular_velocity
        
        print(f"  t={imu_msg.timestamp - imu_data[0].timestamp:.2f}s:")
        print(f"    Accel predicted: [{z_imu_pred[0]:7.2f}, {z_imu_pred[1]:7.2f}, {z_imu_pred[2]:7.2f}]")
        print(f"    Accel actual:    [{z_imu_actual[0]:7.2f}, {z_imu_actual[1]:7.2f}, {z_imu_actual[2]:7.2f}]")
        print(f"    Gyro predicted:  [{z_gyro_pred[0]:7.3f}, {z_gyro_pred[1]:7.3f}, {z_gyro_pred[2]:7.3f}]")
        print(f"    Gyro actual:     [{z_gyro_actual[0]:7.3f}, {z_gyro_actual[1]:7.3f}, {z_gyro_actual[2]:7.3f}]")
    
    # ==================== Initialize State ====================
    print(f"\n{'Initializing State':-^80}")
    
    # Work with relative times
    mocap_times_abs = np.array([s.timestamp for s in agiros_states])
    t_ref = mocap_times_abs[0]
    mocap_times_rel = mocap_times_abs - t_ref
    
    print(f"Reference time: {t_ref:.2f} (absolute)")
    print(f"Relative time range: [{mocap_times_rel[0]:.2f}, {mocap_times_rel[-1]:.2f}]")
    
    # Load orientation data (needed for Phase 2 and Phase 3)
    mocap_orientations = np.array([s.orientation for s in agiros_states])  # quaternions
    mocap_rotations = np.array([quat_to_rotation_matrix(q) for q in mocap_orientations])
    mocap_positions = np.array([s.position for s in agiros_states])
    
    # Prepare sensor extrinsics for Phase 2
    sensor_translation = TRANSLATION
    sensor_rotation = SENSOR_ROTATION
    
    # Create position B-spline with fixed knot spacing (independent of window duration)
    BOUNDARY_ORDER = 2
    n_interior_pos = int(np.ceil((mocap_times_rel[-1] - mocap_times_rel[0]) / DT_POS)) + 1
    n_pos_points = n_interior_pos + 2 * BOUNDARY_ORDER
    pos_bspline = UniformBSpline(np.zeros((n_pos_points, 3)), BSPLINE_DEGREE, DT_POS)
    pos_bspline.t_ref = t_ref
    
    if USE_PHASE2_INIT:
        print("\n" + "="*80)
        print("RUNNING PHASE 2 FOR INITIALIZATION")
        print("="*80)
        
        # Import Phase 2 solver
        from validate_linear_solver import (
            build_radar_jacobian,
            build_accelerometer_jacobian,
            solve_trajectory_linear
        )
        
        # Initialize from MoCap interpolation
        pos_interp = interp1d(mocap_times_rel, mocap_positions, axis=0, kind='cubic',
                              fill_value='extrapolate')
        init_times = np.linspace(pos_bspline.t_start, pos_bspline.t_end, n_pos_points)
        pos_bspline.control_points = pos_interp(init_times)
        
        # Build Jacobians for Phase 2
        print("\nBuilding Phase 2 Jacobians...")
        J_radar, r_radar, n_radar = build_radar_jacobian(
            pos_bspline, radar_frames, agiros_states, t_ref,
            sensor_rotation, sensor_translation, time_offset=0.0
        )
        
        J_accel, r_accel, n_accel = build_accelerometer_jacobian(
            pos_bspline, imu_data, agiros_states, t_ref, g_world=np.array([0, 0, -9.81])
        )
        
        # Build velocity boundary priors for Phase 2 (use B-spline domain, not MoCap)
        mocap_vel_interp_p2 = interp1d(mocap_times_rel,
                                        np.array([s.velocity for s in agiros_states]),
                                        axis=0, kind='linear', fill_value='extrapolate')
        phase2_vel_priors = []
        t_ws = max(pos_bspline.t_start, mocap_times_rel[0])   # Clip to data range
        n_bnd = max(1, int(BOUNDARY_WINDOW * 50))
        for t_r in np.linspace(t_ws, t_ws + BOUNDARY_WINDOW, n_bnd):
            phase2_vel_priors.append((t_r, mocap_vel_interp_p2(t_r)))
        
        # Solve Phase 2
        print("\nSolving Phase 2 (linear)...")
        x_opt = solve_trajectory_linear(
            pos_bspline, J_radar, r_radar, J_accel, r_accel,
            lambda_accel=0.01, lambda_snap=0.1, lambda_position=0.05,
            velocity_priors=phase2_vel_priors,
            lambda_velocity=LAMBDA_BOUNDARY_VEL,
            verbose=True
        )
        
        # Update control points with Phase 2 result
        pos_bspline.control_points = x_opt.reshape(-1, 3)
        
        # Diagnostic: check velocity at boundaries after Phase 2
        v_start = pos_bspline(pos_bspline.t_start, derivative=1)
        v_end = pos_bspline(pos_bspline.t_end, derivative=1)
        v_gt_start = mocap_vel_interp_p2(mocap_times_rel[0])
        v_gt_end = mocap_vel_interp_p2(mocap_times_rel[-1])
        print(f"\n  Phase 2 velocity at t_start: [{v_start[0]:.2f}, {v_start[1]:.2f}, {v_start[2]:.2f}] (|v|={np.linalg.norm(v_start):.2f})")
        print(f"  MoCap velocity at t_start:   [{v_gt_start[0]:.2f}, {v_gt_start[1]:.2f}, {v_gt_start[2]:.2f}] (|v|={np.linalg.norm(v_gt_start):.2f})")
        print(f"  Phase 2 velocity at t_end:   [{v_end[0]:.2f}, {v_end[1]:.2f}, {v_end[2]:.2f}] (|v|={np.linalg.norm(v_end):.2f})")
        print(f"  MoCap velocity at t_end:     [{v_gt_end[0]:.2f}, {v_gt_end[1]:.2f}, {v_gt_end[2]:.2f}] (|v|={np.linalg.norm(v_gt_end):.2f})")
        
        print("\n[OK] Phase 2 initialization complete!")
        print("="*80 + "\n")
    else:
        # Simple MoCap initialization
        pos_interp = interp1d(mocap_times_rel, mocap_positions, axis=0, kind='cubic',
                              fill_value='extrapolate')
        init_times = np.linspace(pos_bspline.t_start, pos_bspline.t_end, n_pos_points)
        pos_bspline.control_points = pos_interp(init_times)
    
    print(f"Position spline: {n_pos_points} control points, dt={pos_bspline.dt:.4f}s")
    
    # Create orientation B-spline initialized from MoCap
    # Use tangent space parameterization around nominal rotations
    
    # Create orientation spline with fixed knot spacing (independent of window duration)
    ori_degree = min(3, BSPLINE_DEGREE)  # Cubic for orientation
    n_interior_ori = int(np.ceil(DURATION / DT_ORI)) + 1
    n_ori_points = max(ori_degree + 2, n_interior_ori + 2 * BOUNDARY_ORDER)
    
    # Initialize with dummy control points (will be overwritten)
    ori_bspline = UniformBSpline(np.zeros((n_ori_points, 3)), ori_degree, DT_ORI)
    ori_bspline.t_ref = t_ref
    
    # Initialize with zero perturbations (identity in tangent space)
    ori_bspline.control_points = np.zeros((n_ori_points, 3))
    
    # Sample nominal rotations at control point times
    rot_interp_func = lambda t: Rotation.from_quat(
        interp1d(mocap_times_rel, mocap_orientations, axis=0,
                 kind='linear', fill_value='extrapolate')(t)
    )
    nominal_times = np.linspace(ori_bspline.t_start, ori_bspline.t_end, n_ori_points)
    nominal_rotations = np.array([
        rot_interp_func(t).as_matrix() for t in nominal_times
    ])
    
    # Create dense SLERP interpolation from MoCap (much better than nearest-neighbor)
    mocap_rots_scipy = Rotation.from_quat(mocap_orientations)  # [qx, qy, qz, qw]
    mocap_slerp = Slerp(mocap_times_rel, mocap_rots_scipy)
    
    # Create angular velocity interpolation from MoCap
    mocap_angular_velocities = np.array([s.angular_velocity for s in agiros_states])
    mocap_omega_interp = interp1d(mocap_times_rel, mocap_angular_velocities, axis=0,
                                   kind='linear', fill_value='extrapolate')
    
    print(f"Orientation spline: {n_ori_points} control points, dt={ori_bspline.dt:.4f}s")
    print(f"  SLERP from {len(mocap_times_rel)} MoCap samples ({1.0/(mocap_times_rel[1]-mocap_times_rel[0]):.0f} Hz)")
    
    # Pre-estimate biases from initial state (MoCap orientation + Phase 2 position)
    # b_a_init = mean(z_imu - R_mocap^T @ (p''(t) - g))
    # b_g_init = mean(z_gyro - omega_mocap)
    g_world = np.array([0, 0, -9.81])
    accel_residuals_init = []
    gyro_residuals_init = []
    for imu_msg in imu_data:
        t = imu_msg.timestamp
        t_rel = t - t_ref
        # Skip if outside B-spline support
        if t_rel < pos_bspline.t_start or t_rel > pos_bspline.t_end:
            continue
        try:
            a_world = pos_bspline(t_rel, derivative=2)
            R_mocap = mocap_slerp(np.clip(t_rel, mocap_times_rel[0], mocap_times_rel[-1])).as_matrix()
            pred_imu = R_mocap.T @ (a_world - g_world)
            accel_residuals_init.append(imu_msg.linear_acceleration - pred_imu)
            
            omega_mocap = mocap_omega_interp(np.clip(t_rel, mocap_times_rel[0], mocap_times_rel[-1]))
            gyro_residuals_init.append(imu_msg.angular_velocity - omega_mocap)
        except Exception:
            pass
    
    if LOCK_BIASES:
        acc_bias = np.zeros(3)
        gyr_bias = np.zeros(3)
        print(f"\n  Biases LOCKED to zero (forcing orientation correction)")
    else:
        acc_bias = np.mean(accel_residuals_init, axis=0) if accel_residuals_init else np.zeros(3)
        gyr_bias = np.mean(gyro_residuals_init, axis=0) if gyro_residuals_init else np.zeros(3)
    print(f"\n  Pre-estimated biases:")
    print(f"    Acc bias init: [{acc_bias[0]:.3f}, {acc_bias[1]:.3f}, {acc_bias[2]:.3f}] m/s² (norm={np.linalg.norm(acc_bias):.3f})")
    print(f"    Gyr bias init: [{gyr_bias[0]:.4f}, {gyr_bias[1]:.4f}, {gyr_bias[2]:.4f}] rad/s")
    
    # Create initial state
    initial_state = TrajectoryState(
        pos_bspline=pos_bspline,
        ori_bspline=ori_bspline,
        nominal_rotations=nominal_rotations,
        acc_bias=acc_bias,
        gyr_bias=gyr_bias,
        mocap_slerp=mocap_slerp,
        mocap_omega_interp=mocap_omega_interp
    )
    
    print(f"Total state variables: {initial_state.get_state_size()}")
    print(f"  Position: {n_pos_points * 3}")
    print(f"  Orientation: {n_ori_points * 3}")
    print(f"  Biases: 6")
    
    # ==================== Optimize ====================
    # sensor_rotation was already correctly computed above with np.radians()
    
    # Build boundary priors from MoCap ground truth
    # IMPORTANT: Use B-spline domain boundaries, NOT MoCap time boundaries.
    # The B-spline valid domain is [t_start, t_end] which is a subset of the
    # MoCap time range due to boundary padding for degree-5 B-splines.
    mocap_vel_interp = interp1d(mocap_times_rel,
                                 np.array([s.velocity for s in agiros_states]),
                                 axis=0, kind='linear', fill_value='extrapolate')
    
    boundary_vel_priors = []
    boundary_pos_priors = []
    boundary_ori_priors = []    # list of t_abs (delta=0 target)
    boundary_accel_priors = []  # list of (t_abs, a_target)
    boundary_gyro_priors = []   # list of t_abs (delta_dot=0 target)
    # Clip to intersection of B-spline domain and MoCap data range
    t_spline_start = max(pos_bspline.t_start, mocap_times_rel[0])
    t_spline_end = min(pos_bspline.t_end, mocap_times_rel[-1])
    
    print(f"\n  B-spline valid domain: [{t_spline_start:.4f}, {t_spline_end:.4f}]")
    print(f"  MoCap time range:     [{mocap_times_rel[0]:.4f}, {mocap_times_rel[-1]:.4f}]")
    
    # MoCap acceleration interpolation (for boundary accel prior)
    mocap_accel_interp = interp1d(mocap_times_rel,
                                   np.gradient(np.array([s.velocity for s in agiros_states]),
                                               mocap_times_rel, axis=0),
                                   axis=0, kind='linear', fill_value='extrapolate')
    
    # Sample boundary points at ~50 Hz within BOUNDARY_WINDOW of START edge only
    n_boundary_samples = max(1, int(BOUNDARY_WINDOW * 50))
    for t_rel in np.linspace(t_spline_start, t_spline_start + BOUNDARY_WINDOW, n_boundary_samples):
        t_abs = t_rel + t_ref
        # Velocity prior
        v_gt = mocap_vel_interp(t_rel)
        boundary_vel_priors.append((t_abs, v_gt))
        # Position prior
        p_gt = pos_interp(t_rel)
        boundary_pos_priors.append((t_abs, p_gt))
        # Orientation prior: delta(t) = 0 (nominal IS MoCap SLERP)
        boundary_ori_priors.append(t_abs)
        # Acceleration prior
        a_gt = mocap_accel_interp(t_rel)
        boundary_accel_priors.append((t_abs, a_gt))
        # Angular velocity prior: delta_dot(t) = 0 (omega = omega_nominal)
        boundary_gyro_priors.append(t_abs)
    
    print(f"\n  Boundary priors (START only): {n_boundary_samples} sample points")
    print(f"    vel: {len(boundary_vel_priors)}, pos: {len(boundary_pos_priors)}, ori: {len(boundary_ori_priors)}, acc: {len(boundary_accel_priors)}, gyr: {len(boundary_gyro_priors)}")
    if len(boundary_vel_priors) > 0:
        v0 = boundary_vel_priors[0][1]
        print(f"    Start vel GT: [{v0[0]:.2f}, {v0[1]:.2f}, {v0[2]:.2f}] m/s (|v|={np.linalg.norm(v0):.2f})")
    
    optimized_state = solve_trajectory_nonlinear(
        initial_state=initial_state,
        radar_frames=radar_frames,
        imu_data=imu_data,
        sensor_translation=TRANSLATION,
        sensor_rotation=sensor_rotation,
        lambda_accel=LAMBDA_ACCEL,
        lambda_gyro=LAMBDA_GYRO,
        lambda_snap_pos=LAMBDA_SNAP_POS,
        lambda_snap_ori=LAMBDA_SNAP_ORI,
        huber_delta=HUBER_DELTA,
        huber_delta_accel=HUBER_DELTA_ACCEL,
        max_iterations=MAX_ITERATIONS,
        lock_biases=LOCK_BIASES,
        use_jacobi_precond=USE_JACOBI_PRECOND,
        verbose=True,
        mocap_times_abs=mocap_times_abs,
        mocap_rotations=mocap_rotations,
        boundary_vel_priors=boundary_vel_priors,
        boundary_pos_priors=boundary_pos_priors,
        lambda_boundary_vel=LAMBDA_BOUNDARY_VEL,
        lambda_boundary_pos=LAMBDA_BOUNDARY_POS,
        boundary_ori_priors=boundary_ori_priors,
        boundary_accel_priors=boundary_accel_priors,
        boundary_gyro_priors=boundary_gyro_priors,
        lambda_boundary_ori=LAMBDA_BOUNDARY_ORI,
        lambda_boundary_accel=LAMBDA_BOUNDARY_ACCEL,
        lambda_boundary_gyro=LAMBDA_BOUNDARY_GYRO,
    )
    
    # ==================== Evaluate Results ====================
    print(f"\n{'Evaluating Results':-^80}")
    
    # Check boundary velocities after optimization
    t_eval_start_abs = max(pos_bspline.t_start + t_ref, mocap_times_abs[0])
    t_eval_end_abs = min(pos_bspline.t_end + t_ref, mocap_times_abs[-1])
    v_opt_start = optimized_state.get_position(t_eval_start_abs, derivative=1)
    v_opt_end = optimized_state.get_position(t_eval_end_abs, derivative=1)
    # Find closest MoCap states to eval boundaries
    i_start = np.argmin(np.abs(mocap_times_abs - t_eval_start_abs))
    i_end = np.argmin(np.abs(mocap_times_abs - t_eval_end_abs))
    v_gt_s = agiros_states[i_start].velocity
    v_gt_e = agiros_states[i_end].velocity
    print(f"  Optimized vel at spline_start: [{v_opt_start[0]:.2f}, {v_opt_start[1]:.2f}, {v_opt_start[2]:.2f}] (|v|={np.linalg.norm(v_opt_start):.2f})")
    print(f"  MoCap vel at spline_start:     [{v_gt_s[0]:.2f}, {v_gt_s[1]:.2f}, {v_gt_s[2]:.2f}] (|v|={np.linalg.norm(v_gt_s):.2f})")
    print(f"  Optimized vel at spline_end:   [{v_opt_end[0]:.2f}, {v_opt_end[1]:.2f}, {v_opt_end[2]:.2f}] (|v|={np.linalg.norm(v_opt_end):.2f})")
    print(f"  MoCap vel at spline_end:       [{v_gt_e[0]:.2f}, {v_gt_e[1]:.2f}, {v_gt_e[2]:.2f}] (|v|={np.linalg.norm(v_gt_e):.2f})")
    
    # Sample trajectory at MoCap times WITHIN the valid B-spline domain
    # (outside the domain, spline returns zeros which distorts RMSE and plots)
    t_eval_start = max(pos_bspline.t_start + t_ref, mocap_times_abs[0])
    t_eval_end = min(pos_bspline.t_end + t_ref, mocap_times_abs[-1])
    spline_valid_mask = (mocap_times_abs >= t_eval_start) & (mocap_times_abs <= t_eval_end)
    eval_times = mocap_times_abs[spline_valid_mask]
    agiros_eval = [agiros_states[i] for i in range(len(agiros_states)) if spline_valid_mask[i]]
    mocap_positions_eval = mocap_positions[spline_valid_mask]
    mocap_rotations_eval = mocap_rotations[spline_valid_mask]
    
    print(f"  Evaluation range: [{eval_times[0]-t_ref:.3f}, {eval_times[-1]-t_ref:.3f}] s ({len(eval_times)}/{len(mocap_times_abs)} MoCap points)")
    
    estimated_positions = np.array([optimized_state.get_position(t, 0) for t in eval_times])
    estimated_velocities = np.array([optimized_state.get_position(t, 1) for t in eval_times])
    estimated_accelerations = np.array([optimized_state.get_position(t, 2) for t in eval_times])
    estimated_rotations = np.array([optimized_state.get_rotation(t) for t in eval_times])
    
    mocap_velocities = np.array([s.velocity for s in agiros_eval])
    
    # Compute MoCap acceleration via numerical differentiation of velocity
    # Use FULL MoCap data for differentiation, then interpolate to eval times
    all_mocap_velocities = np.array([s.velocity for s in agiros_states])
    dt_mocap = np.diff(mocap_times_abs)
    valid_mask = np.ones(len(mocap_times_abs), dtype=bool)
    for i in range(1, len(dt_mocap)):
        if dt_mocap[i-1] < 1e-3:  # dt < 1ms = near-duplicate
            valid_mask[i] = False
    
    # Differentiate on clean samples, then interpolate back to eval times
    clean_times = mocap_times_abs[valid_mask]
    clean_vel = all_mocap_velocities[valid_mask]
    dt_clean = np.diff(clean_times)
    clean_accel = np.zeros_like(clean_vel)
    clean_accel[1:-1] = (clean_vel[2:] - clean_vel[:-2]) / (dt_clean[1:] + dt_clean[:-1])[:, None]
    clean_accel[0] = (clean_vel[1] - clean_vel[0]) / dt_clean[0]
    clean_accel[-1] = (clean_vel[-1] - clean_vel[-2]) / dt_clean[-1]
    
    # Apply SavGol smoothing (window=15, order=3) to suppress differentiation noise
    from scipy.signal import savgol_filter
    win = min(15, len(clean_accel) - (1 if len(clean_accel) % 2 == 0 else 0))
    if win >= 5:
        for dim in range(3):
            clean_accel[:, dim] = savgol_filter(clean_accel[:, dim], win, 3)
    
    # Interpolate back to evaluation timestamps
    mocap_accelerations = np.zeros_like(mocap_velocities)
    for dim in range(3):
        mocap_accelerations[:, dim] = np.interp(eval_times, clean_times, clean_accel[:, dim])
    
    # Compute errors
    pos_errors = np.linalg.norm(estimated_positions - mocap_positions_eval, axis=1)
    vel_errors = np.linalg.norm(estimated_velocities - mocap_velocities, axis=1)
    
    # Rotation errors (angle between matrices)
    rot_errors = []
    for i, mocap_rot in enumerate(mocap_rotations_eval):
        est_rot = estimated_rotations[i]
        R_error = mocap_rot.T @ est_rot
        angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        rot_errors.append(np.degrees(angle_error))
    rot_errors = np.array(rot_errors)
    
    print(f"\nPosition Errors:")
    print(f"  Mean: {pos_errors.mean():.4f} m")
    print(f"  Std:  {pos_errors.std():.4f} m")
    print(f"  RMSE: {np.sqrt(np.mean(pos_errors**2)):.4f} m")
    
    # Acceleration errors
    accel_errors = np.linalg.norm(estimated_accelerations - mocap_accelerations, axis=1)
    
    print(f"\nVelocity Errors:")
    print(f"  Mean: {vel_errors.mean():.4f} m/s")
    print(f"  RMSE: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s")
    
    print(f"\nAcceleration Errors:")
    print(f"  Mean: {accel_errors.mean():.4f} m/s²")
    print(f"  RMSE: {np.sqrt(np.mean(accel_errors**2)):.4f} m/s²")
    
    print(f"\nOrientation Errors:")
    print(f"  Mean: {rot_errors.mean():.4f} deg")
    print(f"  RMSE: {np.sqrt(np.mean(rot_errors**2)):.4f} deg")
    
    print(f"\nEstimated Biases:")
    print(f"  Accelerometer: [{optimized_state.acc_bias[0]:.4f}, {optimized_state.acc_bias[1]:.4f}, {optimized_state.acc_bias[2]:.4f}] m/s²")
    print(f"  Gyroscope: [{optimized_state.gyr_bias[0]:.4f}, {optimized_state.gyr_bias[1]:.4f}, {optimized_state.gyr_bias[2]:.4f}] rad/s")
    
    # ==================== Plotting ====================
    print(f"\n{'Generating Plots':-^80}")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Nonlinear Solver Validation Results', fontsize=14, fontweight='bold')
    
    time_rel = eval_times - eval_times[0]
    
    # 1. Trajectory (X-Y)
    ax = axes[0, 0]
    ax.plot(mocap_positions_eval[:, 0], mocap_positions_eval[:, 1], 'b-', label='MoCap', linewidth=2)
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'r--', label='Estimated', linewidth=2)
    ax.scatter(optimized_state.pos_bspline.control_points[:, 0],
               optimized_state.pos_bspline.control_points[:, 1],
               c='orange', marker='x', s=30, label='Control Points')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory (X-Y Plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 2. Position error
    ax = axes[0, 1]
    ax.plot(time_rel, pos_errors, 'r-', linewidth=2)
    ax.axhline(pos_errors.mean(), color='b', linestyle='--', label=f'Mean: {pos_errors.mean():.4f}m')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Orientation error
    ax = axes[0, 2]
    ax.plot(time_rel, rot_errors, 'g-', linewidth=2)
    ax.axhline(rot_errors.mean(), color='b', linestyle='--', label=f'Mean: {rot_errors.mean():.4f}°')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Orientation Error (deg)')
    ax.set_title('Orientation Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Velocity comparison
    ax = axes[1, 0]
    mocap_speeds = np.linalg.norm(mocap_velocities, axis=1)
    est_speeds = np.linalg.norm(estimated_velocities, axis=1)
    ax.plot(time_rel, mocap_speeds, 'b-', label='MoCap', linewidth=2)
    ax.plot(time_rel, est_speeds, 'r--', label='Estimated', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Velocity error
    ax = axes[1, 1]
    ax.plot(time_rel, vel_errors, 'r-', linewidth=2)
    ax.axhline(vel_errors.mean(), color='b', linestyle='--', label=f'Mean: {vel_errors.mean():.4f}m/s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity Error (m/s)')
    ax.set_title('Velocity Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Acceleration comparison
    ax = axes[1, 2]
    mocap_accel_norm = np.linalg.norm(mocap_accelerations, axis=1)
    est_accel_norm = np.linalg.norm(estimated_accelerations, axis=1)
    ax.plot(time_rel, mocap_accel_norm, 'b-', label='MoCap', linewidth=1, alpha=0.7)
    ax.plot(time_rel, est_accel_norm, 'r--', label='Estimated', linewidth=1, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Acceleration Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Acceleration error
    ax = axes[2, 0]
    ax.plot(time_rel, accel_errors, 'r-', linewidth=1)
    ax.axhline(accel_errors.mean(), color='b', linestyle='--', label=f'Mean: {accel_errors.mean():.4f}m/s²')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration Error (m/s²)')
    ax.set_title('Acceleration Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Acceleration per-axis comparison
    ax = axes[2, 1]
    labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    for i, (lbl, clr) in enumerate(zip(labels, colors)):
        ax.plot(time_rel, mocap_accelerations[:, i], f'{clr}-', alpha=0.5, linewidth=1, label=f'MoCap {lbl}')
        ax.plot(time_rel, estimated_accelerations[:, i], f'{clr}--', alpha=0.7, linewidth=1, label=f'Est {lbl}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Acceleration per Axis')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 9. Error summary + hyperparameters (for reproducibility)
    accel_rmse = np.sqrt(np.mean(accel_errors**2))
    ax = axes[2, 2]
    summary_lines = [
        f"RESULTS",
        f"  Pos  RMSE: {np.sqrt(np.mean(pos_errors**2)):.4f} m",
        f"  Vel  RMSE: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s",
        f"  Acc  RMSE: {accel_rmse:.4f} m/s²",
        f"  Ori  RMSE: {np.sqrt(np.mean(rot_errors**2)):.4f}°",
        f"",
        f"HYPERPARAMETERS",
        f"  bag={bag_key}  t={START_TIME_OFFSET:.0f}s+{DURATION:.0f}s",
        f"  flip={FLIP_BODY_FRAME}  lock_bias={LOCK_BIASES}",
        f"  dt_pos={DT_POS}  dt_ori={DT_ORI}  deg={BSPLINE_DEGREE}",
        f"  λ_accel={LAMBDA_ACCEL}  λ_gyro={LAMBDA_GYRO}",
        f"  λ_snap_pos={LAMBDA_SNAP_POS}  λ_snap_ori={LAMBDA_SNAP_ORI}",
        f"  huber_radar={HUBER_DELTA}  huber_accel={HUBER_DELTA_ACCEL}",
        f"  λ_bnd_vel={LAMBDA_BOUNDARY_VEL}  λ_bnd_pos={LAMBDA_BOUNDARY_POS}",
        f"  λ_bnd_ori={LAMBDA_BOUNDARY_ORI}  λ_bnd_acc={LAMBDA_BOUNDARY_ACCEL}",
        f"  λ_bnd_gyr={LAMBDA_BOUNDARY_GYRO}",
        f"  bnd_window={BOUNDARY_WINDOW}s (start only)",
        f"  max_iter={MAX_ITERATIONS}  precond={USE_JACOBI_PRECOND}",
    ]
    summary_text = "\n".join(summary_lines)
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=8, fontfamily='monospace', verticalalignment='top')
    ax.axis('off')
    ax.set_title('Summary & Config')
    
    plt.tight_layout()
    output_filename = f'nonlinear_solver_validation_{bag_key}_{timestamp_str}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_filename}")

    # ==================== Zoomed X-Y Trajectory Plot ====================
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    ax2.plot(mocap_positions_eval[:, 0], mocap_positions_eval[:, 1], 'b-', label='MoCap', linewidth=2)
    ax2.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'r--', label='Estimated', linewidth=2)
    ax2.scatter(optimized_state.pos_bspline.control_points[:, 0],
                optimized_state.pos_bspline.control_points[:, 1],
                c='orange', marker='x', s=30, alpha=0.5, label='Control Points')
    # Start markers
    ax2.plot(mocap_positions_eval[0, 0], mocap_positions_eval[0, 1], 'bs', markersize=10, label='Start (MoCap)')
    ax2.plot(estimated_positions[0, 0], estimated_positions[0, 1], 'rs', markersize=10, label='Start (Est.)')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('Trajectory (X-Y Plane) — Zoomed to Trajectory', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    # Compute axis limits from the larger of ground truth / estimated trajectory
    all_traj = np.vstack([mocap_positions_eval[:, :2], estimated_positions[:, :2]])
    x_min, x_max = all_traj[:, 0].min(), all_traj[:, 0].max()
    y_min, y_max = all_traj[:, 1].min(), all_traj[:, 1].max()
    x_pad = max(0.1, (x_max - x_min) * 0.08)
    y_pad = max(0.1, (y_max - y_min) * 0.08)
    ax2.set_xlim(x_min - x_pad, x_max + x_pad)
    ax2.set_ylim(y_min - y_pad, y_max + y_pad)
    zoomed_filename = f'nonlinear_trajectory_zoomed_{bag_key}_{timestamp_str}.png'
    fig2.savefig(zoomed_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {zoomed_filename}")
    
    # ==================== Summary ====================
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'VALIDATION SUMMARY':#^80}")
    
    pos_rmse = np.sqrt(np.mean(pos_errors**2))
    vel_rmse = np.sqrt(np.mean(vel_errors**2))
    ori_rmse = np.sqrt(np.mean(rot_errors**2))
    
    if pos_errors.mean() < 0.5 and vel_errors.mean() < 0.5 and rot_errors.mean() < 5.0:
        print("[OK] NONLINEAR SOLVER VALIDATION SUCCESSFUL!")
        print("   - Position, velocity, and orientation errors are low")
        print("   - Bias estimation is working")
        print("   - Ready for real-time deployment")
    else:
        print("[!!] NONLINEAR SOLVER RESULTS")
        print(f"   - Position RMSE: {pos_rmse:.4f} m")
        print(f"   - Velocity RMSE: {vel_rmse:.4f} m/s")
        print(f"   - Orientation RMSE: {ori_rmse:.4f} deg")
    
    print(f"\n   Runtime: {hours}h {minutes}m {seconds}s")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
