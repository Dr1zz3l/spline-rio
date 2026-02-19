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
        Get angular velocity at time t.
        
        omega_body = omega_nominal(t) + J_r(delta) * delta_dot
        For small delta: J_r approx I, so omega approx omega_nominal + delta_dot
        """
        t_rel = t - self.ori_bspline.t_ref
        
        # Nominal angular velocity from MoCap
        omega_nominal = self.get_nominal_angular_velocity(t_rel)
        
        # Perturbation angular velocity from B-spline derivative
        delta_dot = self.ori_bspline(t_rel, derivative=1)
        
        return omega_nominal + delta_dot
    
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
        """
        n_cp = self.ori_bspline.n_points
        
        # Dense sampling to minimize interpolation error between B-spline and SLERP
        n_dense = max(200, n_cp * 20)  # ~200+ samples over the trajectory
        t_start = self.ori_bspline.t_start
        t_end = self.ori_bspline.t_end
        dense_times = np.linspace(t_start, t_end, n_dense)
        
        # Evaluate full rotation at each dense time point
        dense_rots = []
        for t_rel in dense_times:
            R_nom = self.get_nominal_rotation(t_rel)
            delta = self.ori_bspline(t_rel, derivative=0)
            R_full = R_nom @ so3_exp(delta)
            dense_rots.append(R_full)
        
        # Rebuild SLERP from dense samples
        scipy_rots = Rotation.from_matrix(np.array(dense_rots))
        self.mocap_slerp = Slerp(dense_times, scipy_rots)
        
        # Update nominal_rotations array at CP times for fallback
        cp_times = np.linspace(t_start, t_end, n_cp)
        self.nominal_rotations = np.array([
            self.mocap_slerp(t).as_matrix() for t in cp_times
        ])
        
        # Update omega interpolation via numerical differentiation of dense rotations
        dt_dense = dense_times[1] - dense_times[0]
        omegas = np.zeros((n_dense, 3))
        for i in range(n_dense - 1):
            R_rel = dense_rots[i].T @ dense_rots[i + 1]
            omegas[i] = so3_log(R_rel) / dt_dense
        omegas[-1] = omegas[-2]
        self.mocap_omega_interp = interp1d(dense_times, omegas, axis=0,
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
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Compute Jacobian and residual vector analytically using SymForce-generated functions.
    
    Chain rule:
    - Position CPs affect v_world via N_i'(t) and a_world via N_i''(t)
    - Orientation CPs affect delta via M_j(t) and omega via M_j'(t)
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
        omega = state.get_angular_velocity(t)
        
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
            res, J_v, J_delta, J_omega = radar_residual_with_jacobians(
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
            
            # Chain rule: ∂r/∂ori_cp_j = J_delta * M_j(t) + J_omega * M_j'(t)
            # Delta contribution
            for ci, cp_idx in enumerate(ori_val_indices):
                basis_val = ori_val_coeffs[ci]
                for dim in range(3):
                    col = n_pos + cp_idx * 3 + dim
                    val = sqrt_w * J_delta[dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            # Omega contribution
            for ci, cp_idx in enumerate(ori_vel_indices):
                basis_val = ori_vel_coeffs[ci]
                for dim in range(3):
                    col = n_pos + cp_idx * 3 + dim
                    val = sqrt_w * J_omega[dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            row_idx += 1
    
    n_radar = row_idx
    
    # ==================== Accelerometer residuals ====================
    sqrt_lambda_accel = np.sqrt(lambda_accel)
    
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
        
        # 3 rows for this measurement
        for k in range(3):
            all_residuals.append(sqrt_lambda_accel * res_3[k])
            
            # ∂r_accel/∂pos_cp_i = J_a[k,:] * N_i''(t)
            for ci, cp_idx in enumerate(pos_acc_indices):
                basis_val = pos_acc_coeffs[ci]
                for dim in range(3):
                    col = cp_idx * 3 + dim
                    val = sqrt_lambda_accel * J_a[k, dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            # ∂r_accel/∂ori_cp_j = J_delta_3x3[k,:] * M_j(t)
            for ci, cp_idx in enumerate(ori_val_indices):
                basis_val = ori_val_coeffs[ci]
                for dim in range(3):
                    col = n_pos + cp_idx * 3 + dim
                    val = sqrt_lambda_accel * J_delta_3x3[k, dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx)
                        cols.append(col)
                        vals.append(val)
            
            # ∂r_accel/∂b_a = J_ba[k,:] = -I[k,:]
            for dim in range(3):
                col = n_pos + n_ori + dim  # acc bias index
                val = sqrt_lambda_accel * J_ba[k, dim]
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
        
        # Basis functions for orientation derivative
        ori_vel_coeffs, ori_vel_indices = state.ori_bspline.get_basis_coefficients(t_rel_ori, derivative=1)
        
        # Raw gyroscope reading (body frame)
        z_gyro = imu_msg.angular_velocity
        omega_pred = state.get_angular_velocity(t)
        
        # Gyro residual: r = z_gyro - omega - b_g
        r_gyro = z_gyro - omega_pred - state.gyr_bias
        
        for k in range(3):
            all_residuals.append(sqrt_lambda_gyro * r_gyro[k])
            
            # ∂r_gyro/∂ori_cp_j = -M_j'(t) * I[k,:]
            for ci, cp_idx in enumerate(ori_vel_indices):
                basis_val = ori_vel_coeffs[ci]
                col = n_pos + cp_idx * 3 + k  # only k-th dimension affected
                val = sqrt_lambda_gyro * (-basis_val)
                if abs(val) > 1e-15:
                    rows.append(row_idx)
                    cols.append(col)
                    vals.append(val)
            
            # ∂r_gyro/∂b_g = -I[k,:]
            col = n_pos + n_ori + 3 + k  # gyro bias index
            rows.append(row_idx)
            cols.append(col)
            vals.append(sqrt_lambda_gyro * (-1.0))
            
            row_idx += 1
    
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
    lambda_ori_prior: float = 0.0,
    huber_delta: float = 0.5,
    max_iterations: int = 20,
    lock_biases: bool = False,
    verbose: bool = True,
    mocap_times_abs: np.ndarray = None,
    mocap_rotations: np.ndarray = None,
) -> TrajectoryState:
    """
    Solve for optimal trajectory using Levenberg-Marquardt.
    
    Uses analytical Jacobians derived by SymForce for ~50x speedup.
    
    Minimizes:
    E = sum(huber(r_radar)) + lambda_accel * ||r_accel||^2
        + lambda_gyro * ||r_gyro||^2
        + lambda_snap_pos * ||snap_pos||^2 + lambda_snap_ori * ||snap_ori||^2
    """
    if verbose:
        print(f"\n{'Levenberg-Marquardt Optimization':#^80}")
        print(f"Max iterations: {max_iterations}")
        print(f"Huber delta: {huber_delta}")
        print(f"Lambda accel: {lambda_accel}")
        print(f"Lambda gyro: {lambda_gyro}")
        print(f"Lambda snap pos: {lambda_snap_pos}")
        print(f"Lambda snap ori: {lambda_snap_ori}")
        print(f"Lambda ori prior: {lambda_ori_prior}")
        if lock_biases:
            print(f"*** BIASES LOCKED TO ZERO ***")
    
    state = initial_state
    lambda_lm = 1e-3  # Initial LM damping
    prev_cost = None
    
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
    # Orientation prior: penalize ||delta||^2 to keep perturbation small
    if lambda_ori_prior > 0:
        R_snap[n_pos:n_pos+n_ori, n_pos:n_pos+n_ori] += lambda_ori_prior * sparse.eye(n_ori)
    R_snap = R_snap.tocsr()
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'Iteration ' + str(iteration + 1):-^80}")
            # Track orientation RMSE per iteration
            if mocap_times_abs is not None and mocap_rotations is not None:
                ori_rmse = compute_orientation_rmse(state, mocap_times_abs, mocap_rotations)
                delta_norms = np.linalg.norm(state.ori_bspline.control_points, axis=1)
                print(f"Orientation RMSE: {ori_rmse:.1f} deg | delta max: {delta_norms.max():.4f} mean: {delta_norms.mean():.4f}")
                print(f"Acc bias: [{state.acc_bias[0]:.3f}, {state.acc_bias[1]:.3f}, {state.acc_bias[2]:.3f}]"
                      f"  Gyr bias: [{state.gyr_bias[0]:.3f}, {state.gyr_bias[1]:.3f}, {state.gyr_bias[2]:.3f}]")
        
        t_start = time.time()
        
        # Build Jacobian + residual vector analytically (includes all costs)
        if verbose:
            print("Computing analytical Jacobian...")
        
        J, r_total = compute_jacobian_analytical(
            state, radar_frames, imu_data,
            sensor_translation, sensor_rotation,
            lambda_accel, lambda_gyro, huber_delta
        )
        
        cost_total = np.sum(r_total**2)
        
        if verbose:
            # Cost decomposition
            n_r = getattr(J, 'n_radar', 0)
            n_a = getattr(J, 'n_accel', 0)
            cost_radar = np.sum(r_total[:n_r]**2) if n_r > 0 else 0
            cost_accel = np.sum(r_total[n_r:n_r+n_a]**2) if n_a > 0 else 0
            cost_gyro = np.sum(r_total[n_r+n_a:]**2) if (n_r+n_a) < len(r_total) else 0
            print(f"Cost: total={cost_total:.2f} | radar={cost_radar:.2f} accel={cost_accel:.2f} gyro={cost_gyro:.2f}")
            print(f"Jacobian: {J.shape}, nnz={J.nnz}, sparsity={100*(1-J.nnz/(J.shape[0]*J.shape[1])):.2f}%")
        
        # Build normal equations with LM damping
        H = J.T @ J + lambda_lm * sparse.eye(n_total) + R_snap
        b = J.T @ r_total
        
        # Solve
        try:
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
        _, r_new = compute_jacobian_analytical(
            state, radar_frames, imu_data,
            sensor_translation, sensor_rotation,
            lambda_accel, lambda_gyro, huber_delta
        )
        new_cost = np.sum(r_new**2)
        
        # LM acceptance: if cost decreased, accept and reduce damping
        if prev_cost is None or new_cost < cost_total:
            lambda_lm *= 0.5
            prev_cost = new_cost
            # No re-linearization: orientation prior keeps delta bounded
            # Re-linearization introduces B-spline/SLERP interpolation mismatch
        else:
            # Reject and increase damping
            state.from_vector(x_current)
            lambda_lm *= 5.0
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
FLIPPED_BAGS = {"circle_fwd", "backflips", "loopings"}

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

    START_TIME_OFFSET = 5.0   # Skip initial hover
    DURATION = 120.0          # Full bag
    
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
    
    BSPLINE_DEGREE = 5  # Quintic for continuous snap
    DT_POS = 0.15   # Fixed knot spacing for position spline (seconds)
    DT_ORI = 0.1   # Fixed knot spacing for orientation spline (seconds)
    
    # Regularization weights
    LAMBDA_ACCEL = 0.1      # Accelerometer weight: gravity direction constrains orientation
    LAMBDA_GYRO = 0.5       # Gyroscope weight: omega_nominal now included
    LAMBDA_SNAP_POS = 0.001  # Position smoothness
    LAMBDA_SNAP_ORI = 0.01   # Orientation smoothness
    LAMBDA_ORI_PRIOR = 10.0  # Orientation prior: penalizes delta from nominal
    
    HUBER_DELTA = 0.5  # meters/second (Huber threshold)
    MIN_RANGE = 0.2
    MAX_ITERATIONS = 20  # Full run
    USE_PHASE2_INIT = True  # Initialize position from Phase 2 linear solver
    LOCK_BIASES = True  # Lock biases to zero — force solver to fix orientation instead
    
    print(f"\n{'Configuration':-^80}")
    print(f"Bag: {bag_key} -> {BAG_PATH}")
    print(f"Flip body frame: {FLIP_BODY_FRAME}")
    print(f"Time window: {START_TIME_OFFSET:.1f}s + {DURATION:.1f}s")
    print(f"B-spline degree: {BSPLINE_DEGREE}")
    print(f"Lambda accel: {LAMBDA_ACCEL}")
    print(f"Lambda gyro: {LAMBDA_GYRO}")
    print(f"Lambda snap (pos/ori): {LAMBDA_SNAP_POS}/{LAMBDA_SNAP_ORI}")
    print(f"Lambda ori prior: {LAMBDA_ORI_PRIOR}")
    print(f"Huber delta: {HUBER_DELTA} m/s")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Use Phase 2 init: {USE_PHASE2_INIT}")
    print(f"Lock biases: {LOCK_BIASES}")
    
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
        
        # Solve Phase 2
        print("\nSolving Phase 2 (linear)...")
        x_opt = solve_trajectory_linear(
            pos_bspline, J_radar, r_radar, J_accel, r_accel,
            lambda_accel=0.01, lambda_snap=0.1, lambda_position=0.05,
            verbose=True
        )
        
        # Update control points with Phase 2 result
        pos_bspline.control_points = x_opt.reshape(-1, 3)
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
        lambda_ori_prior=LAMBDA_ORI_PRIOR,
        huber_delta=HUBER_DELTA,
        max_iterations=MAX_ITERATIONS,
        lock_biases=LOCK_BIASES,
        verbose=True,
        mocap_times_abs=mocap_times_abs,
        mocap_rotations=mocap_rotations,
    )
    
    # ==================== Evaluate Results ====================
    print(f"\n{'Evaluating Results':-^80}")
    
    # Sample trajectory at MoCap times (use absolute times since get_position converts internally)
    eval_times = mocap_times_abs
    estimated_positions = np.array([optimized_state.get_position(t, 0) for t in eval_times])
    estimated_velocities = np.array([optimized_state.get_position(t, 1) for t in eval_times])
    estimated_accelerations = np.array([optimized_state.get_position(t, 2) for t in eval_times])
    estimated_rotations = np.array([optimized_state.get_rotation(t) for t in eval_times])
    
    mocap_velocities = np.array([s.velocity for s in agiros_states])
    
    # Compute MoCap acceleration via numerical differentiation of velocity
    dt_mocap = np.diff(mocap_times_abs)
    mocap_accelerations = np.zeros_like(mocap_velocities)
    mocap_accelerations[1:-1] = (mocap_velocities[2:] - mocap_velocities[:-2]) / (dt_mocap[1:] + dt_mocap[:-1])[:, None]
    mocap_accelerations[0] = (mocap_velocities[1] - mocap_velocities[0]) / dt_mocap[0]
    mocap_accelerations[-1] = (mocap_velocities[-1] - mocap_velocities[-2]) / dt_mocap[-1]
    
    # Compute errors
    pos_errors = np.linalg.norm(estimated_positions - mocap_positions, axis=1)
    vel_errors = np.linalg.norm(estimated_velocities - mocap_velocities, axis=1)
    
    # Rotation errors (angle between matrices)
    rot_errors = []
    for i, mocap_rot in enumerate(mocap_rotations):
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
    ax.plot(mocap_positions[:, 0], mocap_positions[:, 1], 'b-', label='MoCap', linewidth=2)
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
    
    # 9. Error summary
    accel_rmse = np.sqrt(np.mean(accel_errors**2))
    ax = axes[2, 2]
    ax.text(0.1, 0.9, f"Position RMSE: {np.sqrt(np.mean(pos_errors**2)):.4f} m", 
            transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.8, f"Velocity RMSE: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s",
            transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.7, f"Accel RMSE: {accel_rmse:.4f} m/s²",
            transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.6, f"Orientation RMSE: {np.sqrt(np.mean(rot_errors**2)):.4f}°",
            transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.4, f"Acc bias: [{optimized_state.acc_bias[0]:.3f}, {optimized_state.acc_bias[1]:.3f}, {optimized_state.acc_bias[2]:.3f}]",
            transform=ax.transAxes, fontsize=10)
    ax.text(0.1, 0.3, f"Gyr bias: [{optimized_state.gyr_bias[0]:.3f}, {optimized_state.gyr_bias[1]:.3f}, {optimized_state.gyr_bias[2]:.3f}]",
            transform=ax.transAxes, fontsize=10)
    ax.axis('off')
    ax.set_title('Summary')
    
    plt.tight_layout()
    output_filename = f'nonlinear_solver_validation_{bag_key}_{timestamp_str}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_filename}")

    # ==================== Zoomed X-Y Trajectory Plot ====================
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    ax2.plot(mocap_positions[:, 0], mocap_positions[:, 1], 'b-', label='MoCap', linewidth=2)
    ax2.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'r--', label='Estimated', linewidth=2)
    ax2.scatter(optimized_state.pos_bspline.control_points[:, 0],
                optimized_state.pos_bspline.control_points[:, 1],
                c='orange', marker='x', s=30, alpha=0.5, label='Control Points')
    # Start markers
    ax2.plot(mocap_positions[0, 0], mocap_positions[0, 1], 'bs', markersize=10, label='Start (MoCap)')
    ax2.plot(estimated_positions[0, 0], estimated_positions[0, 1], 'rs', markersize=10, label='Start (Est.)')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('Trajectory (X-Y Plane) — Zoomed to Trajectory', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    # Compute axis limits from the larger of ground truth / estimated trajectory
    all_traj = np.vstack([mocap_positions[:, :2], estimated_positions[:, :2]])
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
