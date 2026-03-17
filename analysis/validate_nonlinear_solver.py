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
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
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
    predict_doppler_velocity,
    compute_aliasing_summary
)
from bspline_utils import (
    UniformBSpline,
    create_uniform_bspline_from_times,
    build_minimum_snap_regularization
)
from cumulative_so3_bspline import (
    CumulativeSO3BSpline,
    so3_exp as _so3_exp_cs,
    so3_log as _so3_log_cs,
)
from codegen.generated_jacobians import (
    radar_residual_with_jacobians,
    accel_residual_with_jacobians,
    gyro_residual_with_jacobians,
    Rot3
)
from config_loader import load_config


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

    Orientation uses cumulative B-splines on SO(3):
        R(t) = R_base[k-3] · ∏ exp(B̃_j(t) · Ω_{k-2+j})
    No nominal trajectory or relinearization needed.

    State variables:
    - Position control points (N_pos × 3)
    - SO(3) increment knots (N_ori × 3) inside CumulativeSO3BSpline
    - Accelerometer bias (3,)
    - Gyroscope bias (3,)
    - Radar extrinsic delta (3,)
    """

    def __init__(
        self,
        pos_bspline: UniformBSpline,
        ori_spline: CumulativeSO3BSpline,
        acc_bias: np.ndarray = None,
        gyr_bias: np.ndarray = None,
        radar_extrinsic_delta: np.ndarray = None,
    ):
        self.pos_bspline = pos_bspline
        self.ori_spline = ori_spline

        self.acc_bias = acc_bias if acc_bias is not None else np.zeros(3)
        self.gyr_bias = gyr_bias if gyr_bias is not None else np.zeros(3)
        self.radar_extrinsic_delta = (radar_extrinsic_delta
                                      if radar_extrinsic_delta is not None
                                      else np.zeros(3))

    def get_position(self, t: float, derivative: int = 0) -> np.ndarray:
        """Get position (or velocity/acceleration) at absolute time t."""
        t_rel = t - self.pos_bspline.t_ref
        return self.pos_bspline(t_rel, derivative=derivative)

    def get_rotation(self, t: float) -> np.ndarray:
        """Return 3×3 rotation matrix at absolute time t."""
        t_rel = t - self.ori_spline.t_ref
        return self.ori_spline.evaluate(t_rel)

    def get_angular_velocity(self, t: float) -> np.ndarray:
        """Return body-frame angular velocity at absolute time t."""
        t_rel = t - self.ori_spline.t_ref
        _, omega, _, _, _ = self.ori_spline.evaluate_with_jacobians(t_rel)
        return omega

    def to_vector(self) -> np.ndarray:
        """
        Flatten state to optimization vector.

        Layout:
          [pos_control_points (N_pos*3),
           ori_omega_knots (N_ori*3),
           acc_bias (3),
           gyr_bias (3),
           radar_extrinsic_delta (3)]
        """
        pos_flat = self.pos_bspline.control_points.flatten()
        ori_flat = self.ori_spline.omega_knots.flatten()
        return np.concatenate([pos_flat, ori_flat,
                                self.acc_bias, self.gyr_bias,
                                self.radar_extrinsic_delta])

    def from_vector(self, x: np.ndarray):
        """Update state from optimization vector (recomputes base rotations)."""
        n_pos = self.pos_bspline.n_points * 3
        n_ori = self.ori_spline.n_knots * 3

        self.pos_bspline.control_points = x[:n_pos].reshape(-1, 3)
        self.ori_spline.from_flat(x[n_pos:n_pos + n_ori])  # also calls recompute_base_rotations
        self.acc_bias = x[n_pos + n_ori:n_pos + n_ori + 3].copy()
        self.gyr_bias = x[n_pos + n_ori + 3:n_pos + n_ori + 6].copy()
        self.radar_extrinsic_delta = x[n_pos + n_ori + 6:n_pos + n_ori + 9].copy()

    def get_state_size(self) -> int:
        """Total number of optimization variables."""
        return self.pos_bspline.n_points * 3 + self.ori_spline.n_knots * 3 + 9


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
    lambda_ori_reg: float = 0.0,
    lambda_bias_prior: float = 0.0,
    lambda_bias_prior_accel: float = None,
    lambda_bias_prior_gyro: float = None,
    bias_prior_mean: np.ndarray = None,
    lock_extrinsics: bool = False,
    optimize_pitch_only: bool = True,
    lambda_extrinsic_prior: float = 0.0,
    v_max: float = None,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Compute Jacobian and residual vector analytically using SymForce-generated functions.

    Chain rule (cumulative SO(3) B-spline):
    - Position CPs affect v_world via N_i'(t) and a_world via N_i''(t)
    - Orientation knots Ω_j: evaluate_with_jacobians() returns J_R_list, J_omega_list
      ∂r/∂Ω_j = J_R_res @ J_R_list[j] + J_omega_res @ J_omega_list[j]
    - Biases affect residuals directly

    Key trick: SymForce residuals are called with R_full as R_nominal and delta=0.
    At delta=0, J_delta from SymForce equals the right-tangent Jacobian at R_full,
    which is exactly what J_R_list expects.

    Returns:
        J: Sparse Jacobian matrix (N_residuals, N_variables)
        r: Residual vector (N_residuals,)
    """
    _zeros3 = np.zeros(3)
    n_pos = state.pos_bspline.n_points * 3
    n_ori = state.ori_spline.n_knots * 3
    n_total = state.get_state_size()

    # Pre-compute sensor rotation quaternion with extrinsic perturbation
    R_bs_nominal = sensor_rotation
    R_bs_delta_matrix = so3_exp(state.radar_extrinsic_delta)
    R_bs_quat = Rot3.from_rotation_matrix(R_bs_nominal @ R_bs_delta_matrix)

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
        t_rel_ori = t - state.ori_spline.t_ref

        pos_vel_coeffs, pos_vel_indices = state.pos_bspline.get_basis_coefficients(
            t_rel_pos, derivative=1)

        v_world = state.get_position(t, derivative=1)

        # Cumulative SO(3) spline: full Jacobians (base + active span).
        # Base knots j <= k-3 affect R via R_base; active span knots affect R and omega.
        R_full, omega, J_R_list, J_omega_list, active_ori = \
            state.ori_spline.evaluate_full_jacobians(t_rel_ori)
        R_nom_quat = Rot3.from_rotation_matrix(R_full)

        n_points = frame.num_points()
        for i in range(n_points):
            p_s = frame.positions[i]
            range_val = frame.ranges[i] if frame.ranges is not None else np.linalg.norm(p_s)
            if range_val < min_range:
                continue

            u_sensor = p_s / np.linalg.norm(p_s)
            v_meas = frame.velocities[i]

            # Use SymForce residual with R_full as R_nominal and delta=0.
            # At delta=0, J_delta equals the right-tangent Jacobian at R_full.
            res, J_v, J_delta_radar, J_omega_radar, J_Rbs = radar_residual_with_jacobians(
                v_world, R_nom_quat, _zeros3, omega,
                u_sensor, sensor_translation, R_bs_quat,
                v_meas, 1e-10
            )

            r = res[0]

            # Doppler unwrapping
            if v_max is not None:
                v_pred = v_meas - r
                k_alias = round((v_pred - v_meas) / (2.0 * v_max))
                if k_alias != 0:
                    r += k_alias * 2.0 * v_max

            w = huber_weight(r, delta=huber_delta)
            sqrt_w = np.sqrt(w)
            all_residuals.append(r * sqrt_w)

            # ∂r/∂pos_cp_i
            for ci, cp_idx in enumerate(pos_vel_indices):
                basis_val = pos_vel_coeffs[ci]
                for dim in range(3):
                    col = cp_idx * 3 + dim
                    val = sqrt_w * J_v[dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx); cols.append(col); vals.append(val)

            # ∂r/∂Ω_j  (chain through cumulative spline Jacobians)
            for jj, knot_idx in enumerate(active_ori):
                # (3,) @ (3,3) + (3,) @ (3,3) → (3,)
                J_wrt = J_delta_radar @ J_R_list[jj] + J_omega_radar @ J_omega_list[jj]
                for dim in range(3):
                    col = n_pos + knot_idx * 3 + dim
                    val = sqrt_w * J_wrt[dim]
                    if abs(val) > 1e-15:
                        rows.append(row_idx); cols.append(col); vals.append(val)

            # ∂r/∂radar_extrinsic_delta
            if not lock_extrinsics:
                for dim in range(3):
                    if optimize_pitch_only and dim != 1:
                        continue
                    col = n_pos + n_ori + 6 + dim
                    val = sqrt_w * J_Rbs[dim]
                    if abs(val) > 1e-15:
                        rows.append(row_idx); cols.append(col); vals.append(val)

            row_idx += 1

    n_radar = row_idx
    
    # ==================== Accelerometer residuals ====================
    sqrt_lambda_accel = np.sqrt(lambda_accel)
    use_huber_accel = huber_delta_accel > 0
    g_world = np.array([0, 0, -9.81])

    for imu_msg in imu_data:
        t = imu_msg.timestamp
        t_rel_pos = t - state.pos_bspline.t_ref
        t_rel_ori = t - state.ori_spline.t_ref

        pos_acc_coeffs, pos_acc_indices = state.pos_bspline.get_basis_coefficients(
            t_rel_pos, derivative=2)

        a_world = state.get_position(t, derivative=2)

        # Cumulative spline: full Jacobians for accel (R-dependent, not omega-dependent).
        # Base knots j <= k-3 also contribute via R_base.
        R_full, _, J_R_list, _, active_ori = \
            state.ori_spline.evaluate_full_jacobians(t_rel_ori)
        R_nom_quat = Rot3.from_rotation_matrix(R_full)

        z_acc = imu_msg.linear_acceleration

        # delta=0: J_delta_3x3 is right-tangent Jacobian at R_full
        res_3, J_a, J_delta_3x3, J_ba = accel_residual_with_jacobians(
            a_world, R_nom_quat, _zeros3, g_world, z_acc, state.acc_bias, 1e-10
        )

        if use_huber_accel:
            accel_res_norm = np.linalg.norm(res_3)
            sqrt_w_accel = np.sqrt(huber_weight(accel_res_norm, delta=huber_delta_accel))
        else:
            sqrt_w_accel = 1.0

        scale = sqrt_lambda_accel * sqrt_w_accel

        for k in range(3):
            all_residuals.append(scale * res_3[k])

            # ∂r_accel/∂pos_cp_i
            for ci, cp_idx in enumerate(pos_acc_indices):
                basis_val = pos_acc_coeffs[ci]
                for dim in range(3):
                    col = cp_idx * 3 + dim
                    val = scale * J_a[k, dim] * basis_val
                    if abs(val) > 1e-15:
                        rows.append(row_idx); cols.append(col); vals.append(val)

            # ∂r_accel/∂Ω_j = J_delta_3x3[k,:] @ J_R_list[j]
            for jj, knot_idx in enumerate(active_ori):
                J_wrt = J_delta_3x3[k, :] @ J_R_list[jj]  # (3,)
                for dim in range(3):
                    col = n_pos + knot_idx * 3 + dim
                    val = scale * J_wrt[dim]
                    if abs(val) > 1e-15:
                        rows.append(row_idx); cols.append(col); vals.append(val)

            # ∂r_accel/∂b_a
            for dim in range(3):
                col = n_pos + n_ori + dim
                val = scale * J_ba[k, dim]
                if abs(val) > 1e-15:
                    rows.append(row_idx); cols.append(col); vals.append(val)

            row_idx += 1

    n_accel_rows = row_idx - n_radar
    
    # ==================== Gyroscope residuals ====================
    # Cumulative spline: r = z_gyro - omega(t) - b_g
    # J_omega = -I  (trivial), J_bg = -I
    sqrt_lambda_gyro = np.sqrt(lambda_gyro)

    for imu_msg in imu_data:
        t = imu_msg.timestamp
        t_rel_ori = t - state.ori_spline.t_ref

        z_gyro = imu_msg.angular_velocity

        _, omega, _, J_omega_list, active_ori = \
            state.ori_spline.evaluate_with_jacobians(t_rel_ori)

        res_gyro_3 = z_gyro - omega - state.gyr_bias  # (3,)

        for k in range(3):
            all_residuals.append(sqrt_lambda_gyro * res_gyro_3[k])

            # ∂r_gyro/∂Ω_j = -J_omega_list[j][k,:]
            for jj, knot_idx in enumerate(active_ori):
                J_wrt = -J_omega_list[jj][k, :]  # (3,)
                for dim in range(3):
                    col = n_pos + knot_idx * 3 + dim
                    val = sqrt_lambda_gyro * J_wrt[dim]
                    if abs(val) > 1e-15:
                        rows.append(row_idx); cols.append(col); vals.append(val)

            # ∂r_gyro/∂b_g = -I[k,:]
            col = n_pos + n_ori + 3 + k
            val = -sqrt_lambda_gyro
            rows.append(row_idx); cols.append(col); vals.append(val)

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
    
    # ==================== Boundary orientation priors ====================
    # Residual: so3_log(R_target^T @ R_est) penalizes rotation error.
    # Jacobian: ∂r/∂(right tangent) ≈ J_r(phi)^{-1} ≈ I (valid when prior is tight).
    n_boundary_ori_rows = 0
    if boundary_ori_priors and lambda_boundary_ori > 0:
        sqrt_lbo = np.sqrt(lambda_boundary_ori)
        for entry in boundary_ori_priors:
            # Support both old format (t_abs,) and new format (t_abs, R_target)
            if isinstance(entry, (tuple, list)):
                t_abs, R_target = entry
            else:
                # Legacy: entry is just t_abs; R_target comes from MoCap interpolation
                # stored externally — skip if not provided
                continue

            t_rel_ori = t_abs - state.ori_spline.t_ref
            R_est, _, J_R_list, _, active_ori = \
                state.ori_spline.evaluate_full_jacobians(t_rel_ori)
            r_ori = _so3_log_cs(R_target.T @ R_est)  # (3,)

            for k in range(3):
                all_residuals.append(sqrt_lbo * r_ori[k])
                # ∂r/∂Ω_j ≈ J_R_list[j][k,:]  (J_r^{-1} ≈ I for small residual)
                for jj, knot_idx in enumerate(active_ori):
                    J_wrt = J_R_list[jj][k, :]  # (3,)
                    for dim in range(3):
                        col = n_pos + knot_idx * 3 + dim
                        val = sqrt_lbo * J_wrt[dim]
                        if abs(val) > 1e-15:
                            rows.append(row_idx); cols.append(col); vals.append(val)
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
    
    # ==================== Boundary angular velocity priors ====================
    # Residual: omega_est - omega_target (anchors angular velocity at trajectory start).
    n_boundary_gyro_rows = 0
    if boundary_gyro_priors and lambda_boundary_gyro > 0:
        sqrt_lbg = np.sqrt(lambda_boundary_gyro)
        for entry in boundary_gyro_priors:
            if isinstance(entry, (tuple, list)):
                t_abs, omega_target = entry
            else:
                # Legacy: entry is just t_abs with implicit omega_target=0
                t_abs = entry
                omega_target = np.zeros(3)

            t_rel_ori = t_abs - state.ori_spline.t_ref
            _, omega_est, _, J_omega_list, active_ori = \
                state.ori_spline.evaluate_with_jacobians(t_rel_ori)
            r_gyro_bnd = omega_est - omega_target  # (3,)

            for k in range(3):
                all_residuals.append(sqrt_lbg * r_gyro_bnd[k])
                for jj, knot_idx in enumerate(active_ori):
                    J_wrt = J_omega_list[jj][k, :]  # (3,)
                    for dim in range(3):
                        col = n_pos + knot_idx * 3 + dim
                        val = sqrt_lbg * J_wrt[dim]
                        if abs(val) > 1e-15:
                            rows.append(row_idx); cols.append(col); vals.append(val)
                row_idx += 1
                n_boundary_gyro_rows += 1
    
    # ==================== Orientation increment regularization (penalize ||Ω_j||²) ====================
    # Encourages small relative rotations between consecutive virtual keyframes.
    n_ori_reg_rows = 0
    if lambda_ori_reg > 0:
        sqrt_lor = np.sqrt(lambda_ori_reg)
        for knot_idx in range(state.ori_spline.n_knots):
            omega_j = state.ori_spline.omega_knots[knot_idx]
            for k in range(3):
                all_residuals.append(sqrt_lor * omega_j[k])
                col = n_pos + knot_idx * 3 + k
                rows.append(row_idx); cols.append(col); vals.append(sqrt_lor)
                row_idx += 1
                n_ori_reg_rows += 1
    
    # ==================== Bias prior ====================
    # Penalize deviation from initial bias: r = sqrt(lambda) * (b[k] - b_init[k])
    # This anchors the solver near the stationary-detected bias instead of zero.
    # Separate weights for accel and gyro to allow different constraint strengths.
    # lambda_bias_prior_accel/gyro override lambda_bias_prior if set.
    bp_mean = bias_prior_mean if bias_prior_mean is not None else np.zeros(6)
    lbp_accel = lambda_bias_prior_accel if lambda_bias_prior_accel is not None else lambda_bias_prior
    lbp_gyro = lambda_bias_prior_gyro if lambda_bias_prior_gyro is not None else lambda_bias_prior
    n_bias_prior_rows = 0
    if lbp_accel > 0:
        sqrt_lbp_a = np.sqrt(lbp_accel)
        for k in range(3):
            all_residuals.append(sqrt_lbp_a * (state.acc_bias[k] - bp_mean[k]))
            rows.append(row_idx)
            cols.append(n_pos + n_ori + k)
            vals.append(sqrt_lbp_a)
            row_idx += 1
            n_bias_prior_rows += 1
    if lbp_gyro > 0:
        sqrt_lbp_g = np.sqrt(lbp_gyro)
        for k in range(3):
            all_residuals.append(sqrt_lbp_g * (state.gyr_bias[k] - bp_mean[3 + k]))
            rows.append(row_idx)
            cols.append(n_pos + n_ori + 3 + k)
            vals.append(sqrt_lbp_g)
            row_idx += 1
            n_bias_prior_rows += 1
    
    # ==================== Extrinsic Prior ====================
    n_extrinsic_prior_rows = 0
    prior_cost_extrinsic = 0.0
    if lambda_extrinsic_prior > 0:
        sqrt_lep = np.sqrt(lambda_extrinsic_prior)
        for k in range(3):
            r = state.radar_extrinsic_delta[k]
            all_residuals.append(sqrt_lep * r)
            prior_cost_extrinsic += lambda_extrinsic_prior * r * r
            rows.append(row_idx)
            cols.append(n_pos + n_ori + 6 + k)
            vals.append(sqrt_lep)
            row_idx += 1
            n_extrinsic_prior_rows += 1
    
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
    J.n_ori_reg = n_ori_reg_rows
    J.n_bias_prior = n_bias_prior_rows
    J.n_extrinsic_prior = n_extrinsic_prior_rows
    J.prior_cost_extrinsic = prior_cost_extrinsic
    
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
    relinearize_threshold_deg: float = 15.0,
    lambda_ori_reg: float = 0.0,
    lambda_bias_prior: float = 0.0,
    lambda_bias_prior_accel: float = None,
    lambda_bias_prior_gyro: float = None,
    bias_prior_mean: np.ndarray = None,
    lock_extrinsics: bool = False,
    optimize_pitch_only: bool = True,
    lambda_extrinsic_prior: float = 0.0,
    v_max: float = None,
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
        print(f"Max iterations: {max_iterations} (re-linearize when delta > {relinearize_threshold_deg}°)")
        print(f"Huber delta (radar): {huber_delta}")
        print(f"Huber delta (accel): {huber_delta_accel if huber_delta_accel > 0 else 'OFF (L2)'}")
        if v_max is not None:
            print(f"Doppler unwrapping: ON (v_max={v_max:.2f} m/s)")
        else:
            print(f"Doppler unwrapping: OFF")
        print(f"Lambda accel: {lambda_accel}")
        print(f"Lambda gyro: {lambda_gyro}")
        print(f"Lambda snap pos: {lambda_snap_pos}")
        print(f"Lambda snap ori: {lambda_snap_ori}")
        if lock_biases:
            print(f"*** BIASES LOCKED TO INITIAL VALUES ***")
        if lock_extrinsics:
            print(f"*** EXTRINSICS LOCKED TO INITIAL VALUES ***")
        if optimize_pitch_only:
            print(f"*** EXTRINSICS: OPTIMIZING PITCH ONLY ***")
        if lambda_extrinsic_prior > 0:
            print(f"Extrinsic prior strength: {lambda_extrinsic_prior}")
        if bias_prior_mean is not None:
            print(f"Bias prior mean: acc=[{bias_prior_mean[0]:.4f}, {bias_prior_mean[1]:.4f}, {bias_prior_mean[2]:.4f}]  gyr=[{bias_prior_mean[3]:.5f}, {bias_prior_mean[4]:.5f}, {bias_prior_mean[5]:.5f}]")
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

    # Separate regularization matrices for position and orientation blocks
    n_pos = state.pos_bspline.n_points * 3
    n_ori = state.ori_spline.n_knots * 3
    n_total = state.get_state_size()

    # Position minimum-snap only; orientation uses increment regularization via lambda_ori_reg
    R_snap = sparse.lil_matrix((n_total, n_total))
    R_snap[:n_pos, :n_pos] = lambda_snap_pos * (R_snap_pos.T @ R_snap_pos)
    R_snap = R_snap.tocsr()
    
    # Helper: build Jacobian with all params
    # Accel warm-up: first 5 iterations use lambda_accel=0 (radar+gyro only for orientation),
    # then ramp to full lambda_accel so the accelerometer doesn't corrupt orientation early.
    ACCEL_WARMUP_ITERS = 0
    
    def _build_jacobian(st, iteration_idx=None):
        la = lambda_accel
        if iteration_idx is not None and iteration_idx < ACCEL_WARMUP_ITERS:
            la = 0.0
        return compute_jacobian_analytical(
            st, radar_frames, imu_data,
            sensor_translation, sensor_rotation,
            la, lambda_gyro, huber_delta,
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
            lambda_ori_reg=lambda_ori_reg,
            lambda_bias_prior=lambda_bias_prior,
            lambda_bias_prior_accel=lambda_bias_prior_accel,
            lambda_bias_prior_gyro=lambda_bias_prior_gyro,
            bias_prior_mean=bias_prior_mean,
            lock_extrinsics=lock_extrinsics,
            optimize_pitch_only=optimize_pitch_only,
            lambda_extrinsic_prior=lambda_extrinsic_prior,
            v_max=v_max,
        )

    lambda_lm = 1e-3
    prev_cost = None
    n_consecutive_rejects = 0
    MAX_CONSECUTIVE_REJECTS = 3

    for iteration in range(max_iterations):
        # Reset LM state when transitioning from warmup to full accel
        if iteration == ACCEL_WARMUP_ITERS:
            prev_cost = None
            lambda_lm = 1e-3
            n_consecutive_rejects = 0
            if verbose:
                print(f"\n  >>> Accel warm-up complete. Enabling accelerometer (lambda_accel={lambda_accel})")
        
        if verbose:
            print(f"\n{'Iteration ' + str(iteration + 1):-^80}")
            # Track orientation RMSE per iteration
            if mocap_times_abs is not None and mocap_rotations is not None:
                ori_rmse = compute_orientation_rmse(state, mocap_times_abs, mocap_rotations)
                omega_norms = np.linalg.norm(state.ori_spline.omega_knots, axis=1)
                print(f"Orientation RMSE: {ori_rmse:.1f} deg | |Ω| max: {np.degrees(omega_norms.max()):.2f}° mean: {np.degrees(omega_norms.mean()):.2f}°")
                print(f"Acc bias: [{state.acc_bias[0]:.3f}, {state.acc_bias[1]:.3f}, {state.acc_bias[2]:.3f}]"
                      f"  Gyr bias: [{state.gyr_bias[0]:.3f}, {state.gyr_bias[1]:.3f}, {state.gyr_bias[2]:.3f}]")
        
        t_start = time.time()
        
        # Build Jacobian + residual vector analytically (includes all costs)
        if verbose:
            if iteration < ACCEL_WARMUP_ITERS:
                print(f"Computing analytical Jacobian... [ACCEL WARMUP: off, iter {iteration+1}/{ACCEL_WARMUP_ITERS}]")
            else:
                print("Computing analytical Jacobian...")
        
        J, r_total = _build_jacobian(state, iteration_idx=iteration)
        
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
            n_or = getattr(J, 'n_ori_reg', 0)
            n_bp_prior = getattr(J, 'n_bias_prior', 0)
            n_g = len(r_total) - n_r - n_a - n_bv - n_bp - n_bo - n_bac - n_bg - n_or - n_bp_prior
            idx = 0
            cost_radar = np.sum(r_total[idx:idx+n_r]**2); idx += n_r
            cost_accel = np.sum(r_total[idx:idx+n_a]**2); idx += n_a
            cost_gyro = np.sum(r_total[idx:idx+n_g]**2); idx += n_g
            cost_bv = np.sum(r_total[idx:idx+n_bv]**2); idx += n_bv
            cost_bp = np.sum(r_total[idx:idx+n_bp]**2); idx += n_bp
            cost_bo = np.sum(r_total[idx:idx+n_bo]**2); idx += n_bo
            cost_bac = np.sum(r_total[idx:idx+n_bac]**2); idx += n_bac
            cost_bg = np.sum(r_total[idx:idx+n_bg]**2); idx += n_bg
            cost_or = np.sum(r_total[idx:idx+n_or]**2); idx += n_or
            cost_bp_prior = np.sum(r_total[idx:idx+n_bp_prior]**2); idx += n_bp_prior
            n_ep = getattr(J, 'n_extrinsic_prior', 0)
            cost_ep = np.sum(r_total[idx:idx+n_ep]**2) if n_ep > 0 else 0.0; idx += n_ep
            
            bnd_parts = []
            if n_bv > 0: bnd_parts.append(f"bnd_vel={cost_bv:.1f}")
            if n_bp > 0: bnd_parts.append(f"bnd_pos={cost_bp:.1f}")
            if n_bo > 0: bnd_parts.append(f"bnd_ori={cost_bo:.1f}")
            if n_bac > 0: bnd_parts.append(f"bnd_acc={cost_bac:.1f}")
            if n_bg > 0: bnd_parts.append(f"bnd_gyr={cost_bg:.1f}")
            if n_or > 0: bnd_parts.append(f"ori_reg={cost_or:.1f}")
            if n_bp_prior > 0: bnd_parts.append(f"bias_prior={cost_bp_prior:.1f}")
            if n_ep > 0: bnd_parts.append(f"ext_prior={cost_ep:.1f}")
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
                print("[WARN] Solver raised exception, increasing damping...")
            lambda_lm *= 10
            continue

        # spsolve returns NaN for singular H without raising an Exception (MatrixRankWarning).
        # Catch it here before NaN can corrupt the state.
        if not np.isfinite(delta_x).all():
            if verbose:
                print(f"[WARN] Solver returned NaN/Inf (singular H, damping={lambda_lm:.2e}), increasing damping...")
            lambda_lm = max(lambda_lm * 100, 1e-4)
            continue

        # Try update
        x_current = state.to_vector()

        # Zero out updates for locked components
        if lock_biases:
            delta_x[n_pos + n_ori : n_pos + n_ori + 6] = 0.0
        if lock_extrinsics:
            delta_x[n_pos + n_ori + 6 : n_pos + n_ori + 9] = 0.0
        elif optimize_pitch_only:
            # Zero out roll (x) and yaw (z) updates to only calibrate pitch (y)
            delta_x[n_pos + n_ori + 6] = 0.0
            delta_x[n_pos + n_ori + 8] = 0.0
            
        x_new = x_current + delta_x
        state.from_vector(x_new)
        
        # Evaluate new cost
        _, r_new = _build_jacobian(state, iteration_idx=iteration)
        new_cost = np.sum(r_new**2)
        
        # LM acceptance: accept only if cost decreased
        if new_cost < cost_total:
            lambda_lm = max(1e-6, lambda_lm * 0.1)
            prev_cost = new_cost
            n_consecutive_rejects = 0
            if verbose:
                omega_norms = np.linalg.norm(state.ori_spline.omega_knots, axis=1)
                print(f"  Accepted: cost {cost_total:.1f} -> {new_cost:.1f} "
                      f"(max |Ω|={np.degrees(omega_norms.max()):.1f}°)")
        else:
            # Reject and increase damping
            state.from_vector(x_current)
            lambda_lm *= 10.0  # Strong punishment for bad step
            n_consecutive_rejects += 1
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
                print("\n[OK] Converged (delta_norm < 1e-4)!")
            break

        if n_consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
            if verbose:
                print(f"\n[OK] Early stop: {MAX_CONSECUTIVE_REJECTS} consecutive rejected steps.")
            break
    
    if verbose:
        print(f"\n{'Optimization Complete':#^80}")
    
    return state


# ==================== Stationary Bias Detection ====================

def detect_stationary_bias(
    imu_data_full,
    window_sec: float = 1.0,
    min_stationary_sec: float = 2.0,
    accel_std_threshold: float = 0.15,
    gyro_std_threshold: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Variance-based stationary detection for IMU bias initialization.

    Scans the FULL bag IMU data with a rolling window. When the standard
    deviation of both accelerometer and gyroscope stays below tight thresholds
    for at least `min_stationary_sec`, that window is flagged as stationary.
    The gyro and accel means from that window give the bias estimate.

    For accelerometer, the bias is computed as:
        b_a = mean(z_accel) - [0, 0, +9.81]  (gravity in NED→FLU body frame
        when the drone is level and z-up)

    Returns dict with keys:
        'gyr_bias': (3,) mean gyro during stationary period
        'acc_bias': (3,) accel bias (mean - gravity)
        'stationary_start': float timestamp
        'stationary_end': float timestamp
        'n_samples': int
    """
    if len(imu_data_full) < 10:
        if verbose:
            print("  [WARN] Not enough IMU data for stationary detection")
        return None

    times = np.array([d.timestamp for d in imu_data_full])
    accels = np.array([d.linear_acceleration for d in imu_data_full])
    gyros = np.array([d.angular_velocity for d in imu_data_full])

    # Estimate sample rate
    dt_median = np.median(np.diff(times))
    fs = 1.0 / dt_median
    win_samples = max(10, int(window_sec * fs))

    if verbose:
        print(f"  Stationary detection: {len(times)} samples, fs={fs:.0f} Hz, window={win_samples} samples ({window_sec}s)")

    # Rolling standard deviation (per-axis, take max across axes)
    n = len(times)
    best_start = None
    best_end = None
    best_length = 0

    # Sliding window: compute std for each window position
    current_start = None
    for i in range(0, n - win_samples, win_samples // 2):  # 50% overlap steps
        j = i + win_samples
        accel_window = accels[i:j]
        gyro_window = gyros[i:j]

        accel_std = np.std(accel_window, axis=0)
        gyro_std = np.std(gyro_window, axis=0)

        is_stationary = (np.max(accel_std) < accel_std_threshold and
                         np.max(gyro_std) < gyro_std_threshold)

        if is_stationary:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                # End of stationary block
                block_duration = times[i] - times[current_start]
                if block_duration > best_length:
                    best_length = block_duration
                    best_start = current_start
                    best_end = i
                current_start = None

    # Handle case where stationary extends to end of data
    if current_start is not None:
        block_duration = times[n - 1] - times[current_start]
        if block_duration > best_length:
            best_length = block_duration
            best_start = current_start
            best_end = n - 1

    if best_start is None or best_length < min_stationary_sec:
        if verbose:
            print(f"  [WARN] No stationary period >= {min_stationary_sec}s found (best: {best_length:.1f}s)")
        return None

    # Extract bias from the validated stationary window
    stat_accels = accels[best_start:best_end]
    stat_gyros = gyros[best_start:best_end]

    gyr_bias = np.mean(stat_gyros, axis=0)
    # Accel bias: subtract expected gravity vector (assuming drone is level, z-up)
    # In a level IMU frame, gravity reads as [0, 0, +9.81] (z-up convention FLU)
    # We detect the actual gravity direction from the mean accel during stationary
    gravity_measured = np.mean(stat_accels, axis=0)
    gravity_norm = np.linalg.norm(gravity_measured)
    acc_bias = gravity_measured - (gravity_measured / gravity_norm) * 9.81

    if verbose:
        print(f"  Stationary period: {times[best_start] - times[0]:.1f}s to {times[best_end] - times[0]:.1f}s ({best_length:.1f}s, {best_end - best_start} samples)")
        print(f"  Accel mean (stationary): [{gravity_measured[0]:.4f}, {gravity_measured[1]:.4f}, {gravity_measured[2]:.4f}] m/s² (|a|={gravity_norm:.4f})")
        print(f"  Accel std  (stationary): [{np.std(stat_accels[:,0]):.4f}, {np.std(stat_accels[:,1]):.4f}, {np.std(stat_accels[:,2]):.4f}] m/s²")
        print(f"  Gyro  mean (stationary): [{gyr_bias[0]:.5f}, {gyr_bias[1]:.5f}, {gyr_bias[2]:.5f}] rad/s")
        print(f"         = [{np.degrees(gyr_bias[0]):.3f}, {np.degrees(gyr_bias[1]):.3f}, {np.degrees(gyr_bias[2]):.3f}] deg/s")
        print(f"  Gyro  std  (stationary): [{np.std(stat_gyros[:,0]):.5f}, {np.std(stat_gyros[:,1]):.5f}, {np.std(stat_gyros[:,2]):.5f}] rad/s")
        print(f"  Accel bias (mean - g):   [{acc_bias[0]:.4f}, {acc_bias[1]:.4f}, {acc_bias[2]:.4f}] m/s²")

    return {
        'gyr_bias': gyr_bias,
        'acc_bias': acc_bias,
        'stationary_start': times[best_start],
        'stationary_end': times[best_end],
        'n_samples': best_end - best_start,
    }


# ==================== Bag Catalogue (loaded from config/bags.yaml) ====================
_cfg = load_config()
BAGS = _cfg['bags']['bags']
FLIPPED_BAGS = set(_cfg['bags']['flipped'])
_BAG_TIMING_CFG = _cfg['bags']['timing']
_RADAR_CFG = _cfg['bags'].get('radar_config', {})
_EXTRINSICS_CFG = _cfg['extrinsics']
_SOLVER_CFG = _cfg['solver']
del _cfg

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

    # Per-bag flight windows (from config/bags.yaml timing section)
    if bag_key in _BAG_TIMING_CFG:
        START_TIME_OFFSET, DURATION = _BAG_TIMING_CFG[bag_key]
    else:
        START_TIME_OFFSET = 30.0
        DURATION = 5.0

    # Sensor extrinsics (from config/extrinsics.yaml — confirmed correct values)
    ROTATION_EULER_DEG = np.array(_EXTRINSICS_CFG['rotation_euler_deg'])  # [180, 30, 0]
    _t_base = np.array(_EXTRINSICS_CFG['translation_body_m'])              # [0, 0.02, -0.01]
    IMU_MOCAP_OFFSET = _EXTRINSICS_CFG['imu_mocap_offset_sec']
    RADAR_IMU_OFFSET = _EXTRINSICS_CFG['radar_imu_offset_sec']

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
    R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)
    if FLIP_BODY_FRAME:
        TRANSLATION = R_yaw_flip @ _t_base
        SENSOR_ROTATION = R_yaw_flip @ R_base
        print(f"  Body frame FLIPPED (R_z(180 deg) applied) for bag '{bag_key}'")
    else:
        TRANSLATION = _t_base.copy()
        SENSOR_ROTATION = R_base

    BSPLINE_DEGREE = _SOLVER_CFG['bspline_degree']
    DT_POS = _SOLVER_CFG['dt_pos']
    DT_ORI = _SOLVER_CFG['dt_ori']

    # Regularization weights (from config/solver.yaml)
    LAMBDA_ACCEL = _SOLVER_CFG['lambda_accel']
    LAMBDA_GYRO = _SOLVER_CFG['lambda_gyro']
    LAMBDA_SNAP_POS = _SOLVER_CFG['lambda_snap_pos']
    LAMBDA_SNAP_ORI = _SOLVER_CFG['lambda_snap_ori']
    LAMBDA_ORI_REG = _SOLVER_CFG['lambda_ori_reg']
    LAMBDA_BIAS_PRIOR = _SOLVER_CFG['lambda_bias_prior']
    LAMBDA_BIAS_PRIOR_ACCEL = _SOLVER_CFG['lambda_bias_prior_accel']
    LAMBDA_BIAS_PRIOR_GYRO = _SOLVER_CFG['lambda_bias_prior_gyro']
    HUBER_DELTA = _SOLVER_CFG['huber_delta']
    HUBER_DELTA_ACCEL = _SOLVER_CFG['huber_delta_accel']
    MIN_RANGE = _SOLVER_CFG['min_range']
    MAX_ITERATIONS = _SOLVER_CFG['max_iterations']
    USE_PHASE2_INIT = _SOLVER_CFG['use_phase2_init']
    USE_STATIONARY_BIAS = _SOLVER_CFG['use_stationary_bias']
    LOCK_BIASES = _SOLVER_CFG['lock_biases']
    LOCK_EXTRINSICS = _SOLVER_CFG['lock_extrinsics']
    OPTIMIZE_PITCH_ONLY = _SOLVER_CFG['optimize_pitch_only']
    LAMBDA_EXTRINSIC_PRIOR = _SOLVER_CFG['lambda_extrinsic_prior']
    RELINEARIZE_THRESHOLD_DEG = _SOLVER_CFG['relinearize_threshold_deg']
    USE_JACOBI_PRECOND = '--precond' in sys.argv  # CLI override
    NO_RADAR = '--no-radar' in sys.argv  # CLI override
    NO_UNWRAP = '--no-unwrap' in sys.argv  # CLI override: disable Doppler unwrapping
    USE_UNWRAP = not NO_UNWRAP

    # Boundary priors: pin spline state at START to MoCap ground truth (no end priors)
    LAMBDA_BOUNDARY_VEL = _SOLVER_CFG['lambda_boundary_vel']
    LAMBDA_BOUNDARY_POS = _SOLVER_CFG['lambda_boundary_pos']
    LAMBDA_BOUNDARY_ORI = _SOLVER_CFG['lambda_boundary_ori']
    LAMBDA_BOUNDARY_ACCEL = _SOLVER_CFG['lambda_boundary_accel']
    LAMBDA_BOUNDARY_GYRO = _SOLVER_CFG['lambda_boundary_gyro']
    BOUNDARY_WINDOW = _SOLVER_CFG['boundary_window']
    
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
    print(f"Max iterations: {MAX_ITERATIONS} (re-linearize when delta > {RELINEARIZE_THRESHOLD_DEG}°)")
    print(f"IMU-MoCap time offset: {IMU_MOCAP_OFFSET*1000:.0f} ms  |  Radar-IMU offset: {RADAR_IMU_OFFSET*1000:.0f} ms")
    print(f"Use Phase 2 init: {USE_PHASE2_INIT}")
    print(f"Stationary bias: {USE_STATIONARY_BIAS}  |  Lock biases: {LOCK_BIASES}")
    print(f"Extrinsics lock: {LOCK_EXTRINSICS} | Optimize pitch only: {OPTIMIZE_PITCH_ONLY} | Extrinsics prior λ={LAMBDA_EXTRINSIC_PRIOR}")
    print(f"Jacobi preconditioning: {USE_JACOBI_PRECOND}")
    if NO_RADAR:
        print(f"*** RADAR DISABLED (--no-radar) — IMU/MoCap-only trajectory ***")
    print(f"Boundary priors (START only): window={BOUNDARY_WINDOW}s")
    print(f"  λ_bnd: vel={LAMBDA_BOUNDARY_VEL} pos={LAMBDA_BOUNDARY_POS} ori={LAMBDA_BOUNDARY_ORI} acc={LAMBDA_BOUNDARY_ACCEL} gyr={LAMBDA_BOUNDARY_GYRO}")
    print(f"  λ_ori_reg: {LAMBDA_ORI_REG}  λ_bias_prior: accel={LAMBDA_BIAS_PRIOR_ACCEL} gyro={LAMBDA_BIAS_PRIOR_GYRO}")
    
    # ==================== Load Data ====================
    print(f"\n{'Loading Data':-^80}")
    
    bag_data = load_bag_topics(BAG_PATH, verbose=True)
    
    t_start = bag_data.start_time + START_TIME_OFFSET
    t_end = t_start + DURATION
    
    agiros_states = [s for s in bag_data.agiros_state if t_start <= s.timestamp <= t_end]
    radar_frames = [f for f in bag_data.radar_velocity if t_start <= f.timestamp <= t_end]
    imu_data = [d for d in bag_data.imu_data if t_start <= d.timestamp <= t_end]
    
    # Apply time offsets to align all sensors with MoCap time axis
    # IMU is 20ms behind MoCap (FINDINGS.md §3)
    # Radar is 19ms behind IMU (notebook 02_radar_time_sync §4), so 39ms behind MoCap total
    if IMU_MOCAP_OFFSET != 0:
        for d in imu_data:
            d.timestamp += IMU_MOCAP_OFFSET
        print(f"  Applied IMU-MoCap time offset: {IMU_MOCAP_OFFSET*1000:.0f} ms to {len(imu_data)} IMU samples")
    radar_total_offset = IMU_MOCAP_OFFSET - RADAR_IMU_OFFSET  # 0.020 - 0.140 = -0.120 s (USB radar latency)
    for f in radar_frames:
        f.timestamp += radar_total_offset
    print(f"  Applied radar time offset: {radar_total_offset*1000:.0f} ms to {len(radar_frames)} radar samples (IMU:{IMU_MOCAP_OFFSET*1000:.0f} + radar-IMU:{-RADAR_IMU_OFFSET*1000:.0f})")
    
    # Filter near-duplicate MoCap timestamps (dt < 1ms) that cause interpolation spikes
    # See FINDINGS.md Section 8: MoCap has samples with dt ~5us causing vel/accel spikes
    n_before = len(agiros_states)
    filtered_agiros = [agiros_states[0]]
    for i in range(1, len(agiros_states)):
        if agiros_states[i].timestamp - filtered_agiros[-1].timestamp >= 1e-3:
            filtered_agiros.append(agiros_states[i])
    agiros_states = filtered_agiros
    n_removed = n_before - len(agiros_states)
    
    print(f"\nFiltered data:")
    print(f"  MoCap states: {len(agiros_states)} ({n_removed} near-duplicate timestamps removed)")
    print(f"  Radar frames: {len(radar_frames)}")
    print(f"  IMU samples: {len(imu_data)}")
    
    if len(agiros_states) == 0 or len(radar_frames) == 0 or len(imu_data) == 0:
        print("ERROR: Insufficient data!")
        return
    
    # Aliasing risk check (uses MoCap ground truth to detect wrapped Doppler)
    # Per-bag v_max: bags with "best_velocity" in name use 3.84 m/s config
    _rc = _RADAR_CFG.get('best_velocity' if 'best_velocity' in bag_key else 'default', {})
    V_MAX = _rc.get('v_max', 4.99)  # ±m/s unambiguous velocity (from config/bags.yaml)
    aliasing_info = compute_aliasing_summary(
        agiros_states, radar_frames,
        TRANSLATION, SENSOR_ROTATION,
        v_max=V_MAX, min_range=MIN_RANGE,
    )
    
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
    ori_degree = min(3, BSPLINE_DEGREE)  # Cubic for orientation (higher degrees destabilize)
    n_interior_ori = int(np.ceil(DURATION / DT_ORI)) + 1
    n_ori_points = max(ori_degree + 2, n_interior_ori + 2 * BOUNDARY_ORDER)
    
    # Enforce quaternion hemisphere consistency BEFORE any interpolation.
    n_flips = 0
    for i in range(1, len(mocap_orientations)):
        if np.dot(mocap_orientations[i], mocap_orientations[i - 1]) < 0:
            mocap_orientations[i] *= -1
            n_flips += 1
    if n_flips > 0:
        print(f"  Fixed {n_flips} quaternion sign flips for hemisphere consistency")

    # Create dense SLERP interpolation from MoCap
    mocap_rots_scipy = Rotation.from_quat(mocap_orientations)  # [qx, qy, qz, qw]
    mocap_slerp = Slerp(mocap_times_rel, mocap_rots_scipy)

    # Sample rotations at control knot times for cumulative spline initialization
    knot_times = np.linspace(0.0, (n_ori_points - 1) * DT_ORI, n_ori_points)
    # A temporary dummy spline to get the valid t_start
    _dummy_bspline = UniformBSpline(np.zeros((n_ori_points, 3)), ori_degree, DT_ORI)
    knot_times_abs_rel = knot_times  # relative to B-spline t=0

    # Sample MoCap rotations at knot times (clamped to MoCap data range)
    # The B-spline time axis starts at 0; t_ref maps to absolute time.
    mocap_sample_times = np.clip(knot_times_abs_rel, mocap_times_rel[0], mocap_times_rel[-1])
    R_knot_samples = np.array([mocap_slerp(t).as_matrix() for t in mocap_sample_times])

    # Initialize cumulative SO(3) spline from MoCap rotation samples
    ori_spline = CumulativeSO3BSpline.from_rotation_samples(R_knot_samples, dt=DT_ORI, t_ref=t_ref)

    # Also keep mocap_slerp for bias fallback and boundary prior construction
    mocap_angular_velocities = np.array([s.angular_velocity for s in agiros_states])
    mocap_omega_interp = interp1d(mocap_times_rel, mocap_angular_velocities, axis=0,
                                   kind='cubic', fill_value='extrapolate')

    print(f"Orientation spline (cumulative SO3): {n_ori_points} knots, dt={DT_ORI:.4f}s")
    print(f"  Initialized from {len(mocap_times_rel)} MoCap samples ({1.0/(mocap_times_rel[1]-mocap_times_rel[0]):.0f} Hz)")
    
    # ==================== Bias Initialization ====================
    # Use variance-based stationary detection on the FULL bag IMU data
    # to extract gyro/accel bias from the ground period (before takeoff).
    g_world = np.array([0, 0, -9.81])
    print(f"\n{'Stationary Bias Detection':-^80}")
    if USE_STATIONARY_BIAS:
        stationary_result = detect_stationary_bias(bag_data.imu_data, verbose=True)
    else:
        stationary_result = None
        print("  Stationary detection DISABLED (USE_STATIONARY_BIAS=False)")

    if USE_STATIONARY_BIAS and stationary_result is not None:
        acc_bias = stationary_result['acc_bias']
        gyr_bias = stationary_result['gyr_bias']
        print(f"\n  Using STATIONARY-DETECTED biases (autonomous, no MoCap needed)")
    elif not USE_STATIONARY_BIAS:
        acc_bias = np.zeros(3)
        gyr_bias = np.zeros(3)
        print(f"\n  Biases initialized to ZERO (stationary detection disabled)")
    else:
        # Fallback: estimate from flight window (MoCap-based, less reliable)
        print(f"\n  [FALLBACK] Stationary detection failed, estimating from flight window...")
        accel_residuals_init = []
        gyro_residuals_init = []
        for imu_msg in imu_data:
            t = imu_msg.timestamp
            t_rel = t - t_ref
            if t_rel < pos_bspline.t_start or t_rel > pos_bspline.t_end:
                continue
            try:
                a_world = pos_bspline(t_rel, derivative=2)
                R_mocap = mocap_slerp(np.clip(t_rel, mocap_times_rel[0], mocap_times_rel[-1])).as_matrix()
                pred_imu = R_mocap.T @ (a_world - g_world)
                accel_residuals_init.append(imu_msg.linear_acceleration - pred_imu)
                t_clamped = np.clip(t_rel, mocap_times_rel[0], mocap_times_rel[-1])
                omega_mocap = mocap_omega_interp(t_clamped)
                gyro_residuals_init.append(imu_msg.angular_velocity - omega_mocap)
            except Exception:
                pass
        acc_bias = np.mean(accel_residuals_init, axis=0) if accel_residuals_init else np.zeros(3)
        gyr_bias = np.mean(gyro_residuals_init, axis=0) if gyro_residuals_init else np.zeros(3)

    # bias_prior_mean: the bias prior penalizes (b - b_init), anchoring to this value
    bias_prior_mean = np.concatenate([acc_bias, gyr_bias])

    print(f"\n  Bias initialization:")
    print(f"    Acc bias init: [{acc_bias[0]:.4f}, {acc_bias[1]:.4f}, {acc_bias[2]:.4f}] m/s² (norm={np.linalg.norm(acc_bias):.4f})")
    print(f"    Gyr bias init: [{gyr_bias[0]:.5f}, {gyr_bias[1]:.5f}, {gyr_bias[2]:.5f}] rad/s")
    print(f"                 = [{np.degrees(gyr_bias[0]):.3f}, {np.degrees(gyr_bias[1]):.3f}, {np.degrees(gyr_bias[2]):.3f}] deg/s")
    print(f"    Lock biases: {LOCK_BIASES} (if True, solver cannot change these values)")
    
    # Create initial state
    initial_state = TrajectoryState(
        pos_bspline=pos_bspline,
        ori_spline=ori_spline,
        acc_bias=acc_bias,
        gyr_bias=gyr_bias,
        radar_extrinsic_delta=np.zeros(3),
    )
    
    print(f"Total state variables: {initial_state.get_state_size()}")
    print(f"  Position: {n_pos_points * 3}")
    print(f"  Orientation (Ω knots): {ori_spline.n_knots * 3}")
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
    boundary_ori_priors = []    # list of (t_abs, R_target)
    boundary_accel_priors = []  # list of (t_abs, a_target)
    boundary_gyro_priors = []   # list of (t_abs, omega_target)
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
        # Orientation prior: R_est should match MoCap
        t_clamped = np.clip(t_rel, mocap_times_rel[0], mocap_times_rel[-1])
        R_gt = mocap_slerp(t_clamped).as_matrix()
        boundary_ori_priors.append((t_abs, R_gt))
        # Acceleration prior
        a_gt = mocap_accel_interp(t_rel)
        boundary_accel_priors.append((t_abs, a_gt))
        # Angular velocity prior: omega_est should match MoCap omega
        omega_gt = mocap_omega_interp(t_clamped)
        boundary_gyro_priors.append((t_abs, omega_gt))
    
    print(f"\n  Boundary priors (START only): {n_boundary_samples} sample points")
    print(f"    vel: {len(boundary_vel_priors)}, pos: {len(boundary_pos_priors)}, ori: {len(boundary_ori_priors)}, acc: {len(boundary_accel_priors)}, gyr: {len(boundary_gyro_priors)}")
    if len(boundary_vel_priors) > 0:
        v0 = boundary_vel_priors[0][1]
        print(f"    Start vel GT: [{v0[0]:.2f}, {v0[1]:.2f}, {v0[2]:.2f}] m/s (|v|={np.linalg.norm(v0):.2f})")
    
    # ==================== Test 1: --no-radar flag ====================
    solver_radar_frames = radar_frames
    if NO_RADAR:
        print(f"\n  *** --no-radar: Disabling all {len(radar_frames)} radar frames ***")
        solver_radar_frames = []

    # ==================== Test 3: Initial residual statistics ====================
    print(f"\n{'Initial Radar Residual Statistics (at MoCap GT)':=^80}")
    n_huber_suppressed = 0
    n_total_pts = 0
    init_residuals = []
    _zeros3_init = np.zeros(3)
    for frame in radar_frames:
        t = frame.timestamp
        try:
            v_world = initial_state.get_position(t, derivative=1)
            t_rel_ori = t - initial_state.ori_spline.t_ref
            R_full, omega, _, _, _ = initial_state.ori_spline.evaluate_with_jacobians(t_rel_ori)
            R_nom_quat = Rot3.from_rotation_matrix(R_full)
            R_bs_quat = Rot3.from_rotation_matrix(sensor_rotation)
        except Exception:
            continue

        for i in range(frame.num_points()):
            p_s = frame.positions[i]
            range_val = frame.ranges[i] if frame.ranges is not None else np.linalg.norm(p_s)
            if range_val < MIN_RANGE:
                continue
            u_sensor = p_s / np.linalg.norm(p_s)
            v_meas = frame.velocities[i]

            res, _, _, _, _ = radar_residual_with_jacobians(
                v_world, R_nom_quat, _zeros3_init, omega,
                u_sensor, TRANSLATION, R_bs_quat,
                v_meas, 1e-10
            )
            init_residuals.append(res[0])
            n_total_pts += 1
            if abs(res[0]) > HUBER_DELTA:
                n_huber_suppressed += 1

    init_residuals = np.array(init_residuals) if init_residuals else np.array([0.0])
    print(f"  Total radar points: {n_total_pts}")
    print(f"  Mean residual:      {init_residuals.mean():+.4f} m/s")
    print(f"  Std residual:       {init_residuals.std():.4f} m/s")
    print(f"  Median |residual|:  {np.median(np.abs(init_residuals)):.4f} m/s")
    print(f"  Max |residual|:     {np.abs(init_residuals).max():.4f} m/s")
    print(f"  Huber-suppressed:   {n_huber_suppressed}/{n_total_pts} ({100*n_huber_suppressed/max(1,n_total_pts):.1f}%) with |r| > {HUBER_DELTA} m/s")
    if abs(init_residuals.mean()) > 0.5:
        print(f"  WARNING: Large mean residual suggests sign convention mismatch!")

    optimized_state = solve_trajectory_nonlinear(
        initial_state=initial_state,
        radar_frames=solver_radar_frames,
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
        relinearize_threshold_deg=RELINEARIZE_THRESHOLD_DEG,
        lambda_ori_reg=LAMBDA_ORI_REG,
        lambda_bias_prior=LAMBDA_BIAS_PRIOR,
        lambda_bias_prior_accel=LAMBDA_BIAS_PRIOR_ACCEL,
        lambda_bias_prior_gyro=LAMBDA_BIAS_PRIOR_GYRO,
        bias_prior_mean=bias_prior_mean,
        lock_extrinsics=LOCK_EXTRINSICS,
        optimize_pitch_only=OPTIMIZE_PITCH_ONLY,
        lambda_extrinsic_prior=LAMBDA_EXTRINSIC_PRIOR,
        v_max=V_MAX if USE_UNWRAP else None,
    )
    
    # ==================== Evaluate Results ====================
    print(f"\n{'Evaluating Results':-^80}")
    
    # Print calibrated radar extrinsics
    calibrated_R_bs = Rot3.from_rotation_matrix(sensor_rotation @ so3_exp(optimized_state.radar_extrinsic_delta))
    calibrated_euler = np.degrees(Rotation.from_quat(calibrated_R_bs.data).as_euler('xyz'))
    print(f"  Initial Radar Extrinsics (Euler xyz): [{ROTATION_EULER_DEG[0]:.2f}, {ROTATION_EULER_DEG[1]:.2f}, {ROTATION_EULER_DEG[2]:.2f}] deg")
    print(f"  Optimized Radar Extrinsics (Euler xyz): [{calibrated_euler[0]:.2f}, {calibrated_euler[1]:.2f}, {calibrated_euler[2]:.2f}] deg")
    delta_deg = np.degrees(optimized_state.radar_extrinsic_delta)
    print(f"  Extrinsic Delta (Tangent xyz): [{delta_deg[0]:.2f}, {delta_deg[1]:.2f}, {delta_deg[2]:.2f}] deg")
    
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
    
    # Lowpass filter MoCap velocity (filtfilt, 4th-order Butterworth, 10 Hz cutoff)
    # MoCap runs at ~300 Hz; we want to remove high-freq noise before error/plotting.
    from scipy.signal import butter, filtfilt, savgol_filter
    _eval_dt = np.median(np.diff(eval_times))
    _fs = 1.0 / _eval_dt
    _fc = min(10.0, _fs * 0.4)  # cutoff 10 Hz (or 40% Nyquist if fs is low)
    _b, _a = butter(4, _fc / (_fs / 2), btype='low')
    if len(mocap_velocities) > 3 * 9:  # need enough samples for filtfilt (3 * padlen)
        for dim in range(3):
            mocap_velocities[:, dim] = filtfilt(_b, _a, mocap_velocities[:, dim])

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
    win = min(15, len(clean_accel) - (1 if len(clean_accel) % 2 == 0 else 0))
    if win >= 5:
        for dim in range(3):
            clean_accel[:, dim] = savgol_filter(clean_accel[:, dim], win, 3)
    
    # Interpolate back to evaluation timestamps
    mocap_accelerations = np.zeros_like(mocap_velocities)
    for dim in range(3):
        mocap_accelerations[:, dim] = np.interp(eval_times, clean_times, clean_accel[:, dim])
    
    # Lowpass filter MoCap acceleration (same filter as velocity)
    if len(mocap_accelerations) > 3 * 9:
        for dim in range(3):
            mocap_accelerations[:, dim] = filtfilt(_b, _a, mocap_accelerations[:, dim])
    
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
    vel_diff = estimated_velocities - mocap_velocities
    print(f"  Per-axis RMSE: x={np.sqrt(np.mean(vel_diff[:,0]**2)):.3f}  y={np.sqrt(np.mean(vel_diff[:,1]**2)):.3f}  z={np.sqrt(np.mean(vel_diff[:,2]**2)):.3f}")
    print(f"  Per-axis mean:  x={np.mean(vel_diff[:,0]):.3f}  y={np.mean(vel_diff[:,1]):.3f}  z={np.mean(vel_diff[:,2]):.3f}")
    
    print(f"\nAcceleration Errors (post-fit eval vs differentiated MoCap velocity):")
    print(f"  Mean: {accel_errors.mean():.4f} m/s²")
    print(f"  RMSE: {np.sqrt(np.mean(accel_errors**2)):.4f} m/s²")

    print(f"\nAccelerometer Optimization Residual Model:")
    print(f"  During fitting, accel residuals use raw IMU specific force measurements")
    print(f"  compared against predicted body-frame specific force R_bw @ (a_world - g) + b_a.")
    print(f"  The post-fit acceleration RMSE above instead compares estimated a_world")
    print(f"  against MoCap-velocity-differentiated acceleration for evaluation only.")
    
    print(f"\nOrientation Errors:")
    print(f"  Mean: {rot_errors.mean():.4f} deg")
    print(f"  RMSE: {np.sqrt(np.mean(rot_errors**2)):.4f} deg")
    
    print(f"\nEstimated Biases:")
    print(f"  Accelerometer: [{optimized_state.acc_bias[0]:.4f}, {optimized_state.acc_bias[1]:.4f}, {optimized_state.acc_bias[2]:.4f}] m/s²")
    print(f"  Gyroscope: [{optimized_state.gyr_bias[0]:.4f}, {optimized_state.gyr_bias[1]:.4f}, {optimized_state.gyr_bias[2]:.4f}] rad/s")
    
    # ==================== Plotting ====================
    print(f"\n{'Generating Plots':-^80}")

    time_rel = eval_times - eval_times[0]

    # --- Derived quantities for new plots ---
    pos_diff = estimated_positions - mocap_positions_eval
    accel_diff = estimated_accelerations - mocap_accelerations
    accel_errors_per_axis = accel_diff  # alias for clarity

    mocap_euler = np.degrees(Rotation.from_matrix(mocap_rotations_eval).as_euler('xyz'))
    est_euler = np.degrees(Rotation.from_matrix(estimated_rotations).as_euler('xyz'))
    euler_diff = ((est_euler - mocap_euler) + 180) % 360 - 180  # wrap to [-180, 180]

    mocap_rot_mag = np.degrees(np.linalg.norm(
        Rotation.from_matrix(mocap_rotations_eval).as_rotvec(), axis=1))
    est_rot_mag = np.degrees(np.linalg.norm(
        Rotation.from_matrix(estimated_rotations).as_rotvec(), axis=1))

    mocap_ang_vel = np.array([s.angular_velocity for s in agiros_eval])
    est_ang_vel = np.array([optimized_state.get_angular_velocity(t) for t in eval_times])
    ang_vel_diff = est_ang_vel - mocap_ang_vel
    ang_vel_abs_error = np.linalg.norm(ang_vel_diff, axis=1)
    mocap_ang_speed = np.linalg.norm(mocap_ang_vel, axis=1)
    est_ang_speed = np.linalg.norm(est_ang_vel, axis=1)

    mocap_speed = np.linalg.norm(mocap_velocities, axis=1)
    est_speed = np.linalg.norm(estimated_velocities, axis=1)
    mocap_accel_norm = np.linalg.norm(mocap_accelerations, axis=1)
    est_accel_norm = np.linalg.norm(estimated_accelerations, axis=1)
    accel_abs_error = np.linalg.norm(accel_diff, axis=1)
    accel_rmse = np.sqrt(np.mean(accel_abs_error**2))
    ang_vel_rmse = np.sqrt(np.mean(ang_vel_abs_error**2))

    axis_labels = ['x', 'y', 'z']
    euler_names = ['roll', 'pitch', 'yaw']

    # Style constants
    AXIS_COLORS = ['#c85050', '#4e9e4e', '#4878c8']  # muted r/g/b for x/y/z per-axis
    C_MOCAP = 'royalblue'    # absolute ground truth
    C_EST   = 'crimson'      # absolute estimated
    LW_AXIS = 0.9            # per-axis line width
    LW_ABS  = 2.0            # absolute curve line width
    A_AXIS  = 0.7            # per-axis alpha

    def _comparison(a, t, mocap_data, est_data, mocap_abs, est_abs,
                    axis_labels, ylabel, abs_label_pair):
        """Plot per-axis (thin, muted) + absolute (thick, blue/red) comparison."""
        for i, lbl in enumerate(axis_labels):
            a.plot(t, mocap_data[:, i], color=AXIS_COLORS[i], linewidth=LW_AXIS,
                   alpha=A_AXIS, label=f'MoCap {lbl}')
            a.plot(t, est_data[:, i], color=AXIS_COLORS[i], linewidth=LW_AXIS,
                   alpha=A_AXIS, linestyle='--')
        a.plot(t, mocap_abs, color=C_MOCAP, linewidth=LW_ABS, label=f'MoCap {abs_label_pair}')
        a.plot(t, est_abs,   color=C_EST,   linewidth=LW_ABS, linestyle='--',
               label=f'Est {abs_label_pair}')
        a.set_ylabel(ylabel); a.legend(fontsize=6, ncol=2); a.grid(True, alpha=0.3)

    def _error(a, t, per_axis_diff, abs_error, rmse, axis_labels, ylabel, abs_label):
        """Plot per-axis errors (thin, muted) + absolute error (thick black) + RMSE line."""
        for i, lbl in enumerate(axis_labels):
            a.plot(t, per_axis_diff[:, i], color=AXIS_COLORS[i], linewidth=LW_AXIS,
                   alpha=A_AXIS, label=f'Δ{lbl}')
        a.plot(t, abs_error, color='k', linewidth=LW_ABS, label=f'|{abs_label}|')
        a.axhline(rmse, color='royalblue', linewidth=1.5, linestyle='--',
                  label=f'RMSE: {rmse:.4f}')
        a.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        a.set_ylabel(ylabel); a.legend(fontsize=7); a.grid(True, alpha=0.3)

    # --- Figure layout: 5 rows x 3 cols, rows 1-4 col 2 merged for summary ---
    fig = plt.figure(figsize=(21, 28))
    fig.suptitle(f'Nonlinear Solver Validation — {bag_key}', fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(5, 3, hspace=0.45, wspace=0.35)

    axd = {}
    for row in range(5):
        for col in range(2):
            axd[(row, col)] = fig.add_subplot(gs[row, col])
    axd[(0, 2)] = fig.add_subplot(gs[0, 2])
    ax_summary = fig.add_subplot(gs[1:, 2])

    pos_rmse  = np.sqrt(np.mean(pos_errors**2))
    vel_rmse  = np.sqrt(np.mean(vel_errors**2))
    ori_rmse  = np.sqrt(np.mean(rot_errors**2))

    # --- Row 0: Position ---
    # [0,0] X-Y trajectory
    a = axd[(0, 0)]
    a.plot(mocap_positions_eval[:, 0], mocap_positions_eval[:, 1],
           color=C_MOCAP, linewidth=LW_ABS, label='MoCap')
    a.plot(estimated_positions[:, 0], estimated_positions[:, 1],
           color=C_EST, linewidth=LW_ABS, linestyle='--', label='Estimate')
    a.set_xlabel('X (m)'); a.set_ylabel('Y (m)')
    a.set_title('Trajectory (X-Y)')
    a.legend(fontsize=8); a.grid(True, alpha=0.3); a.axis('equal')

    # [0,1] X, Y, Z positions over time
    a = axd[(0, 1)]
    _comparison(a, time_rel, mocap_positions_eval, estimated_positions,
                np.linalg.norm(mocap_positions_eval, axis=1),
                np.linalg.norm(estimated_positions, axis=1),
                axis_labels, 'Position (m)', '|pos|')
    a.set_xlabel('Time (s)'); a.set_title('Position vs Time')

    # [0,2] Per-axis position error + absolute + RMSE
    a = axd[(0, 2)]
    _error(a, time_rel, pos_diff, pos_errors, pos_rmse, axis_labels, 'Error (m)', 'err')
    a.set_xlabel('Time (s)'); a.set_title('Position Error per Axis + Abs')

    # --- Row 1: Orientation ---
    # [1,0] Euler angles (roll/pitch/yaw) estimate + MoCap; rotation magnitude on twin axis
    a = axd[(1, 0)]
    for i, lbl in enumerate(euler_names):
        a.plot(time_rel, mocap_euler[:, i], color=AXIS_COLORS[i], linewidth=LW_AXIS,
               alpha=A_AXIS, label=f'MoCap {lbl}')
        a.plot(time_rel, est_euler[:, i], color=AXIS_COLORS[i], linewidth=LW_AXIS,
               alpha=A_AXIS, linestyle='--')
    a2 = a.twinx()
    a2.plot(time_rel, mocap_rot_mag, color=C_MOCAP, linewidth=LW_ABS,
            linestyle='--', alpha=0.8, label='|R| MoCap')
    a2.plot(time_rel, est_rot_mag,   color=C_EST,   linewidth=LW_ABS,
            linestyle='--', alpha=0.8, label='|R| Est')
    a2.set_ylabel('Rotation magnitude (deg)', fontsize=7)
    a2.legend(fontsize=6, loc='lower right')
    a.set_xlabel('Time (s)'); a.set_ylabel('Euler angle (deg)')
    a.set_title('Orientation (Euler xyz) + Abs Magnitude')
    a.legend(fontsize=6, ncol=2); a.grid(True, alpha=0.3)

    # [1,1] Per-axis euler error + absolute orientation error + RMSE
    a = axd[(1, 1)]
    _error(a, time_rel, euler_diff, rot_errors, ori_rmse, euler_names, 'Error (deg)', 'Δori')
    a.set_xlabel('Time (s)'); a.set_title('Orientation Error per Axis + Abs')

    # --- Row 2: Linear Velocity ---
    # [2,0] Per-axis + absolute velocity comparison
    a = axd[(2, 0)]
    _comparison(a, time_rel, mocap_velocities, estimated_velocities,
                mocap_speed, est_speed, axis_labels, 'Velocity (m/s)', '|v|')
    a.set_xlabel('Time (s)'); a.set_title('Linear Velocity Comparison')

    # [2,1] Per-axis velocity error + absolute + RMSE
    a = axd[(2, 1)]
    _error(a, time_rel, vel_diff, vel_errors, vel_rmse, axis_labels, 'Error (m/s)', 'Δv')
    a.set_xlabel('Time (s)'); a.set_title('Linear Velocity Error per Axis + Abs')

    # --- Row 3: Angular Velocity ---
    # [3,0] Per-axis + absolute angular velocity comparison
    a = axd[(3, 0)]
    _comparison(a, time_rel, mocap_ang_vel, est_ang_vel,
                mocap_ang_speed, est_ang_speed, axis_labels, 'Angular vel (rad/s)', '|ω|')
    a.set_xlabel('Time (s)'); a.set_title('Angular Velocity Comparison')

    # [3,1] Per-axis angular velocity error + absolute + RMSE
    a = axd[(3, 1)]
    _error(a, time_rel, ang_vel_diff, ang_vel_abs_error, ang_vel_rmse,
           axis_labels, 'Error (rad/s)', 'Δω')
    a.set_xlabel('Time (s)'); a.set_title('Angular Velocity Error per Axis + Abs')

    # --- Row 4: Linear Acceleration ---
    # [4,0] Per-axis + absolute acceleration comparison
    a = axd[(4, 0)]
    _comparison(a, time_rel, mocap_accelerations, estimated_accelerations,
                mocap_accel_norm, est_accel_norm, axis_labels, 'Accel (m/s²)', '|a|')
    a.set_xlabel('Time (s)'); a.set_title('Acceleration Comparison (vs diff(MoCap vel))')

    # [4,1] Per-axis acceleration error + absolute + RMSE
    a = axd[(4, 1)]
    _error(a, time_rel, accel_diff, accel_abs_error, accel_rmse,
           axis_labels, 'Error (m/s²)', 'Δa')
    a.set_xlabel('Time (s)'); a.set_title('Acceleration Error per Axis + Abs')

    # --- Summary & Config (merged rows 1-4, col 2) ---
    summary_lines = [
        f"RESULTS",
        f"  Pos  RMSE: {np.sqrt(np.mean(pos_errors**2)):.4f} m",
        f"  Vel  RMSE: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s",
        f"  AngV RMSE: {ang_vel_rmse:.4f} rad/s",
        f"  Acc  RMSE: {accel_rmse:.4f} m/s² (vs diff(MoCap vel))",
        f"  Ori  RMSE: {np.sqrt(np.mean(rot_errors**2)):.4f}°",
        f"",
        f"INIT BIASES",
        f"  Acc: [{bias_prior_mean[0]:+.4f}, {bias_prior_mean[1]:+.4f}, {bias_prior_mean[2]:+.4f}] m/s²",
        f"  Gyr: [{np.degrees(bias_prior_mean[3]):+.2f}, {np.degrees(bias_prior_mean[4]):+.2f}, {np.degrees(bias_prior_mean[5]):+.2f}] deg/s",
        f"FINAL BIASES",
        f"  Acc: [{optimized_state.acc_bias[0]:+.4f}, {optimized_state.acc_bias[1]:+.4f}, {optimized_state.acc_bias[2]:+.4f}] m/s²",
        f"  Gyr: [{np.degrees(optimized_state.gyr_bias[0]):+.2f}, {np.degrees(optimized_state.gyr_bias[1]):+.2f}, {np.degrees(optimized_state.gyr_bias[2]):+.2f}] deg/s",
        f"",
        f"HYPERPARAMETERS",
        f"  bag={bag_key}  t={START_TIME_OFFSET:.0f}s+{DURATION:.0f}s",
        f"  flip={FLIP_BODY_FRAME}  lock_bias={LOCK_BIASES}  stat_bias={USE_STATIONARY_BIAS}",
        f"  dt_pos={DT_POS}  dt_ori={DT_ORI}  deg={BSPLINE_DEGREE}",
        f"  λ_accel={LAMBDA_ACCEL}  λ_gyro={LAMBDA_GYRO}",
        f"  λ_snap_pos={LAMBDA_SNAP_POS}  λ_snap_ori={LAMBDA_SNAP_ORI}",
        f"  huber_radar={HUBER_DELTA}  huber_accel={HUBER_DELTA_ACCEL}",
        f"  λ_bnd_vel={LAMBDA_BOUNDARY_VEL}  λ_bnd_pos={LAMBDA_BOUNDARY_POS}",
        f"  λ_bnd_ori={LAMBDA_BOUNDARY_ORI}  λ_bnd_acc={LAMBDA_BOUNDARY_ACCEL}",
        f"  λ_bnd_gyr={LAMBDA_BOUNDARY_GYRO}",
        f"  bnd_window={BOUNDARY_WINDOW}s (start only)",
        f"  max_iter={MAX_ITERATIONS}  precond={USE_JACOBI_PRECOND}",
        f"  relin_thr={RELINEARIZE_THRESHOLD_DEG}°  imu_offset={IMU_MOCAP_OFFSET*1000:.0f}ms  radar_offset={radar_total_offset*1000:.0f}ms",
        f"  λ_ori_reg={LAMBDA_ORI_REG}  λ_bp_a={LAMBDA_BIAS_PRIOR_ACCEL}  λ_bp_g={LAMBDA_BIAS_PRIOR_GYRO}",
        f"",
        f"EXTRINSICS (rotation [roll,pitch,yaw] deg)",
        f"  lock={LOCK_EXTRINSICS}  pitch_only={OPTIMIZE_PITCH_ONLY}  λ_prior={LAMBDA_EXTRINSIC_PRIOR}",
        f"  Init:  [{ROTATION_EULER_DEG[0]:.2f}, {ROTATION_EULER_DEG[1]:.2f}, {ROTATION_EULER_DEG[2]:.2f}]",
        f"  Δ:     [{delta_deg[0]:+.3f}, {delta_deg[1]:+.3f}, {delta_deg[2]:+.3f}]",
        f"  Final: [{calibrated_euler[0]:.2f}, {calibrated_euler[1]:.2f}, {calibrated_euler[2]:.2f}]",
        f"  Trans: [{TRANSLATION[0]:.3f}, {TRANSLATION[1]:.3f}, {TRANSLATION[2]:.3f}] m",
    ]
    ax_summary.text(0.02, 0.98, "\n".join(summary_lines),
                    transform=ax_summary.transAxes,
                    fontsize=8, fontfamily='monospace', verticalalignment='top')
    ax_summary.axis('off')
    ax_summary.set_title('Summary & Config')

    # --- Radar frame tick marks on all time-based subplots ---
    # Draw small light-yellow ticks at the bottom of each time axis to show
    # when radar frames contributed (sparse near stationary segments).
    radar_tick_times = np.array([f.timestamp for f in solver_radar_frames]) - eval_times[0]
    radar_tick_counts = np.array([f.num_points() for f in solver_radar_frames], dtype=float)
    radar_tick_heights = 0.04 * radar_tick_counts / 13.5  # normalised to avg ~13.5 returns/frame
    _time_axes = [axd[(r, c)] for r in range(5) for c in range(2) if (r, c) != (0, 0)] + [axd[(0, 2)]]
    for _ax in _time_axes:
        _trans = mtransforms.blended_transform_factory(_ax.transData, _ax.transAxes)
        _ax.vlines(radar_tick_times, 0, radar_tick_heights, transform=_trans,
                   color='#ffe566', linewidth=0.8, alpha=0.85, zorder=0)

    plots_dir = Path(f'plots/{bag_key}/validation')
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_filename = plots_dir / f'nonlinear_solver_validation_{bag_key}_{timestamp_str}.png'
    fig.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_filename}")

    # ==================== Multi-View Trajectory Plot ====================
    fig2 = plt.figure(figsize=(14, 12))
    gt = mocap_positions_eval
    est = estimated_positions

    # Helper for 2D subplots
    def _setup_2d(ax, xi, yi, xlabel, ylabel, title):
        ax.plot(gt[:, xi], gt[:, yi], 'b-', label='MoCap', linewidth=2)
        ax.plot(est[:, xi], est[:, yi], 'r--', label='Estimated', linewidth=1.5)
        ax.plot(gt[0, xi], gt[0, yi], 'bs', markersize=8)
        ax.plot(est[0, xi], est[0, yi], 'rs', markersize=8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    ax_xy = fig2.add_subplot(2, 2, 1)
    _setup_2d(ax_xy, 0, 1, 'X (m)', 'Y (m)', 'X-Y Plane')

    ax_xz = fig2.add_subplot(2, 2, 2)
    _setup_2d(ax_xz, 0, 2, 'X (m)', 'Z (m)', 'X-Z Plane')

    ax_yz = fig2.add_subplot(2, 2, 3)
    _setup_2d(ax_yz, 1, 2, 'Y (m)', 'Z (m)', 'Y-Z Plane')

    # 3D view
    ax3d = fig2.add_subplot(2, 2, 4, projection='3d')
    ax3d.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'b-', label='MoCap', linewidth=2)
    ax3d.plot(est[:, 0], est[:, 1], est[:, 2], 'r--', label='Estimated', linewidth=1.5)
    ax3d.plot([gt[0, 0]], [gt[0, 1]], [gt[0, 2]], 'bs', markersize=8)
    ax3d.plot([est[0, 0]], [est[0, 1]], [est[0, 2]], 'rs', markersize=8)
    ax3d.set_xlabel('X (m)', fontsize=10)
    ax3d.set_ylabel('Y (m)', fontsize=10)
    ax3d.set_zlabel('Z (m)', fontsize=10)
    ax3d.set_title('3D View', fontsize=12, fontweight='bold')
    ax3d.legend(fontsize=9)

    fig2.suptitle(f'Trajectory Views — {bag_key}', fontsize=14, fontweight='bold')
    fig2.tight_layout()
    views_dir = Path(f'plots/{bag_key}/trajectory_views')
    views_dir.mkdir(parents=True, exist_ok=True)
    views_filename = views_dir / f'nonlinear_trajectory_views_{bag_key}_{timestamp_str}.png'
    fig2.savefig(views_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {views_filename}")
    
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
