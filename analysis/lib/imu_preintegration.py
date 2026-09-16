"""
IMU Preintegration for Radar-Inertial Odometry.

Implements the on-manifold IMU preintegration from Forster et al.,
"On-Manifold Preintegration for Real-Time Visual-Inertial Odometry",
TRO 2017.

Preintegrates raw IMU measurements (full-rate, before any downsampling)
between consecutive radar frames to produce compact 9D factors
(ΔR, Δv, Δp) with first-order bias correction Jacobians.

Convention
----------
Rotations are R_wb (body-to-world) in Forster notation, which matches
the SymForce convention in generated_jacobians.py where R.inverse()
maps world → body.

In our solver, R is stored as R_wb and R.inverse() = R^T is used to
rotate world vectors to body frame (see accel_residual in codegen).

Preintegrated residuals (see generated_jacobians.py):
    r_R = Log(ΔR̃_corr^T @ R_i^T @ R_j)        # rotation
    r_v = R_i^T @ (v_j - v_i - g*Δt) - Δṽ_c    # velocity (R_i^T = R_i.inverse())
    r_p = R_i^T @ (p_j - p_i - v_i*Δt - 0.5*g*Δt²) - Δp̃_c  # position

Usage
-----
    from lib.imu_preintegration import preintegrate, PreintegratedIMU

    # Use imu_data_full (original-rate), NOT the downsampled imu_data!
    meas = preintegrate(imu_samples_full_rate, ba, bg, t_start, t_end)

    # Build factors for all radar frame intervals
    factors = build_preintegrated_factors(imu_data_full, radar_times, ba0, bg0)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import List, Sequence


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross product: skew(v) @ w = v × w."""
    return np.array([
        [ 0.0,   -v[2],  v[1]],
        [ v[2],   0.0,  -v[0]],
        [-v[1],   v[0],  0.0 ],
    ])


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map: Exp(phi) = rotation matrix for axis-angle phi."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) + _skew(phi)
    axis = phi / theta
    K = _skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _right_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """
    Right Jacobian J_r(phi) of SO(3).

    J_r(phi) = I - (1-cos||phi||)/||phi||² [phi]× + (||phi||-sin||phi||)/||phi||³ [phi]×²

    Satisfies: Exp(phi + δ) ≈ Exp(phi) @ Exp(J_r(phi) @ δ)  for small δ.
    At phi=0: J_r = I.
    """
    theta = np.linalg.norm(phi)
    K = _skew(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * K
    theta2 = theta * theta
    c1 = (1.0 - np.cos(theta)) / theta2
    c2 = (theta - np.sin(theta)) / (theta * theta2)
    return np.eye(3) - c1 * K + c2 * (K @ K)


@dataclass
class PreintegratedIMU:
    """
    Preintegrated IMU measurement between two times t_i and t_j.

    All quantities are expressed relative to body frame at t_i.

    Bias correction (first-order):
        ΔR̃_corr = ΔR̃ @ Exp(d_R_d_bg @ (b_g - b_g0))
        Δṽ_corr = Δṽ + d_v_d_ba @ (b_a - b_a0) + d_v_d_bg @ (b_g - b_g0)
        Δp̃_corr = Δp̃ + d_p_d_ba @ (b_a - b_a0) + d_p_d_bg @ (b_g - b_g0)
    """
    t_i: float             # start time (absolute, seconds)
    t_j: float             # end time (absolute, seconds)
    dt: float              # total integration time = t_j - t_i

    # Preintegrated measurements (at linearization biases b_a0, b_g0)
    delta_R: np.ndarray    # (3,3) accumulated rotation ΔR̃
    delta_v: np.ndarray    # (3,)  accumulated velocity Δṽ
    delta_p: np.ndarray    # (3,)  accumulated position Δp̃

    # Linearization biases
    b_a0: np.ndarray       # (3,)
    b_g0: np.ndarray       # (3,)

    # First-order bias correction Jacobians
    d_R_d_bg: np.ndarray   # (3,3)
    d_v_d_ba: np.ndarray   # (3,3)
    d_v_d_bg: np.ndarray   # (3,3)
    d_p_d_ba: np.ndarray   # (3,3)
    d_p_d_bg: np.ndarray   # (3,3)

    def corrected(self, b_a: np.ndarray, b_g: np.ndarray):
        """Apply first-order bias correction to get corrected measurements."""
        dba = b_a - self.b_a0
        dbg = b_g - self.b_g0
        delta_R_c = self.delta_R @ _exp_so3(self.d_R_d_bg @ dbg)
        delta_v_c = self.delta_v + self.d_v_d_ba @ dba + self.d_v_d_bg @ dbg
        delta_p_c = self.delta_p + self.d_p_d_ba @ dba + self.d_p_d_bg @ dbg
        return delta_R_c, delta_v_c, delta_p_c

    def bias_jacobians_residual(self) -> tuple:
        """
        Returns Jacobian blocks for bias update in the assembled residual.

        Returns (J_b_a, J_b_g) each (9, 3), stacking [r_R; r_v; r_p].

        Approximate (standard simplification, assumes small residual):
            ∂r_R/∂b_g ≈ -d_R_d_bg
            ∂r_v/∂b_a = -d_v_d_ba,   ∂r_v/∂b_g = -d_v_d_bg
            ∂r_p/∂b_a = -d_p_d_ba,   ∂r_p/∂b_g = -d_p_d_bg
        """
        J_b_a = np.zeros((9, 3))
        J_b_g = np.zeros((9, 3))
        # r_R (rows 0-2): only depends on b_g via ΔR_corr
        J_b_g[0:3, :] = -self.d_R_d_bg
        # r_v (rows 3-5)
        J_b_a[3:6, :] = -self.d_v_d_ba
        J_b_g[3:6, :] = -self.d_v_d_bg
        # r_p (rows 6-8)
        J_b_a[6:9, :] = -self.d_p_d_ba
        J_b_g[6:9, :] = -self.d_p_d_bg
        return J_b_a, J_b_g


def preintegrate(
    imu_samples: Sequence,
    b_a0: np.ndarray,
    b_g0: np.ndarray,
    t_start: float,
    t_end: float,
) -> PreintegratedIMU:
    """
    Preintegrate IMU samples between t_start and t_end.

    Args:
        imu_samples : Sequence of IMU dataclass objects with fields:
                      .timestamp, .linear_acceleration, .angular_velocity
                      Must be the FULL-RATE data (NOT downsampled).
        b_a0        : (3,) accelerometer bias at linearization point
        b_g0        : (3,) gyroscope bias at linearization point
        t_start     : integration start time (absolute seconds)
        t_end       : integration end time (absolute seconds)

    Returns:
        PreintegratedIMU measurement with accumulated ΔR, Δv, Δp
        and first-order bias correction Jacobians.
    """
    # Filter samples in [t_start, t_end]
    samples = [s for s in imu_samples if t_start <= s.timestamp <= t_end]

    dt_total = t_end - t_start

    if len(samples) < 2:
        # Degenerate: return identity/zero measurement
        return PreintegratedIMU(
            t_i=t_start, t_j=t_end, dt=dt_total,
            delta_R=np.eye(3), delta_v=np.zeros(3), delta_p=np.zeros(3),
            b_a0=b_a0.copy(), b_g0=b_g0.copy(),
            d_R_d_bg=np.zeros((3, 3)),
            d_v_d_ba=np.zeros((3, 3)), d_v_d_bg=np.zeros((3, 3)),
            d_p_d_ba=np.zeros((3, 3)), d_p_d_bg=np.zeros((3, 3)),
        )

    # Accumulated state
    delta_R = np.eye(3)
    delta_v = np.zeros(3)
    delta_p = np.zeros(3)

    # Bias correction Jacobians (accumulated)
    d_R_d_bg = np.zeros((3, 3))
    d_v_d_ba = np.zeros((3, 3))
    d_v_d_bg = np.zeros((3, 3))
    d_p_d_ba = np.zeros((3, 3))
    d_p_d_bg = np.zeros((3, 3))

    for k in range(len(samples) - 1):
        s0 = samples[k]
        s1 = samples[k + 1]
        dt = s1.timestamp - s0.timestamp
        if dt <= 0.0:
            continue

        # Mid-point interpolation for better integration accuracy
        a_m = 0.5 * (np.array(s0.linear_acceleration) + np.array(s1.linear_acceleration))
        w_m = 0.5 * (np.array(s0.angular_velocity) + np.array(s1.angular_velocity))

        # Bias-corrected measurements
        a_corr = a_m - b_a0
        w_corr = w_m - b_g0

        # Rotation update: ΔR_{k+1} = ΔR_k @ Exp(w_corr * dt)
        phi = w_corr * dt
        dR = _exp_so3(phi)
        Jr = _right_jacobian_so3(phi)

        # Update position and velocity BEFORE rotation update (uses current ΔR)
        delta_p += delta_v * dt + 0.5 * (delta_R @ a_corr) * dt * dt
        delta_v += (delta_R @ a_corr) * dt

        # Bias Jacobian updates (Forster Eq. A.7 - A.9)
        # d_p_d_ba: ∂Δp/∂b_a
        d_p_d_ba += d_v_d_ba * dt - 0.5 * delta_R * dt * dt
        # d_p_d_bg: ∂Δp/∂b_g
        d_p_d_bg += d_v_d_bg * dt - 0.5 * (delta_R @ _skew(a_corr) @ d_R_d_bg) * dt * dt
        # d_v_d_ba: ∂Δv/∂b_a
        d_v_d_ba -= delta_R * dt
        # d_v_d_bg: ∂Δv/∂b_g
        d_v_d_bg -= delta_R @ _skew(a_corr) @ d_R_d_bg * dt
        # d_R_d_bg: ∂ΔR/∂b_g (propagated through the chain)
        d_R_d_bg = dR.T @ d_R_d_bg - Jr * dt

        # Apply rotation step
        delta_R = delta_R @ dR

    return PreintegratedIMU(
        t_i=t_start, t_j=t_end, dt=dt_total,
        delta_R=delta_R, delta_v=delta_v, delta_p=delta_p,
        b_a0=b_a0.copy(), b_g0=b_g0.copy(),
        d_R_d_bg=d_R_d_bg,
        d_v_d_ba=d_v_d_ba, d_v_d_bg=d_v_d_bg,
        d_p_d_ba=d_p_d_ba, d_p_d_bg=d_p_d_bg,
    )


def build_preintegrated_factors(
    imu_data_full: Sequence,
    radar_times: np.ndarray,
    b_a0: np.ndarray,
    b_g0: np.ndarray,
) -> List[PreintegratedIMU]:
    """
    Build one PreintegratedIMU factor for each consecutive radar frame interval.

    Args:
        imu_data_full : Full-rate IMU samples (NOT downsampled).
        radar_times   : (N,) absolute timestamps of radar frames.
        b_a0, b_g0    : Linearization biases for preintegration.

    Returns:
        List of N-1 PreintegratedIMU factors (one per consecutive pair).
    """
    factors = []
    for k in range(len(radar_times) - 1):
        t_i = radar_times[k]
        t_j = radar_times[k + 1]
        meas = preintegrate(imu_data_full, b_a0, b_g0, t_i, t_j)
        factors.append(meas)
    return factors
