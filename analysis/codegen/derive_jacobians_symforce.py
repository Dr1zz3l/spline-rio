#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""
Derive analytical Jacobians for radar-inertial odometry using SymForce.

Run in Docker:
    docker exec iwr6843-dev /workspace/.venv_docker/bin/python \
        /workspace/analysis/derive_jacobians_symforce.py

Generates: analysis/generated_jacobians.py (pure NumPy, no SymForce dependency)

The generated file contains:
- radar_residual_with_jacobians():   Doppler residual + ∂r/∂v_world, ∂r/∂delta, ∂r/∂omega
- accel_residual_with_jacobians():   Accel residual  + ∂r/∂a_world, ∂r/∂delta, ∂r/∂b_a
- gyro_residual_with_jacobians():    Gyro residual   + ∂r/∂delta, ∂r/∂delta_dot, ∂r/∂b_g
- gravity_residual_with_jacobians(): Gravity-direction residual + ∂r/∂delta, ∂r/∂b_a
- heading_residual_with_jacobians(): Heading (yaw) residual + ∂r/∂delta
"""

import symforce
symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce.codegen import Codegen, PythonConfig
import os
import re
import glob
import shutil


# ============================================================
# Residual definitions
# ============================================================

def radar_residual(
    v_world: sf.V3,
    R_nominal: sf.Rot3,
    delta: sf.V3,
    omega: sf.V3,
    u_sensor: sf.V3,
    t_body_sensor: sf.V3,
    R_body_sensor: sf.Rot3,
    v_meas: sf.Scalar,
    epsilon: sf.Scalar,
) -> sf.V1:
    """
    Radar Doppler residual.

    Forward model:
        R(t) = R_nominal * exp(delta)
        v_body = R^T * v_world
        v_ant = v_body + omega x t_body_sensor
        u_body = R_body_sensor * u_sensor
        v_pred = u_body . v_ant
        residual = v_meas - v_pred
    """
    # Reconstruct rotation from nominal + tangent perturbation
    R = R_nominal * sf.Rot3.from_tangent(delta, epsilon=epsilon)

    # Velocity in body frame
    v_body = R.inverse() * v_world

    # Lever arm contribution (angular velocity cross lever arm)
    v_lever = omega.cross(t_body_sensor)
    v_ant = v_body + v_lever

    # Unit direction in body frame
    u_body = R_body_sensor * u_sensor

    # Predicted Doppler velocity
    # TI IWR6843 convention: v_meas = -dot(u, v) (positive = receding target)
    v_pred = -u_body.dot(v_ant)

    return sf.V1(v_meas - v_pred)


def accel_residual(
    a_world: sf.V3,
    R_nominal: sf.Rot3,
    delta: sf.V3,
    g_world: sf.V3,
    z_acc: sf.V3,
    b_a: sf.V3,
    epsilon: sf.Scalar,
) -> sf.V3:
    """
    Accelerometer residual.

    Forward model:
        R(t) = R_nominal * exp(delta)
        predicted_specific_force = R^T * (a_world - g_world)
        residual = z_acc - predicted_specific_force - b_a
    """
    R = R_nominal * sf.Rot3.from_tangent(delta, epsilon=epsilon)
    a_body_pred = R.inverse() * (a_world - g_world)
    return z_acc - a_body_pred - b_a


def gyro_residual(
    omega_nominal: sf.V3,
    delta: sf.V3,
    delta_dot: sf.V3,
    z_gyro: sf.V3,
    b_g: sf.V3,
    epsilon: sf.Scalar,
) -> sf.V3:
    """
    Gyroscope residual with proper SO(3) right Jacobian.

    Full angular velocity model:
        R(t) = R_nominal * exp(delta)
        omega_body = exp(-[delta]_x) * omega_nominal + J_r(delta) * delta_dot

    where J_r(delta) is the right Jacobian of SO(3):
        J_r(phi) = I - (1-cos||phi||)/||phi||^2 [phi]_x
                     + (||phi||-sin||phi||)/||phi||^3 [phi]_x^2

    For small delta: omega ≈ omega_nominal + delta_dot (recovers linear model).

    Residual: r = z_gyro - omega_body - b_g
    """
    # Rotation perturbation exp(delta)
    R_delta = sf.Rot3.from_tangent(delta, epsilon=epsilon)

    # Rotated nominal angular velocity: exp(-[delta]_x) * omega_nominal
    omega_rot = R_delta.inverse() * omega_nominal

    # Right Jacobian J_r(delta) via Rodrigues-like formula
    dx = delta[0]
    dy = delta[1]
    dz = delta[2]
    skew = sf.Matrix33([[0, -dz, dy], [dz, 0, -dx], [-dy, dx, 0]])

    theta_sq = delta.dot(delta)
    # Use epsilon to avoid division by zero at theta=0
    safe_theta_sq = theta_sq + epsilon ** 2
    theta = sf.sqrt(safe_theta_sq)

    c1 = (1 - sf.cos(theta)) / safe_theta_sq
    c2 = (theta - sf.sin(theta)) / (theta * safe_theta_sq)

    J_r = sf.Matrix33.eye() - c1 * skew + c2 * skew * skew

    # Full angular velocity in body frame
    omega_pred = omega_rot + J_r * delta_dot

    return z_gyro - omega_pred - b_g


def gravity_residual(
    R_nominal: sf.Rot3,
    delta: sf.V3,
    z_acc: sf.V3,
    b_a: sf.V3,
    g_norm: sf.Scalar,
    epsilon: sf.Scalar,
) -> sf.V3:
    """
    Gravity-direction residual (Mahony-style roll/pitch constraint).

    Compares the normalized accelerometer reading (specific force direction in body frame)
    against the predicted specific force direction from the current attitude estimate.
    Decoupled from linear acceleration / position — no a_world dependency.

    At hover, z_acc ≈ R^T @ [0,0,+g] (accel measures reaction force, pointing UP).
    residual = normalize(z_acc - b_a) * g_norm - R^T @ [0, 0, +g]
    """
    R = R_nominal * sf.Rot3.from_tangent(delta, epsilon=epsilon)
    g_world = sf.V3(0, 0, g_norm)

    z_debiased = z_acc - b_a
    z_normed = z_debiased / z_debiased.norm(epsilon=epsilon) * g_norm
    g_body_predicted = R.inverse() * g_world

    return z_normed - g_body_predicted


def heading_residual(
    R_nominal: sf.Rot3,
    delta: sf.V3,
    R_mocap: sf.Rot3,
    epsilon: sf.Scalar,
) -> sf.V1:
    """
    Heading (yaw) residual: MoCap pseudo-magnetometer.

    Extracts yaw error by comparing the body x-axis projected onto the
    world horizontal plane between estimated and MoCap orientations.

    residual = atan2(cross_z, dot) where cross/dot are between
    the horizontal projections of R_est @ [1,0,0] and R_mocap @ [1,0,0].
    """
    R = R_nominal * sf.Rot3.from_tangent(delta, epsilon=epsilon)
    x_body = sf.V3(1, 0, 0)

    # Body x-axis in world frame (estimated and MoCap)
    x_est = R * x_body
    x_moc = R_mocap * x_body

    # Signed angle between horizontal projections of the two x-axes
    cross_z = x_est[0] * x_moc[1] - x_est[1] * x_moc[0]
    dot_xy  = x_est[0] * x_moc[0] + x_est[1] * x_moc[1]

    return sf.V1(sf.atan2(cross_z, dot_xy, epsilon=epsilon))


def preintegrated_rotation_residual(
    R_i_nominal: sf.Rot3,
    delta_i: sf.V3,
    R_j_nominal: sf.Rot3,
    delta_j: sf.V3,
    delta_R_corr: sf.Rot3,
    epsilon: sf.Scalar,
) -> sf.V3:
    """
    Preintegrated IMU rotation residual (Forster et al., TRO 2017).

    Convention: R is body-to-world (R_wb), matching SymForce usage
    (R.inverse() maps world → body, R * v rotates body vector to world).
    This matches the existing accel/gyro residuals in this file.

    Preintegrated rotation: ΔR̃ = ∏ Exp(ω_k * dt_k)   (accumulated body rotation)
    Predicted relation:     ΔR_pred = R_i^T @ R_j      (in R_wb convention)
    Bias-corrected meas:    ΔR̃_corr = ΔR̃ @ Exp(d_R_d_bg @ δb_g)

    Residual: r_R = Log(ΔR̃_corr^T @ R_i^T @ R_j)

    Jacobians computed w.r.t. delta_i and delta_j (SO3 tangent perturbations).
    Bias Jacobians (∂r_R/∂b_g ≈ -d_R_d_bg) are handled analytically outside SymForce.
    """
    R_i = R_i_nominal * sf.Rot3.from_tangent(delta_i, epsilon=epsilon)
    R_j = R_j_nominal * sf.Rot3.from_tangent(delta_j, epsilon=epsilon)
    # R_rel = ΔR̃_corr^T @ R_i^T @ R_j = delta_R_corr.inverse() @ R_i.inverse() @ R_j
    R_rel = delta_R_corr.inverse() * R_i.inverse() * R_j
    return sf.V3(R_rel.to_tangent(epsilon=epsilon))


def preintegrated_velocity_residual(
    R_i_nominal: sf.Rot3,
    delta_i: sf.V3,
    v_i: sf.V3,
    v_j: sf.V3,
    g_world: sf.V3,
    dt: sf.Scalar,
    delta_v_corr: sf.V3,
    epsilon: sf.Scalar,
) -> sf.V3:
    """
    Preintegrated IMU velocity residual (Forster et al., TRO 2017).

    Preintegrated velocity: Δṽ = ∫ ΔR(s) @ (a_m - b_a) ds
    Predicted:              Δv_pred = R_i^T @ (v_j - v_i - g * dt)
    Bias-corrected meas:    Δṽ_corr = Δṽ + J_v_ba @ δb_a + J_v_bg @ δb_g

    Residual: r_v = R_i^T @ (v_j - v_i - g * dt) - Δṽ_corr

    Jacobians computed w.r.t. delta_i, v_i, v_j.
    Bias Jacobians (∂r_v/∂b_a = -J_v_ba, ∂r_v/∂b_g = -J_v_bg) handled analytically.
    """
    R_i = R_i_nominal * sf.Rot3.from_tangent(delta_i, epsilon=epsilon)
    e_v = v_j - v_i - g_world * dt
    return R_i.inverse() * e_v - delta_v_corr


def preintegrated_position_residual(
    R_i_nominal: sf.Rot3,
    delta_i: sf.V3,
    p_i: sf.V3,
    v_i: sf.V3,
    p_j: sf.V3,
    g_world: sf.V3,
    dt: sf.Scalar,
    delta_p_corr: sf.V3,
    epsilon: sf.Scalar,
) -> sf.V3:
    """
    Preintegrated IMU position residual (Forster et al., TRO 2017).

    Preintegrated position: Δp̃ = ∫∫ ΔR(s) @ (a_m - b_a) ds²
    Predicted:              Δp_pred = R_i^T @ (p_j - p_i - v_i*dt - 0.5*g*dt²)
    Bias-corrected meas:    Δp̃_corr = Δp̃ + J_p_ba @ δb_a + J_p_bg @ δb_g

    Residual: r_p = R_i^T @ (p_j - p_i - v_i*dt - 0.5*g*dt²) - Δp̃_corr

    Jacobians computed w.r.t. delta_i, p_i, v_i, p_j.
    Bias Jacobians (∂r_p/∂b_a = -J_p_ba, ∂r_p/∂b_g = -J_p_bg) handled analytically.
    """
    R_i = R_i_nominal * sf.Rot3.from_tangent(delta_i, epsilon=epsilon)
    half = sf.Rational(1, 2)
    e_p = p_j - p_i - v_i * dt - g_world * (half * dt * dt)
    return R_i.inverse() * e_p - delta_p_corr


# ============================================================
# Code generation
# ============================================================

def generate_and_read(func, name, which_args):
    """Generate code with Jacobians and read the output file."""
    output_dir = f"/tmp/sf_codegen_{name}"

    # Clean previous output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print(f"  Generating {name}...")
    codegen = Codegen.function(func, config=PythonConfig())
    codegen_jac = codegen.with_jacobians(which_args=which_args)
    data = codegen_jac.generate_function(output_dir=output_dir)

    # Find the generated Python file(s)
    pyfiles = [
        str(f) for f in sorted(data.generated_files)
        if str(f).endswith('.py') and '__init__' not in str(f)
    ]

    if not pyfiles:
        raise RuntimeError(f"No generated files found for {name}")

    print(f"  Generated: {pyfiles}")

    # Read the file content
    contents = {}
    for f in pyfiles:
        with open(f) as fh:
            contents[os.path.basename(f)] = fh.read()

    return contents


def post_process(code: str) -> str:
    """
    Post-process SymForce generated code to remove sym dependency.

    The generated code uses:
    - `import sym` (for Rot3 type)
    - `sym.Rot3` (in type annotations)
    - `R.data` (to extract quaternion components [x,y,z,w])

    We replace these with a lightweight Rot3 shim defined in the output file.
    """
    # Remove 'import sym' line
    code = re.sub(r'^import sym\s*$', '', code, flags=re.MULTILINE)

    # Replace sym.Rot3 in type comments with our shim
    code = code.replace('sym.Rot3', 'Rot3')

    # Remove the auto-generated header (we'll add our own)
    code = re.sub(
        r'^# ----.*?# ----[^\n]*\n',
        '',
        code,
        flags=re.DOTALL | re.MULTILINE
    )

    # Remove ruff noqa line
    code = re.sub(r'^# ruff:.*$\n', '', code, flags=re.MULTILINE)

    return code.strip()


def main():
    print("=" * 60)
    print("SymForce Jacobian Code Generation")
    print("=" * 60)

    # Generate radar residual with Jacobians w.r.t. v_world, delta, omega, R_body_sensor
    print("\n[1/5] Radar residual:")
    radar_code = generate_and_read(
        radar_residual,
        "radar",
        which_args=["v_world", "delta", "omega", "R_body_sensor"],
    )

    # Generate accel residual with Jacobians w.r.t. a_world, delta, b_a
    print("\n[2/5] Accelerometer residual:")
    accel_code = generate_and_read(
        accel_residual,
        "accel",
        which_args=["a_world", "delta", "b_a"],
    )

    # Generate gyro residual with Jacobians w.r.t. delta, delta_dot, b_g
    print("\n[3/5] Gyroscope residual:")
    gyro_code = generate_and_read(
        gyro_residual,
        "gyro",
        which_args=["delta", "delta_dot", "b_g"],
    )

    # Generate gravity residual with Jacobians w.r.t. delta, b_a
    print("\n[4/5] Gravity-direction residual:")
    gravity_code = generate_and_read(
        gravity_residual,
        "gravity",
        which_args=["delta", "b_a"],
    )

    # Generate heading residual with Jacobians w.r.t. delta
    print("\n[5/5] Heading (yaw) residual:")
    heading_code = generate_and_read(
        heading_residual,
        "heading",
        which_args=["delta"],
    )

    # Generate preintegrated IMU rotation residual
    print("\n[6/8] Preintegrated IMU rotation residual:")
    preint_rot_code = generate_and_read(
        preintegrated_rotation_residual,
        "preint_rot",
        which_args=["delta_i", "delta_j"],
    )

    # Generate preintegrated IMU velocity residual
    print("\n[7/8] Preintegrated IMU velocity residual:")
    preint_vel_code = generate_and_read(
        preintegrated_velocity_residual,
        "preint_vel",
        which_args=["delta_i", "v_i", "v_j"],
    )

    # Generate preintegrated IMU position residual
    print("\n[8/8] Preintegrated IMU position residual:")
    preint_pos_code = generate_and_read(
        preintegrated_position_residual,
        "preint_pos",
        which_args=["delta_i", "p_i", "v_i", "p_j"],
    )

    # Post-process all generated code
    print("\nPost-processing generated code...")
    processed_radar = {}
    for name, code in radar_code.items():
        processed_radar[name] = post_process(code)

    processed_accel = {}
    for name, code in accel_code.items():
        processed_accel[name] = post_process(code)

    processed_gyro = {}
    for name, code in gyro_code.items():
        processed_gyro[name] = post_process(code)

    processed_gravity = {}
    for name, code in gravity_code.items():
        processed_gravity[name] = post_process(code)

    processed_heading = {}
    for name, code in heading_code.items():
        processed_heading[name] = post_process(code)

    processed_preint_rot = {}
    for name, code in preint_rot_code.items():
        processed_preint_rot[name] = post_process(code)

    processed_preint_vel = {}
    for name, code in preint_vel_code.items():
        processed_preint_vel[name] = post_process(code)

    processed_preint_pos = {}
    for name, code in preint_pos_code.items():
        processed_preint_pos[name] = post_process(code)

    # Assemble the output file
    print("Assembling generated_jacobians.py...")

    header = '''"""
Auto-generated analytical Jacobians for radar-inertial odometry.

Generated by: derive_jacobians_symforce.py (using SymForce {sf_version})
This file has NO dependency on SymForce — it uses only numpy and math.

Contains:
- radar_residual_with_jacobians():        Doppler residual + Jacobians w.r.t. v_world, delta, omega
- accel_residual_with_jacobians():        Accel residual + Jacobians w.r.t. a_world, delta, b_a
- gyro_residual_with_jacobians():         Gyro residual  + Jacobians w.r.t. delta, delta_dot, b_g
- gravity_residual_with_jacobians():      Gravity-direction residual + Jacobians w.r.t. delta, b_a
- heading_residual_with_jacobians():      Heading (yaw) residual + Jacobians w.r.t. delta
- preint_rot_residual_with_jacobians():   Preintegrated rotation 3D residual + Jacobians
- preint_vel_residual_with_jacobians():   Preintegrated velocity 3D residual + Jacobians
- preint_pos_residual_with_jacobians():   Preintegrated position 3D residual + Jacobians
- Rot3: Lightweight quaternion wrapper matching SymForce convention [x, y, z, w]

Usage:
    from generated_jacobians import (
        radar_residual_with_jacobians,
        accel_residual_with_jacobians,
        gyro_residual_with_jacobians,
        gravity_residual_with_jacobians,
        heading_residual_with_jacobians,
        Rot3,
    )

    R_nom = Rot3(quat_xyzw)
    R_bs = Rot3(quat_xyzw)
    res, J_v, J_delta, J_omega = radar_residual_with_jacobians(
        v_world, R_nom, delta, omega, u_sensor, t_body_sensor, R_bs, v_meas, epsilon
    )
"""

import math
import typing as T
import numpy


class Rot3:
    """
    Lightweight quaternion wrapper matching SymForce Rot3 interface.

    Stores quaternion as [x, y, z, w] (Hamilton convention).
    Only provides the .data property needed by generated code.
    """
    __slots__ = ['data']

    def __init__(self, quat_xyzw):
        """
        Args:
            quat_xyzw: Quaternion as [x, y, z, w] array or list.
        """
        if hasattr(quat_xyzw, 'tolist'):
            self.data = quat_xyzw.tolist()
        else:
            self.data = list(quat_xyzw)

    @staticmethod
    def from_rotation_matrix(R):
        """
        Create Rot3 from a 3x3 rotation matrix.

        Uses Shepperd's method for numerical stability.
        Returns quaternion in [x, y, z, w] format.
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        # Normalize
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        return Rot3([x/norm, y/norm, z/norm, w/norm])

'''.format(sf_version=symforce.__version__)

    output_parts = [header]

    # Add separator and radar functions
    output_parts.append("\n# " + "=" * 70)
    output_parts.append("# RADAR DOPPLER RESIDUAL + JACOBIANS")
    output_parts.append("# " + "=" * 70 + "\n")

    for name, code in processed_radar.items():
        output_parts.append(f"\n# --- From: {name} ---\n")
        # Remove duplicate imports (already in header)
        code = re.sub(r'^import math\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import typing as T\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import numpy\s*$', '', code, flags=re.MULTILINE)
        # Normalize function name (remove numeric suffix from with_jacobians)
        code = re.sub(r'def radar_residual_with_jacobians\d+\(',
                       'def radar_residual_with_jacobians(', code)
        output_parts.append(code.strip())

    # Add separator and accel functions
    output_parts.append("\n\n# " + "=" * 70)
    output_parts.append("# ACCELEROMETER RESIDUAL + JACOBIANS")
    output_parts.append("# " + "=" * 70 + "\n")

    for name, code in processed_accel.items():
        output_parts.append(f"\n# --- From: {name} ---\n")
        code = re.sub(r'^import math\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import typing as T\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import numpy\s*$', '', code, flags=re.MULTILINE)
        # Normalize function name
        code = re.sub(r'def accel_residual_with_jacobians\d+\(',
                       'def accel_residual_with_jacobians(', code)
        output_parts.append(code.strip())

    # Add separator and gyro functions
    output_parts.append("\n\n# " + "=" * 70)
    output_parts.append("# GYROSCOPE RESIDUAL + JACOBIANS")
    output_parts.append("# " + "=" * 70 + "\n")

    for name, code in processed_gyro.items():
        output_parts.append(f"\n# --- From: {name} ---\n")
        code = re.sub(r'^import math\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import typing as T\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import numpy\s*$', '', code, flags=re.MULTILINE)
        # Normalize function name
        code = re.sub(r'def gyro_residual_with_jacobians\d+\(',
                       'def gyro_residual_with_jacobians(', code)
        output_parts.append(code.strip())

    # Add separator and gravity functions
    output_parts.append("\n\n# " + "=" * 70)
    output_parts.append("# GRAVITY-DIRECTION RESIDUAL + JACOBIANS")
    output_parts.append("# " + "=" * 70 + "\n")

    for name, code in processed_gravity.items():
        output_parts.append(f"\n# --- From: {name} ---\n")
        code = re.sub(r'^import math\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import typing as T\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import numpy\s*$', '', code, flags=re.MULTILINE)
        # Normalize function name
        code = re.sub(r'def gravity_residual_with_jacobians\d+\(',
                       'def gravity_residual_with_jacobians(', code)
        output_parts.append(code.strip())

    # Add separator and heading functions
    output_parts.append("\n\n# " + "=" * 70)
    output_parts.append("# HEADING (YAW) RESIDUAL + JACOBIANS")
    output_parts.append("# " + "=" * 70 + "\n")

    for name, code in processed_heading.items():
        output_parts.append(f"\n# --- From: {name} ---\n")
        code = re.sub(r'^import math\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import typing as T\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import numpy\s*$', '', code, flags=re.MULTILINE)
        # Normalize function name (SymForce may truncate to heading_residual_with_jacobian)
        code = re.sub(r'def heading_residual_with_jacobians?\d*\(',
                       'def heading_residual_with_jacobians(', code)
        output_parts.append(code.strip())

    # Add separator and preintegrated rotation functions
    output_parts.append("\n\n# " + "=" * 70)
    output_parts.append("# PREINTEGRATED IMU ROTATION RESIDUAL + JACOBIANS")
    output_parts.append("# " + "=" * 70 + "\n")

    for name, code in processed_preint_rot.items():
        output_parts.append(f"\n# --- From: {name} ---\n")
        code = re.sub(r'^import math\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import typing as T\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import numpy\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'def preintegrated_rotation_residual_with_jacobians\d*\(',
                       'def preint_rot_residual_with_jacobians(', code)
        output_parts.append(code.strip())

    # Add separator and preintegrated velocity functions
    output_parts.append("\n\n# " + "=" * 70)
    output_parts.append("# PREINTEGRATED IMU VELOCITY RESIDUAL + JACOBIANS")
    output_parts.append("# " + "=" * 70 + "\n")

    for name, code in processed_preint_vel.items():
        output_parts.append(f"\n# --- From: {name} ---\n")
        code = re.sub(r'^import math\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import typing as T\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import numpy\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'def preintegrated_velocity_residual_with_jacobians\d*\(',
                       'def preint_vel_residual_with_jacobians(', code)
        output_parts.append(code.strip())

    # Add separator and preintegrated position functions
    output_parts.append("\n\n# " + "=" * 70)
    output_parts.append("# PREINTEGRATED IMU POSITION RESIDUAL + JACOBIANS")
    output_parts.append("# " + "=" * 70 + "\n")

    for name, code in processed_preint_pos.items():
        output_parts.append(f"\n# --- From: {name} ---\n")
        code = re.sub(r'^import math\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import typing as T\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^import numpy\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'def preintegrated_position_residual_with_jacobians\d*\(',
                       'def preint_pos_residual_with_jacobians(', code)
        output_parts.append(code.strip())

    # Write output file
    output_path = os.path.join(os.path.dirname(__file__), "generated_jacobians.py")
    output_content = "\n".join(output_parts) + "\n"

    with open(output_path, 'w') as f:
        f.write(output_content)

    print(f"\n✅ Written: {output_path}")
    print(f"   Size: {len(output_content)} chars, {output_content.count(chr(10))} lines")

    # Quick validation: try importing the generated file
    print("\nValidating generated code...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("generated_jacobians", output_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Check that key functions exist
    assert hasattr(mod, 'Rot3'), "Missing Rot3 class"
    assert hasattr(mod, 'radar_residual_with_jacobians'), "Missing radar_residual_with_jacobians"
    assert hasattr(mod, 'accel_residual_with_jacobians'), "Missing accel_residual_with_jacobians"
    assert hasattr(mod, 'gyro_residual_with_jacobians'), "Missing gyro_residual_with_jacobians"
    assert hasattr(mod, 'gravity_residual_with_jacobians'), "Missing gravity_residual_with_jacobians"
    assert hasattr(mod, 'heading_residual_with_jacobians'), "Missing heading_residual_with_jacobians"
    assert hasattr(mod, 'preint_rot_residual_with_jacobians'), "Missing preint_rot_residual_with_jacobians"
    assert hasattr(mod, 'preint_vel_residual_with_jacobians'), "Missing preint_vel_residual_with_jacobians"
    assert hasattr(mod, 'preint_pos_residual_with_jacobians'), "Missing preint_pos_residual_with_jacobians"
    print("   Found: radar, accel, gyro, gravity, heading, "
          "preint_rot, preint_vel, preint_pos, Rot3")

    # Quick numerical test
    import numpy as np
    R_nom = mod.Rot3([0, 0, 0, 1])  # identity
    R_bs = mod.Rot3([0, 0, 0, 1])   # identity

    # Test radar
    result = mod.radar_residual_with_jacobians(
        np.array([1.0, 0.0, 0.0]),   # v_world
        R_nom,                       # R_nominal
        np.array([0.0, 0.0, 0.0]),   # delta (zero = identity)
        np.array([0.0, 0.0, 0.0]),   # omega
        np.array([1.0, 0.0, 0.0]),   # u_sensor
        np.array([0.07, 0.0, 0.0]),  # t_body_sensor
        R_bs,                        # R_body_sensor
        0.5,                         # v_meas
        1e-10,                       # epsilon
    )
    print(f"   radar: residual={result[0]}, "
          f"J_v={result[1].shape}, J_delta={result[2].shape}, "
          f"J_omega={result[3].shape}, J_Rbs={result[4].shape}")

    # Verify: v_world=[1,0,0], R=I, u=[1,0,0] => v_pred=-dot=-1.0, residual=0.5-(-1.0)=1.5
    assert abs(result[0][0] - 1.5) < 1e-6, f"Radar residual sanity check failed: {result[0]}"

    # Verify J_v_world: ∂r/∂v_world = +u_body^T R^T (with R=I, u=[1,0,0] => [+1,0,0])
    # (negated v_pred means ∂r/∂v = +∂dot/∂v = u_body^T R^T)
    assert abs(result[1][0] - 1.0) < 1e-6, f"J_v_world[0] should be +1: {result[1]}"

    # Test accel
    result_accel = mod.accel_residual_with_jacobians(
        np.array([0.0, 0.0, 9.81]),    # a_world
        R_nom,                            # R_nominal
        np.array([0.0, 0.0, 0.0]),      # delta
        np.array([0.0, 0.0, -9.81]),    # g_world
        np.array([0.0, 0.0, 19.62]),    # z_acc
        np.array([0.0, 0.0, 0.0]),      # b_a
        1e-10,                            # epsilon
    )
    print(f"   accel: residual={result_accel[0].flatten()}, "
          f"J_a={result_accel[1].shape}, J_delta={result_accel[2].shape}, J_ba={result_accel[3].shape}")

    # Verify: a_world - g = [0,0,19.62], R=I, so pred = [0,0,19.62], z=[0,0,19.62] => res=[0,0,0]
    assert np.allclose(result_accel[0].flatten(), [0, 0, 0], atol=1e-6), \
        f"Accel residual sanity check failed: {result_accel[0]}"

    # Verify J_b_a = -I (bias Jacobian)
    assert np.allclose(result_accel[3], -np.eye(3), atol=1e-6), \
        f"J_b_a should be -I: {result_accel[3]}"

    # Test gyro (identity rotation, zero delta)
    result_gyro = mod.gyro_residual_with_jacobians(
        np.array([0.0, 0.0, 1.0]),      # omega_nominal
        np.array([0.0, 0.0, 0.0]),      # delta (zero)
        np.array([0.1, 0.2, 0.3]),      # delta_dot
        np.array([0.1, 0.2, 1.3]),      # z_gyro = omega_nom + delta_dot = [0.1, 0.2, 1.3]
        np.array([0.0, 0.0, 0.0]),      # b_g (zero)
        1e-10,                            # epsilon
    )
    print(f"   gyro: residual={result_gyro[0].flatten()}, "
          f"J_delta={result_gyro[1].shape}, J_delta_dot={result_gyro[2].shape}, J_bg={result_gyro[3].shape}")

    # At delta=0: J_r=I, exp(-[0]_x)=I, so omega = omega_nom + delta_dot = [0.1, 0.2, 1.3]
    # z_gyro = [0.1, 0.2, 1.3], b_g=0 => residual should be [0, 0, 0]
    assert np.allclose(result_gyro[0].flatten(), [0, 0, 0], atol=1e-6), \
        f"Gyro residual sanity check failed: {result_gyro[0]}"

    # At delta=0: J_delta_dot should be -J_r(0) = -I
    assert np.allclose(result_gyro[2], -np.eye(3), atol=1e-6), \
        f"J_delta_dot at delta=0 should be -I: {result_gyro[2]}"

    # J_b_g should be -I
    assert np.allclose(result_gyro[3], -np.eye(3), atol=1e-6), \
        f"J_b_g should be -I: {result_gyro[3]}"

    # Test gyro with large delta (pi/2 rotation around z)
    delta_large = np.array([0.0, 0.0, np.pi / 2])
    omega_nom = np.array([1.0, 0.0, 0.0])
    delta_dot_large = np.array([0.0, 0.0, 0.0])
    # exp(-[delta]_x) * omega_nom rotates omega_nom by -pi/2 around z: [1,0,0] -> [0,-1,0]...
    # Actually exp(-delta_hat)*omega = R(-pi/2 around z)*[1,0,0] = [0,1,0]
    # R_z(-pi/2) = [[0,1,0],[-1,0,0],[0,0,1]] when applied as R@v
    # Wait: R_delta = exp([0,0,pi/2]_x) = Rz(pi/2). R_delta^T = Rz(-pi/2).
    # Rz(-pi/2) @ [1,0,0] = [0,-1,0]... let me compute:
    # Rz(theta) = [[cos,-sin,0],[sin,cos,0],[0,0,1]]
    # Rz(pi/2) = [[0,-1,0],[1,0,0],[0,0,1]]
    # Rz(-pi/2) = [[0,1,0],[-1,0,0],[0,0,1]]
    # Rz(-pi/2) @ [1,0,0] = [0,-1,0]
    expected_omega = np.array([0.0, -1.0, 0.0])
    result_gyro_large = mod.gyro_residual_with_jacobians(
        omega_nom, delta_large, delta_dot_large,
        expected_omega,  # z_gyro = expected omega
        np.array([0.0, 0.0, 0.0]), 1e-10,
    )
    assert np.allclose(result_gyro_large[0].flatten(), [0, 0, 0], atol=1e-4), \
        f"Gyro residual at large delta failed: {result_gyro_large[0].flatten()}"
    print(f"   gyro (large delta pi/2): residual={result_gyro_large[0].flatten()} ✓")

    # Test gravity residual
    # At identity rotation, hovering (z_acc ≈ [0,0,9.81] in body=world frame):
    # g_world = [0, 0, -9.81], R=I => g_body_predicted = R^T @ g_world = [0, 0, -9.81]
    # z_debiased = [0, 0, 9.81], normalized * 9.81 = [0, 0, 9.81]
    # At hover with R=I and zero bias:
    #   z_acc = [0, 0, +9.81]  (accel measures reaction/contact force, pointing UP)
    #   g_body_predicted = R^T @ [0, 0, +9.81] = [0, 0, +9.81]
    #   residual = [0,0,+9.81] - [0,0,+9.81] = [0, 0, 0]  ✓
    result_grav = mod.gravity_residual_with_jacobians(
        R_nom,                             # R_nominal (identity)
        np.array([0.0, 0.0, 0.0]),         # delta (zero)
        np.array([0.0, 0.0, 9.81]),        # z_acc (hover: accel reads +g upward)
        np.array([0.0, 0.0, 0.0]),         # b_a (zero bias)
        9.81,                              # g_norm
        1e-10,                             # epsilon
    )
    print(f"   gravity: residual={result_grav[0].flatten()}, "
          f"J_delta={result_grav[1].shape}, J_ba={result_grav[2].shape}")
    assert result_grav[1].shape == (3, 3), f"J_delta shape should be (3,3): {result_grav[1].shape}"
    assert result_grav[2].shape == (3, 3), f"J_ba shape should be (3,3): {result_grav[2].shape}"
    assert np.allclose(result_grav[0].flatten(), [0, 0, 0], atol=1e-6), \
        f"Gravity residual (z_acc=+g, R=I) should be [0,0,0]: {result_grav[0].flatten()}"
    print(f"   gravity (z_acc=+g, R=I): residual={result_grav[0].flatten()} ✓")

    # Test heading residual
    # At identity rotations (both R_est and R_mocap = I), yaw error = 0
    result_hdg = mod.heading_residual_with_jacobians(
        R_nom,                              # R_nominal (identity)
        np.array([0.0, 0.0, 0.0]),          # delta (zero)
        R_nom,                              # R_mocap (identity)
        1e-10,                              # epsilon
    )
    print(f"   heading: residual={result_hdg[0].flatten()}, J_delta={result_hdg[1].shape}")
    assert result_hdg[1].shape in ((1, 3), (3,)), f"J_delta shape should be (1,3) or (3,): {result_hdg[1].shape}"
    assert abs(result_hdg[0][0]) < 1e-6, f"Heading residual (both I) should be 0: {result_hdg[0]}"
    print(f"   heading (both identity): residual={result_hdg[0].flatten()} ✓")

    # Test heading with 90-deg yaw difference: R_est = Rz(pi/2) vs R_mocap = I
    # x_est = Rz(pi/2) @ [1,0,0] = [0,1,0]; x_moc = [1,0,0]
    # cross_z = 0*0 - 1*1 = -1; dot = 0*1 + 1*0 = 0
    # atan2(-1, 0) = -pi/2
    from scipy.spatial.transform import Rotation as _Rot
    R_90z = _Rot.from_rotvec([0, 0, np.pi/2]).as_matrix()
    R_90z_quat = mod.Rot3.from_rotation_matrix(R_90z)
    result_hdg_90 = mod.heading_residual_with_jacobians(
        R_90z_quat,
        np.array([0.0, 0.0, 0.0]),
        R_nom,
        1e-10,
    )
    assert abs(result_hdg_90[0][0] - (-np.pi/2)) < 1e-4, \
        f"Heading residual (Rz90 vs I) should be -pi/2: {result_hdg_90[0][0]}"
    print(f"   heading (Rz(90) vs I): residual={np.degrees(result_hdg_90[0][0]):.1f} deg ✓")

    print("\n✅ Code generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
