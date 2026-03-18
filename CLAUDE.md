# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Radar-inertial odometry (RIO) research system using a TI IWR6843AOPEVM mmWave radar on an Agiros quadrotor with Pixhawk IMU and Vicon MoCap ground truth. The goal is to fit full 6-DOF trajectories from radar Doppler velocity measurements, accelerometer, and gyroscope data using B-spline parameterization and factor graph optimization.

## Environment Setup

```bash
# Python analysis (from analysis/ directory)
uv pip install -r requirements.txt
# or use the existing .venv
source .venv/bin/activate

# ROS1 driver (C++, requires Docker)
docker compose -f docker/docker-compose.yml up
cd mmwave_ti_ros/ros1_driver
catkin_make
source devel/setup.bash
roslaunch ti_mmwave_rospkg 6843AOP_velocity_3d.launch
```

## Running Analysis Scripts

All scripts run from `analysis/` and take a bag name as the first argument:

```bash
cd analysis/

# Bag name aliases: original, circle, circle_fast, circle_fwd, loopings, backflips
# (plus newer bags: backflips_best_velocity, circle_best_velocity, fast_racing_*, slow_racing_*)
python validate_physics.py original          # Ground truth forward model validation (no optimization)
python validate_forward_model.py             # Doppler prediction accuracy
python validate_linear_solver.py             # Phase 2: sparse linear LS for position only
python validate_nonlinear_solver.py          # Phase 3: full LM solver (main script)

# Diagnostics (in diagnostics/ subdir — run from analysis/)
python diagnostics/diagnose_doppler.py circle
python diagnostics/diagnose_gyro.py circle

# Visualization
python viz/plot_radar_map.py circle_fwd      # Interactive 3D radar map (Open3D)
python viz/plot_extrinsics.py                # Show extrinsic calibration frames
python viz/plot_extrinsics.py circle_fwd     # Show with yaw-flip applied
```

Outputs go to `../plots/`. No test framework — validation is script-driven.

## Regenerating Jacobians

```bash
# From the repo root (SymForce is in the root .venv, not the analysis venv)
source .venv/bin/activate
python analysis/codegen/derive_jacobians_symforce.py   # Overwrites analysis/codegen/generated_jacobians.py
```

`codegen/generated_jacobians.py` has zero runtime dependency on SymForce — it's pure NumPy.

## Architecture

### Three-Phase Optimization Pipeline

**Phase 1 — Physics Diagnostics** (`validate_physics.py`): Feed MoCap ground truth through the forward model to verify coordinate transforms, Doppler sign, and extrinsic calibration. No optimization.

**Phase 2 — Linear Solver** (`validate_linear_solver.py`): Use known orientation (MoCap SLERP) and solve sparse least squares for position B-spline control points only. Validates observability and Jacobian construction.

**Phase 3 — Nonlinear Solver** (`validate_nonlinear_solver.py`): Full Levenberg-Marquardt with sparse Cholesky. Jointly optimizes:
- Position trajectory (quintic B-spline control points)
- Orientation trajectory (cumulative SO(3) B-spline on Lie groups with incremental rotation control points Ω_j)
- Constant accelerometer and gyroscope biases

Flags: `--no-flip` (override per-bag yaw flip), `--flip` (force flip on), `--no-radar` (disable radar, IMU-only), `--precond` (Jacobi preconditioning)

### Key Modules

| File | Role |
|------|------|
| `lib/radar_velocity_utils.py` | Forward model, WLS ego-velocity solver, Huber loss, extrinsic calibration |
| `lib/bspline_utils.py` | Uniform B-splines (Cox-de Boor), derivatives, min-snap regularization |
| `lib/cumulative_so3_bspline.py` | Cumulative SO(3) B-spline on Lie groups: evaluate R(t), body-rate ω(t), Jacobians, initialization from rotation samples |
| `codegen/generated_jacobians.py` | SymForce-generated residuals + Jacobians for radar, accel, gyro factors |
| `codegen/derive_jacobians_symforce.py` | Source for regenerating `generated_jacobians.py` |
| `lib/rosbag_loader/loader.py` | Unified API to load 7 ROS topics into typed dataclasses |
| `config_loader.py` | Loads all YAML configs from `config/` as a dict-of-dicts |
| `config/extrinsics.yaml` | Canonical extrinsics: rotation [180,30,0] deg, translation [0.08,0.02,-0.01] m |
| `config/bags.yaml` | Bag aliases → paths, flipped bag set, per-bag timing windows |
| `config/solver.yaml` | LM hyperparameters, B-spline config |
| `diagnostics/diagnose_doppler_sign.py` | Compares both Doppler code paths at MoCap ground truth; tests sign convention |

### State Representation

```
Position:     Quintic B-spline (degree 5) with control points P_i, knot spacing 0.05s
Orientation:  R(t) = R_base[k-3] · ∏ exp(B̃_j(t) · Ω_j)   (cumulative product over j=k-3..k)
              Ω_j ∈ so(3): incremental rotation control points, initialized from MoCap SLERP
              R_base[k-3]: left anchor rotation for the active spline segment
              (no relinearization needed — exact on-manifold representation)
Biases:       Constant b_a (accel), b_g (gyro)
Regularization: position: minimum-snap (∫||P⁴(t)||² dt)
                orientation: per-knot increment penalty (λ · ∑||Ω_j||²)
```

### Sensor Models / Residuals

- **Radar**: `r = v_meas - v_pred` where `v_pred = -dot(u_body, v_ant)`. TI IWR6843 convention: positive Doppler = receding target. Huber loss, δ = 1.0 m/s.
- **Accelerometer**: L2 loss on specific force residual `z_acc - R_bw(a_world - g) - b_a`
- **Gyroscope**: L2 loss on angular velocity residual `z_gyro - ω_body - b_g`

**Critical**: The negation in `v_pred = -dot(u,v)` is physically correct and confirmed by experiment (see FINDINGS.md §11). Do not remove it.

### Rosbag Datasets

Located at `../rosbags/`. Alias → filename mapping is in `config/bags.yaml`. Some bags are listed in the `flipped` set (applies `R_z(180°)` to extrinsics). After the Doppler sign fix, `slow_racing_best_velocity` was confirmed to work without the flip. The other flipped bags (`circle_fwd`, `loopings`, `backflips`) are pending re-evaluation.

### Coordinate Frames & Calibration

Extrinsic calibration lives in `config/extrinsics.yaml` (single source of truth):
- **Rotation**: `[roll=180°, pitch=30°, yaw=0°]` — 180° roll (upside-down mount) + 30° downtilt
- **Translation**: `[0.08, +0.02, -0.01]` m in body frame (8 cm forward, 2 cm left, 1 cm down)
- Body frame: x=forward, y=left, z=up

The radar has limited elevation diversity (2 TX antennas), causing a systematic z-velocity bias of −0.5 to −0.65 m/s. Doppler quantization is 0.63 m/s per bin — keep Huber δ ≥ 1.0 m/s.

## Key Hyperparameters (in `validate_nonlinear_solver.py` and `config/solver.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `HUBER_DELTA` | 1.0 m/s | Must be ≥ radar Doppler bin size (0.63 m/s) |
| `IMU_MOCAP_OFFSET` | +0.020 s | Empirically determined time alignment |
| `LAMBDA_ACCEL` | 0.01 | Accelerometer weight |
| `LAMBDA_GYRO` | 1.0 | Gyroscope weight (primary orientation sensor) |
| `LAMBDA_ORI_REG` | 0.001 | Orientation increment regularization (penalizes `‖Ω_j‖²` per knot) |
| `ORI_BASE_JACOBIAN_WINDOW` | 20 | Max base knots per measurement for orientation Jacobian (~1 s); 0 = full exact |
| `LAMBDA_BIAS_PRIOR_ACCEL` | 1000.0 | Strong prior prevents gravity leakage into bias |
| `LAMBDA_BIAS_PRIOR_GYRO` | 10000.0 | Very strong prior on gyro bias |
| `LAMBDA_BOUNDARY_VEL/POS` | 1000.0 | Anchor start of trajectory to MoCap |
| `LAMBDA_BOUNDARY_ORI` | 10000.0 | Anchor start orientation to MoCap |

## ROS Topics

| Topic | Content |
|-------|---------|
| `/mmWaveDataHdl/RScanVelocity` | Radar point cloud (x, y, z, velocity, intensity, range, noise, frame_number) |
| `/agiros_pilot/imu` | IMU (accel + gyro) |
| `/mocap_node/Agiros/pose` | MoCap 6-DOF pose |
| `/mocap_node/Agiros/accel` | MoCap linear acceleration |
| `/agiros_pilot/state` | Full Agiros state |
| `/agiros_pilot/odometry` | Agiros odometry |

See `analysis/lib/rosbag_loader/RADAR_FIELDS.md` for field-level documentation and `analysis/lib/rosbag_loader/README.md` for module overview.
