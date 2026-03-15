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
python validate_physics.py original          # Ground truth forward model validation (no optimization)
python validate_forward_model.py             # Doppler prediction accuracy
python validate_linear_solver.py             # Phase 2: sparse linear LS for position only
python validate_nonlinear_solver.py          # Phase 3: full LM solver (main script)

# Diagnostics
python diagnose_doppler.py original
python diagnose_gyro.py original
```

Outputs go to `../plots/`. No test framework — validation is script-driven.

## Regenerating Jacobians

```bash
cd analysis/
python derive_jacobians_symforce.py   # Overwrites generated_jacobians.py
```

`generated_jacobians.py` has zero runtime dependency on SymForce — it's pure NumPy.

## Architecture

### Three-Phase Optimization Pipeline

**Phase 1 — Physics Diagnostics** (`validate_physics.py`): Feed MoCap ground truth through the forward model to verify coordinate transforms, Doppler sign, and extrinsic calibration. No optimization.

**Phase 2 — Linear Solver** (`validate_linear_solver.py`): Use known orientation (MoCap SLERP) and solve sparse least squares for position B-spline control points only. Validates observability and Jacobian construction.

**Phase 3 — Nonlinear Solver** (`validate_nonlinear_solver.py`): Full Levenberg-Marquardt with sparse Cholesky. Jointly optimizes:
- Position trajectory (cubic B-spline control points)
- Orientation trajectory (SO(3) B-spline as tangent-space perturbations `δ(t)` around MoCap nominal `R_nom(t)`)
- Constant accelerometer and gyroscope biases

### Key Modules

| File | Role |
|------|------|
| `radar_velocity_utils.py` | Forward model, WLS ego-velocity solver, Huber loss, extrinsic calibration |
| `bspline_utils.py` | Uniform B-splines (Cox-de Boor), derivatives, min-snap regularization |
| `generated_jacobians.py` | SymForce-generated residuals + Jacobians for radar, accel, gyro factors |
| `derive_jacobians_symforce.py` | Source for regenerating `generated_jacobians.py` |
| `rosbag_loader/loader.py` | Unified API to load 7 ROS topics into typed dataclasses |

### State Representation

```
Position:     B-spline with control points P_i (degree 3 or 5)
Orientation:  R(t) = R_nom(t) · exp(δ(t))
              R_nom = SLERP of MoCap quaternions
              δ(t) = B-spline in so(3) tangent space
Biases:       Constant b_a (accel), b_g (gyro)
Regularization: minimum-snap (∫||P⁴(t)||² dt)
```

### Sensor Models / Residuals

- **Radar**: Doppler residual with Huber loss (outlier-robust, δ = 1.0 m/s)
- **Accelerometer**: L2 loss on specific force residual
- **Gyroscope**: L2 loss on angular velocity residual

### Rosbag Datasets

Located at `../rosbags/`. Alias → filename mapping is in `validate_nonlinear_solver.py`. Notable: `circle_fwd` and `loopings` bags require a 180° yaw correction (documented in the solver).

### Coordinate Frames & Calibration

Extrinsic calibration (radar-to-IMU) is embedded in `radar_velocity_utils.py`. The radar has limited elevation diversity (2 TX antennas), causing a systematic z-velocity bias of −0.5 to −0.65 m/s. Doppler quantization is 0.63 m/s per bin — keep Huber δ ≥ 1.0 m/s.

## Key Hyperparameters (hardcoded in `validate_nonlinear_solver.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `HUBER_DELTA` | 1.0 m/s | Must be ≥ radar Doppler bin size (0.63 m/s) |
| `IMU_MOCAP_OFFSET` | +0.020 s | Empirically determined time alignment |
| `LAMBDA_ACCEL` | 0.01 | Lower = less accel influence on orientation |
| `GYRO_PRIOR` | 1.0 | Tight bias prior for gyro |
| `ACCEL_PRIOR` | 10.0 | Loose bias prior for accel |

## ROS Topics

| Topic | Content |
|-------|---------|
| `/mmWaveDataHdl/RScanVelocity` | Radar point cloud (x, y, z, velocity, intensity, range, noise, frame_number) |
| `/agiros_pilot/imu` | IMU (accel + gyro) |
| `/mocap_node/Agiros/pose` | MoCap 6-DOF pose |
| `/mocap_node/Agiros/accel` | MoCap linear acceleration |
| `/agiros_pilot/state` | Full Agiros state |
| `/agiros_pilot/odometry` | Agiros odometry |

See `analysis/rosbag_loader/RADAR_FIELDS.md` for field-level documentation.
