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
cd analysis/
python codegen/derive_jacobians_symforce.py   # Overwrites codegen/generated_jacobians.py
```

`codegen/generated_jacobians.py` has zero runtime dependency on SymForce — it's pure NumPy.

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
| `lib/radar_velocity_utils.py` | Forward model, WLS ego-velocity solver, Huber loss, extrinsic calibration |
| `lib/bspline_utils.py` | Uniform B-splines (Cox-de Boor), derivatives, min-snap regularization |
| `codegen/generated_jacobians.py` | SymForce-generated residuals + Jacobians for radar, accel, gyro factors |
| `codegen/derive_jacobians_symforce.py` | Source for regenerating `generated_jacobians.py` |
| `lib/rosbag_loader/loader.py` | Unified API to load 7 ROS topics into typed dataclasses |
| `config_loader.py` | Loads all YAML configs from `config/` as a dict-of-dicts |
| `config/extrinsics.yaml` | Canonical extrinsics: rotation [180,30,0] deg, translation [0,0.02,-0.01] m |
| `config/bags.yaml` | Bag aliases → paths, flipped bag set |
| `config/timing.yaml` | Per-bag flight window (start offset, duration) |
| `config/solver.yaml` | LM hyperparameters, B-spline config |

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

Located at `../rosbags/`. Alias → filename mapping is in `config/bags.yaml`. Notable: `circle_fwd`, `loopings`, `backflips`, `slow_racing_best_velocity` require a 180° yaw flip (see `flipped` list in `config/bags.yaml`).

### Coordinate Frames & Calibration

Extrinsic calibration lives in `config/extrinsics.yaml` (single source of truth):
- **Rotation**: `[roll=180°, pitch=30°, yaw=0°]` — 180° roll + 30° downtilt
- **Translation**: `[0.0, +0.02, -0.01]` m in body frame (2 cm left, 1 cm down)
- Body frame: x=forward, y=left, z=up

The radar has limited elevation diversity (2 TX antennas), causing a systematic z-velocity bias of −0.5 to −0.65 m/s. Doppler quantization is 0.63 m/s per bin — keep Huber δ ≥ 1.0 m/s.

## Key Hyperparameters (in `validate_nonlinear_solver.py` and `config/solver.yaml`)

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

See `analysis/lib/rosbag_loader/RADAR_FIELDS.md` for field-level documentation and `analysis/lib/rosbag_loader/README.md` for module overview.
