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

# Live RIO solver — C++ backend, batch (CURRENT BEST, ~15-20s, 2-3× better than Python)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp
# --cpp loads config/solver_cpp.yaml overrides automatically (full-rate IMU, tighter priors)
# --set key=value overrides any solver.yaml param at runtime (repeatable)
# --imu-hz N overrides IMU rate (default: 1000 for --cpp, 200 for Python)

# Live RIO solver — C++ sliding window (Phase 4b: Schur complement marginalization)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp --sliding-window
# --sliding-window: fixed-lag smoother with Schur complement marginalization
# Default: window=3.0s, stride=0.3s, marg_prior_scale=2e-4
# slow_racing: 0.205m / 1.57°  (vs batch 0.180m / 1.07°)  ~1.5s/window
# fast_racing: 0.676m / 3.01°  (vs batch 1.110m / 2.56° — better in position!)
# Evaluation trims last window_duration seconds (no subsequent window to correct final drift)
# marg_prior_scale key: raw Schur info O(10^5) >> lambda_boundary=1000; scale down to match
# Tune: --set marg_prior_scale=X  (sweep 1e-4 to 1e-3 for new datasets)

# Live RIO solver — Python backend (~10 min)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw
# Flags: --mocap-yaw (heading+pos priors), --no-plot, --gnc, --preintegrate
# Multi-bag eval: ../.venv/bin/python3 eval_bags.py --label baseline --flags "--mocap-yaw"

# Batch solver (MoCap-initialized, older pipeline)
python validate_nonlinear_solver.py circle_fwd   # Phase 3: full LM solver

# Earlier pipeline phases
python validate_physics.py original              # Ground truth forward model validation (no optimization)
python validate_forward_model.py                 # Doppler prediction accuracy
python validate_linear_solver.py                 # Phase 2: sparse linear LS for position only

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
| `validate_live_solver.py` | **Live RIO: MoCap-free P1-P3 init + solver (main entry point)** |
| `validate_nonlinear_solver.py` | Batch solver: full LM with MoCap init (shared solver core) |
| `lib/radar_velocity_utils.py` | Forward model, WLS ego-velocity solver, Huber loss, extrinsic calibration |
| `lib/bspline_utils.py` | Uniform B-splines (Cox-de Boor), derivatives, min-snap regularization |
| `lib/cumulative_so3_bspline.py` | Cumulative SO(3) B-spline on Lie groups: evaluate R(t), body-rate ω(t), Jacobians, initialization from rotation samples |
| `lib/imu_preintegration.py` | Forster TRO-2017 on-manifold preintegration (--preintegrate flag) |
| `codegen/generated_jacobians.py` | SymForce-generated residuals + Jacobians for radar, accel, gyro factors |
| `codegen/derive_jacobians_symforce.py` | Source for regenerating `generated_jacobians.py` |
| `lib/rosbag_loader/loader.py` | Unified API to load 7 ROS topics into typed dataclasses |
| `config_loader.py` | Loads all YAML configs from `config/` as a dict-of-dicts |
| `config/extrinsics.yaml` | Extrinsics: [180, 25.5, 0] deg (solver-calibrated pitch, physical 30°), translation [0.08,0.02,-0.01] m |
| `config/bags.yaml` | Bag aliases → paths, flipped bag set, per-bag timing windows |
| `config/solver.yaml` | Python solver hyperparameters (default) |
| `config/solver_cpp.yaml` | C++ solver overrides loaded on `--cpp` (IMU rate, snap, gyro weight, bias prior) |
| `diagnostics/diagnose_doppler_sign.py` | Compares both Doppler code paths at MoCap ground truth; tests sign convention |

### C++ Ceres Solver (`rio_solver_cpp/`)

Batch C++ solver called via pybind11 from `validate_live_solver.py --cpp`.

| File | Role |
|------|------|
| `CMakeLists.txt` | Build (Release): `cd build_release && cmake .. && cmake --build . -j$(nproc)` |
| `include/rio/trajectory.h` | Trajectory state: pos CPs + quaternion knots + biases |
| `include/rio/factors/` | Ceres cost functors: radar Doppler, accel, gyro, gravity, heading, regularization, bias prior |
| `include/rio/solver.h` | Public API (`SolverConfig`, `SolverResult`, `solve()`) |
| `src/solver.cpp` | Problem construction + Ceres LM solve |
| `src/pybind_module.cpp` | Python↔C++ bridge |
| `tests/test_spline.cpp` | Phase 1 validation: spline eval + trajectory indexing |
| `scripts/build.sh` | Convenience build script |

**Orientation convention**: quaternion knots [x,y,z,w] = `_base_rotations[i]` from Python's `CumulativeSO3BSpline`. Uses basalt `CeresSplineHelper<N>::evaluate_lie()`.

**C++ results by spline config** (--mocap-yaw --cpp, lever arm correction, eval trims last 3s)

Two configs compared (both use solver_cpp.yaml overrides on top):
- **old**: `dt_pos=0.005s, dt_ori=0.008s` — original spacing, best for racing
- **new**: `dt_pos=0.010s, dt_ori=0.0008s` — coarser pos / 10× finer ori, best for backflips

| Bag | Config | Mode | Pos RMSE | Vel RMSE | Ori RMSE | Acc bias (m/s²) | Gyr bias (rad/s) |
|-----|--------|------|----------|----------|----------|-----------------|-----------------|
| slow_racing | old | batch | **0.160m** | 0.143 | **1.08°** | [-0.005, 0.049, 0.236] | [0.005, -0.003, 0.000] |
| slow_racing | old | sliding-win | 0.210m | 0.137 | 1.58° | [-0.007, 0.046, 0.215] | [0.009, -0.003, 0.006] |
| slow_racing | new | batch | 0.289m | 0.449 | 3.46° | [-0.008, 0.038, 0.236] | [-0.000, -0.003, 0.003] |
| slow_racing | new | sliding-win | 1.867m | 1.081 | 19.33° ⚠️ | [-0.009, 0.047, 0.207] | [0.155, -1.009, -0.243] ⚠️ |
| fast_racing | old | batch | 0.791m | 0.397 | 2.57° | [-0.009, 0.007, 0.064] | [0.006, 0.001, -0.002] |
| fast_racing | old | sliding-win | **0.676m** | 0.376 | 3.01° | [-0.016, 0.006, 0.088] | [0.005, 0.007, 0.007] |
| fast_racing | new | batch | 1.612m | 0.797 | 9.90° | [-0.007, 0.005, 0.058] | [0.004, -0.003, 0.000] |
| fast_racing | new | sliding-win | 1.561m | 0.839 | 21.71° ⚠️ | [-0.007, 0.004, 0.060] | [0.559, 2.527, -2.944] ⚠️ |
| backflips | old | batch | 5.060m | 4.006 | 7.61° | [0.041, 0.082, 0.246] | [0.002, 0.000, -0.001] |
| backflips | old | sliding-win | 3.557m | 2.791 | 11.28° | [0.043, 0.083, 0.483] | [-0.019, -0.071, -0.003] |
| backflips | new | batch | **1.817m** | 1.953 | 8.41° | [0.044, 0.082, 0.220] | [-0.001, 0.000, 0.001] |
| backflips | new | sliding-win | 1.936m | 2.475 | 41.93° ⚠️ | [0.047, 0.082, 0.357] | [-0.189, 0.098, 1.006] ⚠️ |

⚠️ Sliding window unstable with new config: dt_ori=0.0008s → ~3750 ori knots per 3s window
vs ~3000 IMU samples → underconstrained. lambda_ori_reg=0.01 (10× increase) made no difference.
New config only viable for batch mode.

Note: lever arm (ω × r_antenna) added to C++ RadarDopplerFunctor in Phase 4b.
Pre-lever-arm batch (old config): slow 0.146m/0.96°, fast 0.925m/2.35°, backflips 2.93m/10.7°.

**C++ solver config** (`config/solver_cpp.yaml` overrides vs `solver.yaml`):
- `lambda_gyro`: 1.0 → **4.0** (tighter orientation constraint)
- `lambda_snap_pos`: 1e-4 → **2e-5** (less over-smoothing for racing dynamics)
- `lambda_bias_prior_accel/gyro`: 1.0 → **10000** (full-rate IMU makes tight prior safe; prevents bias trash-can)

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
- **Rotation**: `[roll=180°, pitch=25.5°, yaw=0°]` — solver-calibrated pitch (physical mount 30°, self-adjusts per run when optimize_pitch_only=true)
- **Translation**: `[0.08, +0.02, -0.01]` m in body frame (8 cm forward, 2 cm left, 1 cm down)
- Body frame: x=forward, y=left, z=up

**Critical**: `optimize_pitch_only: true` in `solver.yaml` must stay enabled. Only pitch is
observable from Doppler; free roll/yaw optimization drifts 5–7° and corrupts orientation.

The radar has limited elevation diversity (2 TX antennas), causing a systematic z-velocity bias of −0.5 to −0.65 m/s. Doppler quantization is 0.63 m/s per bin — keep Huber δ ≥ 1.0 m/s.

## Key Hyperparameters

### `config/solver.yaml` — Python solver defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| `huber_delta` | 1.0 m/s | Must be ≥ radar Doppler bin size (0.63 m/s) |
| `lambda_accel` | 0.01 | Accelerometer weight |
| `lambda_gyro` | 1.0 | Gyroscope weight |
| `lambda_snap_pos` | 0.0001 | Min-snap position regularization |
| `lambda_ori_reg` | 0.001 | Orientation increment regularization |
| `lambda_bias_prior_accel` | 1.0 | Relaxed — biases free to adjust |
| `lambda_bias_prior_gyro` | 1.0 | Same |
| `lambda_boundary_vel/pos/ori` | 1000.0 | Anchor start of trajectory |
| `optimize_pitch_only` | **true** | **Must stay true** — only pitch is Doppler-observable |
| `max_iterations` | 400 | Used by C++ solver; Python uses early-stop criteria |

### `config/solver_cpp.yaml` — C++ overrides (applied on `--cpp`)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lambda_gyro` | **4.0** | Tighter orientation; reduces acc_bias blow-up |
| `lambda_snap_pos` | **2e-5** | Less over-smoothing for racing dynamics |
| `lambda_bias_prior_accel` | **10000** | Full-rate IMU provides enough constraints; prevents trash-can |
| `lambda_bias_prior_gyro` | **10000** | Same |

## Known Gaps / TODO

### Orientation regularization: angular acceleration (∫||dω/dt||²) replaces angular velocity

**Implemented.** `lambda_ori_reg` (min-ω) disabled; `lambda_ori_accel` (min-α) active.

The old `OrientationRegFunctor` penalized `||log(q_i^{-1}·q_{i+1})||²` (minimum angular velocity),
which fought every banked turn and the entire backflip maneuver.

`AngularAccelRegFunctor` penalizes the second finite difference:
```
r = log(q_{i-1}^{-1}·q_i) - log(q_i^{-1}·q_{i+1})
```
Zero for constant angular rate. Only fires at maneuver onset/offset.

**Lambda sweep results** (--mocap-yaw --cpp, batch, backflips uses dt_ori=0.0008):

| lambda_ori_accel | slow_racing pos | fast_racing pos | backflips ori |
|---|---|---|---|
| 0.0 (none) | **0.150m** | 0.875m | 9.59° |
| 0.001 | 0.178m | 0.786m | 8.61° |
| 0.01 | 0.178m | 0.790m | 8.43° |
| **0.1** ← default | 0.170m | **0.740m** | **8.31°** |
| 1.0 | 0.174m | 0.796m | 44.2° ⚠️ |

0.1 is best compromise. 1.0 blows up backflips (regularizer too tight to represent rapid angular
acceleration of the flip itself). Orientation RMSE insensitive across all racing bags — 1000 Hz
gyro dominates; regularizer only matters for position and backflip stability.

Current defaults: `lambda_ori_reg: 0.0`, `lambda_ori_accel: 0.1`

### C++ solver: extrinsic pitch optimization not implemented

The Python solver optimizes radar extrinsics (pitch angle) as a free parameter during solve.
The C++ solver does **not** — `R_radar_to_body` is baked in as a constant from `ExtrinsicConfig`
and never added as a Ceres parameter block, even though `SolverConfig` has `lock_extrinsics` and
`optimize_pitch_only` fields that are read from YAML but currently ignored.

**Effect:** C++ always uses the fixed `pitch_deg=25.5°`. If the true pitch differs per-run, the
systematic Doppler error accumulates across all radar measurements. Python self-calibrates this
each run.

**What needs to be done:**
- Add a 1-DOF pitch parameter (or 3-DOF so(3) delta with roll/yaw frozen) to `solver.cpp`
- Pass it into `RadarDopplerFunctor` as a parameter block instead of constant
- The comment in `radar_doppler.h:34-36` already anticipates this design
- When `optimize_pitch_only=true`: use a 1-param block for pitch increment only
- When `lock_extrinsics=true`: skip entirely (current behaviour)
- Also port to `sliding_window_solver.cpp`

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
