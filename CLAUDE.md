# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Map

| Document | Role |
|----------|------|
| **CLAUDE.md** (this file) | Operational hub: how to run, current config, current results |
| `report/IEEE-conference-template-062824.tex` | IEEE conference paper: methodology, ablations, negative results |
| `documentation/Forward Model.md` | Math reference: sensor forward models, coordinate frames, Doppler sign convention |
| `documentation/Backward Model.md` | Math reference: state parameterization (cumulative SO(3) B-spline), factor-graph MAP, Ceres LM, Schur-complement marginalization |
| `documentation/FINDINGS.md` | Foundational calibration findings: body frame, time offsets, Doppler sign fix (§11), extrinsics |
| `documentation/RESEARCH_NOTES.md` | Design rationale: solver perf profile, Doppler unwrapping, preintegration investigation, BandedSchurSolver post-mortem, SW timing analysis |
| `documentation/SW_DEVELOPMENT.md` | SW solver development history: phase ablations, sweep tables, backflips analysis, marg_prior_scale tuning |
| `analysis/lib/rosbag_loader/README.md` | Rosbag loader API and ROS topic reference |
| `rio_solver_cpp/README.md` | C++ Ceres solver: build instructions, Phase status, Python API |

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

# Live RIO solver — C++ backend, batch (CURRENT BEST, ~15-20s)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp
# --cpp loads config/solver_cpp.yaml overrides automatically (full-rate IMU, tighter priors)
# --set key=value overrides any solver.yaml param at runtime (repeatable)
# --imu-hz N overrides IMU rate (default: 1000 for --cpp, 200 for Python)

# Live RIO solver — C++ sliding window (Phase 4b: Schur complement marginalization)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp --sliding-window
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp --sliding-window
# Default: window=3.0s, stride=0.3s
# Per-bag marg_prior_scale: slow_racing=1e-7 (per-bag override), fast_racing=2e-4 (default)
# Avoid 1e-5–1e-6 range: harmful intermediate regime (worse than both extremes)
# Per-window diagnostics printed: jac/res/lin/other timing, iter count, prior cond/rank, tr(S⁻¹)/tr(H⁻¹)

# Backflips sliding window (Phase 3 config — must use --set overrides):
../.venv/bin/python3 validate_live_solver.py backflips_best_velocity --mocap-yaw --cpp --sliding-window \
  --set dt_ori=0.008 --set lambda_ori_accel=0.001 --set lock_gyro_bias=0 \
  --set marg_prior_scale=0.0 --set lambda_pos_init_prior=1000.0
# bags.yaml retains dt_ori=0.0008 for batch; --set dt_ori=0.008 overrides for SW

# Live RIO solver — Python backend (~10 min)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw
# Flags: --mocap-yaw (heading+pos priors), --no-plot, --gnc, --preintegrate
# Multi-bag eval: ../.venv/bin/python3 eval_bags.py --label baseline --flags "--mocap-yaw"

# Earlier pipeline phases (historical, superseded by --cpp)
python validate_nonlinear_solver.py circle_fwd   # MoCap-initialized batch solver
python validate_physics.py original              # Ground truth forward model validation
python validate_linear_solver.py                 # Sparse linear LS for position only

# Diagnostics and visualization
python diagnostics/diagnose_doppler.py circle
python viz/plot_radar_map.py circle_fwd      # Interactive 3D radar map (Open3D)
python viz/plot_extrinsics.py
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

### Key Modules

| File | Role |
|------|------|
| `validate_live_solver.py` | **Live RIO: MoCap-free P1-P3 init + solver (main entry point)** |
| `validate_nonlinear_solver.py` | Batch solver: full LM with MoCap init (shared solver core) |
| `lib/radar_velocity_utils.py` | Forward model, WLS ego-velocity solver, Huber loss, extrinsic calibration |
| `lib/bspline_utils.py` | Uniform B-splines (Cox-de Boor), derivatives, min-snap regularization |
| `lib/cumulative_so3_bspline.py` | Cumulative SO(3) B-spline on Lie groups: R(t), ω(t), Jacobians |
| `lib/imu_preintegration.py` | Forster TRO-2017 on-manifold preintegration (--preintegrate flag) |
| `codegen/generated_jacobians.py` | SymForce-generated residuals + Jacobians for radar, accel, gyro factors |
| `lib/rosbag_loader/loader.py` | Unified API to load 7 ROS topics into typed dataclasses |
| `config/extrinsics.yaml` | Extrinsics: [180, 25.5, 0] deg, translation [0.08, 0.02, -0.01] m |
| `config/bags.yaml` | Bag aliases → paths, flipped bag set, per-bag timing windows + solver_overrides |
| `config/solver.yaml` | Python solver hyperparameters (default) |
| `config/solver_cpp.yaml` | C++ solver overrides loaded on `--cpp` |

### Import Convention

Root `analysis/` scripts: `sys.path.insert(0, str(Path(__file__).parent / 'lib'))`, bare imports.
Subdirectory scripts (`diagnostics/`, `viz/`): add both `analysis/` and `analysis/lib/`.

### C++ Ceres Solver (`rio_solver_cpp/`)

| File | Role |
|------|------|
| `CMakeLists.txt` | Build: `cd build_release && cmake .. && cmake --build . -j$(nproc)` |
| `include/rio/solver.h` | Public API: `SolverConfig`, `SolverResult`, `solve()`, `SlidingWindowSolver` |
| `include/rio/factors/` | Cost functors: radar Doppler, accel/gyro (analytic + AutoDiff), regularization |
| `include/rio/factors/analytic/` | `GyroAnalyticFactor`, `AccelAnalyticFactor`, `RadarAnalyticFactor` — bypass Jet arithmetic |
| `include/sym/rot3.h` | Minimal `sym::Rot3` shim for SymForce-generated C++ headers |
| `include/rio/factors/analytic/radar_sensor_jac_gen.h` | SymForce-generated radar sensor-model Jacobians (re-run `derive_jacobians_symforce.py` to regenerate) |
| `src/solver.cpp` | Batch problem construction + Ceres LM solve |
| `src/sliding_window_solver.cpp` | SW problem construction, Schur complement marginalization |
| `src/pybind_module.cpp` | Python↔C++ bridge |

**Orientation convention**: quaternion knots [x,y,z,w] = `_base_rotations[i]` from Python's
`CumulativeSO3BSpline`. Uses basalt `CeresSplineHelper<N>::evaluate_lie()`.

### State Representation

```
Position:     Quintic B-spline (degree 5), control points P_i, knot spacing dt_pos
Orientation:  R(t) = R_base[k-3] · ∏ exp(B̃_j(t) · Ω_j)   (cumulative product, j=k-3..k)
              Ω_j ∈ so(3): incremental rotation control points
Biases:       Constant b_a (accel), b_g (gyro)
Regularization: position: minimum-snap (∫||P⁴(t)||² dt);  orientation: angular-accel penalty
```

### Sensor Models / Residuals

- **Radar**: `r = v_meas - v_pred` where `v_pred = -dot(u_body, v_ant)` (includes lever arm ω×r).
  TI IWR6843 convention: positive Doppler = receding target. Huber loss, δ = 1.0 m/s.
- **Accelerometer**: L2 on `z_acc - R_bw(a_world - g) - b_a`
- **Gyroscope**: L2 on `z_gyro - ω_body - b_g`

**Critical**: The negation in `v_pred = -dot(u,v)` is physically correct (see FINDINGS.md §11). Do not remove it.

### Coordinate Frames & Calibration

Extrinsic calibration lives in `config/extrinsics.yaml` (single source of truth):
- **Rotation**: `[roll=180°, pitch=25.5°, yaw=0°]` — solver-calibrated pitch (physical mount 30°, converges to ~27–28°)
- **Translation**: `[0.08, +0.02, -0.01]` m in body frame
- Body frame: x=forward, y=left, z=up

**Critical**: `optimize_pitch_only: true` in `solver.yaml` must stay enabled. Only pitch is
observable from Doppler; free roll/yaw optimization drifts 5–7° and corrupts orientation.

The radar has limited elevation diversity (2 TX antennas), causing a systematic z-velocity bias of
−0.5 to −0.65 m/s. Doppler quantization is 0.63 m/s per bin — keep Huber δ ≥ 1.0 m/s.

### Rosbag Datasets

Located at `../rosbags/`. Alias → filename mapping in `config/bags.yaml`. Some bags in the
`flipped` set apply `R_z(180°)` to extrinsics. After the Doppler sign fix, `slow_racing_best_velocity`
works without the flip. The other flipped bags (`circle_fwd`, `loopings`, `backflips`) are pending re-evaluation.

## Key Hyperparameters

### `config/solver.yaml` — Python solver defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| `huber_delta` | 1.0 m/s | Must be ≥ radar Doppler bin size (0.63 m/s) |
| `lambda_accel` | 0.01 | Accelerometer weight |
| `lambda_gyro` | 1.0 | Gyroscope weight |
| `lambda_snap_pos` | 0.0001 | Min-snap position regularization |
| `lambda_ori_reg` | 0.0 | Angular velocity reg (disabled; use lambda_ori_accel instead) |
| `lambda_ori_accel` | 0.1 | Angular acceleration reg — best across all bags; see SW_DEVELOPMENT §1 |
| `lambda_bias_prior_accel` | 1.0 | Relaxed — biases free to adjust |
| `lambda_bias_prior_gyro` | 1.0 | Same |
| `lambda_boundary_vel/pos/ori` | 1000.0 | Anchor start of trajectory |
| `lambda_pos_init_prior` | 0.0 | SW only: per-CP anchor to P1-P3 init; 1000 for backflips SW |
| `optimize_pitch_only` | **true** | **Must stay true** — only pitch is Doppler-observable |
| `max_iterations` | 40 | C++ SW uses this; batch Python uses early-stop |

### `config/solver_cpp.yaml` — C++ overrides (applied on `--cpp`)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lambda_gyro` | **4.0** | Tighter orientation; reduces acc_bias blow-up |
| `lambda_snap_pos` | **2e-5** | Less over-smoothing for racing dynamics |
| `lambda_bias_prior_accel` | **10000** | Full-rate IMU makes tight prior safe; prevents bias trash-can |
| `lambda_bias_prior_gyro` | **10000** | Same |
| `window_duration` | **3.0s** | 3s window, 0.3s stride |
| `marg_prior_scale` | **2e-4** | Default; overridden per-bag for slow_racing (1e-7) |

## Current Results

### C++ batch (--mocap-yaw --cpp)

Per-bag config auto-selected via `bags.yaml` solver_overrides:
- racing bags: `dt_pos=0.005s, dt_ori=0.008s`
- backflips: `dt_pos=0.010s, dt_ori=0.0008s, lock_extrinsics=1`

| Bag | Pos RMSE | Vel RMSE | Ori RMSE | Ext pitch |
|-----|----------|----------|----------|-----------|
| slow_racing | **0.174m** | 0.151 | **1.08°** | 27.1° |
| fast_racing | 0.758m | 0.386 | 2.58° | 27.6° |
| backflips | **1.817m** | 1.951 | **8.31°** | locked 25.5° |

### C++ sliding window (--mocap-yaw --cpp --sliding-window)

| Bag | Settled pos | Settled ori | Live pos | Live vel | Live ori | marg_prior_scale |
|-----|-------------|-------------|----------|----------|----------|-----------------|
| slow_racing | 0.218m | 1.57° | **0.393m** | 0.391 m/s | **2.21°** | 1e-7 (per-bag) |
| fast_racing | 0.804m | 3.10° | **0.877m** | 0.478 m/s | **4.16°** | 2e-4 (default) |
| backflips¹ | 2.56m | 10.87° | 3.33m | — | 9.33° | 0 (Phase 3 config) |

Note: settled vel for slow_racing is 0.886 m/s — eval artifact from near-zero prior causing
position jumps at stride boundaries in retrospective eval. Real-time live edge is continuous.

¹ backflips SW requires `--set` overrides; see Running Analysis Scripts above.
Pre-lever-arm batch (historical): slow 0.146m/0.96°, fast 0.925m/2.35°, backflips 2.93m/10.7°.

## Sliding Window Timing Benchmark (2026-06-03)

Both racing bags, `--mocap-yaw --cpp --sliding-window`. Per-window breakdown from Ceres
internal timers + `num_iterations = summary.num_successful_steps`.

### Before analytic radar Jacobians (AutoDiff DynamicAutoDiff for radar)

| Component | slow_racing | fast_racing | Share |
|---|---|---|---|
| Jacobian eval | ~0.7s | ~0.5s | ~35% |
| Linear solve | ~0.7s | ~0.6s | ~35% |
| Residual eval | ~0.07s | ~0.06s | ~3% |
| Other (compute_prior) | ~0.7s | ~0.6s | ~27% |
| **Total** | **~2.1s** | **~1.7s** | — |

### After analytic radar Jacobians (RadarAnalyticFactor, 2026-06-03)

| Component | slow_racing | fast_racing |
|---|---|---|
| Jacobian eval (avg) | ~0.81s | ~0.39s |
| Linear solve (avg) | ~0.92s | ~0.70s |
| Other (compute_prior, avg) | ~0.79s | ~0.62s |
| **Total (avg)** | **~2.6s** | **~1.77s** |
| **LM iterations** | **~28** | **~30** |

fast_racing Jacobian time improved ~30% (0.39s vs 0.5s); linear solve and compute_prior unchanged.
Total time similar due to Jacobian eval being only ~29% of total. The `slow_racing` result may
reflect higher radar point density (more observations per window) or system load variance.

**Key findings**: iter ≈ 28–30 every window regardless of warm/cold start or marg_prior_scale.
`function_tolerance` is the stopping criterion; cond(H) ≈ 5.5×10¹⁰ causes slow LM convergence.
"Other" ≈ 0.65s is `compute_prior()` calling `problem.Evaluate()` outside the LM loop.
All factors now analytic: GyroAnalyticFactor, AccelAnalyticFactor, RadarAnalyticFactor; -O3 -march=native.
Real-time gap: stride 0.3s vs ~1.7–2.1s solve → 5–7× too slow.
Remaining speedup levers: reduce compute_prior (accounts for ~35%), and linear solve.
See `documentation/RESEARCH_NOTES.md §9–10` for the full analysis and speedup path assessment.

## ROS Topics

| Topic | Content |
|-------|---------|
| `/mmWaveDataHdl/RScanVelocity` | Radar point cloud (x, y, z, velocity, intensity, range, noise, frame_number) |
| `/angrybird2/imu` | IMU (accel + gyro) |
| `/mocap/angrybird2/pose` | MoCap 6-DOF pose |
| `/mocap/angrybird2/accel` | MoCap linear acceleration (TwistStamped despite topic name) |
| `/angrybird2/agiros_pilot/state` | Full Agiros state |
| `/angrybird2/agiros_pilot/odometry` | Agiros odometry |

See `analysis/lib/rosbag_loader/RADAR_FIELDS.md` for field-level documentation.
