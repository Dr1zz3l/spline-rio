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

**C++ results — current best** (--mocap-yaw --cpp, lambda_ori_accel=0.1, extrinsic opt enabled)

Per-bag config auto-selected via `bags.yaml` solver_overrides:
- racing bags: `dt_pos=0.005s, dt_ori=0.008s` (default)
- backflips: `dt_pos=0.010s, dt_ori=0.0008s, lock_extrinsics=1` (per-bag override)

| Bag | Mode | Pos RMSE | Vel RMSE | Ori RMSE | Ext pitch | Acc bias (m/s²) | Gyr bias (rad/s) |
|-----|------|----------|----------|----------|-----------|-----------------|-----------------|
| slow_racing | batch | **0.174m** | 0.151 | **1.08°** | 27.1° | [-0.005, 0.049, 0.236] | [0.005, -0.003, 0.001] |
| fast_racing | batch | 0.758m | 0.386 | 2.58° | 27.6° | [-0.008, 0.007, 0.064] | [0.006, 0.001, -0.002] |
| backflips | batch | **1.817m** | 1.951 | **8.31°** | locked 25.5° | [0.044, 0.082, 0.220] | [-0.001, 0.000, 0.001] |

Extrinsic optimization note: racing bags converge to 27–28° from either 25.5° or 30° init.
Backflips locks extrinsics — 22630 dense ori knots underconstrain pitch_delta (drifts to +18°).

**Sliding window** (batch results above are better; sliding-window not re-tuned since angular accel change):
- slow_racing: ~0.21m / 1.6° (sliding-win with old config)
- fast_racing: ~0.68m / 3.0° (sliding-win with old config — better position than batch)
- backflips: unstable with dt_ori=0.0008s (underconstrained per window), use batch

Note: lever arm (ω × r_antenna) added to C++ RadarDopplerFunctor in Phase 4b.
Pre-lever-arm batch: slow 0.146m/0.96°, fast 0.925m/2.35°, backflips 2.93m/10.7°.

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

### Per-bag solver overrides (implemented)

`config/bags.yaml` has a `solver_overrides` section. Any key valid in `solver.yaml` can be
overridden per bag. Applied after `solver_cpp.yaml`, before `--set` (so `--set` always wins).

```yaml
solver_overrides:
  my_bag:
    dt_ori: 0.0008
    dt_pos: 0.010
```

**Why backflips needs different dt_ori** — swept dt_ori 0.002–0.0008 on both bags
(lambda_ori_accel scaled as λ ∝ dt_ori³ to keep continuous ∫||α||²dt equivalent):

| dt_ori | slow_racing pos | backflips pos | backflips ori |
|---|---|---|---|
| 0.008 (default) | **0.170m** | 3.369m | 8.01° |
| 0.006 | 0.151m | 3.662m | 8.17° |
| 0.004 | 0.224m | 4.772m | 9.55° |
| 0.002 | 0.459m | 3.355m | 49.8° ⚠️ |
| **0.0008** (backflips default) | — | **1.814m** | 8.57° |

No intermediate value helps backflips — the position drops below 2m only at 0.0008 (hard cliff,
not a gradient). Every step denser than 0.008 hurts slow_racing. Root cause: 0.008s doesn't have
enough bandwidth to represent the rapid angular acceleration ramp-up of the backflip. The
orientation bandwidth requires 0.0008s; there is no universal dt_ori that works for both.

**Why extrinsic optimization always breaks on backflips (any density)**: the flip maneuver
produces rapid large-amplitude orientation changes that the spline can only partially
represent. The systematic orientation model error creates a systematic Doppler residual,
which the optimizer reduces by drifting pitch_delta (cheaper than fixing orientation DOF
en masse). Tested: at default density (dt_ori=0.008s, 2268 knots, well-constrained ratio
8:1), pitch_delta still drifts to +29.6° (55.1°) with catastrophic pos RMSE 3.96m. The
root cause is not the DOF/constraint ratio — it is that the backflip creates conditions
where pitch_delta acts as a surrogate for orientation model error.
Fix: `lock_extrinsics: 1` per-bag, regardless of density.

**Additional issue at dt_ori=0.0008s**: 22630 knots × 3 = 67890 DOF vs ~54000 gyro constraints
(ratio 1.26 DOF/constraint) makes the system *technically* underconstrained, compounding the
above. This is a secondary reason and also explains sliding window instability:
- **Sliding window**: a 3s window has ~3750 ori knots (11250 DOF) vs ~3000 gyro samples
  (9000 constraints) → underconstrained per window, Ceres exits after 2 iterations every
  window (rank-deficient Jacobian), gyro bias runs away.
  Fix: batch-only for backflips.
- Batch works because lambda_ori_accel=0.1 stiffens the underdetermined modes across 18s.
  A 3s window lacks enough chain length; no amount of prior tuning can replicate global coupling.

**marg_prior_scale sweep for backflips SW at dt_ori=0.0008** (all equally broken):

| marg_prior_scale | Pos RMSE | Ori RMSE | Gyr bias z |
|---|---|---|---|
| 2e-6 | 2.37m | 59.1° | −0.22 rad/s |
| 2e-5 | 2.37m | 58.7° | ~0 (P1-P3 init only) |
| 2e-4 (default) | 2.37m | 59.4° | −1.04 rad/s |
| 2e-3 | 2.37m | 57.5° | −0.16 rad/s |
| window=5s (2e-4) | 2.57m | 59.1° | +0.30 rad/s |

Position is locked at ~2.37m across 3 orders of magnitude. The failure is rank-deficiency,
not prior miscaling. Neither wider windows nor preintegration can fix this.

**Note: dt_ori=0.0008 is NOT a spline bandwidth limit.** A cubic B-spline at dt_ori=0.008
has 62.5 Hz Nyquist — well above the ~10–20 Hz bandwidth needed for a 0.5s backflip. The
non-monotonic dt_ori sweep (worse at 0.004, worse still at 0.002, suddenly good at 0.0008)
is an optimizer convergence/regularizer-scaling artifact, not a Nyquist argument. The jump
at 0.0008 reflects a specific balance between knot density and lambda_ori_accel (scaled
λ ∝ dt_ori³) that the batch optimizer can exploit but no fixed-lag smoother can replicate.

### C++ solver: extrinsic pitch optimization (implemented)

**Implemented.** `RadarDopplerWithPitchFunctor` in `radar_doppler.h` accepts a 1-DOF scalar
`pitch_delta` as an extra parameter block; `solver.cpp` and `sliding_window_solver.cpp` add it
when `lock_extrinsics=false` (default). Composition: `R_total = R_nominal * Ry(pitch_delta)`.
A `PitchDeltaPriorFunctor` with `lambda_extrinsic_prior=10.0` keeps it near nominal.

**Convergence:** Racing bags consistently converge to 27–28° from either 25.5° or 30° init
(+1.6–2.1° from 25.5° nominal). Metrics are essentially unchanged vs locked baseline.

**Backflips uses `lock_extrinsics: 1` per-bag override** — the dense ori knots (22630 for 18s)
create too many DOF for the weak prior to anchor pitch_delta; it drifts to +18° and causes
catastrophic orientation error (43.8° pitch vs 8.3° with locked). Added to `bags.yaml`
`solver_overrides.backflips_best_velocity`.

### C++ solver: IMU preintegration (implemented, disabled by default)

**Implemented** but **disabled** (`use_preintegration: false` in `solver_cpp.yaml`). The full
Forster TRO-2017 preintegration pipeline is in place:

- `IMUPreintegrationFunctor` in `rio_solver_cpp/include/rio/factors/imu_preintegration.h`
- `PreintFactor` struct in `solver.h` with full Jacobians (d_R_d_bg, d_v_d_ba, d_v_d_bg, d_p_d_ba, d_p_d_bg)
- `SolverConfig` fields: `use_preintegration`, `lambda_preint`, `lambda_preint_v`, `lambda_preint_p`, `preint_hz`
- Functor has separate `scale_v`, `scale_p` scalars (default 0.0) to independently enable r_v and r_p
- In the preint path, raw accel factors are still added (preint r_R replaces gyro only)
- Python wiring in `validate_live_solver.py`: `_build_preint_factors_cpp()`, dt_ori coupling override
- `PreintFactor` bound in pybind11; `preint_factors` arg added to `solve()` and `solve_window()`

**Why it's disabled — fundamental constraint-density issue:**

Preint at 100 Hz (dt_ori=0.01) replaces 1000 Hz raw gyro with 100 Hz preintegration factors.
For racing bags at dt_ori=0.008s:
- Raw gyro: ~8 samples per knot → 8:1 overconstrained (strong, noise-resistant)
- Preint at 100 Hz: 1 factor per knot → 1:1 (critically constrained → orientation degrades)
- Tested: slow_racing orientation RMSE 1.09° (raw gyro) → 6.0° (preint at 100 Hz)

**Why preint can't fix backflips sliding window instability:**
Two conflicting requirements make preint fundamentally unable to help:
1. Model bandwidth: backflips require dt_ori ≤ 0.001s (empirically ~0.0008s from dt_ori sweep)
2. Preintegration requires dt_preint ≥ 1/IMU_hz = 0.001s (need ≥2 samples per interval)
At 100 Hz (dt_ori=0.01s): model bandwidth insufficient — same bad results as dt_ori=0.008
At 1250 Hz (dt_ori=0.0008s): dt_preint=0.0008 < IMU period=0.001 → degenerate factors

**Backflips sliding window** remains documented as batch-only (see section above).

**If enabling preint in the future**, start with `lambda_preint_v=0, lambda_preint_p=0` (r_R only).
r_v residuals are ~0.1 m/s at init (P1-P3 velocity ≠ IMU-integrated velocity) and corrupt
orientation through ∂r_v/∂R_i if enabled before the optimizer has converged.

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
