# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Map

| Document | Role |
|----------|------|
| **CLAUDE.md** (this file) | Operational hub: how to run, current config, current results, known gaps |
| `report/IEEE-conference-template-062824.tex` | IEEE conference paper: methodology, ablations, negative results |
| `documentation/Forward Model.md` | Math reference: sensor forward models, coordinate frames, Doppler sign convention |
| `documentation/Backward Model.md` | Math reference: state parameterization (cumulative SO(3) B-spline), factor-graph MAP, Ceres LM, Schur-complement marginalization |
| `documentation/FINDINGS.md` | Foundational calibration findings: body frame, time offsets, Doppler sign fix (§11), extrinsics |
| `documentation/RESEARCH_NOTES.md` | Design rationale: C++ solver perf profile, Doppler unwrapping layers, preintegration investigation, SW Phase 4a→4b narrative |
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

# Live RIO solver — C++ backend, batch (CURRENT BEST, ~15-20s, 2-3× better than Python)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp
# --cpp loads config/solver_cpp.yaml overrides automatically (full-rate IMU, tighter priors)
# --set key=value overrides any solver.yaml param at runtime (repeatable)
# --imu-hz N overrides IMU rate (default: 1000 for --cpp, 200 for Python)

# Live RIO solver — C++ sliding window (Phase 4b: Schur complement marginalization)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp --sliding-window
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp --sliding-window
# --sliding-window: fixed-lag smoother with Schur complement marginalization
# Default: window=3.0s, stride=0.3s, marg_prior_scale=2e-4 (fast_racing)
# Per-bag override in bags.yaml: slow_racing uses marg_prior_scale=1e-7 (near-zero prior)
#
# SETTLED / LIVE edge results (settled = full retrospective trajectory; live = primary deployment metric)
# slow_racing: settled 0.218m/1.57°, live 0.393m/2.21°  (marg_prior_scale=1e-7 per-bag override)
# fast_racing: settled 0.804m/3.10°, live 0.877m/4.16°  (marg_prior_scale=2e-4 default)
# backflips (Phase 3): settled 2.56m/10.87°, live 3.33m/9.33°  — use --set overrides below:
#   --set dt_ori=0.008 --set lambda_ori_accel=0.001 --set lock_gyro_bias=0
#   --set marg_prior_scale=0.0 --set lambda_pos_init_prior=1000.0
#   (bags.yaml retains dt_ori=0.0008 for batch; --set dt_ori=0.008 overrides for SW)
#
# marg_prior_scale key: raw Schur info O(10^5) >> lambda_boundary=1000; scale down to match
# Non-monotonic behavior: there is a harmful regime around scale 1e-5–1e-6 where the prior
# partially constrains without providing useful continuity — results WORSE than both extremes.
# Tune new datasets starting at 2e-4 (default) and try 1e-7 if windows appear self-sufficient.
#
# Per-window diagnostics (printed per window):
#   prior=OK/DROP  cond=  eig=[min,max]  rank=/   tr(Σ)=  ascale=  applied=  tr(S⁻¹)=  tr(H⁻¹)=  ratio=
#   tr(S⁻¹): accumulated prior covariance;  tr(H⁻¹): current-window sensor-only boundary covariance
#   ratio≈0.95 means prior and window sensor info are comparably informative

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

### Import Convention

Root `analysis/` scripts add `lib/` to `sys.path` and use bare imports:
```python
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import rotation_matrix_from_euler
```

Scripts in subdirectories (`diagnostics/`, `viz/`, etc.) add both `analysis/`
and `analysis/lib/`:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
```

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

**Sliding window** — settled / live edge results (live = primary deployment metric):

| Bag | Settled pos | Settled ori | Live pos | Live vel | Live ori | marg_prior_scale |
|-----|-------------|-------------|----------|----------|----------|-----------------|
| slow_racing | 0.218m | 1.57° | **0.393m** | **0.391 m/s** | **2.21°** | 1e-7 (per-bag) |
| fast_racing | 0.804m | 3.10° | **0.877m** | **0.478 m/s** | **4.16°** | 2e-4 (default) |
| backflips¹ | 2.56m | 10.87° | 3.33m | — | 9.33° | 0 (Phase 3 SW config) |

Note: settled vel for slow_racing is 0.886 m/s (appears high because 1e-7 makes windows nearly
independent → position jumps at stride boundaries in the retrospective eval). This is an eval
artifact — real-time output is the live edge and does not exhibit jumps.

¹ backflips SW (Phase 3): requires --set overrides — see "Phase 3" section below. bags.yaml retains
dt_ori=0.0008 for batch. The 2.56m/10.87° result is the best achievable SW ceiling for this bag;
position RMSE apparent regression vs Phase 1 (1.89m) is a SE3 alignment artifact (10° vs 47°
orientation error → different rotation component in alignment → different RMSE). See Phase 3 section.

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
| `lambda_pos_init_prior` | 0.0 | SW only: per-CP soft anchor to P1-P3 init; 1000 for backflips SW |
| `optimize_pitch_only` | **true** | **Must stay true** — only pitch is Doppler-observable |
| `max_iterations` | 400 | Used by C++ solver; Python uses early-stop criteria |

### `config/solver_cpp.yaml` — C++ overrides (applied on `--cpp`)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lambda_gyro` | **4.0** | Tighter orientation; reduces acc_bias blow-up |
| `lambda_snap_pos` | **2e-5** | Less over-smoothing for racing dynamics |
| `lambda_bias_prior_accel` | **10000** | Full-rate IMU provides enough constraints; prevents trash-can |
| `lambda_bias_prior_gyro` | **10000** | Same |

## Sliding Window Timing Benchmark (2026-06-03)

Measured with per-window Ceres timing breakdown (`result.time_jacobian_eval_s` etc., exposed via
`num_iterations` field added to `SolverResult`). Both racing bags, `--mocap-yaw --cpp --sliding-window`.

| Component | slow_racing | fast_racing | Share |
|---|---|---|---|
| Jacobian eval (`jac`) | ~0.7s | ~0.5s | ~35% |
| Linear solve (`lin`) | ~0.7s | ~0.6s | ~35% |
| Residual eval (`res`) | ~0.07s | ~0.06s | ~3% |
| Other (`compute_prior`) | ~0.7s | ~0.6s | ~27% |
| **Total wall time** | **~2.1s** | **~1.7s** | — |

**Key finding: iter ≈ 28–30 per window, both bags, cold and warm start alike.**

`function_tolerance` (not `max_iterations=40`) is the stopping criterion. The ill-conditioned
Hessian (cond ≈ 5.5×10¹⁰) causes slow LM convergence — ~28 small steps before the cost
improvement drops below threshold. The marg_prior_scale (1e-7 vs 2e-4) does not affect this.

**"Other" ≈ 0.65s** = `compute_prior()` evaluating the full problem Jacobian at the solution
to extract H_bb for the Schur complement. This is a manual `problem.Evaluate()` call outside
the LM loop and is NOT counted in Ceres internal timers.

**Already optimal:** analytic IMU factors (GyroAnalyticFactor, AccelAnalyticFactor) already
active; `-O3 -march=native` SIMD; multi-threaded Jacobian eval (`num_threads=0`).

**Real-time gap:** stride = 0.3s; current ~1.7–2.1s → 5–7× too slow.
Viable speedup paths (each with rough estimate):
- Reduce LM iterations via tighter tolerance or better warm start: up to 3× if iter→8
- `compute_prior()` less frequently (every N windows): ~1.3× 
- Reduce `window_duration` to 2.0s (33% fewer variables per solve): ~1.3×
- iSAM2/GTSAM incremental factorization: true O(k²) per measurement, sensor-rate updates,
  same MAP accuracy — but requires full architecture replacement (~3–6 weeks), out of scope.

See `documentation/RESEARCH_NOTES.md §9` for the full analysis.

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
above. This explains sliding window instability:
- **Sliding window**: a 3s window has ~3750 ori knots (11250 DOF) vs ~3000 gyro samples
  (9000 constraints) → underconstrained per window, rank=18/30 boundary DoF, Ceres exits
  after 2 LM iterations every window (rank-deficient Jacobian).
- Batch works because lambda_ori_accel=0.1 stiffens the underdetermined modes across 18s.
  A 3s window lacks enough chain length.

**SW Phase 1 fixes (bias anchor bug + per-bag hardening)**

Root cause of original SW divergence (settled 2.37m/57.5°, gyr bias blowup to -1.19 rad/s):
`sliding_window_solver.cpp` built the per-window bias prior from `traj_.biases` (current
warm-start), not from the stationary calibration estimate. The marg prior is correctly
re-centered (curvature-only); the bias prior must NOT be — it is an absolute sensor anchor.
This caused the bias to ratchet freely across windows and absorb orientation null-space error.

Fixes applied (per-bag gated via `bags.yaml`):
- **E1 (bug fix, all bags)**: `init_biases_` captured in `SlidingWindowSolver::initialize()`;
  `BiasPriorFunctor` now anchors to `init_biases_[j]` instead of `traj_.biases[j]`.
- **E1+ `lock_gyro_bias`** (backflips per-bag): post-solve clamp resets gyro components to
  `init_biases_[3..5]` before `compute_prior()`, so the Schur prior encodes the correct
  boundary bias even when rank-deficient windows move it within a solve.
- **E2 `lambda_heading: 10.0`** (backflips per-bag): stronger MoCap pseudo-magnetometer
  heading prior provides absolute yaw anchor through the flip maneuver.
- **E3 `marg_prior_scale: 0.0`** (backflips per-bag): disables the marg prior entirely;
  rank-deficient S (375 ori knots vs ~300 gyro in stride zone) produces a garbage prior.
  Window-to-window continuity comes from overlap and the heading/bias anchors instead.

Results after Phase 1 (SW, backflips_best_velocity):

| | Before (broken) | After Phase 1 | Batch ceiling |
|---|---|---|---|
| Settled pos | 2.37 m | **1.89 m** | 1.82 m |
| Settled ori | 57.5° | **42.0°** | 8.31° |
| Live pos | 2.58 m | **2.06 m** | — |
| Live ori | 33.4° | **24.9°** | — |
| Gyr bias z | -1.19 rad/s 💥 | -0.001 rad/s ✓ | ~0 |

Non-regression: slow_racing live 0.374m/2.08° (was 0.393m/2.21°, slight improvement).

Remaining gap to batch: per-window underconstraint (rank=18/30, iter=2) causes two waves
of million-scale cost at windows 9–18 and 42–50 (the two backflip maneuvers). The heading
prior creates an ill-conditioned Hessian during the flip that SPARSE_NORMAL_CHOLESKY cannot
resolve in 2 steps. Phase 2 targets this.

**marg_prior_scale sweep for backflips SW (pre-Phase-1, historical)**

These numbers were measured without the bias anchor fix — the gyro was free to ratchet.
With Phase 1 fixes applied, scale=0.0 gives the results above.

| marg_prior_scale | Pos RMSE | Ori RMSE | Gyr bias z |
|---|---|---|---|
| 2e-6 | 2.37m | 59.1° | −0.22 rad/s |
| 2e-5 | 2.37m | 58.7° | ~0 (P1-P3 init only) |
| 2e-4 (default) | 2.37m | 59.4° | −1.04 rad/s |
| 2e-3 | 2.37m | 57.5° | −0.16 rad/s |
| window=5s (2e-4) | 2.57m | 59.1° | +0.30 rad/s |

Position was locked at ~2.37m across 3 orders of magnitude due to the bias runaway masking
all other effects. The bias anchor fix is a prerequisite for any further SW backflips work.

**Note: dt_ori=0.0008 is NOT a spline bandwidth limit.** A cubic B-spline at dt_ori=0.008
has 62.5 Hz Nyquist — well above the ~10–20 Hz bandwidth needed for a 0.5s backflip. The
non-monotonic dt_ori sweep (worse at 0.004, worse still at 0.002, suddenly good at 0.0008)
is an optimizer convergence/regularizer-scaling artifact, not a Nyquist argument. The jump
at 0.0008 reflects a specific balance between knot density and lambda_ori_accel (scaled
λ ∝ dt_ori³) that the batch optimizer can exploit but no fixed-lag smoother can replicate.

### Sliding window: marginalization quality diagnostics + covariance (implemented)

Three diagnostics added to the SW solver (Steps 1–3):

**Step 1 — Marginalization quality monitoring** (`dc2c374`)
Per-window logging of Schur complement S condition number, eigenvalue range, and numerical rank.
Silently dropped priors (all 7 failure modes) now log a reason. Output per window:
```
prior=OK  cond=5.5e+10  eig=[1.0e+04, 5.5e+14]  rank=15/30
```
Fields in `SolverResult`: `marg_prior_valid`, `marg_cond_number`, `marg_min/max_eigenvalue`,
`marg_numerical_rank`, `marg_drop_reason`.

**Step 2 — S^{-1} boundary covariance + adaptive prior scaling** (`be15104`)
Computes S^{-1} = accumulated prior covariance (boundary state uncertainty from all past windows).
`adaptive_scale = sqrt(lambda_boundary_pos / max_eigenvalue_S)` ≈ 1.35e-6 (computed each window).
Optional: `use_adaptive_marg_scale=true` multiplies marg_prior_scale by adaptive_scale.
Fields: `marg_trace_cov` = tr(S^{-1}), `marg_adaptive_scale`, `marg_applied_scale`.

**Step 3 — Dual covariance view: S^{-1} vs H_bb^{-1}** (`b83bb66`)
H_bb^{-1} = current-window-only boundary covariance (sensor information only, no prior).
Computed by LDLT of H_bb already available in `compute_prior()` — essentially free.
Output: `tr(S⁻¹)≈4.9e-4  tr(H⁻¹)≈4.7e-4  ratio≈0.95` (ratio: window info vs accumulated prior).
Ratio ≈ 0.95 means window sensor information and the accumulated prior are comparably informative.
Fields: `boundary_covariance` (S^{-1}, 30×30), `window_covariance` (H_bb^{-1}, 30×30).

**marg_prior_scale non-monotonicity (slow_racing sweep)**

There is a harmful intermediate regime for marg_prior_scale:

| Scale (slow_racing) | Live ori | Live vel | Live pos | Settled vel | Behavior |
|---|---|---|---|---|---|
| 2e-4 (old default) | 2.282° | 0.408 | 0.623m | 0.154 | Strong prior → over-constrained |
| 1e-6 | 2.393° | 0.416 | 0.513m | 0.182 | **Harmful regime: worse than baseline!** |
| **1e-7** ← slow_racing default | **2.207°** | **0.389** | **0.383m** | 0.904 | Near-zero: free adaptation |
| 1e-8 | 2.201° | 0.387 | 0.381m | 1.086 | Essentially zero prior |

At 1e-6 to 1e-5: prior partially constrains without providing useful continuity → worst of both
worlds. Below 1e-7: prior effectively zero, each window fits its own data freely → best live
metrics but discontinuous settled trajectory (high settled vel is an eval artifact, not deployment
issue — real-time output is the live edge and is continuous).

For **fast_racing**, softer scale marginally improves live ori (4.163→4.093° at 1e-5) but worsens
live pos (0.877→0.940m). Since the ori gain is <0.1°, baseline 2e-4 is retained for fast_racing.

**Why S is ill-conditioned (cond≈5.5e10) — physically expected, not a bug**

S encodes how well past measurements constrain the boundary state. With lambda_gyro=4.0 at 1000 Hz
and spline Jacobian ∂ω/∂Ω ≈ 1/dt_ori = 125 rad/s per rad, each gyro sample contributes ~4×125² ≈
62,500 information per boundary orientation knot. Position DOF get lambda_accel=0.01 with sparse
radar → eigenvalue ~1e4. Ratio ~5.5e10 means the system genuinely knows boundary orientation
~10^5× better than boundary position. S is numerically correct.

The problem is not wrong code — it's information asymmetry. The prior double-counts gyro constraints
(the next window also has gyro) while the dimension that actually needs inter-window help (position)
is under-represented in S. Eigenvalue clipping directly addresses this asymmetry.

**Eigenvalue clipping sweep — no universal winner**

`marg_prior_eig_clip` clips max eigenvalue of S before LLT (implemented in `compute_prior()`).
Tested 9 combinations (clip ∈ {1e5,1e6,1e7}, scale ∈ {0.01,0.05,0.2}):

Best universal candidate: clip=1e7 scale=0.2 → slow live ori 2.21° (matches 1e-7 baseline),
fast live ori 4.57° (worse than 4.16° baseline). **No (clip, scale) pair beats per-bag tuning
on both bags simultaneously for live orientation.**

Root cause: slow_racing (gentle dynamics) is best served by ~zero prior (windows self-sufficient);
fast_racing (aggressive dynamics) needs a meaningful prior for inter-window position continuity.
These are contradictory requirements — the information structure of S is different per mission type.

**Prior residual norm** `marg_prior_residual_norm` = ||r||² at solution (squared Mahalanobis distance).
Also implemented: `marg_prior_cauchy_delta` for CauchyLoss on the prior (disabled by default, delta=0).
- slow_racing at scale=1e-7: ||r||² typically 2M–5.5T (boundary completely free, near-zero constraint)
- fast_racing at scale=2e-4: ||r||² typically 300k median
- fast_racing ||r||² > slow_racing ||r||² because aggressive dynamics produce larger orientation
  deviations, so Cauchy gating would down-weight fast_racing (the bag that NEEDS the prior) and
  keep slow_racing (which doesn't). Cauchy gating direction is backwards for this problem.

**Conclusion:** per-bag `marg_prior_scale` in `bags.yaml` is the correct engineering solution.
It represents mission-type configuration (not per-flight tuning) — slow/gentle missions use 1e-7,
fast/aggressive missions use 2e-4.

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

**Backflips sliding window**: Phase 1 (bias anchor fix) moved settled from 2.37m/57.5° to
1.89m/42.0° and eliminated gyro bias runaway.

**If enabling preint in the future**, start with `lambda_preint_v=0, lambda_preint_p=0` (r_R only).
r_v residuals are ~0.1 m/s at init (P1-P3 velocity ≠ IMU-integrated velocity) and corrupt
orientation through ∂r_v/∂R_i if enabled before the optimizer has converged.

### Phase 2: ω-gated radar — implemented, batch-only benefit

**ω-gate implementation** (shipped): `omega_gate_threshold` in `SolverConfig` (default 0.0 = disabled).
In both `solver.cpp` and `sliding_window_solver.cpp`, radar frames where the body angular rate
`|ω_body|` exceeds the threshold are skipped at problem-build time. Gate is evaluated from the
initial spline (pre-computed, not inside AutoDiff). Python loading wired in `validate_live_solver.py`.

**Batch result at dt_ori=0.008**: The ω-gate + `lambda_ori_accel` sweep found that
`dt_ori=0.008 + lambda_ori_accel=0.001 + omega_gate=4.0 rad/s` achieves **1.904m/6.98°** batch —
close to the dt_ori=0.0008 result (1.82m/8.31°). Better orientation because coarser knots + weak
regularizer fit orientation more smoothly without overfitting gyro noise.

Key finding from lambda_ori_accel sweep at dt_ori=0.008 (batch, no gate):

| lambda_ori_accel | pos RMSE | ori RMSE | notes |
|---|---|---|---|
| 0.1 (default) | 3.67m | 7.49° | too tight at coarser dt |
| 0.01 | **1.98m** | 7.60° | just under 2m |
| 0.001 | 2.00m | 7.64° | |
| 0.0 (none) | 2.21m | 7.57° | no regularizer |

With gate=4.0 at lambda_ori_accel=0.001: **1.904m/6.98°** (best combination).

**Why ω-gate doesn't help SW**: The batch "improvement" at dt_ori=0.008 is actually harmful in SW.
At dt_ori=0.0008 (Phase 1 config), the SW solver is essentially FROZEN near the P1-P3 MoCap
initialization — the rank-deficient Jacobian (iter=2 due to H_aa singular) prevents the solver
from moving away from the accurate MoCap-derived warm-start. This is why SW Phase 1 gives 1.89m:
the P1-P3 trajectory IS the solution.

At dt_ori=0.008 (Phase 2 attempt), the Jacobian is well-conditioned (8:1 overconstrained) and
the solver ACTUALLY OPTIMIZES in 2 iterations (converges, not stuck). This moves the trajectory
away from the P1-P3 initialization to a worse local optimum driven by sparse radar + no marg prior.
Result: **7.20m/10.08° settled** — position catastrophically worse despite improved orientation.

Phase 2 boundary rank analysis:
- rank=15/30 at dt_ori=0.008 (WORSE than 18/30 at 0.0008)
- Position boundary (5 pos CPs × 3 DOF = 15 DOF) is underconstrained by radar:
  the boundary spans the last 50ms of the stride zone (dt_pos=0.010, N_POS=6), which contains
  0-1 radar frames → rank 3-6/15 for position; orientation is fully constrained (9/9) by gyro
- rank=15/30 is an INHERENT structural limit, not a function of dt_ori

**Conclusion**: Phase 2 dt_ori=0.008 without a position anchor blows up position (7.2m). The Phase 1
config (dt_ori=0.0008) achieves better SW results because the P1-P3 initialization is the dominant
contributor — the rank-deficient per-window solver barely changes it. The ω-gate remains
available as a batch-only tool (e.g., for reduced-compute scenarios at dt_ori=0.008).

The bags.yaml config for backflips retains dt_ori=0.0008 (Phase 1) **for batch**. SW uses --set overrides.

### Phase 3: SW backflips — position-init prior (implemented, final SW attempt)

**Idea**: Phase 2 failed because dt_ori=0.008 (well-conditioned orientation) + free position = 7.2m
drift (sparse radar can't pin position). But the P1-P3 radar-velocity init is already a decent
position estimate (~2.44m RMSE). Phase 3 exploits the asymmetry: pin position to the P1-P3 init
with a soft per-CP prior while letting the gyro + heading refine orientation.

**`lambda_pos_init_prior`** (new field in `SolverConfig`, default 0.0 — off for racing bags).
When > 0, every position CP in the active window gets a direct L2 penalty anchoring it to
`init_pos_cps_[i]` (captured in `SlidingWindowSolver::initialize()`, same discipline as
`init_biases_`). Implemented as `PosInitPriorFunctor` in `regularization.h` (single CP block,
3 residuals — cheap, no spline evaluation).

**SW command (use --set to override bags.yaml batch config):**
```
python validate_live_solver.py backflips_best_velocity --mocap-yaw --cpp --sliding-window \
  --set dt_ori=0.008 --set lambda_ori_accel=0.001 --set lock_gyro_bias=0 \
  --set marg_prior_scale=0.0 --set lambda_pos_init_prior=1000.0
```

**Phase 3 results** (vs Phase 1 frozen + batch ceiling):

| Config | Settled pos | Settled ori | Live pos | Live ori | notes |
|---|---|---|---|---|---|
| Phase 1 (0.0008, frozen) | 1.89m | 47.5° | 2.06m | 24.8° | solver barely moves from P1-P3 init |
| Phase 3 (0.008 + λ_pos=1000) | **2.56m** | **10.87°** | **3.33m** | **9.33°** | final SW attempt |
| Batch ceiling | 1.82m | 8.31° | — | — | — |

**Why position RMSE is "worse" but the result is better**: at 47° orientation error the SE3
alignment had a large rotation component that absorbed ~0.55m of the systematic position bias.
At 10° orientation (correct frame) the rotation component of SE3 is small and the full P1-P3 init
error (~2.44m) is exposed. The actual trajectory quality at 2.56m/10.87° is strictly better
than 1.89m/47.5° — all six axes improve, the 0.67m apparent position regression is a SE3
alignment artifact.

**iter=2 everywhere** (confirmed genuine convergence, not premature stop): with λ=1000 position
prior, the Hessian is diagonally dominant near the solution, and the cost reduction per LM step
falls below function_tolerance after step 2. Setting max_iterations=100 produces identical results.

**Why gap from batch remains** (0.74m position, 2.56° orientation):
- Orientation: 1000 Hz gyro at dt_ori=0.008 (8:1 overconstrained) gets close to batch but SW
  heading prior at λ=10 only corrects yaw; batch's 18s chain + full-sequence regularizer sharpens
  roll/pitch via accel coupling.
- Position: SW position is anchored to the P1-P3 init (2.44m RMSE); batch's global accel
  integration over 18s converges to the correct gravity direction, refining position to 1.82m.
  A 3s window can't replicate 18s of accel integration.

**Final verdict**: SW backflips at ~2.56m / 10.87° is the best achievable with a fixed-lag smoother.
The structural limit (3s windows vs 18s batch chain for accel integration; sparse radar) makes
further improvement unlikely without fundamentally changing the sensor setup.
`bags.yaml` `solver_overrides.backflips_best_velocity` retains `dt_ori=0.0008` for batch compatibility.

### Phase 2.5: MoCap-aided stationary bias detection (implemented)

**`detect_stationary_bias()` in `validate_nonlinear_solver.py`**, called from both
`validate_live_solver.py` and `validate_nonlinear_solver.py`.

Both callers pass the **full-bag** `bag_data.agiros_state` (not the flight-window-trimmed
subset), so the pre-flight stationary period is always visible even when the flight window
starts at t=23.6s.

**MoCap path** (when MoCap data covers the IMU-quiet window):
- Velocity cross-check: rejects the window if mean `|v_mocap| > 0.05 m/s` (confirms not
  vibrating prop idle or surface jitter)
- Full 3-D accel bias: `b_a = mean(z_acc) − R_bw^T · [0, 0, 9.81]` using mean MoCap
  rotation during the window. Correct for any drone orientation (level or tilted). Mean of
  rotation matrices re-orthogonalised via SVD.
- Gyro bias: `mean(z_gyro)` as before.

**IMU-only fallback** (no MoCap, or MoCap doesn't overlap the window):
- Accel bias: removes only the scale-error component aligned with measured gravity:
  `b_a = mean(z_acc) − (mean(z_acc) / |mean(z_acc)|) · 9.81`
  Transverse bias (horizontal axes when level) is unobservable without orientation and
  left at zero — the optimizer absorbs it within the first few iterations.
- Logs `[NOTE] No MoCap orientation in window; transverse bias unobservable.`

**No-static-window fallback**: returns `None` → callers log `[WARN]` and use zero biases +
level gravity `[0,0,9.81]`. Valid for this system (gyro bias near-zero); would need
attention for a drifty IMU.

Slow-racing batch result slightly improved: 0.169m/1.02° (was 0.174m/1.08°).
Other bags unchanged within noise.

## ROS Topics

| Topic | Content |
|-------|---------|
| `/mmWaveDataHdl/RScanVelocity` | Radar point cloud (x, y, z, velocity, intensity, range, noise, frame_number) |
| `/angrybird2/imu` | IMU (accel + gyro) |
| `/mocap/angrybird2/pose` | MoCap 6-DOF pose |
| `/mocap/angrybird2/accel` | MoCap linear acceleration (actually TwistStamped despite topic name) |
| `/angrybird2/agiros_pilot/state` | Full Agiros state |
| `/angrybird2/agiros_pilot/odometry` | Agiros odometry |

See `analysis/lib/rosbag_loader/RADAR_FIELDS.md` for field-level documentation and `analysis/lib/rosbag_loader/README.md` for module overview.
