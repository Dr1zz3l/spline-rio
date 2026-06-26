# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Map

| Document | Role |
|----------|------|
| **CLAUDE.md** (this file) | Operational hub: how to run, current config, current results |
| `report/main.tex` | **T-RO journal submission — the single maintained paper. Trimmed to the hard 12pp limit (2026-06-25).** Thin root: preamble + `\input{acronyms}` + `\input{sections/*.tex}` (one file per section) + bib/bios. Figures in `report/figures/`; acronyms in `report/acronyms.tex`. **Figure set (post-trim, 2026-06-26):** Fig 1 pipeline (TikZ inline, scalebox 0.78), Fig 2 traj overlays (`gen_combined_traj.py`, figsize h=2.45), Fig 3 error-over-time (`gen_error_time.py`, **3 rows incl. backflips**, live-edge only — batch curve dropped). **Removed for the page budget / content trim:** the SW-marginalization TikZ diagram (`fig:sw_diagram` — Proposition 1 + §IV-E text carry it), the RPE figure (`fig:rpe`/`gen_rpe.py` — summary numbers now in §VI-B, full data repo-only) and the prior-scale plot (`fig:prior_scale`/`gen_prior_scale.py` — redundant with Table VII). Also dropped: GNC + full-extrinsic negative-result paragraphs, IMU-preint ablation row (kept the §VII-D paragraph). **Table V (`tab:baselines`) is single-column** ((a) drift%/vel/ori `\footnotesize`, (b) decomposition `\scriptsize`); 22 refs (dropped Hug, Yang/GNC). **`paper/` (conference cut) is FROZEN (2026-06-24): do NOT edit it.** Regenerate a conference cut from `report/` if ever needed. |
| `documentation/Forward Model.md` | Math reference: sensor forward models, coordinate frames, Doppler sign convention |
| `documentation/Backward Model.md` | Math reference: state parameterization (cumulative SO(3) B-spline), factor-graph MAP, Ceres LM, Schur-complement marginalization |
| `documentation/FINDINGS.md` | Foundational calibration findings: body frame, time offsets, Doppler sign fix (§11), extrinsics |
| `documentation/RESEARCH_NOTES.md` | Design rationale: solver perf profile, Doppler unwrapping, preintegration investigation, BandedSchurSolver post-mortem, SW timing analysis, real-time/speedup path (§1/§7-§10; absorbed the deleted `Realtime_Options.md`) |
| `documentation/SW_DEVELOPMENT.md` | SW solver development history: phase ablations, sweep tables, backflips analysis, marg_prior_scale tuning |
| `documentation/ROADMAP.md` | Design-rationale **archive** (2026-06-11 audit, largely executed): the "ROADMAP Part X / N.M" targets that CLAUDE.md's inline comments reference. Open item: iSAM2 back-end |
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
# Default: window=3.0s, stride=0.3s
# 2026-06-12 consistency fixes (see documentation/ROADMAP.md Part 1 results):
#   marg_markov_blanket=1 + warm_start_align=1 are new C++ defaults (--set ...=0 for legacy).
#
# UNIVERSAL WEIGHTING CONFIG (2026-06-12, ROADMAP Part 5 "universal config" +
# velocity-metric rebalance): one weighting set for ALL bags — ω-adaptive gyro
# λ_eff = 4·(1+(|z_gyro|/4)⁴), radar/accel ω-soft-gates 4/8, per-point SNR weighting:
UNIV="--set marg_prior_scale=1.0 --set lambda_gyro_omega_sigma=4.0 --set lambda_gyro_omega_pow=4.0 \
  --set omega_soft_sigma=4.0 --set accel_soft_sigma=8.0 --set radar_intensity_weight=1.0"
# (gates 2/4 variant = fast-ori-priority: fast 2.94° live ori but backflips vel 2.59 vs 2.25
#  and slow live ori 2.17 vs 1.97 — see ROADMAP "Velocity as a primary metric")
# Per-bag EXTRAS (grids = platform/dynamics params; the rest see ROADMAP):
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp --sliding-window \
  $UNIV --set dt_pos=0.04 --set dt_ori=0.016 --set lock_extrinsics=1 \
  --set-ext 'rotation_euler_deg=[180.0,27.5,0.0]'
# → 0.285/2.31° settled, 0.39m/2.84° live, vel 0.32, ~0.35s/window (RANSAC default)
# Pitch locked at the measured 27.5° (inclinometer 27–28°). Locking vs in-solver
# self-cal is RMSE-neutral (the old "locking beats by −21%" was a pre-RANSAC artifact,
# retracted); we freeze because pitch is weakly observable (self-cal init-dependent).
# DO NOT add radar_zbias_fixed on racing (b=−0.5 → 3.3m): b is a flip-regime proxy, not physical.
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp --sliding-window \
  $UNIV --set max_iterations=12 --set lambda_heading=10.0
# → live: 0.30m/1.94°, vel 0.45, drift 0.63%, 0.70s (RANSAC default)
#   (mapping variant: drop iter cap + λh → settled ~0.31m/1.14°, yaw 0.61°, 2.1s/window
#    [sensor-only, RANSAC default; pre-RANSAC/MoCap-aided was 1.22°/yaw 0.71°/1.64s])
# DO NOT add lambda_pos_init_prior (tether) on racing: poisons fast position (1.5m!)
# and slow full-iter yaw — it is a backflips-only rescue (radar sparsity in flips).
# Per-window diagnostics printed: cost0, jac/res/lin/other timing, iter count, prior cond/rank, tr(S⁻¹)/tr(H⁻¹)

# Backflips sliding window (2026-06-12, universal weighting + backflips extras):
../.venv/bin/python3 validate_live_solver.py backflips_best_velocity --mocap-yaw --cpp --sliding-window \
  $UNIV --set dt_ori=0.008 --set lock_gyro_bias=0 --set lambda_pos_init_prior=0.5 \
  --set radar_zbias_fixed=-1.5 --set-ext 'rotation_euler_deg=[180.0,27.5,0.0]'
# → settled 1.68m/4.82°, live 1.55m/6.26°, vel 2.35, ~0.6s/window (RANSAC default)
# b=−1.5 (ROADMAP 4.2): b is a FLIP-REGIME radar-error proxy, not a physical
# elevation bias (racing explodes under any b even with locked pitch; backflips
# improves monotonically past the WLS-measured −0.5). Curve continues to −2.0
# (1.41 live pos) — capped at −1.5 pending a mechanism.
#   (session start: 1.99/9.22°, live 2.14/8.67°, vel 1.80; Phase 3: 2.56m settled /
#   3.33m live).  λh=10 comes via bags.yaml.  NOTE the vel↔ori gate trade (ROADMAP
#   "Velocity as a primary metric"): no-gates λg400 era gave vel 1.80 at ori 7.62°.
# The ω-adaptive gyro in UNIV (λ_eff = 4·(1+(|z_gyro|/4)⁴): ~4 below 3 rad/s, ~160-500
# in flips) replaces the earlier flat lambda_gyro=400 (that era: 1.80/5.29 | 1.77/6.37);
# ω₀=3 variant gives the best backflips settled ori (4.75°) at a fast-racing ori cost.
# accel_soft_sigma: ω-dependent ACCEL down-weighting (mirror of the radar soft gate) —
# accel distorts orientation mid-flip (ROADMAP Part 5 Tier-1 #6).
# Backflips-only extras: tether λ=0.5 (radar sparsity in flips; POISON on racing — fast
# pos 0.5→1.5m, slow mapping yaw 0.7→2.5°); locked pitch 27.5° (measured as-built; 25.5° was a stale seed);
# λ_ori_accel REMOVED (bit-identical under stiff gyro).
# omega_soft_sigma: ω-dependent radar down-weighting w=1/(1+(|ω|/ω₀)²) — beats the hard
# ω-gate on orientation without discarding data (ROADMAP Part 3c)
# radar_zbias_fixed: per-point elevation bias v_corr = v − b·u_z (ROADMAP Part 4b).
# DO NOT apply on racing bags — b degrades them (b=-0.5/-1.0 → fast batch 0.76→2.6/5.4m);
# on racing it is a regime proxy with no benefit (the "+2° pitch absorbs z-bias" reading is retired).
# NOTE: lambda_pos_init_prior was never plumbed before 2026-06-12 (see SW_DEVELOPMENT §7
# correction); the old Phase 3 command silently ran with tether=0.
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
| `config/extrinsics.yaml` | **As-built pitch measured 27–28°, frozen 27.5°** (deployed via `--set-ext`); yaml keeps 25.5° only as a stale self-cal seed. roll 180°, yaw 0°; translation [0.08, 0.02, -0.01] m |
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

Extrinsic calibration lives in `config/extrinsics.yaml` (pitch 25.5° is now only the
**batch self-cal init**; deployed runs lock **27.5°** via `--set-ext`):
- **Rotation**: `[roll=180°, pitch=27.5° frozen, yaw=0°]` — **physically measured at 27–28°**
  (inclinometer), the SOLE founded anchor. Self-cal is **init-DEPENDENT** (lands at 27.0/27.2°
  only because seeded at 25.5°; 2026-06-24 sweep: 25.5°→27°, 30°→36°, 33°→42°; old
  "init-independent" claim retracted in paper+report) — NOT independent corroboration, so
  frozen for SW, not estimated. `extrinsics.yaml` keeps 25.5° only as the batch self-cal seed
  (stale; don't change — reseeding to 27.5° lands the free-pitch batch at ~31°). SW free pitch
  → 29.5/34.7/40°. Paper/report state only the measured value (no self-cal, no v1/v2 mounts).
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

> **Pre-RANSAC, Huber front-end — superseded by the RANSAC-default headline below
> (2026-06-14). Kept for reference; the deployment metric is the SW live edge.**

Per-bag config auto-selected via `bags.yaml` solver_overrides:
- racing bags: `dt_pos=0.005s, dt_ori=0.008s`
- backflips: `dt_pos=0.010s, dt_ori=0.0008s, lock_extrinsics=1`

| Bag | Pos RMSE | Vel RMSE | Ori RMSE | Ext pitch |
|-----|----------|----------|----------|-----------|
| slow_racing | **0.174m** | 0.151 | **1.08°** | 27.1° |
| fast_racing | 0.758m | 0.386 | 2.58° | 27.6° |
| backflips | **1.817m** | 1.951 | **8.31°** | locked 25.5° |

### C++ sliding window (--mocap-yaw --cpp --sliding-window)

> **Pre-universal, per-bag-tuned prior — superseded by the universal + RANSAC-default
> headline below (2026-06-12/14). Kept for reference only.**

| Bag | Settled pos | Settled ori | Live pos | Live vel | Live ori | marg_prior_scale |
|-----|-------------|-------------|----------|----------|----------|-----------------|
| slow_racing | 0.225m | 1.57° | **0.336m** | 0.390 m/s | **2.08°** | 1e-7 (per-bag) |
| fast_racing | 0.726m | 3.19° | **0.829m** | 0.487 m/s | **3.65°** | 2e-4 (default) |
| backflips¹ | 2.56m | 10.87° | 3.33m | — | 9.33° | 0 (Phase 3 config) |

**2026-06-14 update — RANSAC prefilter is now the DEFAULT front-end** (`solver.yaml`
`radar_ransac_threshold: 0.15`; disable with `--no-radar-ransac`). reve-style 3D-LSQ
RANSAC, seeded `default_rng(0)` (bit-identical reproduction), runs once at frame load
(NOT per window → no timing impact), frames <5 returns bypass. It hard-rejects
elevation-biased single-chip returns before the solve. Verdict (RANSAC vs Huber):
fast_racing live pos −20% (0.50→0.40m), vel 0.41→0.32, ori 3.24→2.88°; slow/backflips
neutral (within +3%); ICINS whole-traj ATE order-of-magnitude (9.6→0.46, 2.9→0.24,
10.9→0.76, 5.5→0.46m); held-out + old-firmware kept-% 46–75% (no starvation), old-fw
backflips ori 10.7→8.1°/9.1°. Config UNCHANGED (universality preserved). Both papers
(report/ master + paper/) rewritten to RANSAC-default; duration/portability hedge +
batch Table II removed. **2026-06-24: accel-bias init is now SENSOR-ONLY by default**
(gravity-aligned scalar correction; legacy MoCap-attitude seed opt-in via `--mocap-accel-bias`,
≤5% on the live edge — review #2 fix). Numbers below are the RANSAC-default, **sensor-only** headline:

| Bag | Per-bag extras | Live vel | Live ori | Live pos (drift) | Settled pos/ori | dt/win |
|-----|--------|---------|---------|------------------|-----------------|--------|
| fast_racing | grids .04/.016 + locked p27.5 | **0.32** | 2.88° | **0.40m** (0.88%) | **0.312/2.35°** | **0.35s** |
| slow_racing (live) | iter12 + λh10 + locked p27.5 | **0.48** | **1.97°** | 0.31m (0.64%) | 0.293/1.53° | 0.70s |
| backflips | tether.5 + b−1.5 + p27.5 + lgb0 | **2.35** | **6.35°** | **1.55m** (2.85%) | **1.67/5.01°** | ~0.6s |

Pre-RANSAC (Huber-only) baseline, 2026-06-12, gates 4/8 — for reference:

| Bag | Per-bag extras | Live vel | Live ori | Live pos (drift) | Settled pos/ori | dt/win |
|-----|--------|---------|---------|------------------|-----------------|--------|
| fast_racing | grids .04/.016 + locked p27.5 | **0.41** | 3.24° | **0.50m** (1.1%) | **0.447/2.66°** | **0.35s** |
| slow_racing (live) | iter12 + λh10 | **0.46** | **1.97°** | 0.30m (0.63%) | 0.286/1.58° | 0.70s |
| slow_racing (mapping) | none (full iter) | 0.36 | — | — | **0.281/1.22°** (yaw 0.71°) | 1.64s |
| backflips | tether.5 + b−1.5 + p27.5 + lgb0 | **2.29** | **6.29°** | **1.51m** (2.8%) | **1.64/4.98°** | ~0.6s |

Gates 2/4 variant (fast-ori-priority): fast 0.566/2.94° live, 0.493/2.39 settled;
backflips vel 2.59. Pre-universal specialized bests: fast 0.639/2.55|0.728/2.88;
slow live 0.287/1.63|0.303/1.92; backflips 1.80/5.29|1.77/6.37 (vel 2.58).
Backflips vel↔ori gate trade: no-gates λg era = vel 1.80 at ori 7.62° (ROADMAP).
dt_pos AND dt_ori were over-dense for fast; slow keeps dt_ori=0.008.
Window 3.0s vs 2.0s for fast is now NEARLY NEUTRAL at the headline dt_pos=40ms config
(2026-06-25 re-run: 2.0s settled 0.380/2.31°, live 0.443/2.95° vs 3.0s settled 0.312/2.35°,
live 0.399/2.88° — ori flat, 3.0s modestly better position). The old "2.0s → roll/yaw ~11°
observability limit" was a dt_pos=5ms/pre-RANSAC artifact; coarsening dt_pos to 40ms removed it.
3.0s kept as default only for the small position gain. Iteration caps must be ≥ natural count for the
chosen dt_pos (slow@20ms needs ~16; capping at 12 explodes position).

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
