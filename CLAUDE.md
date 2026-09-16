# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Map

| Document | Role |
|----------|------|
| **CLAUDE.md** (this file) | Operational hub: how to run, current config, current results |
| `documentation/HANDOFF.md` | **Read first when resuming.** Where the notebook stands, the measured constants, what NOT to re-explain, what is already settled, and the traps that cost time. Overwritten in place, never dated and never forked, so there is only ever one |
| `worklog/README.md` | **Experiment record.** One file per experiment (`YYYY-MM-DD_slug.md`): what was tested, how, results, consequences — with a Corrections block when a conclusion is later overturned. Index lists every experiment whose result is live. **New experiments go here**, not in `worklog/FEEDBACK_PLAN.md` |
| `worklog/FEEDBACK_PLAN.md` | **FROZEN 2026-08-08.** The chronological progress log up to that date, and the source for the backfilled worklog files. Do not append. Its own internal paths are as-of-freeze (docs were reorganised 2026-08-08 after the freeze) |
| `worklog/reports/` | The long-form standalone reports the worklog entries link to but do not copy: `MECHANISM_VERDICT_2026-08-06.md` (the measured per-return noise law, full evidence), the two `AUDIT_NOTEBOOK11_*` adversarial audits, and the prompts that produced them. Dated artifacts: read, do not edit |
| `report/main.tex` | **The guided-research project report (10 ECTS), rewritten 2026-09-03 from the T-RO draft Lukas reviewed** (rev0 comments on branch `feedback_v0`). IEEEtran two-column kept; report title block + AI-use declaration; no page limit; no negative-results section; deployed per-return radar model throughout; results from `worklog/2026-09-03_report-battery.md`. Thin root: preamble + `\input{acronyms}` + `\input{sections/*.tex}` (one file per section) + bib. Structure and figure generators: `report/NOTES.md`; information flow: `documentation/report_notes/REPORT_FLOW.md`; plan/decision record: `documentation/report_notes/REPORT_SYNC_TODO.md`. Figures: `gen_characterization.py` (ladder.pdf, law.pdf), `gen_combined_traj.py`, `gen_error_time.py` (need `--save-arrays` headline runs). Nissov & Alexis 2026 deliberately NOT cited. **`paper/` (conference cut) is FROZEN (2026-06-24): do NOT edit it.** |
| `documentation/Forward Model.md` | Math reference: sensor forward models, coordinate frames, Doppler sign convention |
| `documentation/Backward Model.md` | Math reference: state parameterization (cumulative SO(3) B-spline), factor-graph MAP, Ceres LM, Schur-complement marginalization |
| `documentation/FINDINGS.md` | Foundational calibration findings: body frame, time offsets, Doppler sign fix (§11), extrinsics |
| `documentation/RESEARCH_NOTES.md` | Design rationale: solver perf profile, Doppler unwrapping, preintegration investigation, BandedSchurSolver post-mortem, SW timing analysis, real-time/speedup path (§1/§7-§10; absorbed the deleted `Realtime_Options.md`) |
| `documentation/SW_DEVELOPMENT.md` | SW solver development history: phase ablations, sweep tables, backflips analysis, marg_prior_scale tuning |
| `documentation/ROADMAP.md` | Design-rationale **archive** (2026-06-11 audit, largely executed): the "ROADMAP Part X / N.M" targets that CLAUDE.md's inline comments reference. Open item: iSAM2 back-end |
| `documentation/report_notes/` | Everything *about* writing the paper, as opposed to the paper itself: `REPORT_FLOW.md` + `REPORT_SYNC_TODO.md` (structure and code↔paper sync), `CUT_PLAN.md` (page-budget trim), `reviewer_feedback_v0.txt` + `paper_discussion_notes.md` + `review_decisions_pending.md` + `REVIEW_RESPONSE.md` (the review rounds) |
| `documentation/todo/` | Open items: `NEXT_LEDGER_ITEMS.md` (the optional 3a-3d ledger items), `TODO_MEASUREMENTS.md` (bench measurements Timo owes), `ToDo.md` (older scratch list), **`CLEANUP_LEDGER.md` (stale code and dead parameters logged for the post-report cleanup; append whenever something turns up, never clean up ad hoc)** |
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
# >>> 2026-07-16: the per-bag commands below are PRE-UNIFICATION history.
# >>> Deployed config = the UNIFIED command in "Current Results" (one flag set
# >>> for all bags: grids 40/16ms, lh=0.6, no iter cap, no rescue terms).
# 2026-06-12 consistency fixes (see documentation/ROADMAP.md Part 1 results):
#   marg_markov_blanket=1 + warm_start_align=1 are new C++ defaults (--set ...=0 for legacy).
#
# UNIVERSAL WEIGHTING CONFIG (2026-06-12, ROADMAP Part 5 "universal config" +
# velocity-metric rebalance): one weighting set for ALL bags — ω-adaptive gyro
# λ_eff = 4·(1+(|z_gyro|/4)⁴), radar/accel ω-soft-gates 4/8, per-point SNR weighting:
UNIV="--set marg_prior_scale=1.0 \
  --set omega_soft_sigma=4.5 --set accel_soft_sigma=8.0"
# (2026-08-09: SNR/intensity weight DELETED -- radar_intensity_weight no longer set.
#  Unfounded shape (measured sigma peaks MID-intensity while the law is monotone),
#  no identifiable exponent, ~4% of variance, and NO consistent end-to-end sign:
#  own platform +-3%, ICINS 2/4 better and 2/4 worse at up to +-46%. Median effect
#  of removal -10.8% ATE / -10.4% ori. Solver option remains, default 0 -- same
#  treatment as the gyro omega-boost. See worklog/2026-08-09_snr-weight-deletion.md)
# (2026-07-16: gyro omega-boost DELETED — lambda_gyro_omega_sigma no longer set;
#  unfounded mechanism, see worklog/FEEDBACK_PLAN.md; solver option remains, default 0)
# (2026-07-27: omega_soft_sigma 4.0 -> 4.5 — the radar rate-law omega_0 now deploys
#  the value FITTED on the reference-immune sigma_i (notebook 5b: 4.5±0.5, protocol
#  spread ~3.9-4.6); gate battery flat (racing/ICINS par, backflips drift −0.7pp);
#  _ALIAS_OMEGA0 in validate_live_solver.py tracks it. See FEEDBACK_PLAN 2026-07-27.)
# (gates 2/4 variant = fast-ori-priority: fast 2.94° live ori but backflips vel 2.59 vs 2.25
#  and slow live ori 2.17 vs 1.97 — see ROADMAP "Velocity as a primary metric")
# Per-bag EXTRAS (grids = platform/dynamics params; the rest see ROADMAP):
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp --sliding-window \
  $UNIV --set dt_pos=0.04 --set dt_ori=0.016 --set lock_extrinsics=1 \
  --set-ext 'rotation_euler_deg=[180.0,27.5,0.0]'
# → 0.382/2.34° settled, 0.47m/2.90° live, vel 0.32, ~0.35s/window (sensor-only unwrap, RANSAC default)
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

### Characterization scripts (the research-report measurements)

These produce the numbers the story notebook (`notebooks/11_algorithm_story_speed_law.ipynb`)
and the paper quote. All are estimator-free: residuals are formed against MoCap
through the forward model, never against the solver's own output.

| File | What it measures |
|------|------------------|
| `radar_config_spec.py` | Sensor spec derived from the uploaded chirp `.cfg`: **CPI 49.92 ms**, v_max, Doppler bin, range res, duty cycle, ~30° angular res. Closes against the data (dv 0.0492 vs measured 0.049; dr 0.214 vs 0.214). Also shows why the old-firmware bags *cannot* show the speed law (their 0.174 m/s quantization floor buries it) |
| `characterize_pointwise_noise.py` | **Per-point** noise mechanism test. Winner: a bearing error, `v·cosθ` (Doppler scale error) an exact null, CPI-smear loses. Two estimators (binned robust σ; within-frame pairwise composite likelihood). **Superseded on the constant (2026-08-06):** its δφ = 3.5° was fitted without an off-boresight factor; see `characterize_shape_contradiction.py` |
| `characterize_shape_contradiction.py` | Resolves the frame-level (`σ ∝ v²`) vs per-point (`σ ∝ v sinθ`) disagreement: **both ladders lacked an off-boresight rung**, which is where the whole discrimination lives. Measured law σ_j = (δφ₀/√2)·\|v\|·sinθ_j/cos φ_j, **linear in speed**, δφ₀ ≈ 2° at boresight. Reproduces both published ladders as a gate, then re-scores under corrected statistics, 7 confounder controls, 6 held-out flights, and a whitening-adequacy test. **Gate battery RUN 2026-08-06 and NOT ADOPTED** (`--bearing-weight b`, 8 flights + scale controls): velocity improves everywhere (8–9% racing at two weight scales, 0.8–2.2% ICINS), orientation regresses off the fit set (backflips +0.22°, fast2 +0.09°). **The orientation cost is UNEXPLAINED** — the earlier "the weight thins a frame's angular diversity, cond ↑1.3–1.6×" mechanism was RETRACTED 2026-08-07 (one frame's Jacobian has rank 3 of 9 and carries no attitude information, so a per-frame statistic cannot reach that channel), and its replacement candidate (the mis-composed frame covariance) was gated 2026-08-08 and does not survive. See `worklog/2026-08-06_bearing-weight-battery.md` |
| `characterize_reference_immune.py` | Ladder selection controls (random / oracle / held-out consensus); post-ladder tail decomposition (36–88 % frame-shared); elevation, intensity (α = +1.07 on slow) and the alias floor re-measured reference-immune at matched speed; σ_c vs timestamp jitter (refuted) |
| `characterize_frame_observability.py` | What one frame constrains: conditioning, per-axis noise amplification (per-point worst on **z**, frame-shared worst on **x**), elevation diversity |
| `characterize_imu_noise.py` | Reference-free IMU floors from the in-bag motors-off / motors-on-ground segments. Shows λ\* encodes the **total** residual budget, not sensor noise |
| `derive_stream_weights.py` | λ\* = 1/(σ² f_s τ) for all three streams. Reproduces the deployed 3.13 / 0.00392 / 2.11 (radar via the Huber influence-capped moment) |
| `characterize_speed_law.py` | Frame-level speed-law ladder, held-out and cross-platform transfer (E1–E4) |
| `characterize_frame_correlation.py` | Intra-frame correlation σ_i/σ_c, Kish N_eff, in-band IMU σ |

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

The radar shows a systematic z-velocity bias of −0.5 to −0.65 m/s (measured; mechanism OPEN.
The old “2 TX / limited elevation diversity” attribution was WRONG for the AOP package — its
3TX×4RX on-package array has ~symmetric az/el resolution, 120°×120° FoV. Leading candidate:
floor-multipath / scene anisotropy, consistent with the measured az/el bearing-noise anisotropy
that the isotropic angle-FFT lattice does not explain, and with the SAME sensor (IWR6843AOP,
verified 2026-08-12) showing a larger same-signed bias on the ICINS indoor scenes.)
Doppler quantization is 0.63 m/s per bin — keep Huber δ ≥ 1.0 m/s.

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
| `lambda_gravity` | **0.0** | Gravity-direction (tilt) factor, **disabled**. Inert on the racing/backflips headline flights (verified bit-identical); on the ICINS cross-validation the full-stride gravity=0 re-run reproduces the published baselines to within run-to-run noise (≤0.002 m ATE), so it is omitted from the paper and off by default |
| `max_iterations` | 40 | C++ SW uses this; batch Python uses early-stop |

### `config/solver_cpp.yaml` — C++ overrides (applied on `--cpp`)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lambda_gyro` | **3.13** | **DERIVED** (2026-07-17): 1/(sigma_ib^2 fs tau); was tuned 4.0 |
| `lambda_accel` | **0.00392** | **DERIVED**; was tuned 0.01 |
| `radar_weight` | **2.11** | **DERIVED** (Huber-effective sigma); radar no longer the numeraire 1.0 |
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
fast_racing live pos −8% (0.51→0.47m), vel 0.41→0.32, ori 3.31→2.90°; slow/backflips
neutral (within +3%); ICINS whole-traj ATE order-of-magnitude (9.6→0.46, 2.9→0.24,
10.9→0.76, 5.5→0.46m); held-out + old-firmware kept-% 46–75% (no starvation), old-fw
backflips ori 10.7→8.1°/9.1°. Config UNCHANGED (universality preserved). Both papers
(report/ master + paper/) rewritten to RANSAC-default; duration/portability hedge +
batch Table II removed. **2026-06-24: accel-bias init is now SENSOR-ONLY by default**
(gravity-aligned scalar correction; legacy MoCap-attitude seed opt-in via `--mocap-accel-bias`,
≤5% on the live edge — review #2 fix). **2026-07-07: Doppler alias unwrapping is now
SENSOR-ONLY by default too** (IMU-integrated alias prediction; legacy MoCap-aided opt-in
via `--mocap-unwrap`/`RIO_MOCAP_UNWRAP`). This shifted only fast-racing position (live
0.40→0.47m / 0.88→1.04%, settled 0.31→0.38m; batch 0.64→0.74m); slow/backflips/held-out/ICINS
bit-identical (verified: ICINS 0/103608 pts alias). See
`documentation/UNWRAP_SENSOR_ONLY_CORRECTION.md`. Numbers below are the RANSAC-default, **sensor-only** headline:

| Bag | Per-bag extras | Live vel | Live ori | Live pos (drift) | Settled pos/ori | dt/win |
|-----|--------|---------|---------|------------------|-----------------|--------|
| fast_racing | grids .04/.016 + locked p27.5 | **0.32** | 2.90° | **0.47m** (1.04%) | **0.382/2.34°** | **0.35s** |
| slow_racing (live) | iter12 + λh10 + locked p27.5 | **0.48** | **1.97°** | 0.31m (0.64%) | 0.293/1.53° | 0.70s |
| backflips | tether.5 + b−1.5 + p27.5 + lgb0 | **2.35** | **6.35°** | **1.55m** (2.85%) | **1.67/5.01°** | ~0.6s |

**2026-07-15: whitened per-frame correlated-noise model ADOPTED as the --cpp
default** (`radar_frame_sigma_c: 0.4` in `config/solver_cpp.yaml`; measured
intra-frame correlation, see `analysis/characterize_frame_correlation.py` and
worklog/FEEDBACK_PLAN.md progress log for the full gate battery). One stacked whitened
residual block per radar frame (per-point Huber -> warm-start IRLS weights
inside the stack); works with locked AND free extrinsics. NEW HEADLINE (same
commands as above, sigma_c rides along via the config default):

| Bag | Live vel | Live ori | Live pos (drift) | Settled pos/ori | dt/win |
|-----|---------|---------|------------------|-----------------|--------|
| fast_racing | **0.29** | 2.86° | **0.40m** (0.88%) | **0.334/2.36°** | **0.21s** |
| slow_racing (live) | **0.48** | **1.89°** | 0.31m (0.64%) | 0.292/1.42° | 0.30s |
| backflips (+rescue) | **2.31** | **6.54°** | **1.58m** (2.92%) | **1.65/5.81°** | ~0.53s |
| backflips (universal, no rescue) | 3.03 | 7.09° | 9.8m (18.2%) | — | 0.53s |

Both racing regimes are now REAL-TIME at the 0.3s stride (frame stacking
halves solve time).

**2026-07-16: gyro omega-boost DELETED from UNIV** (principled-weighting
branch; unfounded/wrong-sign as a noise model). New headline: slow
0.31m/0.64%/0.48/1.81deg @0.31s; fast 0.47m/1.05%/0.28/2.55deg @0.22s;
backflips rescue 1.59m/2.94%/2.29/7.07deg; universal 9.9m/18.3%/2.98/7.86deg.
Trade accepted (Timo): fast pos +0.17pp & backflip ori +0.5deg for a fully
founded weighting; racing ori/vel improve; backflip ori NEES 36->12. Held-out/ICINS/NEES all par or better (NEES fast vel
1.0->1.5, backflips vel 9.6->5.8). Set `--set radar_frame_sigma_c=0.0` for the
legacy per-point behaviour. Solver branch history: `stochastic-noise-model`
(kept). Ongoing weighting-foundation experiments: branch `principled-weighting`
(merge to main only on no-regression gates).

**2026-07-19: CAUSAL ALIAS DOWN-WEIGHT ADOPTED (driver default; Timo).**
Aliased returns keep recoverable Doppler but a broken on-chip bearing
(TDM Doppler compensation; repair proven impossible downstream, R^2<=0.14).
Measured: regime-flat error floor sigma_al~0.6 m/s -> causal rule
w(|omega|) = clip(sigma0^2(1+(|w_gyro|/4)^2)/0.6^2, 0.01, 1), sigma0=0.095.
Implemented via a per-point weight channel (RadarPoint.w, Nx6
make_radar_frame); validate_live_solver --alias-weight defaults 'auto'
('off' = legacy). Beats keep AND delete: fast 0.297/2.56deg/0.92%,
fast2 held-out 0.396/2.64deg/1.60% (best of all variants), slow par,
backflips 2.37/8.17deg/11.0% (vel +0.19 for -7.3pp drift, best-tier ori).
NEES at the full default: slow conserv, fast full-state x0.90 OK,
backflips VELOCITY now calibrated x1.04 (ori overconf n=6 caveat stays).
Open refinement: re-derive radar_weight on the clean stream (current 2.11
errs conservative). Notebook ch3 = Gaussian-core discovery ladder,
ch6b = the alias saga; story->solver map audited, no fork. Report
renumber to these values PENDING Timo's notebook read.**

**2026-09-07 (evening): FOUNDED REMOVALS DEPLOYED (Timo) -- WINDOW-START
PINS FIRST WINDOW ONLY, BOTH HUBER KERNELS OFF, RADAR WEIGHT RE-DERIVED.**
`solver_cpp.yaml`: `boundary_first_window_only: 1` (the P3 pos/vel/ori pins,
lambda 1000 hand-set, used to be re-added at EVERY window start anchored to
the previous estimate and marginalized into the next prior; found by the
Fig. 1 audit), `huber_delta: 1e6`, `huber_delta_accel: 1e6` (exactly
quadratic losses; the radar kernel was inert, the accelerometer kernel
costs 4.5 deg flip orientation but backflips is the declared degradation
case), `radar_weight: 2.64` (plain RMS of the non-aliased prefiltered
stream; 2.11 was the Huber-capped moment). Canonical command unchanged
and flag-free; flag-free slow racing bit-identical to the battery arm.
NEW HEADLINE (live vel/ori/drift, `worklog/2026-09-07_founded-removals.md`):
slow **0.165/1.655/0.32**; fast **0.279/2.557/1.33** (vertical 0.51 m: the
pins were masking the vertical channel, accepted); backflips
**1.660/12.14/2.91**; fast2 0.273/2.703/1.18; ICINS 0.087/0.551/0.40,
0.080/1.000/0.97, 0.080/0.530/1.34, 0.076/1.214/1.20 (ICINS drift ~1 %
accepted as "passable"). Solo t_win 0.16/0.18/0.20 s. NEES: racing
conservative (slow x0.26/x0.36, fast x0.38/x0.59), backflips vel x1.36 /
ori x3.16 (n=6). Report renumbered (part 1 pushed; ablation tables from
battery v3). The set-A block below is the PREVIOUS headline.

**2026-09-07: SET A DEPLOYED (Timo: "deploy it") -- MEASURED BIAS PRIORS +
POSITION-ROW STALENESS.** `solver_cpp.yaml`: `lambda_bias_prior_accel: 21`,
`lambda_bias_prior_gyro: 77000` (sigma = the measured error of the pre-flight
bias anchor against the in-flight bias, from the motors-off / motors-on
ground windows; replaces the hand-set 10000/10000) and `marg_stale_pos: 0.12`
(m per swept rad, GT-derived with the `marg_stale_phi` protocol; the
position-row analogue). Regularization UNCHANGED (no-regularization is a
large velocity win alone but diverges on ICINS 4 with the loosened bias
prior: `worklog/2026-09-04_deployment-batteries.md`). Canonical command
unchanged and flag-free. NEW HEADLINE (live vel/ori/drift, battery v2,
`worklog/2026-09-07_set-a-deployment.md`): slow **0.166/1.70/0.64**; fast
**0.288/2.49/0.75** (vertical drift 0.30 -> 0.09 m); backflips
**2.047/7.37/7.03**; fast2 0.277/2.84/1.67; ICINS 0.087/0.671/0.24,
0.079/1.233/0.62, 0.079/0.727/0.58, 0.075/1.390/0.66 (v2 battery at
the flag-free config, identical to the set-A battery). NEES: racing
conservative (slow x0.25/x0.42, fast x0.36/x0.72), backflips vel x1.25 in
CI, ori x2.68 (n=6). The 2026-08-12 block below is the PREVIOUS headline.

**2026-08-12: PER-RETURN RADAR MODEL ADOPTED AS THE UNIFIED DEFAULT (Timo).**
The measured per-return law replaces the per-frame speed law:
`radar_weight_model: bearing_b` + exact frame covariance
(`radar_frame_hetero: 1`, `radar_frame_sigma_c: 1.27`, the bearing-channel
own-platform geomean) + zero-parameter model-sigma Huber knee
(`huber_model_knee: 1`) + GT-derived staleness inflation
(`marg_stale_phi: 0.01`) -- all solver_cpp.yaml defaults, so the canonical
command needs NO radar-model flags (omega soft gate stays 0; CLI
--bearing-weight/--speed-weight override; 'speed' = legacy arm):
```
../.venv/bin/python3 validate_live_solver.py <bag> --mocap-yaw --cpp --sliding-window \
  --set marg_prior_scale=1.0 --set accel_soft_sigma=8.0 \
  --set dt_pos=0.04 --set dt_ori=0.016 --set lambda_heading=0.6 \
  --set lock_extrinsics=1 --set-ext 'rotation_euler_deg=[180.0,27.5,0.0]'
# backflips additionally: --set lock_gyro_bias=0
# ICINS: dataset extrinsics + translation + --imu-hz 400 --whole-traj-align
```
DEPLOYED HEADLINE (live vel/ori/drift): slow **0.177/1.675/0.82**; fast
0.280/2.558/1.01; backflips **2.040/7.773/7.44** (best-ever vel); fast2
0.276/2.836/1.87 (cc default); ICINS 0.087/0.635/0.25, 0.079/1.209/0.59,
0.079/0.689/0.57, 0.075/1.362/0.65. ICINS drift +0.2-0.3pp vs the legacy
per-frame arm is ACCEPTED (one config, zero per-platform calibration; the
bearing-channel sigma_c is measured regime-dependent -- ICINS's own 0.93
closes 35-58% of the gap -- stated in the report as "no param tuned to
their platform"). Fast2's baseline includes the cluster-collapse default.
Full record: worklog/2026-08-12_per-return-adoption.md (+ the gp-port,
zbias-gate-battery, cluster-collapse and zbias-measurement entries of the
same day). Report/notebook renumber PENDING. The 2026-07-17/19 numbers
below are the LEGACY per-frame arm.

**2026-07-17: DERIVED STATIC WEIGHTS ADOPTED (Timo: "any params should be
derived") — lambda_gyro=3.13, lambda_accel=0.00392, radar_weight=2.11 are
now solver_cpp.yaml defaults, each from lambda* = 1/(sigma^2 fs tau) with
measured inputs (analysis/derive_stream_weights.py; radar uses the
influence-capped sigma E[min(|r|,delta)^2] which reconciles the two racing
bags 1.9/2.3 where the naive core-sigma gave 3.1/11.1 and failed gates at
its geomean 5.83). NEW HEADLINE (unified command unchanged, live
vel/ori/drift): slow 0.179/1.71deg/0.59%; fast 0.316/2.58deg/1.17%;
backflips 2.18/8.32deg/18.3% (universal, declared failure). Held-out fast2
0.455/2.68deg/2.18% (+16% vel vs tuned statics = the accepted trade);
ICINS whole-traj ATE 0.179/0.102/0.317/0.213 m (par-to-better). NEES:
slow conservative (0.37/0.61 vs 3), fast CALIBRATED (full-state
sigma_scale x1.00; mean statistic, median conservative). Trade accepted
mirroring the gyro-boost deletion: modest fast/fast2 velocity cost for a
fully derived weight set. The 2026-07-16 numbers below are SUPERSEDED.**

**2026-07-16: CONFIG UNIFICATION ADOPTED (Timo) — one flag-free config for
ALL bags and platforms.** Per-bag extras DELETED: grids unified to
dt_pos=0.04/dt_ori=0.016, lambda_heading=0.6 everywhere, no iteration cap
(all regimes converge in <=12 iters at this grid), pitch locked 27.5 (own
platform) / dataset extrinsics (ICINS), NO rescue terms (backflips =
declared degradation). Command (identical for all three bags; backflips
additionally needs `--set lock_gyro_bias=0` to undo a batch-era bags.yaml
override):
```
UNIFIED="$UNIV --set dt_pos=0.04 --set dt_ori=0.016 --set lambda_heading=0.6 \
  --set lock_extrinsics=1 --set-ext 'rotation_euler_deg=[180.0,27.5,0.0]'"
../.venv/bin/python3 validate_live_solver.py <bag> --mocap-yaw --cpp --sliding-window $UNIFIED
```
NEW HEADLINE (live vel/ori/pos-drift): slow 0.21/1.81deg/1.12%; fast
0.28/2.55deg/1.05%; backflips 2.28/8.63deg/16.1% (universal, honest
failure). ICINS 1-4 improve 2-3x (whole-traj ATE 0.19/0.09/0.31/0.19m,
drift 0.2-0.5%). fast2 held-out unchanged. NEES: slow conservative
(0.69/0.98), fast calibrated (1.30/2.33), backflips overconf (ori ~28,
n=6). lambda_heading ladder 0->10: pos/vel flat everywhere, ori moves
<=0.6deg (lh=0 = GT-free after start gauge). KNOWN LIMIT: circle held-out
(5 m/s >> 3 m/s Doppler ambiguity, 63.8% aliased points) degrades
4.2%->8.4% under the unified config; needs the legacy dense-grid
(5/8ms)+lh10+iter12 recipe — an aliased-regime limit, not a weighting
failure. The old "slow yaw drifts 4.3deg at lh0.6" and "slow +0.3deg at
dt_ori=16ms" were dense-grid artifacts. Paper renumbered on branch
unify-config (tab:config deleted, backflips rescue rows dropped to a
Limitations sentence).

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
(sensor-only unwrap: 2.0s settled 0.455/2.33°, live 0.518/2.95° vs 3.0s settled 0.382/2.34°,
live 0.469/2.90° — ori flat, 3.0s modestly better position). The old "2.0s → roll/yaw ~11°
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
