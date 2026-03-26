# Live RIO Solver — Summary (2026-03-26)

## What it does

`validate_live_solver.py` is a MoCap-free radar-inertial odometry prototype.
It initialises and optimises a 6-DOF trajectory from radar Doppler velocities
and IMU alone, using the same Levenberg-Marquardt B-spline solver as the
batch pipeline but with sensor-only initialisation.

```
P1  Gyro integration → initial orientation spline (no MoCap SLERP)
P2  IMU-aided radar WLS dead-reckoning → initial position spline
P3  Sensor-only boundary priors at trajectory start
```

MoCap data is loaded but used **only** for final RMSE evaluation.

## How to run

```bash
cd analysis/

# Full sensor-only (no MoCap at all)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity

# With MoCap heading prior + position boundary (best accuracy)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw

# C++ Ceres solver (much faster, better results)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp

# Multi-bag evaluation → eval_results/<label>_<timestamp>.json
../.venv/bin/python3 eval_bags.py --label baseline --flags "--mocap-yaw"
```

MoCap flags:
- `--mocap-init`    — init position origin from MoCap at t_ref
- `--mocap-heading` — heading (yaw) priors from MoCap (pseudo-magnetometer)
- `--mocap-yaw`     — both of the above

C++ flags:
- `--cpp`           — use C++ Ceres solver (loads `config/solver_cpp.yaml` overrides automatically)
- `--imu-hz N`      — override IMU rate (default: 1000 for `--cpp`, 200 for Python)
- `--set key=value` — override any solver.yaml key at runtime (repeatable)

## Benchmark results (--mocap-yaw, 2026-03-26)

### C++ solver (current best)

Uses full-rate IMU (~1000 Hz), `config/solver_cpp.yaml` overrides:
`lambda_gyro=4.0, lambda_snap_pos=2e-5, lambda_bias_prior=10000`

| Solver | slow pos | slow vel | slow ori | fast pos | fast vel | fast ori | time |
|--------|---------|---------|---------|---------|---------|---------|------|
| **C++ (current)** | **0.146 m** | **0.147 m/s** | **0.96°** | **0.925 m** | **0.371 m/s** | **2.35°** | 15–19 s |
| Python baseline | 0.374 m | 0.226 m/s | 3.32° | 1.397 m | 0.412 m/s | 4.38° | ~10 min |

C++ is **2–3× better than Python** on all metrics, and **30× faster**.

Key config findings (all with full-rate IMU, `--cpp`):

| Change | Effect |
|--------|--------|
| IMU 200 Hz → ~1000 Hz (raw rate) | slow: 0.358→0.147m; fast: 1.52→0.925m; faster convergence |
| lambda_snap_pos: 1e-4 → 2e-5 | min-snap was over-constraining racing dynamics |
| lambda_gyro: 1.0 → 4.0 | better orientation tracking, reduces acc_bias blow-up |
| lambda_bias_prior: 1.0 → 10000 | prevents bias "trash-can" at full IMU rate; critical for backflips |

### Python solver variants (200 Hz IMU)

Config: `optimize_pitch_only=true`, `lambda_gravity=0.001`.

| Variant | slow pos | slow ori | fast pos | fast ori |
|---------|---------|---------|---------|---------|
| Baseline (Huber + gravity) | 0.374 m | 3.3° | 1.397 m | 4.4° |
| No gravity (λ=0) | **0.346 m** | 5.8° | 1.417 m | **4.2°** |
| Preintegration (--preintegrate) | 0.346 m | 4.8° | 1.494 m | 4.9° |
| GNC (--gnc, μ_final=1) | 0.396 m | 7.4° | 1.476 m | 5.7° |

**Note:** Old benchmark (2026-03-25, free extrinsics) showed 9.6°/5.7° but was corrupted by
roll/yaw extrinsic drift (+5–7°). See "Extrinsic observability" section below.

Gravity-direction factor (`lambda_gravity=0.001`, default enabled):
- Mahony-style roll/pitch constraint: `r = normalize(a_debiased)·g - R^T·[0,0,g]`
- Down-weighted when `‖a_debiased‖ deviates from g by > gravity_accel_threshold` (3.0 m/s²)
- Applied at every IMU sample (~200 Hz after downsampling)
- **Dramatically improves slow_racing** orientation (5.8°→3.3°, −43%), especially yaw (4.9°→2.6°)
  — gravity anchors roll/pitch absolutely, which lets the heading prior work more effectively for yaw
- **Small regression for fast_racing** (4.2°→4.4°, +5%) — systematic z-velocity bias causes large
  accelerations that the factor partially fights even with the dynamic threshold
- λ=0.001 is the sweet spot; λ≥0.01 diverges fast_racing; λ=0 disables

Preintegration (`--preintegrate`):
- Replaces per-sample accel residuals with Forster TRO-2017 9D factors
- Keeps per-sample gyro at ~200 Hz (orientation knots need dense pinning)
- Jacobian 32% smaller, fast_racing solve 3.3× faster
- **Helps slow_racing** orientation (5.8° → 4.8°, -17%), removes vibration noise
- **Hurts fast_racing** orientation (4.2° → 4.9°, +17%) and velocity (0.41 → 0.49 m/s, +19%)
- For fast flight, per-sample accel provides sub-interval coupling lost in preintegration

GNC (`--gnc`, Geman-McClure loss, Yang et al. RA-L 2020):
- Replaces Huber loss with GM loss `ρ(r; μ) = μr²/(μ+r²)`, annealing μ over phases
- μ_init = (30δ)² = 900 (≈L2), μ_final = δ² = 1 (~TLS at δ); ~10 phases at div=2
- Only applied to radar Doppler residuals; IMU/regularization stay L2
- **Hurts both bags** with corrected extrinsics (5.8°→7.4° slow, 4.2°→5.7° fast)
- Previously appeared to help slow_racing because it was compensating for extrinsic corruption
- ~3× slower than Huber due to phase-based annealing loop; not recommended

## Architecture

### State

| Variable | Parameterisation | Knot spacing |
|---|---|---|
| Position | Quintic B-spline control points | dt_pos = 0.005 s |
| Orientation | Cumulative SO(3) B-spline, incremental Ω_j knots | dt_ori = 0.008 s |
| Biases | Constant b_a (accel), b_g (gyro) | — |

### Residuals

| Factor | Source | Weight |
|---|---|---|
| Radar Doppler | Per-point WLS residual, Huber δ=1 m/s | — |
| Accelerometer | Specific force residual | λ_accel = 0.01 |
| Gyroscope | Angular velocity residual | λ_gyro = 1.0 |
| Min-snap | ∫‖P⁴(t)‖² dt on position | λ_snap = 0.0001 |
| Preintegrated | ΔR/Δv/Δp per radar interval (Forster) | — |
| Heading prior | Yaw-only MoCap pseudo-magnetometer | λ_heading = 3 |
| Boundary pos/vel/ori | Pin start of trajectory to sensor init | λ = 1000 |
| Bias prior | Soft prior on b_a, b_g | λ = 1.0 |

## Key implementation notes

### Doppler unwrapping (two layers)

1. **Pre-unwrapping** (`preunwrap_radar_frames()`): IMU-aided per-point alias
   selection runs before the solver.  Uses accelerometer-integrated world
   velocity, reset by WLS at each frame.  Produces clean `RadarVelocity` copies.
   Fast_racing: 20.9% of points shifted; slow_racing: 1.7%.

2. **In-loop unwrapping**: solver recomputes alias based on current spline
   prediction at each iteration.  `k = round(-r / (2 * v_max))`.

Layer 1 is infrastructure (prevents cascade failure on cold start); layer 2 is
the effective correction.  Both give the same final accuracy — the solver was
already handling aliases correctly from the good initialisation.

### IMU downsampling

Raw IMU rate is ~1000 Hz.  Default target rate is solver-dependent:
- `--cpp`: full rate (stride=1, ~1000 Hz) — dense constraints enable tight bias priors
- Python: ~200 Hz — necessary to keep 10-min runtime

Override with `--imu-hz N`.  Full-rate data is always kept in `imu_data_full`
for preintegration and gyro integration init.

### Position B-spline initialisation

Linear interpolation at control-point times from the radar dead-reckoned
trajectory.  Initial snap cost is ~10^10 (inherent for piecewise-linear input
to a quintic spline) and drops fast in iteration 1.  LS fitting was tried but
is 14× underdetermined (3611 CPs, ~258 radar frames at dt_pos=0.005 s).

### Bias initialisation

Stationary detection (`detect_stationary_bias()`) during the hover phase
before t=19.1 s gives a good accelerometer and gyroscope bias seed.

- Python: biases optimised freely (λ_bias_prior = 1.0) — sparse IMU needs freedom
- C++: tight prior (λ_bias_prior = 10000) — full-rate IMU provides enough constraints that
  biases don't need flexibility; prevents optimizer from absorbing dynamics into bias

## Extrinsic observability

### Only pitch is observable from Doppler

The radar is mounted at 30° downtilt (roll=180°, yaw≈0°).  From Doppler
measurements alone, **only the pitch extrinsic is observable** — rotating the
radar about its boresight (roll/yaw perturbations) does not meaningfully change
radial velocity predictions.

`optimize_pitch_only: true` (default) locks roll and yaw to their initial values
and only optimises pitch.  This prevents 5–7° roll/yaw drift that acts as a
"trash can" for unexplained residuals, which would corrupt the trajectory
(previously caused yaw RMSE to blow up to 8–9°).

With `optimize_pitch_only: false`, the solver drifts to:
- slow_racing: Δroll=+5.4°, Δyaw=+4.5° → yaw RMSE 8.7°
- fast_racing: Δroll=+6.7°, Δyaw=+3.5° → yaw RMSE 8.3°

The physical mount is 30° pitch; solver self-calibrates to:
- slow_racing: pitch ≈ 25.3° (barely moves from 25.5° init)
- fast_racing: pitch ≈ 27.0° (moves ~1.5° toward physical mount)

The partial pitch correction for fast_racing partly absorbs the systematic
z-velocity bias from limited elevation diversity.

## Backflips bag

The backflips bag shows the limits of the batch approach for extreme dynamics:

| Solver | pos | vel | ori |
|--------|-----|-----|-----|
| Python | 1.88 m | 3.8 m/s | 13° |
| C++ default config | 3.86 m | 4.9 m/s | 52° |
| C++ tuned (snap=2e-5, gyro=4, prior=10k) | 2.93 m | 4.6 m/s | 10.7° |
| C++ snap=1e-6 (backflip-optimised) | 1.56 m | 3.2 m/s | 10.2° |

**Root cause: initialization drift from large gyro bias**

Gyro bias z = +0.152 rad/s (8.7°/s).  Over the 5s trajectory the orientation
init already drifts 10.8° by t=2.5s — before optimization starts.

```
Gyro-init angle error vs MoCap:
  slow_racing:  0° → 7.7° over 25.8s  (0.30°/s)
  fast_racing:  0° → 2.4° over 18s    (0.13°/s — stays low, small bias)
  backflips:    0° → 10.8° at t=2.5s  (4.3°/s — large bias + short window)
```

The optimizer can compensate if the init is within the basin of attraction
(~3m position init is similar across bags), but the orientation init for
backflips is already ~11° wrong *for the wrong spline shape*, so the
optimizer converges to a local minimum that does not capture the flips.

**Fix: sliding window (Phase 4)**

In a 1–2s sliding window, the gyro integration runs for at most 1-2 knot
intervals from a known state.  Drift at 4.3°/s over 1s ≤ 5° — recoverable.
The previous window's optimized orientation initializes the next, so errors
do not accumulate.  Backflips (~0.25s each) fit entirely within a 1s window.

## Known limitations

### fast_racing position error (~1.42 m)

Root cause is the radar's limited elevation diversity (2 TX antennas), which
causes a systematic z-velocity underestimate of ~0.5–0.65 m/s.  At fast
racing speeds, z-velocity is larger and changes more rapidly, amplifying this
bias.  The optimiser absorbs it by distorting roll/pitch (fast_racing roll
RMSE 5.7° vs slow_racing 3.0°), which propagates into position error.

Initial residual stats confirm this: fast_racing mean = −0.44 m/s, std = 2.74 m/s
vs slow_racing mean = +0.06 m/s, std = 0.79 m/s.

Fixing this properly would require better elevation diversity or explicit
modelling of the z-velocity bias as an optimised parameter.

### Yaw observability

Yaw is unobservable from Doppler alone (all yaw-equivalent trajectories give
the same Doppler predictions for pure rotation about gravity).  The
`--mocap-heading` flag provides the only yaw reference.  Yaw RMSE is ~5°
for both bags with `optimize_pitch_only=true`; the
`lambda_boundary_ori_yaw = 0.0` setting keeps yaw as a free gauge
(no boundary prior on yaw).

### Orientation jump at iteration 1

Orientation RMSE spikes from ~3° to ~50° in iteration 1 for all runs.  This
is the snap regularisation forcing the initial piecewise-linear control points
smooth, temporarily distorting orientation before it recovers.  Expected
behaviour, not a bug.

## Files

| File | Role |
|---|---|
| `validate_live_solver.py` | Main script: P1-P3 init + LM solver |
| `validate_nonlinear_solver.py` | Solver core (shared with batch pipeline) |
| `eval_bags.py` | Multi-bag evaluation harness |
| `lib/imu_preintegration.py` | Forster TRO-2017 on-manifold preintegration |
| `codegen/generated_jacobians.py` | SymForce-generated residuals + Jacobians |
| `config/solver.yaml` | All hyperparameters |
| `config/bags.yaml` | Bag aliases, timing windows, radar configs |
| `eval_results/` | JSON results from eval_bags.py runs |
| `FINDINGS_PREINTEGRATION.md` | Detailed investigation notes |
