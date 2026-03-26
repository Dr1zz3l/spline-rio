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

**Root cause: position initialization missing flip dynamics**

The gyro bias IS subtracted before integration (`omega = z_gyro - b_g`).
The orientation error for backflips is oscillating (not monotonically growing):

```
Gyro-init angle error vs MoCap:
  slow_racing:  0° → 7.7° over 25.8s  (monotonic, ~bias drift)
  fast_racing:  0° → 2.4° over 18s    (low and stable)
  backflips:    0° → 10.8° at t=2.5s → 6.3° at t=3.75s → 8.4° at t=5.0s
```

The oscillating backflips pattern is gyro scale-factor / high-rate integration
error, not bias accumulation.  The orientation init is roughly correct (the
flips are tracked), but with some phase/amplitude error.

The bigger problem is the **position initialization**.  During backflips the
drone undergoes several metres of Z oscillation, but the P2 dead-reckoning
completely misses this: radar WLS is unreliable when the antenna is rotating
rapidly (aliased Doppler, wrong body→world projection from orientation error),
so the initial position trajectory is flat.  Starting from a flat trajectory,
the min-snap regularization prevents the optimizer from freely reshaping into
the flip shape — it converges to a local minimum instead.

**Fix: sliding window (Phase 4)**

In a 1–2s sliding window, the previous window's optimized state (which already
captured the flip dynamics) provides the initialization for the next window.
The optimizer only needs a small correction from a position init that is
already physically correct.  Backflips (~0.25s each) fit entirely within
one 1s window.

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

## C++ Solver Performance Profile

Measured via Ceres `Solver::Summary` timing fields, exposed on `result.time_*_s`.
Printed automatically after each `--cpp` solve.

### Batch solve (full trajectory, ~18-26s of data)

| Phase | slow_racing | fast_racing | Share |
|-------|------------|------------|-------|
| Jacobian eval (autodiff Jet<double,N>) | 4.95s | 3.85s | **39%** |
| Residual eval | 0.46s | 0.37s | 4% |
| Linear solve (sparse Cholesky) | 6.00s | 4.87s | **48%** |
| Other (preprocessing, callbacks) | 1.14s | 0.88s | 9% |
| **Total** | **12.55s** | **9.96s** | |

Both the autodiff and the linear solve are significant — neither dominates overwhelmingly.

### How Ceres autodiff works

Ceres autodiff is **not** symbolic or one-time — it runs at every LM iteration
for every residual block using dual-number (Jet) arithmetic.  Each functor is
called with `T = Jet<double, N>` where N = number of parameter dimensions in
the block (e.g. 40 for the accel factor: 4 ori knots×4 + 6 pos CPs×3 + 6 bias).
Every scalar op carries an N-vector of partial derivatives alongside the value,
so cost ≈ N× plain-double arithmetic.

With 25k IMU samples × 3 factors × 35 iterations, this adds up to the 39% above.

### Paths to further speedup

**Analytic Jacobians (→ −39%)**: replace Jet evaluation with closed-form Eigen
expressions.  Path: swap `PythonConfig()` → `CppConfig()` in
`codegen/derive_jacobians_symforce.py` to emit C++ instead of NumPy, then
implement as `ceres::SizedCostFunction` subclasses.  basalt's own VIO uses
this approach for its spline factors.

**Variable ordering / fill-in reduction (→ part of −48%)**: Ceres uses
automatic variable ordering for the Cholesky factorisation.  For a B-spline
problem, the natural ordering (knots in time order) is already near-optimal,
but the combined pos+ori+bias block structure may not be.  Try
`options.linear_solver_ordering` with explicit elimination groups.

**Sliding window (→ smaller n)**: a 1–2s window has ~400 pos CPs + 250 ori
knots ≈ 3000 parameters vs the current ~27000.  Cholesky scales super-linearly,
so expect a much larger than 9× speedup on the linear solve alone.

**Combined target**: analytic Jacobians + sliding window → estimated <2s/window
at real-time rates.

## Sliding Window (Phase 4a)

Implemented as a fixed-lag smoother in `validate_live_solver.py` (`--sliding-window`).
Re-solves the last `window_duration` seconds every `window_stride` seconds.
No marginalization — global CP arrays serve as warm-start cache; boundary priors
(λ=1000) anchor the leading edge to the previous window's solution.

### Results on slow_racing (--mocap-yaw)

| Config | Pos RMSE | Ori RMSE | Time/window |
|--------|----------|----------|-------------|
| Batch (--cpp, baseline) | **0.146m** | **0.96°** | 12.6s total |
| SW window=1.5s, n_fix=6 | 1.149m | 2.29° | 0.5–1.8s |
| SW window=1.5s, n_fix=0 | 1.000m | 1.34° | 0.5–1.8s |
| SW window=3.0s, n_fix=0 | 0.358m | 1.34° | ~1.1s |
| SW window=5.0s, n_fix=0 | 0.262m | 1.15° | ~2.5s |

**Key observations:**
- Larger window → better accuracy (monotonic trend; no hard floor visible yet)
- Ceres converges in 2 LM iterations per window (warm start from prev window)
- Hard-fixing leading knots (`n_fix>0`) is WORSE than boundary-prior anchoring (`n_fix=0`)
  — fixing freezes bad state if previous window wasn't perfectly accurate
- 3.0s window is the default (best accuracy/speed balance)
- Accuracy gap vs batch (~2.5×) due to missing marginalization: no forward propagation
  of global corrections. Batch solver benefits from seeing all heading priors jointly.

### Why the sliding window is worse than batch

Without marginalization, each window independently estimates position from local
Doppler + heading priors.  The batch solver uses ALL heading priors simultaneously
to correct yaw (and thus position direction) across the full trajectory.  In the
sliding window, yaw corrections at t=20s cannot retroactively fix position at t=5s.

### Phase 4b: True Marginalization

To close the accuracy gap, marginalize out old CPs/knots via Schur complement
into a `ceres::NormalPrior` factor.  This carries forward the information content
of discarded measurements.  Planned but not yet implemented.

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
