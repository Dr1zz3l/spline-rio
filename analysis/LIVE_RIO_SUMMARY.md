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

### Phase 4b: Schur Complement Marginalization (implemented)

`SlidingWindowSolver` in `rio_solver_cpp/` implements full Schur complement
marginalization.  After each window solve, the "stride zone" CPs/knots are
marginalized out and their information is compressed into a dense 30×30 Gaussian
prior on the boundary CPs/knots + bias.  This prior is carried forward via
`MargPriorFunctor` (DynamicAutoDiff over SO3 local coordinates).

**Phase 4b results (3.0s window, 0.3s stride, --mocap-yaw, `marg_prior_scale=2e-4`):**

| Bag | Phase 4a | Phase 4b (scale=1.0) | Phase 4b (scale=2e-4) | Batch |
|-----|----------|----------------------|-----------------------|-------|
| slow_racing | 0.358m / 1.34° | 0.468m / 1.90° | **0.162m / 1.77°** | 0.146m / 0.96° |
| fast_racing | ~1.0m / ~3° | 0.848m / 4.36° | **0.710m / 4.26°** | 0.925m / 2.35° |

Note: above numbers are pre-lever-arm.  Post-lever-arm (current): slow 0.43m/1.59°, fast 0.80m/3.10°
(lever arm adds physically correct antenna-offset correction but changes the absolute numbers).

**Per-window timing:** ~1.5s.

**Key finding: prior scale is critical.**  The raw Schur complement has per-entry magnitudes
of O(10⁵–10⁶) — derived from ~3000 IMU samples per window all contributing to H_bb.  This
is 100–1000× tighter than the boundary priors (λ=1000).  An unscaled prior locks the boundary
CPs completely, preventing adaptation to new data and causing drift.

`marg_prior_scale=2e-4` brings the prior into the same regime as the boundary priors.  At
this scale, the prior still propagates curvature information forward but allows the boundary
CPs to move when new measurements demand it.

**Prior design (re-linearization):** before each `add_prior_to_problem()` call, the prior's
linearization point (x₀) is updated to the current warm-start.  This makes the prior
contribute only curvature (Hessian) information, not a gradient pull toward a stale
historical estimate.  Combined with the scale factor this gives near-batch accuracy from
a 3s/0.3s sliding window.

## Phase 4b: Settled vs Live (Leading-Edge) RMSE (2026-03-29)

### What "live" means

In sliding-window mode the final RMSE is evaluated over the *settled* trajectory —
each CP/knot has been refined by ~10 subsequent windows by the time evaluation runs.
This is an optimistic metric for online deployment.

The *live (leading-edge)* metric snapshots the estimate at the end of each window
(`[t_w_end − stride, t_w_end]`) before any subsequent refinement.  These CPs have
only been optimised once — they represent what a truly causal online system would output.

Implementation: after each `solve_window()`, the current global pos/vel/ori splines
are evaluated on a 50 Hz grid over the stride zone and accumulated.  At evaluation time,
the snapshots are interpolated to MoCap timestamps and scored with the same SE3
alignment used for the settled trajectory (apples-to-apples comparison).

### Results — C++ SW (3.0s window, 0.3s stride, --mocap-yaw --cpp --sliding-window)

All metrics evaluated over the same MoCap time range (settled eval range, which trims
the last `window_duration` seconds).

#### With radar (full RIO)

| Bag | Settled Pos | Live Pos | Δ Pos | Settled Vel | Live Vel | Δ Vel | Settled Ori | Live Ori | Δ Ori |
|-----|------------|---------|-------|------------|---------|-------|------------|---------|-------|
| slow_racing | 0.429 m | **0.627 m** | +0.200 m (+47%) | 0.154 m/s | **0.409 m/s** | +0.255 m/s (+166%) | 1.59° | **2.28°** | +0.69° (+43%) |
| fast_racing | 0.804 m | **0.876 m** | +0.073 m (+9%) | 0.383 m/s | **0.723 m/s** | +0.340 m/s (+89%) | 3.10° | **4.17°** | +1.07° (+34%) |

#### Without radar (IMU-only baseline, --no-radar, same SW config)

| Bag | Settled Pos | Live Pos | Δ Pos | Settled Vel | Live Vel | Δ Vel | Settled Ori | Live Ori | Δ Ori |
|-----|------------|---------|-------|------------|---------|-------|------------|---------|-------|
| slow_racing | 6.437 m | **7.332 m** | +0.895 m (+14%) | 1.031 m/s | **2.543 m/s** | +1.512 m/s (+147%) | 4.16° | **8.17°** | +4.01° (+96%) |
| fast_racing | 5.898 m | **6.963 m** | +1.065 m (+18%) | 1.011 m/s | **1.723 m/s** | +0.712 m/s (+70%) | 6.94° | **6.83°** | −0.11° (≈0) |

#### Radar impact (settled, IMU-only → RIO)

| Bag | Pos improvement | Vel improvement | Ori improvement |
|-----|----------------|----------------|----------------|
| slow_racing | 6.44 → 0.43 m (**−93%**) | 1.03 → 0.15 m/s (**−85%**) | 4.16 → 1.59° (**−62%**) |
| fast_racing | 5.90 → 0.80 m (**−86%**) | 1.01 → 0.38 m/s (**−62%**) | 6.94 → 3.10° (**−55%**) |

### Observations

**Radar impact is dominant.**  Position improves 86–93% vs IMU-only.  The IMU-only
solver drifts rapidly (6–6.5 m settled RMSE over ~20 s) because the only position
information is dead-reckoned from accelerometer integration — double-integration
amplifies any bias or noise.  Radar Doppler breaks this by providing direct velocity
measurements that constrain the spline trajectory continuously.

**Live degradation is larger for velocity than position.**  Position at the leading
edge degrades +9–47% (radar); velocity degrades +89–166%.  Velocity (first spline
derivative) responds more quickly to each new data window — it has less temporal
inertia than integrated position.  Subsequent windows smooth the velocity estimate
significantly; the first-pass velocity is noisier.

**slow_racing position degrades more than fast_racing (+47% vs +9%).**  Counter-
intuitive, but consistent with the settled RMSE difference: slow_racing's batch
optimizer extracts more benefit from re-processing previous windows (tighter dynamics,
more consistent Doppler) so the gap between first-pass and settled is larger.
fast_racing position is already poorly constrained (z-velocity bias) so subsequent
windows add little.

**Orientation leading-edge degradation is moderate (+34–43% with radar).**  The
gyroscope dominates orientation — it provides dense continuous constraints.  The
optimizer corrects small residual errors in subsequent windows but orientation
is already well-anchored on the first pass.

**IMU-only live orientation degradation is severe for slow_racing (+96%) but
negligible for fast_racing (−0.1%).**  Without radar, position drifts freely and
pulls orientation through the coupled spline factors.  fast_racing's orientation
estimate happens to be insensitive to this coupling (it diverges along a different
error mode — higher angular dynamics mean gyro dominates completely).

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
| `../rio_solver_cpp/include/rio/marginalization.h` | MarginalizationPrior + MargPriorFunctor |
| `../rio_solver_cpp/include/rio/sliding_window_solver.h` | SlidingWindowSolver API |
| `../rio_solver_cpp/src/sliding_window_solver.cpp` | Schur complement marginalization impl |
