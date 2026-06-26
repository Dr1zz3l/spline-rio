# Research Notes — Design Rationale and Investigative Findings

> **Scope:** Design decisions, solver internals, and investigative dead-ends.
> Benchmark numbers and current config live in `CLAUDE.md`; methodology and
> results tables live in the IEEE paper (`report/`).  This file is the
> *why* — architecture rationale that doesn't fit neatly in either.

---

## 1. C++ Solver Performance Profile

Measured via Ceres `Solver::Summary` timing fields (exposed as `result.time_*_s`
in the pybind11 bridge; printed automatically after each `--cpp` solve).

### Batch solve (full trajectory, ~18–26 s of data)

| Phase | slow_racing | fast_racing | Share |
|-------|-------------|-------------|-------|
| Jacobian eval (autodiff Jet<double,N>) | 4.95 s | 3.85 s | **39 %** |
| Residual eval | 0.46 s | 0.37 s | 4 % |
| Linear solve (sparse Cholesky) | 6.00 s | 4.87 s | **48 %** |
| Other (preprocessing, callbacks) | 1.14 s | 0.88 s | 9 % |
| **Total** | **12.55 s** | **9.96 s** | |

Both autodiff and linear solve are significant — neither dominates overwhelmingly.

### How Ceres autodiff works

Ceres autodiff is **not** symbolic or one-time — it runs at every LM iteration
for every residual block using dual-number (Jet) arithmetic.  Each functor is
called with `T = Jet<double, N>` where N = number of parameter dimensions in
the block (e.g. 40 for the accel factor: 4 ori knots×4 + 6 pos CPs×3 + 6
bias).  Every scalar op carries an N-vector of partial derivatives alongside
the value, so cost ≈ N× plain-double arithmetic.

With 25 k IMU samples × 3 factors × 35 iterations, this accounts for the 39 %.

### Paths to further speedup

**Analytic Jacobians (→ −39 %)**: replace Jet evaluation with closed-form Eigen
expressions.  Path: swap `PythonConfig()` → `CppConfig()` in
`codegen/derive_jacobians_symforce.py` to emit C++ instead of NumPy, then
implement as `ceres::SizedCostFunction` subclasses.  basalt's VIO uses this
approach for its spline factors.

**Variable ordering / fill-in reduction (→ part of −48 %)**: Ceres uses
automatic variable ordering for the Cholesky factorisation.  For a B-spline
problem the natural ordering (knots in time order) is already near-optimal,
but the combined pos+ori+bias block structure may not be.  Try
`options.linear_solver_ordering` with explicit elimination groups.

**Sliding window (→ smaller n)**: a 1–2 s window has ~400 pos CPs + 250 ori
knots ≈ 3 000 parameters vs the current ~27 000.  Cholesky scales
super-linearly, so expect a much larger than 9× speedup on the linear solve alone.

**Combined target**: analytic Jacobians + sliding window → estimated <2 s/window
at real-time rates.

---

## 2. Two-Layer Doppler Unwrapping Design

Radar Doppler is periodic with period `2·v_max`.  The IWR6843 firmware aliases
returns outside `[−v_max, +v_max]`.  We handle this with two independent layers:

**Layer 1 — Pre-unwrapping (`preunwrap_radar_frames()`)**: runs before the solver.
Uses IMU-integrated world-frame velocity (reset by WLS at each frame) to pick
the correct alias offset per point.  Produces clean `RadarVelocity` copies.
Fast racing: 20.9 % of points shifted; slow racing: 1.7 %.

**Layer 2 — In-loop unwrapping**: the solver recomputes the alias offset at each
iteration based on the current spline prediction: `k = round(-r / (2·v_max))`.

**Finding**: Layer 1 is protective infrastructure — it prevents cascade failures
on cold start where the first frame is completely aliased and the solver has no
good init velocity.  However, both layers give the same final accuracy: the
in-loop solver was already handling all aliased points correctly from the good
initialisation (confirmed on fast_racing with 20.9 % pre-unwrapping: no accuracy
change vs baseline).

---

## 3. Why We Stayed with Per-Sample IMU, Not Preintegration

The natural "optimization" for RIO is to replace ~1 000 Hz per-sample IMU
factors with Forster TRO-2017 preintegrated factors at radar rate (~11 Hz).
Investigation showed this degrades accuracy severely.

**Root cause**: at `dt_ori = 0.008 s`, there are ~8 raw gyro samples per
orientation knot (8:1 overconstrained).  Replacing raw gyro with one
preintegrated factor per knot drops this to 1:1 (critically constrained), and
orientation RMSE degrades from 0.96° to 6.0° on slow_racing.

**Why preint can't fix the backflips sliding-window failure**:
two conflicting requirements make it fundamentally unable to help:
1. Model bandwidth: backflips require `dt_ori ≤ 0.001 s` (empirically ≈ 0.0008 s)
2. Preintegration requires `dt_preint ≥ 1/IMU_hz = 0.001 s` (need ≥ 2 samples per interval)
At 100 Hz (`dt_ori = 0.01 s`): model bandwidth insufficient (same bad results).
At 1 250 Hz (`dt_ori = 0.0008 s`): `dt_preint = 0.0008 < IMU period = 0.001` → degenerate factors.

**Preint implementation note**: when testing `--preintegrate` on the Python solver
at 200 Hz IMU, the key is that preintegration *replaces* accel (set `lambda_accel=0`)
but keeps per-sample gyro.  Adding preint on top of existing accel residuals only
grows the Jacobian with no benefit.  The gyro must remain at full rate because the
cumulative SO(3) B-spline needs dense pinning of intermediate knots; preintegration
only constrains interval endpoints.

---

## 4. z-Velocity Systematic Bias — Root Cause

Initial radar residuals before optimisation:
- slow_racing: mean = +0.06 m/s, std = 0.79 m/s (nearly unbiased)
- fast_racing: mean = −0.44 m/s, std = 2.74 m/s (systematic bias + 3.5× noise)

The −0.44 m/s mean matches the known z-velocity systematic bias of −0.5 to −0.65 m/s
caused by **limited elevation diversity** (2 TX antennas on the IWR6843).  At
faster flight speeds the z-velocity component is larger and changes more rapidly,
amplifying this bias.  The optimizer absorbs it by adjusting roll/pitch:
- fast_racing roll RMSE: ~7.5° vs slow_racing: ~3.4°

The higher per-point noise (std 2.74 vs 0.79 m/s) reflects degraded radar return
quality at high speeds (more aliased returns in ambiguous bins, lower SNR, more
multi-path).  Fixing this requires better sensor-level modelling (explicit
z-bias parameter, per-elevation attenuation model, or better HW), not
unwrapping improvements.

---

## 5. Phase 4a→4b Sliding Window: Why Marginalization + Re-Centered Prior

**Phase 4a** (simple sliding window, no marginalization): each window estimates
position from local Doppler + heading priors.  Without marginalization, the
batch solver has a large advantage: it uses ALL heading priors simultaneously to
correct yaw (and thus position direction) across the full trajectory.  In SW, a
yaw correction at t=20 s cannot retroactively fix position at t=5 s.
Result: slow_racing settled pos 0.358 m vs batch 0.146 m (pre-lever-arm).

**Phase 4b** (Schur complement marginalization): after each window solve,
the "stride zone" CPs/knots are marginalized out via Schur complement and their
information is compressed into a dense 30×30 Gaussian prior on the boundary
CPs/knots + bias.  This prior is carried forward via `MargPriorFunctor`.

**Critical: re-centered prior.**  Before adding the prior to each window's
problem, the linearization point `x₀` is updated to the current warm-start.
This makes the prior contribute only *curvature* (Hessian shape), not a gradient
pull toward a stale historical estimate.  The residual is
`r = Lᵀ(x − x₀)` in local coordinates.  Without re-centering, the prior
tries to pull the current window back toward old states and the whole thing
diverges.

**Why prior scale is so sensitive**: the raw Schur complement has eigenvalues
spanning [10⁴, 5.5×10¹⁴] (condition number ≈ 5.5×10¹⁰) because 1 kHz gyro
with λ_g=4 contributes ~62 500 information per orientation knot vs ~10 000 for
position.  An unscaled prior locks boundary CPs completely.  The scalar
`marg_prior_scale` is effectively a mission-type parameter (not per-flight
tuning): gentle dynamics → windows are self-sufficient → use near-zero prior
(≈10⁻⁷); aggressive dynamics → windows need inter-window continuity → use
meaningful prior (≈2×10⁻⁴).  See CLAUDE.md and report §V-C for the full sweep.

---

## 6. Why Not a Python Sliding Window

The cumulative SO(3) B-spline has a left-triangular Jacobian structure (each
R(t) depends on all prior Ω knots via `R_base`).  The Python solver builds the
full Jacobian matrix for each window, and the sparse Cholesky cost dominates for
the relevant parameter counts.  More importantly, Schur complement
marginalization requires numerically reliable Hessian factorization — this is
infrastructure that Ceres provides natively.  The Phase 4b implementation went
directly to C++ (`rio_solver_cpp/src/sliding_window_solver.cpp`) rather than
prototyping in Python first.

---

## 7. Preconditioning & Damping — Early Experiments (2026-02-19)

Context: Phase 3 Python solver on backflips, DT_ORI=0.1 s, 288 state variables.

**Jacobi preconditioning** (`M = diag(1/√diag(H))`): identical results to
unscaled — every iteration, cost value, update norm, and damping value matched
exactly.  The Hessian diagonal is already at similar scales for this problem
size; Jacobi scaling doesn't change the search direction.

**Aggressive LM damping** (×0.1 / ×10 vs conservative ×0.5 / ×5): marginal
improvement (<2 % in RMSE).  The aggressive strategy dropped λ to 10⁻¹⁵ by
iteration 10 (pure Gauss-Newton), then stalled for 10 consecutive rejected steps.
Both strategies converged to approximately the same local minimum.

**Root cause (confirmed)**: the failure was not numerical conditioning or damping.
The accelerometer cost dominated (43 000 of 45 000 total cost = 96 %) because
during backflips (ω ≈ 10 rad/s, centripetal accel ≈ 25 m/s²) the spline's
second-derivative acceleration estimate is poor, pulling orientation away from
the MoCap-initialized SLERP.  These experiments motivated the switch to Ceres
C++ with full-rate IMU, tight bias priors, and min-α regularization.

---

## 8. BandedSchurSolver — O(n) Attempt and Abandonment (Sessions 1–3, 2026-06)

### Motivation

The Ceres SPARSE_NORMAL_CHOLESKY / SuiteSparse solver factorizes the full n×n
normal equations H = J^T J + D²I each LM iteration (n ≈ 2890 for a 3 s sliding
window).  CHOLMOD's fill-reducing Cholesky is fast but scales super-linearly.
For realtime (≤ 0.3 s per window), a solver exploiting problem structure was needed.

The structural observation: H_kk (knot subblock, 2883 DOF) is sparse with **bandwidth
bw ≈ 15** because quintic B-spline basis functions are locally supported — each knot
only couples to its 5 nearest neighbours (3 DOF each).  Banded Cholesky on H_kk is
O(n_k × bw²) ≈ O(n_k × 225), vs CHOLMOD's O(n_k × fill²) where fill >> bw.  The
remaining global parameters (biases + pitch_delta, n_c = 7) are dense but tiny.
The arrowhead Schur complement eliminates H_kk first, leaving a 7×7 dense system.

Projected speedup over SuiteSparse: 10–100× depending on AMD fill factor.

### Session 1–2: Banded LL^T — Numerical Failure

Implemented AMD-reordered banded LL^T with Jacobi diagonal preconditioning.

**Fatal finding**: cond(H_kk) ≈ 5.5×10^10.  No-pivot banded LL^T (which the banded
structure requires — any off-band pivot swap destroys the O(n) property) accumulates
catastrophic rounding errors through kd ≈ 64 Schur complement steps.  Off-diagonal
entries drive intermediate pivots negative regardless of Jacobi scaling.

Pivoting would fix the numerics but breaks the band structure, collapsing complexity
back to O(n²) or worse.  The fundamental contradiction: the problem is too
ill-conditioned for no-pivot Cholesky, but pivoting negates the O(n) benefit.

### Session 3: Eigen::SimplicialLDLT — Same Class as Baseline, Gate Fails

Replaced banded LL^T with Eigen::SimplicialLDLT on H_kk (AMD-ordered supernodal
LDL^T) + dense 7×7 Schur complement.  This is algebraically correct but offers no
speedup: SimplicialLDLT is the same algorithmic class as CHOLMOD.  The Schur
elimination of 7 columns does not reduce the n_k × n_k factorization cost, and the
fill pattern of the 2883×2883 subproblem is essentially the same as the full 2890×2890.
Net effect: two factorizations (2883 + 7) + block extraction overhead vs one CHOLMOD
(2890).  Likely slower than baseline.

**Gate outcome**: FAIL (never passed).
- Windows 0–1: cost_rel ≈ 5×10^{-4} (threshold 1×10^{-4}).
- Window 2+: catastrophic divergence (cost ~8×10^5 vs baseline ~1.5×10^3).

**Root cause of gate failure — floating-point J^T J indefiniteness**:
J^T J is mathematically PSD, but the floating-point computed version has small
negative eigenvalues (~7×10^{-2} in LDLT pivots) when the true minimum eigenvalue
is near zero (~3×10^{-11} estimated from step amplification).  With LM damping
D² ≈ (8.75×10^{-7})² = 7.7×10^{-13} at late convergence, D² << |neg. eigenvalue|,
so the factorization is numerically indefinite.

SuiteSparse (CHOLMOD) handles this because it detects near-zero pivots and applies
internal regularisation (`dbound`), effectively clamping them to a small positive
value.  Eigen::SimplicialLDLT has no such mechanism — it returns a factorization
with negative D entries, and the resulting step is uphill (verified: g^T Δx = −3.08 < 0).

Attempted fixes: (a) return LINEAR_SOLVER_FAILURE on negative D, forcing LM to
increase mu — burns through trust-region budget by window 2; (b) clamp negative D
entries to D_max × 1×10^{-8} — produces model_cost_change predictions of 10^{21}
at window 2, catastrophic divergence.  Neither fix produces stable behaviour across
windows because the clamped solve approximates a different (nearby) matrix, breaking
the quadratic model accuracy assumption.

**Key technical artefacts found along the way (preserved for reference)**:
- `SimplicialCholeskyBase::_solve_impl` aliasing bug: `dest = m_Pinv * dest` with
  Eigen's `AliasFreeProduct` tag suppresses the implicit eval, causing in-place
  permutation corruption.  Workaround: explicit index loops for P and P^T steps.
  See header comment in `banded_schur_solver.cc`.
- Permutation convention for Eigen SimplicialLDLT: P.indices()(i) = j means
  (P·v)(i) = v(j).  Factorisation is P^T A P = L D L^T.  Correct solve:
  P^T scatter: tmp[P_idx[i]] = b[i]; L^{-1}; D^{-1}; L^{-T}; P gather: res[i] = tmp[P_idx[i]].
  The opposite order (P gather first, then P^T scatter) is wrong and produces uphill steps.

### What a Real O(n) Path Requires

A truly O(n) sliding-window solver needs **incremental sparse factorisation** — the
iSAM2 / Bayes tree approach (Kaess et al. 2012).  The idea: maintain a symbolic
Cholesky factorisation of the factor graph and update only the affected cliques when
new measurements arrive.  Each incremental update is O(k²) where k is the number of
affected variables, not O(n).  This is a fundamentally different architecture —
not a drop-in Ceres LinearSolver — and would require replacing the Ceres minimiser
with a custom incremental optimiser.  Out of scope for this project.

---

## 9. Sliding Window Timing Analysis (2026-06-03)

Per-window Ceres timing measured by adding `num_iterations = summary.num_successful_steps`
and exposing `time_{jacobian,residual,linear_solver}_eval_s` to the Python print loop.
Run: `--mocap-yaw --cpp --sliding-window --no-plot` on both racing bags.

### Results

| Component | slow_racing | fast_racing |
|---|---|---|
| Jacobian eval | ~0.7s | ~0.5s |
| Linear solve | ~0.7s | ~0.6s |
| Residual eval | ~0.07s | ~0.06s |
| Other (compute_prior) | ~0.7s | ~0.6s |
| **Total** | **~2.1s** | **~1.7s** |
| **LM iterations** | **~28–30** | **~28–30** |

Real-time target: ≤ 0.3s (stride). Current: 5–7× too slow.

### Root causes

**1. LM iteration count (28–30) is the dominant issue.**
Iter count is constant across cold-start (window 1) and warm-start (window 30), and across
both marg_prior_scale values (1e-7 and 2e-4). The stopping criterion is `function_tolerance`,
not `max_iterations=40`. The ill-conditioned Hessian (cond ≈ 5.5×10¹⁰) forces LM to take
many small steps before the per-step cost improvement drops below threshold. With gyro-dominant
information (λ×∂ω/∂Ω ≈ 62,500) and sparse radar position information (~10,000), the optimizer
converges quickly in orientation but slowly in position — requiring ~28 iterations overall.

**2. compute_prior() accounts for the "other" ≈ 0.65s (27% of wall time).**
After each Ceres solve, `compute_prior()` calls `problem.Evaluate()` manually to extract the
full Jacobian at the solution and compute the boundary block H_bb for the Schur complement.
This is one additional full Jacobian evaluation — equivalent to one more LM iteration — but
is NOT counted in Ceres's internal timing fields. It appears as wall-clock overhead in "other".

**3. Already-captured wins: analytic IMU factors.**
`GyroAnalyticFactor` and `AccelAnalyticFactor` are active in both solvers. Without them the
Jacobian eval time would be ~3× higher (Jet arithmetic on 6000 high-frequency IMU samples).

### What remains available

- **Fewer LM iterations**: `--set max_iterations=N` for N ∈ {8, 12} — quick experiment.
  If accuracy holds at 8 iterations: ~3× speedup on Ceres part.
- **compute_prior() less frequently**: compute every 3–5 windows → ~1.3× on total.
- **Window size reduction** (2.0s vs 3.0s): ~1.33× on all per-iteration costs.
- **iSAM2/GTSAM**: O(k²) incremental updates at sensor rate, same MAP accuracy.
  Requires expressing B-spline factors in GTSAM's custom factor API (~3–6 weeks). See §8.

Combining iter→8 + window=2s + compute_prior fix: estimated ~0.5s per window.
True real-time (0.3s) likely requires iSAM2 or a fundamentally different sensor schedule.

---

## §9b Parameter Sweep: max_iterations and window_duration (2026-06-03)

Script: `analysis/sweep_sw_params.py`. Results: `analysis/eval_results/sweep_iter_20260603_175706.json`,
`sweep_window_20260603_185017.json`. Both bags, `--mocap-yaw --cpp --sliding-window`.
Accuracy metric: per-axis orientation RMSE (position RMSE not captured due to regex mismatch
against sliding-window output format; see `parse_rmse_from_output` in `eval_bags.py`).

### Iteration sweep (window_duration=3.0s fixed)

| max_iter | slow yaw RMSE | slow median_dt | fast yaw RMSE | fast median_dt |
|---|---|---|---|---|
| 8  | 4.23° | 0.83s | 12.18° | 0.68s |
| 12 | 4.50° | 1.07s | 9.59°  | 0.83s |
| 16 | 4.66° | 1.28s | 10.69° | 0.97s |
| 20 | 4.44° | 1.46s | 7.31°  | 1.08s |
| 28 | **0.93°** | 1.93s | **3.56°** | 1.38s |
| 40 | **0.93°** | 1.90s | **3.55°** | 1.38s |

**Finding**: There is a sharp cliff at max_iter=28, NOT a gradual elbow. Yaw RMSE drops from
4–12° at iter≤20 to ~0.9–3.6° at iter=28. iter=28 and iter=40 give IDENTICAL results (both
timing and RMSE) — confirming that the solver converges naturally at ~28 steps via
`function_tolerance`. No time savings available from reducing max_iterations below ~28.

### Window duration sweep (max_iter=40 fixed)

| window (s) | slow yaw RMSE | slow median_dt | fast yaw RMSE | fast median_dt |
|---|---|---|---|---|
| 1.0 | 112.3° | 0.88s | 8.87°  | 0.52s |
| 1.5 | **1.75°** | 1.21s | 9.68°  | 0.73s |
| 2.0 | 1.34° | 1.41s | 15.17° | 1.01s |
| 2.5 | 1.06° | 1.90s | 9.96°  | 1.23s |
| 3.0 | **0.93°** | 2.04s | **3.56°** | 1.48s |

**Finding (slow_racing)**: Soft elbow at 1.5s — yaw goes from 0.93° (3.0s) to 1.75° at 1.5s
(+88%), with 1.69× speedup in median_dt. Acceptable accuracy–speed trade-off IF this bag alone
is considered. window=1.0s is broken (112° yaw — marginalization instability).

**Finding (fast_racing)**: Requires 3.0s window. All windows < 3.0s produce 9–15° yaw RMSE
(vs 3.56° at 3.0s). This is due to the stiff marg_prior_scale=2e-4 combined with the short
window not providing enough radar observations to constrain the marginalization prior.
The instability at window=2.0s (15.17°) followed by partial recovery at 2.5s (9.96°) suggests
non-monotone behavior driven by which radar frames land inside the window.

**Overall conclusion**: The 4–5× gap to the 0.3s real-time target cannot be closed by parameter
tuning alone. The iteration count is a hard minimum; window size only helps for slow_racing and
at the cost of yaw accuracy. Structural changes (analytic Jacobians, iSAM2, compute_prior
batching) are required.

---

## §10 Analytic Radar Jacobians — Implementation and Timing (2026-06-03)

### Motivation

After §9 established that Jacobian eval is ~35% of window solve time, replacing the remaining
`DynamicAutoDiffCostFunction` (Jet arithmetic over ~41 parameters) for radar with analytic
Jacobians was the natural next step. IMU factors were already analytic (§9).

### Approach

SymForce's `CppConfig` generates the sensor-model Jacobians (∂r/∂v_world, ∂r/∂R_bw, ∂r/∂ω)
as a dependency-free Eigen template (`RadarSensorJacWithJacobians012`, 146 CSE-optimized ops).
Spline Jacobians come from the existing `spline_jacobians.h` (`rotation_with_jacobian_manual`,
`body_velocity_with_jacobian_manual`). The convention chain is:

1. SymForce emits RIGHT-perturbation Jacobian for `Rot3`: `∂r/∂ε_R` where `R_new = R·Exp(ε_R)`
2. Convert to LEFT: `J_left = J_right · R^T`
3. Chain to basalt LEFT-perturbation spline Jacobians: `J_knot_i = J_left · J_R.d_val_d_knot[i]`
4. Convert to Ceres ambient: `jacobians[i] = J_knot_i · tangent_to_ambient(params[i])`
5. ω Jacobian needs no convention conversion (plain vector output): `J_knot_i = J_ω · J_omega.d_val_d_knot[i]`

Two factor classes: `RadarAnalyticFactor` (fixed extrinsics, pre-computes `u_body = R_rb·u_sensor`)
and `RadarAnalyticWithPitchFactor` (pitch_delta parameter, recomputes `u_body` per LM step;
pitch Jacobian: `∂r/∂pd = v_ant · (R_rb · dRy/dpd · u_sensor)`).

### Results (2026-06-03, `--mocap-yaw --cpp --sliding-window`)

Accuracy (batch): slow_racing 0.177m/1.02°, fast_racing 0.763m/2.64° — both within 5% of
AutoDiff baseline (confirmed correct Jacobians).

Timing comparison (analytic vs AutoDiff radar):

| Metric | slow_racing (before→after) | fast_racing (before→after) |
|---|---|---|
| Jacobian eval (avg) | ~0.7s → ~0.81s | ~0.5s → ~0.39s |
| Linear solve (avg) | ~0.7s → ~0.92s | ~0.6s → ~0.70s |
| Other/compute_prior | ~0.7s → ~0.79s | ~0.6s → ~0.62s |
| **Total** | **~2.1s → ~2.6s** | **~1.7s → ~1.77s** |

fast_racing shows ~30% Jacobian speedup (0.39s vs 0.5s); total unchanged since jac is only ~29%
of total. slow_racing may reflect higher radar point density per window or system load variation.

### Key finding

With analytic factors for ALL sensor residuals (IMU + radar), the Jacobian component is no
longer the dominant bottleneck. The real bottleneck is `compute_prior` (called once per window
OUTSIDE the LM loop, equivalent to +1 full Jacobian evaluation, classified as "other" ≈ 0.62s)
plus the linear solve itself (~0.70s). Both are unaffected by analytic Jacobians.

**Remaining 5-7× real-time gap requires structural changes**: reduce `compute_prior` frequency
(recompute every N>1 windows), or iSAM2 for O(k²) incremental updates.

## §11 iSAM2 Backend Speedup Investigation (2026-06-26)

The iSAM2 backend (`--isam`) already beat the Ceres SW 2-3x (slow ~177ms/update vs SW
350-700ms). Question: can we get a SUBSTANTIAL further speedup (2-10x)? Investigated on
slow_racing (the slowest case). **Answer: no substantial win; ~1.3-1.6x available via
adaptive iterations, the rest is structurally blocked (1 kHz factors + cond(H)~5.5e10),
exactly as §9/§10 concluded for the Ceres SW.**

### Cost structure (slow, extra_iters=3, QR)
`extra_iters` is the dominant cost: it does 1 main + 3 extra empty ISAM2 updates/stride
(re-linearize + re-solve to converge roll/pitch from the weak P1-P3 init).
| config | ms/update | pos RMSE | ori RMSE |
|---|---|---|---|
| extra3 QR (baseline) | 177 | 0.165 | 1.39 |
| extra2 | 155 | 0.210 | 1.66 |
| extra1 | 113 | 0.227 | 2.65 |
| extra0 | 64 | **4.33 / 75deg DIVERGES** | |
So one update floor is ~64ms (full linearize of ~3000 IMU factors + 1 QR solve); each
extra iter ~38ms; extra_iters<1 diverges (roll/pitch never recovers).

### Cholesky vs QR -> QR wins (counter-intuitive)
extra3 Cholesky = 247ms (SLOWER than QR 177ms) AND less accurate (0.200 vs 0.165). At
cond(H)~5.5e10 Cholesky does more work / recovers poorly; QR (cond(J)=sqrt(cond H)~2.3e5)
is both faster and more accurate. Confirms the §1/Caveat-1 conditioning story.

### Approach 1: adaptive extra_iters (step-norm early-stop) -- WORKS, modest
`extra_iters_dnorm`: stop the extra updates when max-abs ISAM2 getDelta() (tangent rad/m)
< dnorm. The earlier COST-based rtol failed (orientation hidden under dominant gyro/radar
residuals); the STEP-norm catches the slow-mode (roll/pitch) convergence. The delta floor
is ~0.2 (NOT ~0.01): every stride's entering knots carry real roll/pitch error, so most
strides still need the iters -- the early-stop only reclaims compute on the easy ones.
| dnorm | slow ms (pos/ori) | fast ms (pos/ori) |
|---|---|---|
| off (extra3) | 177 (0.165/1.39) | 107 (0.517/2.26) |
| 0.2 | **132 -25% (0.173/1.41)** | **66 -38% (0.524/2.33)** |
Beats fixed extra2 on BOTH speed and accuracy (full iters on hard strides, stop on easy).
A real ~1.3x (slow) / ~1.6x (fast) at ~1-5% accuracy cost. Kept (off by default,
extra_iters_dnorm=0). Recommended speed/accuracy knob: extra_iters=3 + dnorm~0.2.

### Approach 2: complementary-filter init (Mahony roll/pitch from accel) -- NEGATIVE
`--comp-filter GAIN` adds a Mahony correction to `integrate_gyro_orientation` (nudge
predicted-gravity toward the accelerometer during near-1g samples). Hypothesis: a better
roll/pitch init -> fewer extra_iters. **Fails on our dynamic data:** (1) the near-1g gate
rarely fires during racing (high linear accel != gravity), so the init barely changes
(roll drift 42->33deg, pitch unchanged); (2) more fundamentally, the extra_iters need is
the ILL-CONDITIONED per-stride convergence (soft modes), not init quality -- the solver
re-derives roll/pitch from its own accel factors regardless of init. comp+extra1
(0.27/2.9) is no better than plain extra1 (0.227/2.65). Code kept (off by default,
correct Mahony filter -- could help a low-dynamics platform; useless here).

### Verdict
No substantial (2-10x) iSAM2 speedup without changing the formulation (preintegration
costs 0.96->6.0deg ori, §3; IMU downsample hurts the spline bandwidth). The cost is the
1 kHz factor density + intrinsic cond(H)~5.5e10. ~1.3-1.6x is available now via adaptive
extra_iters (Approach 1) at small accuracy cost. The honest real-time story is unchanged:
iSAM2 hits the 0.3s stride with modest margin (fast) to none (slow/backflips spikes).
