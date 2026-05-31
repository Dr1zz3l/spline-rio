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
