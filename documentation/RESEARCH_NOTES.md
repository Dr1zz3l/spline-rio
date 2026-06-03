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
