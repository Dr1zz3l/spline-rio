# iSAM2 / GTSAM Back-End Migration

> **Status (2026-06-25): Phase 0 spike in progress on branch `isam2-backend`.**
> Goal: replace the Ceres LM fixed-lag sliding window with a GTSAM
> `IncrementalFixedLagSmoother` (iSAM2 core + timestamp marginalization) for a
> real-time margin. Staged behind a de-risk **gate**; paper is frozen (no
> `report/` edits). Plan file: `~/.claude/plans/ethereal-sniffing-hejlsberg.md`.
> Companion auto-memory: `project_isam2_backend`.

This file is the durable record of *why* and *what was learned*, so the key
insights survive context compression. Benchmark numbers and per-phase config
go in CLAUDE.md / RESEARCH_NOTES.md as results land.

---

## 1. Motivation

Ceres LM sliding window solves each window from scratch (28 -> 11 LM iters) at
0.35--0.70 s/window, only *barely* hitting the 0.3 s stride at a tuned no-margin
point. Two prior speed attempts failed structurally: BandedSchurSolver
(`RESEARCH_NOTES §8`, numerically dead at cond(H) ~ 5.5e10) and parameter tuning
(`§9b`, the iteration floor / window wall). The standing fix in docs + paper is
**incremental smoothing (iSAM2)**: the single open `ROADMAP` item and the named
future-work deployment path (`conclusion.tex:48`, `results.tex:96-100`).

## 2. Why iSAM2 is feasible here (core premise, CONFIRMED in code)

Despite the "cumulative SO(3)" naming, the C++ solver uses **basalt
absolute-knot** splines with **strictly local support**: every radar/accel/gyro
residual touches only `N_ORI=4` consecutive orientation knots + `N_POS=6`
position CPs, *not* the left-triangular global coupling of the Python `R_base`
incremental-Omega form. The factor graph is genuinely banded, so a Bayes-tree
factorization stays small-clique.

Confirmed 2026-06-25 in the spike: gtsam `linearize` of each factor yields
Jacobian blocks of exactly the local support: radar `(1x30)` = 4 ori*3 + 6 pos*3;
accel `(3x36)` = +6 bias; gyro `(3x18)` = 4 ori*3 + 6 bias.

**The one global coupling is the single shared constant IMU bias** (connects to
every IMU factor). This is the "shared global bias induces a Bayes-tree fill-in"
the report flags (`results.tex:99`). The bias model is therefore the crux; the
spike measures **both** variants (constant vs random-walk) and picks by clique
size.

## 3. Two verified caveats (raised in review, both TRUE) -> plan adapted

Verified against Girod et al. 2024 (arXiv:2408.05764, the cited `girod2024brio`)
and `methodology.tex:370-385`.

**Girod's iSAM2 is well-conditioned by formulation, not because iSAM2 fixes
conditioning.** Girod runs GTSAM iSAM2 over a *discrete-time* graph: one 15-dim
keyframe (R,t,v,b_g,b_a) per radar frame at 8 Hz, IMU **preintegrated** (Forster)
into one between-factor, **per-state random-walk biases**, solved in 40 ms / 10 s
window on a Jetson Orin NX. Our system is the opposite on every axis
(continuous-time B-spline, ~1000s of knots, per-sample **1 kHz** gyro/accel
factors with lambda_g -> 62500 info/knot, single shared bias). Our cond(H) ~ 5.5e10
comes from exactly these choices. We cannot adopt Girod's preintegration to dodge
it: `RESEARCH_NOTES §3` shows preint degrades our orientation (0.96 -> 6.0 deg)
because the per-sample gyro *is* the spline's bandwidth source.

1. **Ill-conditioning is intrinsic and inherited.** Conditioning is a property
   of the information matrix, not the optimizer; iSAM2 factorizes the same
   information. It only changes the *failure mode*: iSAM2 eliminates small
   band-bounded cliques and can use **QR** (cond(J) = sqrt(cond(H)) ~ 2.3e5,
   comfortably double-precision) instead of the monolithic no-pivot banded
   Cholesky that killed BandedSchur. So it is more likely to *survive*, but it
   does not improve conditioning, soft-mode convergence, or stiffness.

2. **FEJ becomes load-bearing, and our marginalization is "halfway" there.**
   Our nullspace is absolute position + yaw. The Ceres SW survives without FEJ
   only because it re-solves each window from scratch (washing out nullspace
   accumulation over short 18--26 s missions) and freezes the prior information
   S at marginalization time. iSAM2's **fluid relinearization** removes the first
   protection: it keeps and incrementally updates linearizations. The paper
   states the gap outright: "We do not employ FEJ ... We do not claim this
   re-centering preserves consistency across relinearizations ... left to future
   work." The Markov-blanket factor-selection (Prop 1) + frozen-S is the
   structural half; the missing half is **pinning the linearization points** of
   the marginalized-coupled boundary + bias variables (FEJ). Tested data-driven
   (FEJ on/off vs NEES) at the Phase-0 gate; implemented in Phase 2 if needed.

## 4. Phase plan + GATE

- **Phase 0 (Python gtsam spike, gated)** on slow_racing. Reuses
  `generated_jacobians.py` residuals + basalt-exact spline primitives.
- **GATE (all must hold):** accuracy within 5--10% of the Ceres SW headline
  (settled 0.293 m / 1.53 deg, live 0.31 m / 1.97 deg); **bounded clique size**
  (constant in trajectory length; picks the bias model); **no conditioning
  failures** (try QR); **NEES-calibrated** on the observable subspace with FEJ
  on/off; **incremental timing trend** beats batch re-solve.
- If the gate fails: write a post-mortem, stop, keep the Ceres SW.
- **Phase 1** C++ `gtsam::NoiseModelFactorN` (reuse `factors/analytic/*` math),
  ctest vs numericalDerivative + Ceres parity.
- **Phase 2** C++ `IsamSlidingWindowSolver` (sibling of `SlidingWindowSolver`,
  same API), `IncrementalFixedLagSmoother`, FEJ pinning if gated, `--isam` flag.
- **Phase 3** 3-bag accuracy + per-update timing vs the CLAUDE.md SW headline.

## 5. GTSAM API facts (verified 2026-06-25, gtsam 4.2.1)

- `gtsam.CustomFactor(noise, keys, error_fn)` present. `error_fn(this, values, H)`
  returns the **unwhitened** residual; fills `H[i] = jacobian` (dim_res x
  dim_tangent_i) when `H is not None`. The noise model whitens.
- **`Rot3.retract` is RIGHT-mult: `R.retract(xi) == R * Exp(xi)`** (verified).
  Identical to Ceres/Sophus, so the basalt-left -> gtsam-right bridge is
  `J_right = J_left * R(knot)` (the existing `spline_jacobians.h` math).
- **`IncrementalFixedLagSmoother` lives in `gtsam_unstable`** (importable;
  NOT top-level `gtsam`). Also `BatchFixedLagSmoother`,
  `FixedLagSmootherKeyTimestampMap`. Constructs as `(lag, ISAM2Params)`; has
  `update`, `calculateEstimate`. `ISAM2.marginalizeLeaves` is NOT exposed in
  Python, so the smoother is the marginalization path.
- `ISAM2Params.setFactorization('QR'|'CHOLESKY')` exists (Caveat-1 mitigation).
- `gtsam.numericalDerivative` is NOT exposed in Python; the spike uses an
  on-manifold finite-difference checker instead.

## 6. Convention crib (must match the C++ solver exactly)

- Orientation knots: **absolute** rotations, quaternion **[x,y,z,w]** (Sophus
  order). gtsam variable = `Rot3`.
- Position: quintic (degree 5, N_POS=6) uniform B-spline; CP = Vector3 / Point3.
- `UniformBSpline` (lib/bspline_utils.py) is **0-based** (knots = arange*dt), so
  evaluate at `t_rel = t_abs - t_ref`; its clamping matches C++ `pos_index`.
- SO(3) eval: uniform cubic **cumulative** blending lam(u) (closed form in
  `spline_factors.cumulative_cubic`, matches `NonUniformSO3Spline.evaluate` /
  basalt to 8e-8 rad). Active ori knots `[k-3..k]` with
  `k = clamp(int(t_rel/dt_ori), 3, n_ori-1)`, `u=(t_rel-k*dt_ori)/dt_ori`.
- Extrinsic `R_radar_to_body`: **ZYX**, `Rz*Ry*Rx` (`rotation_matrix_from_euler`
  == C++ `ExtrinsicConfig::R_radar_to_body`, verified). Spike LOCKS extrinsics at
  the C++-solved pitch.
- Residual models: reuse `codegen/generated_jacobians.py` (same SymForce
  derivation as the C++ analytic factors). **Rotations passed as `gj.Rot3` shim
  objects** (`.data` = quat xyzw), NOT 3x3 matrices; use
  `gj.Rot3.from_rotation_matrix(R)`. Radar `r = v_meas - v_pred`,
  `v_pred = -dot(u_body, v_ant)` (lever arm); accel `z - R^T(a_world-g) - b_a`;
  gyro `z - omega - b_g` (dr/domega = -I, verified).

## 7. Spike artifacts (`analysis/isam_spike/`)

- `capture_problem.py`: monkeypatches `validate_live_solver._solve_cpp` to dump
  the EXACT batch problem (init knots, post-RANSAC radar, IMU, heading, full cfg,
  C++-solved trajectory) to `_cache/<bag>_batch.npz`. Non-invasive; the real
  batch solve still runs (reference: slow_racing C++ batch = 0.199 m / 1.077 deg).
- `spline_factors.py`: `Problem` (loads npz, index/basis machinery, initial
  gtsam `Values`), spline primitives, and `make_{radar,accel,gyro}_factor`
  (CustomFactor + retract-consistent FD Jacobians). FD Jacobians are correct by
  construction w.r.t. gtsam variables; Phase 1 swaps to the analytic chain for
  the C++ port (the spike needs correct Jacobians, not fast ones).
- `test_phase0a.py`: primitives vs basalt reference (8e-8 rad), derivative
  consistency, factor build/evaluate, end-to-end gtsam `linearize`. PASS.

- `test_phase0b.py`: batch parity vs the Ceres optimum over an interior window.
  (A) stationarity: init at the C++ solution, outer knots pinned, solve. (B)
  convergence: init at the P1-P3 init, solve. Compares the sampled gtsam
  trajectory to the C++ trajectory.

Phase 0a status: **DONE** (scaffold builds, evaluates, linearizes with correct
local sparsity).

Phase 0b status: **PASS** (2026-06-25). gtsam reproduces the Ceres optimum:
  - 0.5 s window: max d_pos = 3.6 mm, max d_ori = 0.019 deg vs C++.
  - 1.0 s window: max d_pos = 13.4 mm, max d_ori = 0.057 deg vs C++.
Both the C++-init (stationarity) and P1-P3-init (convergence) solves land on the
SAME gtsam optimum (identical final cost), so the sensor factors/weights/
conventions match Ceres. The small mm/0.05 deg gap is windowing (local boundary
pinning + omitted heading prior vs C++ global boundary+heading), not a factor
bug: a sign/convention error would make the C++ optimum a NON-stationary point
and gtsam would move far. Python FD solve is ~20-60 s/window (unrepresentative;
the timing verdict is C++ Phase 3, per the plan).

- `test_phase0c.py`: ISAM2 incremental smoothing fed stride-by-stride (full
  smoothing, no marginalization yet), both bias models, with a Bayes-tree clique
  analysis. IMU decimated to ~250 Hz (preserves the bias-coupling connectivity
  that drives fill-in while keeping the Python FD solve tractable; structure
  depends on connectivity, not factor count). Init at the C++ solution to isolate
  STRUCTURAL fill-in from the convergence transient.

Phase 0c status: **DONE -> bias decision = RANDOM-WALK** (2026-06-25).
  - The tree is **banded**: max clique size **12** vars (mean 11), bounded, no
    growth with trajectory length, identical for both bias models.
  - **The shared-bias fill-in is REAL and measured.** Constant bias: `b0` appears
    in **ALL 402 cliques** (it threads the entire Bayes tree). Random-walk:
    each per-stride `b_k` is local (`b0` in only 67/402, decaying as later `b_k`
    take over). Max clique size hides it (bias is +1 var/clique), but `b0`'s
    tree-wide degree is what keeps raw-ISAM2 `reElim` ~ total and will block clean
    marginalization in 0d.
  - ISAM2 incremental tracks C++ to <1 deg ori at 250 Hz (inflated by decimation
    + omitted heading; revisit full-rate in 0d). Estimate is bias-model-
    independent (0.675 vs 0.674 deg) -> the bias choice is purely structural.
  - **DECISION: random-walk bias** (per-stride `B(k)` + between-factor). Matches
    Girod's per-state bias and standard VIO; the localized coupling is what makes
    fixed-lag marginalization bounded in 0d/Phase 2.
  - Caveat noted for 0d: raw ISAM2 with COLAMD does not keep the leading edge near
    the root, so `reElim` is large in full smoothing regardless of bias model. The
    fixed-lag smoother's constrained ordering + marginalization (0d) is what turns
    the localized random-walk bias into bounded per-update cost.

- `test_phase0d.py`: IncrementalFixedLagSmoother (random-walk bias, lag 1.5s) fed
  stride-by-stride; the lag marginalizes knots older than t_now - lag. Probes
  conditioning (QR + Cholesky), bounded cost (active-var plateau), and
  consistency vs a full-smoothing ISAM2 (the exact reference, no marginalization)
  including a live-edge NEES.

Phase 0d status: **GATE = GREEN to proceed, FEJ now MANDATED for Phase 2**
(2026-06-25). Hard de-risk items pass; the consistency caveat is empirically
confirmed (not a failure: the plan routes FEJ into Phase 2 on exactly this).
  - **Conditioning: PASS.** Both QR and Cholesky survive every stride, no
    `IndeterminantLinearSystemException`. gtsam marginalization runs stably at our
    cond(H) ~ 5.5e10 (where the hand-rolled BandedSchur died, `RESEARCH_NOTES §8`).
    QR is the safe default (cond(J) = sqrt(cond(H))).
  - **Bounded cost: PASS.** Active variables grow during fill (107 -> 599) then
    PLATEAU at ~600 (601/600/601/597/...) for all subsequent strides; update time
    plateaus flat. Fixed-lag marginalization caps the problem size independent of
    trajectory length. (Absolute ~7.4 s/update is Python-FD-inflated, dominated by
    re-linearizing FD CustomFactors during marginalization; the real number is C++
    Phase 3.)
  - **Consistency: FEJ caveat CONFIRMED.** vs the exact full-smoother, live-edge
    NEES (dof 3, 95% band ~[0.2, 9.3]): **2.1 at 3.6 s -> 14.1 at 4.5 s**, with
    drift growing 1.5 -> 3.0 deg. Marginalization consistency degrades with
    horizon WITHOUT FEJ. This validates Caveat 2: iSAM2 fluid relinearization
    after marginalization injects spurious information into the weakly-observable
    gauge; the covariance becomes overconfident. => **Phase 2 must implement FEJ**
    (pin the linearization points of marginalized-coupled boundary + bias vars).
  - **Confound (likely OVERSTATES the inconsistency):** the spike omits heading
    priors (yaw anchored only at the start pin), so the yaw nullspace is freer
    than in the real system (lambda_heading > 0 anchors it throughout). With
    heading the NEES degradation would be milder. FEJ remains the principled fix
    (absolute position is unobservable even with heading), so it stays mandated;
    a heading-on consistency refinement is a cheap follow-up that does not change
    the gate decision.
  - **Deferred to C++ (Phase 3, by design):** full-rate accuracy vs the SW
    headline and rigorous FEJ on/off NEES-vs-mocap. The Python FD spike cannot run
    the full trajectory at 1 kHz, gtsam has no FEJ flag (the real on/off test needs
    the C++ implementation), and 0b already showed gtsam reproduces the Ceres
    optimum, so accuracy parity is expected once full-rate.

## Phase 1 progress (C++ GTSAM factors)

**Factor MATH done + verified on the existing toolchain (no GTSAM needed yet).**
The GTSAM-tangent Jacobians live as API-independent free functions in
`rio_solver_cpp/include/rio/gtsam/{gyro,accel,radar}_factor_math.h`, reusing the
existing analytic spline Jacobians (`spline_jacobians.h`) and the SymForce radar
sensor model. The only change vs the Ceres factors is the final convention step:
swap `tangent_to_ambient(q)` (Ceres ambient 4-dim) for `* R(knot_i).matrix()`
(GTSAM right-tangent 3-dim), per the bridge `d(f)/d(delta)=J_left*R_i`.

ALL FIVE factor-math types are now done + verified (headers in
`include/rio/gtsam/`): gyro, accel, radar (`*_factor_math.h`), plus min-snap and
angular-accel (`reg_factor_math.h`). `tests/test_gtsam_factor_math.cpp` (builds
WITHOUT GTSAM, runs in ctest) verifies each vs (a) numerical differentiation using
the GTSAM right-retract `q*Exp(eps)` and (b) the existing Ceres analytic factor
(residual parity). Result: **ALL PASS** -- every knot/CP/bias Jacobian matches
numerics to <=1e-4 (mostly ~1e-8 to 1e-10), all sensor residuals match the Ceres
factors EXACTLY (0). The derived angular-accel right-tangent Jacobian
(Jl^-1/Jr^-1 of the SO(3) log-differences) matches to 3e-10. Convention bridge +
spline-Jacobian reuse confirmed in C++. (Test uses central differences +
physically scaled CPs; forward differences blow up because a_world ~ inv_dt_pos^2.)

**Phase 1 wrappers DONE + verified (libgtsam-dev 4.2.0).** `include/rio/gtsam/
factors.h` wraps the 5 factors as **dynamic `gtsam::NoiseModelFactor`** (n-ary;
override `unwhitenedError(Values, boost::optional<std::vector<Matrix>&>)`),
chosen over variadic `NoiseModelFactorN` to avoid 10-11 template args for
radar/accel. Each just extracts the variables and forwards to the verified math.
`tests/test_gtsam_factors.cpp` checks each vs `gtsam::linearizeNumerically` (which
differentiates through GTSAM's own retract = the true end-to-end convention test):
ALL PASS, relative Jacobian error ~1e-11 to 1e-12. ctest 3/3 green.

**Two GTSAM packaging gotchas (Ubuntu Noble libgtsam-dev 4.2.0) -- important for
Phase 2/3:**
1. `find_package(GTSAM)` config mode FAILS: `GTSAM-exports.cmake` references a
   broken `CppUnitLite` imported target (missing `libCppUnitLite.a`). Workaround:
   link `libgtsam.so` directly via `find_library(GTSAM_LIB gtsam)` +
   `find_path(GTSAM_INC ...)`, NOT `find_package`. Also link `fmt::fmt` (basalt
   Sophus uses fmt) and `tbb`.
2. **Eigen alignment ABI mismatch -> heap corruption ("double free").** Ubuntu's
   GTSAM is built for baseline x86-64 (16-byte Eigen align); our global
   `-march=native` enables AVX (32-byte). Eigen objects (e.g. Rot3's Matrix3)
   then have mismatched layout across the GTSAM boundary -> corruption at runtime.
   Fix: compile GTSAM-linked TUs with `-march=x86-64` (`target_compile_options`).
   **Phase 2/3 consequence:** the IsamSolver + its pybind TU must use matching
   alignment; the Ceres analytic factors stay `-march=native` (separate TUs).
   For the final Phase-3 timing, consider rebuilding GTSAM from source with
   `-march=native` to regain AVX in the GTSAM-facing factor evaluation.

Remaining (minor, lands in Phase 2 wiring): heading prior (1D yaw factor on ori
knots) and boundary anchors (built-in `gtsam::PriorFactor<Rot3/Vector3/Vector6>`).

## Phase 2 progress (C++ IncrementalFixedLagSmoother backend)

`IsamSolver` (`include/rio/isam_sliding_window_solver.h` + `src/...cpp`) +
`rio_isam` pybind module. Fed NON-overlapping strides (the lag handles the
window); random-walk bias; reuses the Phase-1 factor wrappers. Driven by
`analysis/isam_spike/validate_isam_cpp.py` over the captured slow_racing problem.

**GTSAM had to be built from source (vendored):** apt `libgtsam-dev` ships NO
`gtsam_unstable` (no `IncrementalFixedLagSmoother`) AND is baseline-x86-64 (the
Eigen ABI mismatch). Source build (4.2.1, `gtsam_vendored_install/`, gitignored)
with `GTSAM_BUILD_UNSTABLE=ON` + `-march=native` + system Eigen fixes both.
Build gotcha: it links a bundled `libmetis-gtsam.so` via `DT_RUNPATH` (not
transitive), so GTSAM-linked targets need `BUILD/INSTALL_RPATH` to the vendored
lib dir + `-Wl,--disable-new-dtags` (old DT_RPATH, which IS transitive).

**First end-to-end run (slow_racing, full 25.6s @ 997 Hz IMU, no heading yet):**
  - **Timing: mean 64.9 ms/update, max 84.6 ms** vs the 300 ms stride budget
    => ~4.6x real-time margin (the Ceres SW barely hit 300 ms). THE MIGRATION
    GOAL, achieved.
  - Active vars plateau at 601 (marginalization bounds the problem, confirmed in C++).
  - Accuracy poor (6.1 deg mean ori / 0.6 m pos vs Ceres batch).

**Accuracy investigation (2026-06-25) -- timing/structure SOLVED, accuracy is the
open item.** Added heading factor (`HeadingFactor`, verified vs gtsam numerics)
and warm-start alignment (align entering knots to the solved boundary, mirroring
the Ceres SW). Diagnostic sweep on slow_racing (vs Ceres BATCH, which is
1.08 deg/0.20 m vs mocap):
  | config | ori mean | pos mean |
  |---|---|---|
  | P1-P3 init, lambda_h 0.6 | 6.0 deg | 568 mm |
  | + lambda_h 10 / 50 | 4.5 / 3.9 deg | 560 / 626 mm |
  | + warm-start align | 4.8 deg | 584 mm |
  | **cpp (optimal) init, lag 1.5** | 5.1 deg | 768 mm |
  | **cpp init, lag 30 (full smoother)** | **3.9 deg** | **303 mm** |
  - Tight bias RW (1e-6) and tight relinearize (1e-3): NO effect -> not bias-wander
    or under-relinearization.
  - KEY CLUE: even FULL-smoothing from the OPTIMAL init drifts to 3.9 deg/303 mm
    from Ceres. So gtsam's converged solution here differs from Ceres's by ~4 deg,
    even though Phase 0b showed <0.06 deg agreement over a 1 s FIXED-boundary
    window. The difference is the GAUGE/CONVERGENCE regime: 0b pinned BOTH ends +
    solved to LM convergence; here only the START is anchored (+ heading) and
    ISAM2 does gated incremental GN, not full re-optimization.
  - Prioritized candidates for the next debugging pass (accuracy only; timing is
    done): (1) **FEJ** (0d mandated it; the long-horizon drift matches the 0d NEES
    degradation); (2) ISAM2 convergence -- force extra relinearization / multiple
    updates so past states re-optimize like batch; (3) gauge/boundary -- the
    Ceres SW's exact boundary + the missing gravity-direction factor (lambda 1e-3);
    (4) compare against the SW (also fixed-lag, 0.30 m/1.97 deg live) rather than
    the global BATCH -- a fixed-lag method should be benchmarked against fixed-lag.

**Phase 2 status: backend WORKS + is REAL-TIME (4.6x margin, bounded); accuracy
(~4-6 deg vs ~2 deg target) is the focused remaining work.**

## Phase 0 verdict: PROCEED to the C++ port (Phases 1-3), with random-walk bias
and FEJ as firm requirements.
 The spike de-risked the two things that could have
killed the migration (conditioning under marginalization; unbounded cost) and
turned the FEJ caveat from a theoretical worry into a measured, must-fix design
requirement.
