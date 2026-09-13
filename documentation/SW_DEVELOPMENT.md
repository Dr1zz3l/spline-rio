# Sliding Window Solver — Development History and Ablation Results

> Detailed rationale and experimental data behind the SW solver's current defaults.
> For the high-level performance profile and quick-reference config see `CLAUDE.md`.
> For mathematical derivations see `Backward Model.md`.

---

## 0. Historical context: Three-Phase Optimization Pipeline

Before the live C++ SW solver, the pipeline went through three Python phases:

**Phase 1 — Physics Diagnostics** (`validate_physics.py`): Feed MoCap ground truth through the
forward model to verify coordinate transforms, Doppler sign, and extrinsic calibration. No optimization.

**Phase 2 — Linear Solver** (`validate_linear_solver.py`): Use known orientation (MoCap SLERP)
and solve sparse least squares for position B-spline control points only.

**Phase 3 — Nonlinear Solver** (`validate_nonlinear_solver.py`): Full Levenberg-Marquardt with
sparse Cholesky. Jointly optimizes position, orientation, and biases. MoCap-initialized.
Flags: `--no-flip`, `--flip`, `--no-radar`, `--precond` (Jacobi preconditioning).

These scripts still exist and run, but the current best results come from `validate_live_solver.py --cpp`.

---

## 1. Orientation Regularization: Angular Acceleration Replaces Angular Velocity

The old `OrientationRegFunctor` penalized `||log(q_i^{-1}·q_{i+1})||²` (minimum angular velocity),
which fought every banked turn and the entire backflip maneuver.

`AngularAccelRegFunctor` penalizes the second finite difference:
```
r = log(q_{i-1}^{-1}·q_i) - log(q_i^{-1}·q_{i+1})
```
Zero for constant angular rate. Only fires at maneuver onset/offset.

**Lambda sweep** (--mocap-yaw --cpp, batch, backflips uses dt_ori=0.0008):

| lambda_ori_accel | slow_racing pos | fast_racing pos | backflips ori |
|---|---|---|---|
| 0.0 (none) | **0.150m** | 0.875m | 9.59° |
| 0.001 | 0.178m | 0.786m | 8.61° |
| 0.01 | 0.178m | 0.790m | 8.43° |
| **0.1** ← default | 0.170m | **0.740m** | **8.31°** |
| 1.0 | 0.174m | 0.796m | 44.2° ⚠️ |

0.1 is best compromise. 1.0 blows up backflips (regularizer too tight to represent the rapid
angular acceleration of the flip itself). Orientation RMSE insensitive across racing bags —
1000 Hz gyro dominates; regularizer only matters for position and backflip stability.

Current defaults: `lambda_ori_reg: 0.0`, `lambda_ori_accel: 0.1`

---

## 2. Per-Bag dt_ori: Why Backflips Needs 0.0008s

**dt_ori sweep** on both bags (lambda_ori_accel scaled λ ∝ dt_ori³ to keep continuous ∫||α||²dt equivalent):

| dt_ori | slow_racing pos | backflips pos | backflips ori |
|---|---|---|---|
| 0.008 (default) | **0.170m** | 3.369m | 8.01° |
| 0.006 | 0.151m | 3.662m | 8.17° |
| 0.004 | 0.224m | 4.772m | 9.55° |
| 0.002 | 0.459m | 3.355m | 49.8° ⚠️ |
| **0.0008** (backflips default) | — | **1.814m** | 8.57° |

No intermediate value helps backflips — position drops below 2m only at 0.0008 (hard cliff, not
a gradient). Every step denser than 0.008 hurts slow_racing. Root cause: 0.008s doesn't represent
the rapid angular acceleration ramp-up of the backflip. There is no universal dt_ori for both.

**Note: dt_ori=0.0008 is NOT a Nyquist argument.** A cubic B-spline at dt_ori=0.008 has 62.5 Hz
Nyquist — well above the ~10–20 Hz bandwidth needed for a 0.5s backflip. The non-monotonic sweep
is an optimizer convergence/regularizer-scaling artifact. The jump at 0.0008 reflects a specific
balance between knot density and lambda_ori_accel that the batch optimizer exploits.

**Why extrinsic optimization always breaks on backflips**: rapid large-amplitude orientation changes
that the spline can only partially represent create a systematic Doppler residual, which the optimizer
reduces by drifting pitch_delta (cheaper than fixing orientation DOF en masse). At dt_ori=0.008,
pitch_delta drifts to +29.6° (55.1°) with catastrophic pos RMSE 3.96m. Root cause is not the
DOF/constraint ratio — it is that the backflip creates conditions where pitch_delta acts as a
surrogate for orientation model error. Fix: `lock_extrinsics: 1` per-bag.

**Additional issue at dt_ori=0.0008s**: 22630 knots × 3 = 67890 DOF vs ~54000 gyro constraints
(ratio 1.26 DOF/constraint) makes the system technically underconstrained. In SW:
- 3s window has ~3750 ori knots (11250 DOF) vs ~3000 gyro samples (9000 constraints)
- Underconstrained per window → rank=18/30 boundary DoF → Ceres exits after 2 LM iterations
- Batch works because lambda_ori_accel=0.1 stiffens the underdetermined modes across 18s

---

## 3. SW Phase 1: Bias Anchor Bug Fix + Per-Bag Hardening

**Root cause of original SW divergence** (settled 2.37m/57.5°, gyr bias blowup to -1.19 rad/s):
`sliding_window_solver.cpp` built the per-window bias prior from `traj_.biases` (current warm-start),
not from the stationary calibration estimate. The marg prior is correctly re-centered (curvature-only);
the bias prior must NOT be — it is an absolute sensor anchor. This caused the bias to ratchet freely
across windows and absorb orientation null-space error.

**Fixes applied** (per-bag gated via `bags.yaml`):
- **E1 (bug fix, all bags)**: `init_biases_` captured in `SlidingWindowSolver::initialize()`;
  `BiasPriorFunctor` now anchors to `init_biases_[j]` instead of `traj_.biases[j]`.
- **E1+ `lock_gyro_bias`** (backflips per-bag): post-solve clamp resets gyro components to
  `init_biases_[3..5]` before `compute_prior()`.
- **E2 `lambda_heading: 10.0`** (backflips per-bag): stronger heading prior through the flip.
- **E3 `marg_prior_scale: 0.0`** (backflips per-bag): disables the marg prior entirely;
  rank-deficient S produces a garbage prior; window continuity comes from overlap instead.

**Results after Phase 1** (SW, backflips_best_velocity):

| | Before (broken) | After Phase 1 | Batch ceiling |
|---|---|---|---|
| Settled pos | 2.37 m | **1.89 m** | 1.82 m |
| Settled ori | 57.5° | **42.0°** | 8.31° |
| Live pos | 2.58 m | **2.06 m** | — |
| Gyr bias z | -1.19 rad/s 💥 | -0.001 rad/s ✓ | ~0 |

**Pre-Phase-1 marg_prior_scale sweep for backflips** (historical, without bias anchor fix):

| marg_prior_scale | Pos RMSE | Ori RMSE | Gyr bias z |
|---|---|---|---|
| 2e-4 (default) | 2.37m | 59.4° | −1.04 rad/s |
| 2e-3 | 2.37m | 57.5° | −0.16 rad/s |

Position was locked at ~2.37m across 3 orders of magnitude due to bias runaway masking all
other effects. The bias anchor fix is a prerequisite for any further SW backflips work.

---

## 4. Marginalization Prior Diagnostics (Steps 1–3)

Three diagnostic layers added to the SW solver:

**Step 1 — Marginalization quality monitoring**: per-window logging of Schur complement S
condition number, eigenvalue range, and numerical rank. Output per window:
```
prior=OK  cond=5.5e+10  rank=15/30  applied=2.00e-04  ||r||²=300000
```
Fields in `SolverResult`: `marg_prior_valid`, `marg_cond_number`, `marg_min/max_eigenvalue`,
`marg_numerical_rank`, `marg_drop_reason`.

**Step 2 — S⁻¹ boundary covariance + adaptive prior scaling**: computes S⁻¹ = accumulated
prior covariance. `adaptive_scale = sqrt(lambda_boundary_pos / max_eigenvalue_S)` ≈ 1.35e-6.
Optional: `use_adaptive_marg_scale=true`. Fields: `marg_trace_cov`, `marg_adaptive_scale`,
`marg_applied_scale`.

**Step 3 — Dual covariance view S⁻¹ vs H_bb⁻¹**: H_bb⁻¹ = current-window-only boundary
covariance (sensor only, no prior). Output: `tr(S⁻¹)≈4.9e-4  tr(H⁻¹)≈4.7e-4  ratio≈0.95`.
Ratio ≈ 0.95 means window sensor information and accumulated prior are comparably informative.

---

## 5. marg_prior_scale: Non-Monotonicity, Eigenvalue Clipping, Prior Residual

**Why S is ill-conditioned (cond ≈ 5.5e10) — physically expected, not a bug**

With lambda_gyro=4.0 at 1000 Hz, each gyro sample contributes ~4×125² ≈ 62,500 information per
boundary orientation knot. Position DOF get lambda_accel=0.01 with sparse radar → eigenvalue ~1e4.
Ratio ~5.5e10. The prior double-counts gyro constraints while position (which needs inter-window
help) is under-represented in S. Eigenvalue clipping directly addresses this asymmetry.

**marg_prior_scale sweep — slow_racing**

| Scale | Live ori | Live vel | Live pos | Behavior |
|---|---|---|---|---|
| 2e-4 (old default) | 2.282° | 0.408 | 0.623m | Strong prior → over-constrained |
| 1e-6 | 2.393° | 0.416 | 0.513m | **Harmful regime: worse than baseline!** |
| **1e-7** ← slow_racing default | **2.207°** | **0.389** | **0.383m** | Near-zero: free adaptation |
| 1e-8 | 2.201° | 0.387 | 0.381m | Essentially zero prior |

At 1e-6 to 1e-5: prior partially constrains without providing useful continuity → worst of both
worlds. For fast_racing, softer scale marginally improves live ori (4.163→4.093° at 1e-5) but
worsens live pos (0.877→0.940m); baseline 2e-4 retained.

**Eigenvalue clipping sweep** (`marg_prior_eig_clip`): tested 9 combinations
(clip ∈ {1e5,1e6,1e7}, scale ∈ {0.01,0.05,0.2}). Best universal candidate:
clip=1e7 scale=0.2 → slow live ori 2.21°, fast live ori 4.57° (worse than 4.16% baseline).
**No (clip, scale) pair beats per-bag tuning on both bags simultaneously.**

Root cause: slow_racing (gentle) → windows self-sufficient → near-zero prior optimal;
fast_racing (aggressive) → needs inter-window position continuity → meaningful prior needed.
Contradictory requirements; per-bag `marg_prior_scale` in `bags.yaml` is correct.

**Prior residual norm** (`marg_prior_residual_norm` = ||r||² at solution):
- slow_racing at 1e-7: ||r||² 2M–5.5T (boundary completely free)
- fast_racing at 2e-4: ||r||² ~300k median
- Cauchy gating direction is backwards (would down-weight fast_racing which needs the prior).
  `marg_prior_cauchy_delta` disabled by default.

---

## 6. Phase 2: ω-Gated Radar — Batch-Only Benefit

**ω-gate** (`omega_gate_threshold` in SolverConfig, default 0.0 = disabled): skips radar frames
where |ω_body| exceeds the threshold at problem-build time (from initial spline, not per-iteration).

**Batch result at dt_ori=0.008**: ω-gate=4.0 + lambda_ori_accel=0.001 achieves **1.904m/6.98°**
batch — close to dt_ori=0.0008 result (1.82m/8.31°).

lambda_ori_accel sweep at dt_ori=0.008 (batch, no gate, backflips):

| lambda_ori_accel | pos RMSE | ori RMSE |
|---|---|---|
| 0.1 (default) | 3.67m | 7.49° |
| 0.01 | **1.98m** | 7.60° |
| 0.001 | 2.00m | 7.64° |
| 0.0 | 2.21m | 7.57° |

**Why ω-gate doesn't help SW**: at dt_ori=0.0008 (Phase 1 config), the SW solver is essentially
FROZEN near P1-P3 MoCap init — rank-deficient Jacobian (iter=2) prevents the solver from moving
away from the accurate MoCap warm-start. This is why SW Phase 1 gives 1.89m: the P1-P3 trajectory
IS the solution. At dt_ori=0.008, the Jacobian is well-conditioned (8:1) and the solver
ACTUALLY OPTIMIZES in 2 iterations, moving to a worse local optimum (7.20m/10.08°).

Phase 2 boundary rank analysis:
- rank=15/30 at dt_ori=0.008 (WORSE than 18/30 at 0.0008)
- Position boundary (5 pos CPs × 3 DOF = 15 DOF) underconstrained by radar:
  boundary spans last 50ms of stride zone → 0-1 radar frames → rank 3-6/15 for position
- rank=15/30 is an INHERENT structural limit, not a function of dt_ori

---

## 7. Phase 3: SW Backflips — Position-Init Prior (Final SW Attempt)

**Idea**: Phase 2 failed because dt_ori=0.008 + free position = 7.2m drift. The P1-P3
radar-velocity init is already a decent position estimate (~2.44m RMSE). Phase 3 pins position
to the P1-P3 init with a soft per-CP prior while letting gyro + heading refine orientation.

`lambda_pos_init_prior`: every position CP in the active window gets a direct L2 penalty anchoring
it to `init_pos_cps_[i]`. Implemented as `PosInitPriorFunctor` in `regularization.h`.

**SW command (use --set to override bags.yaml batch config):**
```bash
python validate_live_solver.py backflips_best_velocity --mocap-yaw --cpp --sliding-window \
  --set dt_ori=0.008 --set lambda_ori_accel=0.001 --set lock_gyro_bias=0 \
  --set marg_prior_scale=0.0 --set lambda_pos_init_prior=1000.0
```

**Phase 3 results** vs Phase 1 frozen + batch ceiling:

| Config | Settled pos | Settled ori | Live pos | Live ori |
|---|---|---|---|---|
| Phase 1 (0.0008, frozen) | 1.89m | 47.5° | 2.06m | 24.8° |
| Phase 3 (0.008 + λ_pos=1000) | **2.56m** | **10.87°** | **3.33m** | **9.33°** |
| Batch ceiling | 1.82m | 8.31° | — | — |

**Why position RMSE appears "worse"**: at 47° orientation error the SE3 alignment absorbed ~0.55m
of systematic position bias. At 10° orientation the full P1-P3 init error (~2.44m) is exposed.
The 2.56m/10.87° result is strictly better on all six axes; the apparent regression is an SE3
artifact.

**iter=2 everywhere** (genuine convergence): with λ=1000 position prior the Hessian is diagonally
dominant near the solution. Setting max_iterations=100 produces identical results.

**Why gap from batch remains**: SW position anchored to P1-P3 init (2.44m RMSE); batch's global
accel integration over 18s refines this to 1.82m. A 3s window can't replicate 18s of accel
integration.

**Final verdict**: SW backflips ~2.56m/10.87° is the best achievable with a fixed-lag smoother.
`bags.yaml` `solver_overrides.backflips_best_velocity` retains `dt_ori=0.0008` for batch.

> **CORRECTION (2026-06-12):** `cfg.lambda_pos_init_prior` was never plumbed from
> the Python driver into the C++ `SolverConfig` (`git log -S` confirms the line
> never existed; fixed in `validate_live_solver.py:382`). All Phase 3 SW runs —
> including the documented `--set lambda_pos_init_prior=1000.0` command — silently
> ran with the tether at the C++ default **0.0**. The Phase 3 results above were
> therefore produced by `dt_ori=0.008 + lambda_ori_accel=0.001 + marg_prior_scale=0
> + lock_gyro_bias=0` alone. The "iter=2 / diagonally dominant Hessian" analysis is
> also affected. Backflips SW deserves a re-run with the now-working tether AND the
> 2026-06-12 consistent marginalization prior (see `ROADMAP.md`).

---

## 8. Phase 2.5: MoCap-Aided Stationary Bias Detection

`detect_stationary_bias()` in `validate_nonlinear_solver.py`, called from both
`validate_live_solver.py` and `validate_nonlinear_solver.py`. Both callers pass the
**full-bag** `bag_data.agiros_state` so the pre-flight stationary period is always visible.

**MoCap path**: velocity cross-check (rejects if |v_mocap| > 0.05 m/s); full 3-D accel bias
`b_a = mean(z_acc) − R_bw^T · [0, 0, 9.81]`; mean rotation re-orthogonalised via SVD.

**IMU-only fallback**: removes scale-error component only:
`b_a = mean(z_acc) − (mean(z_acc)/|mean(z_acc)|) · 9.81`. Transverse bias unobservable,
left at zero (optimizer absorbs within first few iterations). Logs a NOTE.

**No-static-window fallback**: returns None → callers use zero biases + level gravity [0,0,9.81].

Slow-racing batch result slightly improved: 0.169m/1.02° (was 0.174m/1.08°).

---

## 9. Extrinsic Pitch Optimization

`RadarDopplerWithPitchFunctor` in `radar_doppler.h` accepts a 1-DOF scalar `pitch_delta`.
`solver.cpp` and `sliding_window_solver.cpp` add it when `lock_extrinsics=false` (default).
Composition: `R_total = R_nominal * Ry(pitch_delta)`. A `PitchDeltaPriorFunctor` with
`lambda_extrinsic_prior=10.0` keeps it near nominal.

Racing bags consistently converge to 27–28° from either 25.5° or 30° init (+1.6–2.1° from
25.5° nominal). Metrics essentially unchanged vs locked baseline.

Backflips uses `lock_extrinsics: 1` per-bag override — 22630 dense ori knots create too many DOF
for the weak prior; pitch_delta drifts to +18° with catastrophic orientation error (43.8° vs 8.3°
locked).

---

## 10. IMU Preintegration (Implemented, Disabled)

See `RESEARCH_NOTES.md §3` for the full analysis. Summary: preint at 100 Hz replaces 1000 Hz raw
gyro with 100 Hz preintegrated factors, degrading constraint density from 8:1 to 1:1 per orientation
knot. slow_racing orientation RMSE 1.09° → 6.0°. Fundamental conflict with backflips (dt_ori=0.0008
requires dt_preint < IMU period). Disabled via `use_preintegration: false` in `solver_cpp.yaml`.

If enabling: start with `lambda_preint_v=0, lambda_preint_p=0` (r_R only) to avoid corrupting
orientation through r_v/∂R_i before the optimizer has converged.
