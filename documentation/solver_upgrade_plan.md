# Plan: Banded-Cholesky + Arrowhead-Schur linear solver (Option 3a)

## Context

The SW spline solver's linear solve (SuiteSparse `SPARSE_NORMAL_CHOLESKY`) is ~58% of wall
time and cannot hit the 0.3 s stride budget on a Jetson Orin (~4–6 s/window). The B-spline
Hessian is **block-banded** (each residual touches 4 consecutive ori knots / 6 pos CPs) with
a small dense **arrowhead** (globally-coupled IMU bias + extrinsic pitch, and in SW the 30-dim
marginalization boundary block). Exploiting this gives an O(n) factorization vs SuiteSparse's
super-linear fill-in.

**Decision: Option 3a (banded Cholesky + correct arrowhead Schur), not 3b (Thomas).** Prior
review confirmed the note's 3b recipe has the Schur direction backwards, mislabels the band as
tridiagonal, and ignores the marg prior; the O(n) thesis is sound but not novel (STEAM, banded
B-splines). The defensible contribution is a **real-time systems** result, not theory.

**Method: three sessions, deliberately isolating math errors (Python) from framework/memory
errors (C++ skeleton) from optimization errors (C++ fast math).** Each session has a hard gate
before the next starts.

**Architecture decision (chosen): Route A — patch the vendored Ceres.** `rio_solver` currently
links the system `libceres-dev 2.2.0`, which exposes **no** public hook for a custom
`ceres::LinearSolver` (the class is in `ceres/internal/`, not installed; `linear_solver_type`
is a closed enum). We re-point the build to the vendored source tree
(`lie-spline-experiments/thirdparty/ceres-solver`) and patch the `LinearSolver` factory to
inject `BandedSchurSolver`. This keeps Ceres' LM trust-region, damping, and robust-loss
handling intact, so "converges exactly as SuiteSparse" is a clean, literal verification.

---

## Two expectation-setters to keep in view

- **Amdahl ceiling.** Linear solve = 58% of wall time ⇒ making it ~free gives only
  `0.42 + 0.58/10 ≈ 2.1×` *overall*, not 10×. On Orin that's ~2–3 s/window — still ~8–10×
  over the 0.3 s budget; the Jacobian/residual eval (~42%) becomes the new floor. "Order of
  magnitude on the linear solve" (Session 3 metric) is correct as a **component** result but
  is **not** real-time on its own. Hitting the stride budget needs this **+ Option 2**
  (shorter window / coarser `dt`). Track both component and wall-clock numbers.
- **Conditioning.** The undamped SW Hessian is cond ≈ 5.5×10¹⁰. Correctness gates compare
  **method agreement** (banded-Schur vs dense on the *same* H), not accuracy vs the true
  solution (which is cond-limited to ~`eps·cond` ≈ 1e-6, a matrix property, not a bug). The
  LM-damped system `JᵀJ + D²` is far better conditioned and is the realistic case.

---

## Session 1 — Python (the math)

**Goal:** a standalone Python script that reproduces a dense `np.linalg.solve(H, g)` step to
numerical precision via the banded + arrowhead-Schur path. No C++ solver logic.

### 1.1 — Matrix-dump hook in C++ (small, low-risk; reuse existing problem build)
- `solver.h`: add `bool dump_system{false}` to `SolverConfig`; add `SolverResult` fields for
  the warm-start linearized system — `jac_values/jac_cols/jac_row_ptr` (Ceres `CRSMatrix`),
  `jac_num_cols`, `residuals`, `gradient`, and a `block_map` (`{type_id, index, col_offset,
  tangent_size}` per parameter block; SO3 manifold tangent = 3).
- `solver.cpp` / `sliding_window_solver.cpp`: when `dump_system`, after building the problem
  call `problem.Evaluate(opts, …)` once with an **explicit ordered `opts.parameter_blocks`**
  (ori → pos → bias → pitch) so column order is deterministic. SW's `MargPriorFunctor` is
  already a residual block, so its 30-dim block lands in J automatically.
- `pybind_module.cpp`: expose the new fields as numpy.

### 1.2 — Dump driver `analysis/dump_linear_system.py`
Run `--cpp --set dump_system=1` for a few windows of: (a) **batch racing** (corner = 7),
(b) **SW racing** (corner = 7 + 30 marg = 37), (c) **SW backflips** (corner = 6 + 30 = 36;
locked extrinsics, denser/variable band). Reconstruct `J` (scipy CSR), `H = JᵀJ`, `g = Jᵀr`;
`np.save` H (sparse `.npz`), g, block_map.

### 1.3 — Prototype `analysis/prototype_banded_solver.py` (pure NumPy/SciPy)
1. **Partition** via block_map: dense corner = trailing bias(+pitch)(+marg-boundary) cols
   (auto-detected, 6/7/36/37); banded knots = the rest.
2. **Reorder** knots with `scipy.sparse.csgraph.reverse_cuthill_mckee`; record achieved
   half-bandwidth (expect ~5 pos / ~3 ori), assert locality; document the temporal-interleave
   equivalent. **Confirm the 30 marg-boundary knots are contiguous at the band end** — the
   one precondition for folding them into a single dense corner.
3. **Banded knot solve** via `scipy.linalg.cholesky_banded` / `solveh_banded`.
4. **Arrowhead Schur into the corner** (correct direction):
   `Y = H_kk⁻¹ H_kc`, `y_g = H_kk⁻¹ g_k`, `S_c = H_cc − H_kcᵀ Y`,
   `δx_c = S_c⁻¹(g_c − H_kcᵀ y_g)`, `δx_k = y_g − Y δx_c`; un-permute.
5. **Compare** to `np.linalg.solve(H_dense, g)`.

### 1.4 — Session-1 gate (must pass before Session 2)
- Test on **both** `H = JᵀJ` (worst, cond≈5.5e10) and `H = JᵀJ + μ·diag` (realistic LM).
- Gate: `‖δx_banded − δx_dense‖ / ‖δx_dense‖ < 1e-8` on every dumped window
  (6/7/36/37-corner, batch + SW). **Assessment: validate all corner sizes** — parametric,
  near-free, and the 6-DoF (locked) case confirms generality.
- Bonus paper figure: sparsity before/after RCM + zero-fill banded factor vs CHOLMOD fill-in.

---

## Session 2 — The Ceres bridge (C++ skeleton, slow but correct)

**Goal:** own the linear-algebra pipeline; ignore fast math. Do **not** write the banded solver.

- **Build switch:** re-point `rio_solver_cpp/CMakeLists.txt` `find_package(Ceres)` to the
  vendored `thirdparty/ceres-solver` (build it from source into a local prefix). Confirm
  `rio_solver` links the patched build, not system `libceres`.
- **Patch the factory:** add a `BANDED_SCHUR` value to Ceres' `LinearSolverType` and a case in
  the `LinearSolver::Create` factory returning `BandedSchurSolver`. Keep the change minimal
  and documented (it is a maintained fork).
- **`BandedSchurSolver` (skeleton):** subclass the appropriate `TypedLinearSolver`. In
  `Solve(A, b, per_solve_options, x)`: assemble the **normal equations including LM damping**
  `H = AᵀA + diag(D²)` (D from `per_solve_options.D`) into a dense `Eigen::MatrixXd`, solve
  with `Eigen::LDLT` (SPD; prefer over `PartialPivLU` for the near-singular undamped case),
  write the step into `x`. Ceres applies the robust-loss `√ρ′` correction upstream, so the
  matrix you receive is already correct — no loss handling needed here.
- **Set `options.linear_solver_type = BANDED_SCHUR`** in `solver.cpp` behind a config switch;
  keep `SPARSE_NORMAL_CHOLESKY` as the default fallback for regression.

**Session-2 gate:** compile; run **one 3 s racing window** (~3k DoF — dense LDLT is seconds;
**not** backflips, whose ~12k DoF dense H is ~1 GB). Verify the LM loop iterates and converges
**identically** to SuiteSparse (same final cost, same trajectory to float noise), just slower.
Cross-check the per-iteration dense step against Session-1's Python dense solve on the same
window. SuiteSparse is now bypassed and you own the pipeline.

---

## Session 3 — The fast math (C++ banded-Schur)

**Goal:** replace the dense solve *inside* `BandedSchurSolver::Solve` with the Session-1 logic.
Everything else (build, factory, LM, loss, damping) is unchanged from Session 2.

- Iterate Ceres' CRS arrays (`cols`, `rows`, `values`) and assemble: the dense corner
  (`Eigen::MatrixXd`, 6/7/36/37), the coupling `H_kc`, and the knot band in LAPACK banded
  storage for `dpbtrf`/`dpbtrs` (LAPACKE). (Header-only alternative = `Eigen::SimplicialLLT`
  on the sparse knot block with a band-preserving permutation — *not* a true banded routine.)
- **Recompute the ordering deterministically in C++** (RCM or fixed temporal interleave) — do
  **not** serialize Python's permutation. Lock the column→block map to the matrix Ceres hands
  the solver (its internal layout, which you re-permute yourself).
- Translate the four-step Schur (`Y`, `y_g`, `S_c`, back-substitute) to Eigen + LAPACK; add D²
  to the band diagonal before factorization.

**Session-3 gate:**
- **Parity:** SW run reproduces CLAUDE.md live-edge metrics (slow_racing 0.393m/2.21°,
  fast_racing 0.877m/4.16°) within float noise of the SuiteSparse baseline.
- **Speed:** `time_linear_solver_s` drops ~order of magnitude; linear scaling vs trajectory
  length (3/6/12/24 s); Orin per-window time. **Report overall wall time too** (Amdahl) and
  note the Option-2 follow-up needed for hard real-time.

---

## Critical files

| File | Session | Change |
|------|---------|--------|
| `rio_solver_cpp/include/rio/solver.h` | 1 | `dump_system` flag + `SolverResult` CRS/blockmap fields |
| `rio_solver_cpp/src/solver.cpp` | 1,2,3 | dump hook; `BANDED_SCHUR` switch |
| `rio_solver_cpp/src/sliding_window_solver.cpp` | 1,3 | dump hook (captures 30-dim marg block) |
| `rio_solver_cpp/src/pybind_module.cpp` | 1 | expose dump fields |
| `analysis/dump_linear_system.py` | 1 | NEW — drive dumps |
| `analysis/prototype_banded_solver.py` | 1 | NEW — banded+Schur prototype + dense gate |
| `rio_solver_cpp/CMakeLists.txt` | 2 | point Ceres at vendored source build |
| vendored `thirdparty/ceres-solver` (`types.h`, `linear_solver.cc`) | 2 | add `BANDED_SCHUR` enum + factory case |
| `rio_solver_cpp/include/rio/banded_schur_solver.{h,cpp}` | 2,3 | NEW — dense skeleton (S2) → banded+Schur (S3) |

## Open assessment notes (carried)
- Corner size varies: 6/7 (batch) ± 30 marg (SW) = 6/7/36/37. Folding the 30 in is valid only
  if those knots are contiguous at the band end — confirm in 1.3.
- Real-time needs Session 3 **+ Option 2**; banded solve alone is ~2.1× overall (Amdahl).
- Session-2 dense skeleton: racing windows only (backflips dense H ≈ 1 GB).
- Gates compare method-agreement, not true accuracy (cond ≈ 5.5e10).
- Fork maintenance: the Ceres patch is version-locked to the vendored tree; pin its commit.
