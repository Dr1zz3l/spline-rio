# Real-Time Deployment Options (Jetson)

**Baseline**: SW solver at 0.3 s stride takes ~2 s per window on a desktop CPU
(SuiteSparse SPARSE_NORMAL_CHOLESKY is the bottleneck, ~58% of wall time).
On Jetson Orin (~2–3× slower for serial sparse linear algebra) → ~4–6 s/window,
roughly **15–20× over the 0.3 s stride budget**.

---

## Option 1 — ITERATIVE_SCHUR + PCG (drop-in Ceres change)

Change `linear_solver_type` from `SPARSE_NORMAL_CHOLESKY` to `ITERATIVE_SCHUR`
with `JACOBI` or `SCHUR_JACOBI` preconditioning.

**Potential gain**: PCG's J^T(Jv) products parallelize across residuals;
12-core Orin could give ~2–4× on that step. Also avoids SuiteSparse fill-in on
the banded Hessian.

**Why it doesn't close the gap**: `ITERATIVE_SCHUR` is designed for bundle
adjustment's arrowhead sparsity (many 3D points, few poses). Our Hessian is
**banded** (each IMU residual touches 4 consecutive knots, no point-like
variables to eliminate). PCG iteration count scales as O(√κ); the per-window
Hessian condition number is O(10⁵–10⁶) (gyro at 1 kHz contributes ~62 500
info per knot vs ~10 for accel). With JACOBI preconditioning, convergence to
Cholesky-equivalent accuracy requires hundreds of iterations per LM step,
burning the parallelism advantage many times over.

**Expected net speedup**: ~1.5–2× at best, possibly slower. Not sufficient.

---

## Option 2 — Problem Size Reduction (accuracy trade-off)

Halving window length (3 s → 1.5 s) or doubling dt_ori (8 ms → 16 ms) roughly
quarters the Hessian size. Cholesky time scales as O(n^1.5–2) on sparse systems,
so ~4–8× faster solve, bringing desktop time to ~0.25–0.5 s.

**Cost**: accuracy degrades (ablation in CLAUDE.md shows 1.5 s window gives
0.675 m live position vs 0.337 m at 3 s). Coarser dt_ori would also hurt
orientation on fast-dynamics bags. Could be viable if accuracy targets loosen.

**How to test**: `--set window=1.5 --set dt_ori=0.016` per the existing CLI.

---

## Option 3 — Exploit Banded Spline Structure (solver rewrite)

The key insight the current solver ignores: the B-spline Hessian is **block
band-diagonal** with bandwidth 4 (each evaluation uses 4 consecutive knots).
The full Hessian H looks like:

```
[ pos_block      | cross      |  0     ]
[ cross^T        | ori_block  |  0     ]
[   0            |   0        | bias   ]
```

where `pos_block` and `ori_block` are individually block-tridiagonal (or
block-pentadiagonal for quintic) with block size 3×3 or 3×9.

### Option 3a — Banded Cholesky via SuiteSparse hint or Eigen

Reorder variables so spline knots are ordered temporally rather than by type.
With AMD/COLAMD constrained to preserve the band, fill-in stays O(bandwidth^2 × n)
instead of O(n^1.5). Effectively: replace the current `problem.Solve()` with a
manual normal-equation assembly → `Eigen::SimplicialLLT` or banded LAPACK
`dpbtrf`. The factorization time drops from O(n^1.5) to **O(n × bw^2)** ≈ linear
in trajectory length.

**Estimated speedup**: 5–10× for the factorization step (dominant term).
Combined with LM overhead: ~3–6× overall, potentially bringing Orin into the
1–2 s/window range. Still not hard real-time at 0.3 s stride but makes a
larger stride (1 s) viable.

**Implementation cost**: medium. Need to assemble H and g manually (bypassing
Ceres's internal normal-equation builder), then call a banded solver. The Ceres
`EvaluationCallback` + `GetCovarianceMatrix` APIs expose enough to do this
without a full rewrite.

### Option 3b — Block-Tridiagonal Direct Solve (Thomas algorithm)

If the Hessian can be cast into pure block-tridiagonal form (one block per
time step containing all variables active at that step), the Thomas algorithm
factorizes it in **O(n × b^3)** where b is the block size (~12–15 DOF/knot if
pos+ori are co-located in time). For n=375 ori knots and b=12: ~375 × 1728 ≈
650 k FLOPs vs SuiteSparse's millions for the factorization.

**Precondition**: requires aligning position and orientation knot spacing.
Currently dt_pos=5 ms, dt_ori=8 ms — different grids break the block structure.
Unifying them (or sub-sampling to a common grid) is a design change with
accuracy implications.

**Implementation cost**: high. Requires rethinking the state representation and
manual block-tridiagonal factorization. Essentially rebuilds the linear solver
from scratch but gains a provably optimal complexity.

---

#### Scientific value and community relevance

**The core insight**: the Hessian of any uniform B-spline trajectory estimator
is block-tridiagonal when (a) state variables are ordered temporally (all DOF at
knot k before knot k+1) and (b) position and orientation share a common knot
spacing. This follows directly from the B-spline locality property: the basis
function B̃_j(t) is non-zero only on the interval [t_j, t_{j+4}), so each
residual evaluated at time t touches exactly 4 consecutive knots. The resulting
J^T J has non-zero blocks only between temporally adjacent knots. The bias term
(6 DOF) creates a dense arrowhead column, but that is eliminated by a single
6×6 Schur complement step before the tridiagonal solve, leaving a clean
block-tridiagonal system.

**Why this is likely novel**: every major continuous-time estimator in the
literature — Furgale et al. (2013), basalt VIO, Hug et al. (2022), this work —
uses SuiteSparse or g2o without noting or exploiting this structure. Solver
complexity in all of them grows as O(n^{1.5}–n^2) with trajectory length.
Nobody has published the O(n) proof or an implementation that achieves it.

**Why it matters beyond this system**: the result is framework-independent and
general — it applies to any continuous-time B-spline estimator regardless of
sensor modality (visual-inertial, LiDAR-inertial, radar-inertial). The practical
implication is that it removes the computational barrier to long-horizon
continuous-time estimation. Currently extending a batch window from 3 s to 30 s
is prohibitive; with O(n) complexity it is a linear cost increase.

**Framework note**: the Thomas algorithm is a linear algebra subroutine, not
tied to Ceres or GTSAM. It solves Hδx = g where H is block-tridiagonal and
δx is the update step — a pluggable component in any solver. In Ceres it is
implemented as a custom `LinearSolver` subclass with no other changes needed.
For a pure sliding-window system without loop closures, it is also strictly
more efficient than iSAM2's Bayes tree: iSAM2's general machinery handles
non-sequential measurements (loop closures) that are absent here, and for the
sequential case reduces to essentially the same sequential factorization but
with more framework overhead. The Thomas algorithm is the minimal, optimal
specialization for this specific problem class.

**Proposed paper framing**:
> *"Linear-time continuous-time trajectory optimization via block-tridiagonal
> structure in B-spline estimators"*
>
> We prove that the Hessian of any uniform B-spline trajectory estimator is
> block-tridiagonal when variables are ordered temporally and knot spacings are
> unified, enabling O(n) factorization via the Thomas algorithm. We demonstrate
> a 10–15× speedup over SuiteSparse on an RIO system, achieving real-time
> sliding-window estimation on a Jetson Orin at full accuracy. The result
> generalizes to any B-spline-based sensor fusion system and removes the
> super-linear complexity barrier to long-horizon continuous-time estimation.

This frames the RIO system as the experimental vehicle for a broader
algorithmic result — the right positioning for an ICRA/RAL submission.

---

#### Steps to proving it

**Step 1 — Analytical proof of block-tridiagonal structure**

Write out H = J^T W J symbolically for a single residual evaluated at time t
in knot interval [t_k, t_{k+1}). The Jacobian ∂r/∂Ω_j is non-zero only for
j ∈ {k−3, k−2, k−1, k}. The outer product J^T J therefore contributes
non-zero blocks only in the 4×4 sub-matrix at rows/columns {k−3..k}. Summing
over all residuals and all time steps, the maximum off-diagonal reach is 3
knots (bandwidth 4). Show formally: H_{ij} = 0 for |i−j| > 3. Then show the
bias block produces an arrowhead extension, not additional bandwidth in the
main knot chain.

**Step 2 — Verify structure empirically on the current system**

Dump the assembled Hessian from a single window solve using
`ceres::Problem::Evaluate()`. Visualize the sparsity pattern — a block-banded
structure should be immediately visible. Measure the actual fill-in produced by
SuiteSparse CHOLMOD and compare to the zero fill-in of the Thomas algorithm.
This takes roughly a day and provides the key paper figure: sparsity pattern
before and after factorization, with fill-in counts.

**Step 3 — Implement the Thomas algorithm as a Ceres LinearSolver**

Ceres exposes a `LinearSolver` interface. Implement a `BlockTridiagonalSolver`
that:
1. Receives the sparse H and g from Ceres's normal-equation builder.
2. Extracts the block-tridiagonal knot chain and the bias arrowhead.
3. Eliminates bias via Schur complement (6×6 dense solve — negligible cost).
4. Applies the Thomas forward sweep: for k = 1..n, factorize diagonal block
   A_k − B_{k−1} L_{k−1}^{−1} C_{k−1} (each step is a b×b Cholesky on a
   modified diagonal block after subtracting the left-neighbour contribution).
5. Back-substitutes for δx.

Total cost: n × b^3 FLOPs. Strictly O(n).

**Step 4 — Benchmark and characterize**

Run both solvers on trajectories of increasing length (3 s, 6 s, 12 s, 24 s).
Plot wall time vs n. SuiteSparse should show super-linear growth; Thomas should
be linear. Benchmark on Jetson Orin to demonstrate the embedded deployment
claim. This is the core empirical result of the paper.

**Step 5 — Accuracy parity**

Solve the same window with both solvers and verify RMSE is identical to
floating-point precision. They solve the same linear system; any difference is
numerical noise. This confirms the Thomas solver is a zero-accuracy-cost
drop-in replacement.

---

#### Indications that this will work

These are observable from the current system without any new experiments:

- **The bandwidth-4 structure is a mathematical certainty, not an
  approximation.** The B-spline evaluation kernel in `cumulative_so3_bspline.py`
  and `CeresSplineHelper` always accesses exactly 4 consecutive knots per
  evaluation. There is no code path that violates this.

- **SuiteSparse fill-in is already the measured bottleneck.** The linear solve
  is 58% of batch wall time. For a banded matrix, CHOLMOD's AMD reordering
  produces O(bandwidth^2 × n) fill-in — already much less than a dense system,
  but still super-linear in n. The Thomas algorithm has zero fill-in by
  construction. The speedup follows from this difference, not from any
  approximation.

- **The bias arrowhead is small.** The bias block is 6×6 with dense connections
  to all knots, but only 6 DOF. The Schur complement step costs
  6^3 + 6^2 × n FLOPs — linear in n and negligible relative to the tridiagonal
  solve. It does not break the structure of the remaining system.

- **Unified knot spacing is low-risk.** dt_pos=5 ms and dt_ori=8 ms are close.
  Unifying at 5 ms gives denser orientation knots (equal or better accuracy);
  unifying at 8 ms gives slightly coarser position knots. The `--set` CLI
  already supports this as a one-line experiment to characterize the accuracy
  delta before committing to the design change.

- **The FLOP margin is large.** For n=375 knots, b=12, the Thomas algorithm
  requires ~650k FLOPs for the forward sweep. The current SuiteSparse solve on
  the same system takes ~1.16 s on a desktop. Even with aggressive overhead
  the Thomas algorithm is 2–3 orders of magnitude fewer FLOPs — the speedup
  margin cannot plausibly be erased by implementation overhead.

### Option 3c — iSAM2 / GTSAM incremental smoothing

GTSAM's iSAM2 (incremental smoothing and mapping) maintains a Bayes tree and
applies only local Cholesky updates when new measurements arrive. For a sliding
window with regularly marginalized variables this gives amortized O(1) per
measurement rather than O(n) re-factorization, plus natural parallelism over
independent subtrees.

**Estimated speedup**: potentially 5–20× if the factor graph is well-structured.
iSAM2 has been demonstrated for RIO/VIO at real-time rates on embedded hardware.

**Implementation cost**: very high. Full rewrite replacing Ceres with GTSAM.
The existing cumulative SO(3) B-spline Jacobian infrastructure has no direct
GTSAM equivalent; custom factors would be needed.

---

## Option 4 — EKF / iEKF Architecture

Drop the sliding-window batch optimizer entirely. Maintain a state vector of
current position, velocity, orientation, and bias; propagate with IMU, update
with each radar frame via an EKF/iEKF measurement step.

**Reference**: Doer & Trommer's EKF-RIO runs in real-time on a laptop. Their
accuracy (0.5–2% drift depending on dynamics) is comparable to our SW live-edge
results, though their radar operates at 76 GHz with different noise characteristics.

**What you lose**: the continuous-time representation (asynchronous fusion,
closed-form derivatives), the global angular-acceleration regularization, and
the Schur complement marginalization propagating information across windows.
Backflips would almost certainly be worse. Extrinsic self-calibration harder.

**What you gain**: O(1) per frame, trivially real-time, well-understood
failure modes, and a large body of prior implementation (ESKF, basalt-style
filter).

---

## Summary

| Option | Est. speedup | Accuracy cost | Implementation effort |
|--------|-------------|---------------|-----------------------|
| ITERATIVE_SCHUR | 1.5–2× | None | Trivial |
| Problem size reduction | 4–8× | Moderate | Trivial (existing flags) |
| Banded Cholesky (3a) | 3–6× | None | Medium |
| Block-tridiagonal (3b) | 8–15× | State design change | High |
| iSAM2 / GTSAM (3c) | 5–20× | None (similar) | Very high |
| EKF rewrite | ∞ (real-time) | Significant | High |

For a Jetson deployment, the most pragmatic path is **Option 2 + Option 3a** in
combination: accept a modest accuracy trade-off from a shorter window, and
invest in banded Cholesky to recover most of that loss. That combination could
plausibly reach 0.5–1 s/window on Orin — viable for a 1 s stride if the latency
is acceptable for the application.
