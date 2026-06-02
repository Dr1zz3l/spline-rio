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
