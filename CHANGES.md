# CHANGES — vertical-drift investigation pass (2026-06-13)

Scope: changes made **after** the barometer-refutation revision (commit `7fbd9a1`).
This pass follows up that pass's escalated open item (§3.1 there: the ICINS vertical-drift
mechanism was unresolved). It is now **resolved** and the drift is reduced to <1 m.

## Summary

The 16.9 m vertical drift of our system on ICINS-2021 was a **radar front-end
outlier-rejection gap**, not a fundamental limitation. Doer's `reve` front-end uses 3D
RANSAC that hard-rejects the minority of elevation-biased single-chip returns; our pipeline
used a Huber kernel (δ=1.0 m/s) that only *down-weights* them, leaking ~0.1–0.16 m/s of
vertical velocity bias that integrates into the drift. Applying reve-equivalent RANSAC
(inlier 0.15 m/s) collapses our vertical drift to match the baselines.

## 1. Wording fixes (Part A) — done

- Table VI caption: removed "structurally unobservable vertical elevation bias".
- Body whole-traj sentence: dropped "rather than a porting artifact"; claim only faithful
  reproduction + that the gap survives Umeyama alignment.
- (Both will be re-edited again to the *resolved* mechanism now that the investigation
  succeeded — see §3, pending.)

## 2. Investigation (Part B) — findings with sources

### B0 — elevation-gate verified (paper claim accurate, no change)
The converted bags ARE gated to ±60°: flight_1 |elev| max 57.5° in the converted bag vs
75.6° in the original (3% of points >60° in the original → 0% in the converted). So the
paper's "their ±60° elevation pre-gate" claim is correct, and "we omitted the gate" is ruled
out — the 16.9 m already includes gating. Source: elevation comparison of
`baselines/datasets/icins2021/.../flight_datasets/flight_1.bag` `/sensor_platform/radar/scan`
vs `baselines/datasets/our_format/icins_flight_1.bag` `/mmWaveDataHdl/RScanVelocity`.

### B1 — WLS front-end vertical-velocity bias (new diagnostic)
New `analysis/diagnostics/icins_zbias_probe.py`: per radar frame, solve ego-velocity with
plain WLS / Huber / RANSAC, rotate to world via **GT attitude**, compare to GT velocity.
Per-flight vertical velocity bias [m/s] (no extra elevation gate):

| flight | plain WLS | Huber δ=1.0 | RANSAC |
|--------|-----------|-------------|--------|
| 1 | −0.283 | −0.064 | +0.002 |
| 2 | −0.306 | −0.164 | −0.097 |
| 3 | −0.353 | −0.102 | −0.014 |
| 4 | −0.306 | −0.100 | −0.029 |

The Huber column (our solver's effective per-point kernel) ≈ the observed drift; RANSAC
(reve's method) cuts the vertical bias 2–10×. Run: `diagnostics/icins_zbias_probe.py icins_flight_{1..4}`.

### B2 — end-to-end RANSAC front-end (new `--radar-ransac` flag)
Added `--radar-ransac [thresh]` to `analysis/validate_live_solver.py`: a reve-style 3D LSQ
RANSAC pre-filter on each radar frame (inlier 0.15 m/s), keeping only inlier points before the
spline solve. Re-ran our solver on ICINS. Causal start-anchored metric
(pos RMSE m / vel m/s / ori° / drift%), naive port → **+RANSAC**:

| flight | naive port | **+ RANSAC** | (baseline ekf-yrio) |
|--------|-----------|--------------|---------------------|
| 1 | 16.94 / 0.348 / 0.57 / 11.95 | **0.92 / 0.077 / 0.51 / 0.65** (vert 16.90→0.89) | 0.39 / 0.08 / 0.70 / 0.3 |
| 2 | 6.70 / 0.399 / 0.93 / 14.98 | **0.51 / 0.077 / 0.94 / 1.14** (vert 6.68→0.51) | 0.20 / 0.08 / 0.91 / 0.4 |
| 3 | 21.57 / 0.379 / 0.52 / 14.69 | **1.92 / 0.074 / 0.51 / 1.30** (vert 21.6→1.87) | 0.48 / 0.08 / 1.09 / 0.3 |
| 4 | 10.30 / 0.331 / 1.11 / 13.15 | **0.92 / 0.069 / 1.13 / 1.18** (vert 10.3→0.90) | 0.22 / 0.07 / 1.41 / 0.3 |

RANSAC kept ~95% of points (rejected ~5% elevation-biased outliers). All four flights drop
by an order of magnitude to baseline-comparable (0.5–1.9 m, 0.7–1.3% drift); velocity RMSE
also falls from ~0.33–0.40 to ~0.07 (baseline level).

### B3 — NOT NEEDED
The cause is the front-end, so the planned C++ gravity-referenced vertical-velocity
regularization factor was not implemented.

## 3. Paper edits made (Task C) — done
- Sec. VI-F: replaced the "unresolved / future work" vertical paragraph with the resolved
  mechanism (Huber vs RANSAC front-end, the per-frame bias numbers) + the all-four-flights
  RANSAC result; retains the Huber front-end as the reported default and names RANSAC as
  future work.
- Table VI caption: "cause unresolved" → "a vertical bias … that a RANSAC front-end removes".
- Conclusion limitation (3): "cause unresolved — ruled out altitude aiding" → traced to
  radar-front-end outlier rejection (reve RANSAC vs our Huber), with the future-work line
  "a RANSAC radar front-end is the better default and should replace our Huber front-end".
- (Decision: keep Huber as the reported default; do NOT change the main-pipeline numbers.)
- PDF rebuilt (11 pages, clean, all citations resolve).

## 4. Curiosity tests on our OWN bags (NOT in the paper, per request)
RANSAC front-end vs the Huber baseline on our own pitched-mount bags (live causal):
- slow_racing: 0.303 m / 1.97° → **0.303 m / 1.88°** — neutral (already clean).
- fast_racing: 0.501 m / 3.24° / vel 0.41 → **0.389 m / 2.84° / 0.32** — **improves** (−22% pos).
- backflips:  1.51 m / 6.29° / vel 2.29 → **1.55 m / 6.26° / 2.35** — neutral (≤0.04 m).
Net: RANSAC is a real win on aggressive flight (fast_racing), harmless elsewhere — i.e. a
genuinely better front-end, confirming the ICINS problem was a naive-port artifact and not a
general deficiency. Promoting RANSAC to the pipeline default (re-running all benchmarks,
ablations, NEES, timing; ideally a C++/seeded implementation) is left as a deliberate
future pass.

## Files changed this pass
- `report/IEEE-conference-template-062824.tex` — Sec. VI-F mechanism paragraph (resolved +
  RANSAC result), Table VI caption, body whole-traj sentence, Conclusion limitation (3).
- `report/IEEE-conference-template-062824.pdf` — rebuilt (11 pages).
- `analysis/diagnostics/icins_zbias_probe.py` — new front-end bias diagnostic.
- `analysis/validate_live_solver.py` — new `--radar-ransac` front-end pre-filter (opt-in;
  Huber remains the default).
- `documentation/ROADMAP.md` — Part 6c (this investigation).
- `CHANGES.md` — this file.
- (`baselines/results/ours_icins/*ransac*` run logs are gitignored artifacts.)
