# CHANGES — RANSAC default front-end + full re-benchmark (2026-06-14)

Scope: the RANSAC ego-velocity prefilter (Sec. VI-F follow-through) was made the **sole
default** radar front-end after a full re-tune + re-benchmark across every radar-dependent
metric. `solver.yaml radar_ransac_threshold: 0.15` (disable `--no-radar-ransac`); seeded,
frame-load, frames <5 bypass — see [[project_ransac_default]] / `memory`.

**Verdict (adopted per user rule):** fast_racing live pos −22% (0.50→0.39m), vel 0.41→0.32,
ori 3.24→2.84°; slow/backflips neutral (≤+3%); ICINS whole-traj ATE order-of-magnitude
(9.6→0.46 / 2.9→0.24 / 10.9→0.76 / 5.5→0.46m); held-out + old-fw kept 46–75% (no
starvation); old-fw backflips ori 10.7→8.1/9.1°. **Config unchanged → universality preserved.**

**Both papers rewritten** (report/ master + paper/): new Sec. III-B RANSAC paragraph;
**batch Table II removed** (demoted to prose ceiling + pitch self-cal); Table III/V/VI +
abstract + conclusion renumbered to RANSAC; VI-F → clean baseline *match* (duration/portability
hedge + future-work framing deleted); backflips batch reframed as bistable, figure → SW-only;
NEES/held-out/old-fw refreshed. Figures regenerated from RANSAC `--save-arrays` npz. report/
12pp, paper/ 11pp (float-bound), both build clean, 0 undefined refs. Re-benchmark logs:
`baselines/results/ransac_default/` + `baselines/results/ours_icins/*_ransac.log`.

---

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
  RANSAC result.
- Table VI caption: "cause unresolved" → "a vertical bias … that a RANSAC front-end rejects,
  closing the gap by an order of magnitude".
- Conclusion limitation (3): "cause unresolved — ruled out altitude aiding" → traced to
  radar-front-end outlier rejection (reve RANSAC vs our Huber).

### Framing reframe (2nd review note) — RANSAC = mount-portability fix, NOT universal default
The first draft of these edits called RANSAC "the better default [that] should replace our
Huber front-end" — an overclaim: the paper *demonstrates* RANSAC superiority only on the
foreign-mount ICINS cross-validation, not on our own pitched-mount results. Reframed to claim
only what is shown:
- RANSAC closes the **portability gap** to a horizontal-boresight mount; on our platform
  Huber is **adequate** (empirically — our reported results, Sec. results, are 0.28–0.5 m) and
  is what we report. Deferral reason stated: adopting RANSAC as default requires re-validating
  the dynamics-adaptive weighting law against it across all regimes — future work.
- Sec. IV-B bridge added: the Huber δ=1.0 choice is reconciled with VI-F (adequate on our
  platform; portability examined in VI-F) so the two sections cohere.

**MECHANISM HEDGED (not asserted) — it contradicted our own Sec. VII-B.** The first draft of
this reframe said "the systematic bias is absorbed by the 27.5° pitched mount via extrinsic
pitch calibration." But Sec. VII-B explicitly **retires** that reading ("the earlier
'self-calibrated +2° absorbs the elevation bias' reading is retired — 27.5° is simply the
correct mount angle"). Asserting absorption in IV-B/VI-F would have contradicted VII-B. Fixed:
lead with the empirically-shown adequacy (our own results), and demote mount geometry to a
hedged "plausibly because on a horizontal-boresight mount the poorly observed elevation
direction aligns with world-vertical" — no claim that pitch calibration does bias-absorption work.

**Optional upgrade attempted, did NOT pan out.** Ran the B1 diagnostic on our own bags
(`icins_zbias_probe.py slow_racing_best_velocity --euler 180,27.5,0`) hoping to show Huber's
vertical bias is near-zero on our platform (a positive demonstration). It produced spurious
nonzero *horizontal* biases (y −0.16 to −0.20 m/s) that the solver's good slow_racing result
(0.28 m) contradicts — the diagnostic's conventions, validated on the *converted* ICINS bags,
don't transfer cleanly to our own bags. Not trustworthy; no claim built on it. The hedge stands.
(The `--euler` flag added to the diagnostic is a harmless generalization.)
- Table VI: added a note row with the RANSAC-corrected whole-traj ATE (0.46/0.24/0.76/0.46 m)
  so the reader does not carry away the 9.6–10.9 m Huber naive-port figure.
- Tightened "comparable to the baselines" → "reaching the same order as the baselines" (we are
  ~2–4× of yaw-aided ekf-yrio, genuinely comparable only to unaided ekf-rio).
- Stated the RANSAC inlier threshold (0.15 m/s) is **reve's stock value** (verified:
  `reve/.../radar_ego_velocity_estimator.py` default + `params_demo_dataset.yaml`), preempting
  "tuned the gate".
- Fixed the RANSAC bias range to "at most 0.10 m/s in magnitude" (flight_1 is +0.002, positive).
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

---

## 5. Own-bag vertical-bias diagnostic — geometry vs duration vs fusion (measurement only)

Follow-up to resolve WHY our own platform avoids the ICINS vertical leak. Generalised
`icins_zbias_probe.py` to run correctly on our own bags: applies the radar→GT time offset
(`radar_total_offset = imu_mocap_offset − radar_imu_offset`, ≈ −120 ms on our bags vs 0 on
ICINS — this was the bug behind the earlier spurious horizontal bias), per-bag extrinsics +
flip set, GT-aided Doppler unwrap, GT-velocity low-pass. **No pipeline constants, solver, or
paper edits.**

**Validation gates — both pass.** (1) Positive control: ICINS flight_1 reproduces the committed
B1 (plain −0.283, Huber −0.064, RANSAC +0.002 m/s vertical). (2) Own-bag sanity: after the
offset fix, slow_racing horizontal bias is ~0 (was −0.16..−0.20); fast_racing RANSAC x/y ≈ 0
(−0.017/−0.045), confirming conventions are correct (its Huber x +0.096 is real outlier leakage,
not a convention bug).

**Per-frame vertical ego-velocity bias [m/s] (rotated to world via GT attitude):**

| bag | window | plain WLS | Huber δ=1.0 | RANSAC | dead-reckon z-drift (∫) | actual solver vert RMSE |
|-----|--------|-----------|-------------|--------|-------------------------|--------------------------|
| ICINS flight_1 | 183 s | −0.283 | −0.064 | +0.002 | −11.8 m | ~16.9 m |
| slow_racing | 22 s | −0.029 | **−0.019** | −0.002 | −0.4 m | 0.198 m |
| fast_racing | 14 s | −0.051 | **−0.054** | −0.008 | −0.8 m | 0.167 m |

(circle, the extrinsics-flip-set code-path test, ran sanely — flip handling works — but is a
short different-day bag and is NOT used for any mechanism conclusion. backflips excluded per
scope: its flip-regime `b`/intra-frame distortion is a different mechanism.)

**VERDICT: DURATION-dominated, NOT mount-geometry.**
- fast_racing's per-frame Huber vertical bias (−0.054 m/s) is **essentially ICINS-level**
  (−0.064) — **on our pitched mount.** So the mount does NOT make the elevation bias vanish;
  the "pitched-mount geometry absorbs it" reading is not supported by the data.
- The reason our reported results avoid the ICINS-scale drift is mainly **flight duration**:
  our flights are 14–22 s vs ICINS 87–186 s (~8×). Projecting fast_racing's bias to ICINS
  duration: −0.054 × 186 ≈ −10 m — i.e. ICINS-scale. If we flew aggressive long flights on our
  own mount, we would drift like ICINS.
- **Partial geometry / outlier-rate:** slow_racing's bias (−0.019) is smaller than ICINS, but
  this tracks motion aggressiveness (outlier rate: benign motion → fewer corrupted returns),
  not mount geometry per se (same mount as fast_racing, which is 3× higher).
- **Modest fusion:** fast_racing actual vertical (0.167 m) is below the dead-reckon RMS (~0.46 m)
  → ~2–3× solver suppression on short windows; slow_racing shows little (0.198 vs ~0.23 m).
  Fusion helps on short flights but is overwhelmed over ICINS-length flights.
- RANSAC drives the bias to ~0 on every bag (slow −0.002, fast −0.008), so it remains the robust
  fix independent of mount/duration.

**Implication for the paper framing (for the author to decide — NO edit made):** the current
VI-F/IV-B hedge ("plausibly because on a horizontal-boresight mount the elevation direction
aligns with world-vertical … our pitched mount") is the WEAKEST-supported of the three; the
measurement says the honest reason is **short flight duration** (+ partial outlier-rate, modest
fusion). A duration-based framing would be more defensible than the mount-geometry one.

---

## 6. Paper updated with the duration finding (3rd review note) — done

Author decision: update the paper with the measured finding now (do NOT switch to RANSAC
default this submission). Edits (PDF now 12 pages):
- **VI-F**: replaced the mount-geometry hedge with the measured story, led by the
  **same-mount dynamics scaling** (slow −0.019 → fast −0.054 m/s on the identical 27.5°
  mount → leak is outlier-driven, not mount-driven). Added: geometry *redistributes* which
  axis the bias lands on (ICINS x +0.002/z −0.064 concentrated; fast_racing x +0.096/z −0.054
  split) but does not eliminate it; **duration** is why our drift stays sub-metre (18–26 s
  vs ICINS 90–190 s; 0.05 m/s × 186 s ≈ 10 m). Added the **protective self-characterization**
  sentence: the same bias is present in our own racing results, under-exposed by short lab
  flights, would dominate position on sustained flight — the by-construction position-drift
  component a RANSAC front-end removes at source.
- **IV-B bridge**: "portability to other mount geometries" → "admits a small residual …
  adequate at our short flight durations but not eliminated; dynamics-/duration-limited
  validity examined in VI-F."
- **Conclusion (3)**: deferral reframed onto duration + re-validation (not "Huber is fine on
  our mount"): the bias scales with dynamics on our own platform, kept sub-metre only by
  short flights; RANSAC removes it at source; default-switch pending weighting-law
  re-validation is future work.

**Reviewer-C caveat applied (fusion leg dropped).** The §5 "fusion suppresses 2–3×" sub-point
is weakly supported — it compared an RMSE to an integrated-displacement (∫), which is not
apples-to-apples (linear-growth RMSE ≈ 0.58× final; and on ICINS actual 16.9 m already
EXCEEDS the ∫ 11.8 m, the tell that the metrics aren't matched). The paper claim therefore
rests on the two solid legs only — (i) per-frame bias is ICINS-level on our own mount and
scales with dynamics, (ii) it integrates to sub-metre only because flights are short. Fusion
is not claimed in the paper.

RANSAC-default still deferred (submittable-now > better-but-incomplete). If the paper reaches
final and time allows, the considered move is to re-run all benchmarks (racing, backflips,
ablations, NEES, Pareto, timing) with RANSAC as the default front-end and update the numbers.

---

## 7. Full-paper read & audit (2026-06-13)

Read the entire paper end-to-end and fixed what was found (length left for later, per request).

**Content fixes:**
- **Citation error (L270):** the intro called the ICINS-2021 datasets "of [doer2020ekf]" (the
  2020 MFI/EKF paper); the datasets belong to the 2021 yaw-aided paper and the baselines section
  itself cites `doer2021yrio` for them. Fixed → `doer2021yrio`.
- **Heading-weight disclosure (IV-B):** methodology stated a bald `λ_ψ=0.6`, but the headline
  slow-racing live result (Table III) and backflips use `λ_heading=10` (bags.yaml; the Table III
  footnote already lists "the heading weight" as per-regime). Tightened to "0.6 by default, raised
  to 10 on the heading-critical bags (slow racing, backflips)".

**Figures (text too small):** root cause — figures authored at 9–13 in then shrunk to the 3.45 in
column (~0.27–0.38×), so 11–13 pt fonts rendered at 3–5 pt. Fixed by raising source fonts in the
four `report/figures/gen_*.py` scripts (load cached `.npz`; no solver re-run) so on-page text is
~9–10 pt, and making the 6-panel `error_over_time` a full-width `figure*`. Regenerated all six
figure PDFs; verified legible by rendering. (Trajectory/RPE/prior-scale/error-time all readable now.)

**Tables:** verified **zero overfull boxes** — no table actually overflows the column. The dense
Table V (4 numbers/cell) is tight but fits; could be promoted to a full-width `table*` for air if
desired (not done — it fits).

**Flagged, NOT changed (author's call):**
- 27.5° vs "physical 30°" mount: L283/L780 say physical 30° tilt and self-cal → 27.1–27.6°, while
  VII-B says "27.5° is simply the correct mount angle". Mild tension (CAD/eyeball 30° vs data-fit
  extrinsic-pitch 27.5°); internally explained but a sharp reader may notice. Left as the author's
  calibration narrative.

PDF 12 pages, clean, all citations resolve. Sections not flagged by reviewers were read and found
consistent with the evolved cross-validation story.
