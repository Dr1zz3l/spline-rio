# Response plan to the Opus T-RO-style review

Each point checked **against the actual report source** (not taken on faith). Verdicts:
**ADAPT** (do it) / **PARTIAL** (do part, decline part) / **REJECT** (with reason).
Ordered by priority. The three contradictions (#2, #3, #1) and the NEES/parity overclaims
(#4, #6) are the must-fixes; #12 and a couple of minors are misreadings.

## Implementation status (2026-06-24) — ALL must-fixes DONE

All ADAPT/PARTIAL items are now in `report/` (modular `sections/*.tex`):
- **#2** sensor-only made the default (`--mocap-accel-bias` opt-in); every headline number
  re-run and replaced throughout (Table III, abstract, conclusion). Completeness grep clean.
- **#3** adaptive-knot study added to Negative Results (VII-D); intro/conclusion wording matched;
  VII-B future-work lever changed to per-chirp timestamps (adaptive knots dropped).
- **#1** marginalization reframed around the shared-bias trap; [15]/standard selection contrasted;
  spline order N=6/N=4 stated; diagnostic + re-centering led as the novel bits.
- **#4** NEES: median reported for both channels, temporal-correlation + 6/50 backflip caveats,
  abstract softened to "approximately calibrated."
- **#6** "parity" removed (results + conclusion) → "same order of magnitude" / "trails ekf-yrio."
- **#9** noise-free MoCap-yaw caveat added (Limitations VII-A heading-sensor).
- **#7** real-time qualified to "aggressive racing at a 0.4 s stride."
- **#8** backflips settled-velocity cell dropped to `---` with note.
- **#10** consolidated per-regime configuration table added (Table II, §V).
- **#5** dedicated "Statistical Scope" subsection (N=1 honest; re-runs = reproducibility,
  not power). New flights declined for now (no data); flagged as the strongest future response.
- **#11** reverse-port relabeled "sensor/stack-coupling characterization."
- **#12** timing clarified: form-once (~10 ms) vs evaluate-per-iteration (`compute_prior`).
- **Minors:** title hyphenated; metric ordering unified; tool-name typography; Dataset table
  95th-pct footnote.

**Minors — final audit (2026-06-24, commit pending):**
- **[8] Ng et al. venue/year — VERIFIED CORRECT, no change.** IROS 2021 (IEEE Xplore doc
  9636014); `arXiv:2201.02437` is the post-conference preprint (Jan 2022). The bib's
  `year=2021`/`booktitle=IROS` is right.
- **Doppler-unwrap scope — CLARIFIED.** Code check (`validate_live_solver.py:930
  preunwrap_radar_frames`, called at :1849) confirms the unwrapped Doppler replaces the frame
  velocities fed to the solver, so it reaches the **main solver's radar factors**, not only the
  P2 WLS seed (the C++ solver's own $v_{\max}$ check is just a safety net). Methodology IV-D P2
  now states this explicitly.
- **`reve` first-mention gloss — ADDED** ("Doer's radar ego-velocity estimator") and
  `\mbox`-wrapped to match its later typography; RVE/basalt/ekf-rio/ekf-yrio/x-rio were already
  defined at first mention.
- **"Doer et al." — CONFIRMED CLEAN** (grep: reads "Doer and Trommer").
- **DECLINED for now:** Fig. 4 backflip error-vs-time panel (optional, low priority); extra
  recordings per regime (#5's strongest answer — needs new flight data, user's call).

Build: 12 pages, 0 undefined refs, 0 overfull. Trimmed to 12pp via redundancy cuts + tightened
float spacing + bios commented out (optional for initial submission).

## v2 work order (updated `opus_review.md`, 2026-06-24) — ALL valid items DONE

Commits `674788e` (edits) + `76773de` (re-run items). Each checked against compiled source first.
- **[4] "Doer et al." → "Doer and Trommer"** — VALID and *I'd missed it in v1*: the v1 grep
  false-negatived because the source line-wrapped "Doer et\nal."; ref [1] is two authors.
- **[1] NEES (P0)** — VALID cherry-picking catch (ori=mean, vel=median). Re-ran both racing bags
  sensor-only; now report BOTH as per-window median: ori 1.2–2.6, vel 1.4–5.4 (consistent 3),
  means kept parenthetically. Orientation calibrated/conservative; abstract claim holds.
- **[2] "fused 9°"** — VALID number clash with the 6.35° headline. Verified provenance
  (ROADMAP: 7.80° gyro-only vs 9.2° fused, diagnostic config); relabeled as the pre-reweighting
  diagnostic value that rate-adaptive gyro then brings to the 6.4° headline.
- **[3]/[6]** — VALID stale VII-C backflip numbers → Table III (1.67m/5.0° settled, 6.4° live).
- **[8] --no-radar** — re-ran both racing bags: confirms 5.9%/2.4 (slow), 8.0 vel + IMU-only ori
  8.9/4.0 (fast); ">90%" made precise to ≈99.5%.
- **[5]** IV-A knot spacings tagged "slow racing; see Table II". **[7]** abstract 6-DOF→6-DoF.
  **[10]** Table I footnote: ~10 rad/s is the 95th-pct sustained rate, not instantaneous max.
- **[9] audit** clean (no stale MoCap numbers, no "parity", all other "et al." ≥3 authors).
- **[11]** reverse-port compute one-liner folded in (EKF near sensor rate vs 0.35–0.70 s/window);
  page cost absorbed by trimming redundancy in VII-B + Conclusion (per user: trim anywhere).
- **DECLINED:** [12] backflip error panel (optional), [13] extra recordings (needs data).
- **No-change (reviewer agrees):** Ng [8] IROS 2021, 6-DoF title, timing, marginalization framing.

Build after v2: 12 pages, 0 undefined refs, 0 overfull, 0 broken refs.

## Verdict summary

| # | Concern | Verified? | Verdict |
|---|---------|-----------|---------|
| 2 | "Sensor-only init" vs MoCap-aided accel bias | TRUE (real contradiction) | **ADAPT (top priority)** |
| 3 | Adaptive-knot "disproof" claimed, not shown, contradicts VII-B | TRUE (real contradiction) | **ADAPT** |
| 1 | Marginalization framed as more novel than it is | Partly true (framing) | **ADAPT (reframe)** |
| 4 | NEES under-powered + statistic-switching | TRUE | **ADAPT** |
| 6 | "Position parity" overstates Table III | TRUE (we trail ekf-yrio on all 4) | **ADAPT** |
| 9 | "Magnetometer surrogate" = noise-free MoCap yaw | TRUE | **ADAPT (caveat)** |
| 7 | Real-time claim is conditional (0.4 s stride) | TRUE | **ADAPT (qualify)** |
| 8 | Settled backflip velocity 3.47 artifact | TRUE | **ADAPT (drop cell)** |
| 10 | Scattered per-regime hyperparameters | TRUE | **ADAPT (config table)** |
| 5 | Statistical thinness (N=1 per regime) | TRUE | **PARTIAL** |
| 11 | No compute comparison; reverse-port value | Partly | **PARTIAL** |
| 12 | Timing "10 ms" vs "dominant" inconsistency | **Misreading** | **REJECT + small clarify** |
| minors | citations / tool names / title / etc. | mixed | mostly **ADAPT**, two **REJECT** |

---

## MAJOR

### #2 — Sensor-only vs MoCap-aided accel bias  →  ADAPT (highest priority)
**Verified.** `sections/methodology.tex:191` (P1): *"MoCap orientation is used for a full 3-D
accelerometer bias estimate when available; otherwise a gravity-aligned scalar correction is
applied."* This directly contradicts `sections/system_overview.tex:89` ("P1–P3 … use no
external pose reference") and the abstract. Since all datasets are MoCap-lab recordings, the
"when available" path was almost certainly active for the headline Table II numbers.
**This is the single biggest honesty exposure — fix first.**

Nuance worth keeping in mind: the accel bias is an *optimized state with a prior*; MoCap only
**seeds** it. So it's a convergence aid, not MoCap entering the estimate directly — but the
**strong "no external pose reference"** wording does not survive it as written.

**RESOLVED (2026-06-24 re-run, `--sensor-only-bias` flag added to
`validate_live_solver.py`).** Ran all three headline SW configs both ways. Withholding MoCap
attitude from the accel-bias seed (gravity-aligned scalar correction only) changes the live-edge
numbers by **≤5%** (max: slow-racing ori 1.88→1.97°); the bias is an optimized state, so the seed
barely matters. **All abstract claims hold under the sensor-only numbers** (racing vel 0.32/0.48 <
0.5, ori 2.88/1.97 < 3°). The claim survives — resolution = reviewer's option (b), show the
headline under the sensor-only path.

| bag | live pos/vel/ori — MoCap-aided | live pos/vel/ori — sensor-only |
|-----|------|------|
| fast | 0.389 / 0.316 / 2.84 | 0.399 / 0.319 / 2.88 |
| slow | 0.303 / 0.463 / 1.88 | 0.310 / 0.476 / 1.97 |
| backflips | 1.554 / 2.346 / 6.26 | 1.545 / 2.350 / 6.35 |

**Remaining paper edit (pick one):**
- **(A) airtight:** make the sensor-only path the default; report the sensor-only numbers as
  headline (≤5% worse, all claims hold); delete the "MoCap orientation … when available"
  parenthetical in IV-D. ~15 number updates (Table II, abstract, conclusion).
- **(B) lower-effort:** keep the MoCap-aided headline numbers but add one sentence + this table:
  "the accel-bias seed may use MoCap attitude (an optional convergence aid for an optimized state);
  withholding it changes live-edge metrics by ≤5%, so the estimate is effectively sensor-only."
  A picky reviewer still notes the headline used MoCap, but it's honest and minimal.

Recommendation: **(A)** for a paper whose differentiator is honesty — the numbers barely move and
the claims survive, so airtight beats a caveat.

### #3 — Adaptive-knot "disproof" claimed but not shown, and contradicts VII-B  →  ADAPT
**Verified contradiction.** Claimed in `introduction.tex:53-54` and `conclusion.tex:29-31`
("a representation-level study that *disproves* the adaptive-knot-density hypothesis"), but
**no body section/table presents the study**, and `limitations.tex:110` (VII-B) proposes
*"adaptive (non-uniform) knot placement … to be the relevant levers, which we leave to future
work."* So the paper both disproves and proposes adaptive knots. We *have* the evidence (the
Phase-0 NO-GO investigation).

**Plan**
1. Add a short **representation-level study** to the body (Negative Results), showing uniform knots
   already saturate **orientation** tracking — denser/adaptive knots don't lower attitude error at
   10 rad/s (gyro-noise-floor argument), with the numbers from the NO-GO sweep.
2. **Separate the two axes explicitly:** (a) orientation knot *density* → disproved as a lever;
   (b) the residual **position/loop-scale** gap is **intra-frame Doppler smearing**, whose lever is
   **per-chirp timestamps**, not knot density.
3. **Fix VII-B**: change the future-work lever from "adaptive (non-uniform) knot placement and
   per-chirp timestamps" to **per-chirp timestamps** (drop adaptive knots, which we disproved).
4. Make intro/conclusion wording match the body study.

### #1 — Marginalization framed as more novel than it is  →  ADAPT (reframe; not an error)
**Partly true — a positioning issue, not a factual one.** A reviewer who has implemented OKVIS/
VINS marginalization can read "make the Schur marginal exactly closed" as "did textbook
marginalization correctly," because correct Schur already includes only marginal-connected
factors. The genuinely novel bit is **narrower and stronger**: in a continuous-time B-spline
window with a **global shared IMU bias**, the shared-bias block tempts inclusion of far-edge IMU
factors; the fix is to gate on *touches a marginalized variable* (not *marginalized OR boundary*).
Reframing to that is defensible and strengthens the paper; over-doing it risks under-claiming.

**Plan**
1. Foreground the **shared-bias trap** as the specific insight (not "we made marginalization exact").
2. State concretely what **[15] / standard factor selection** does and **why it fails here**
   (one or two sentences) — currently [15] gets ~one sentence.
3. **State the spline order N explicitly** (quintic N=6 position, cubic N=4 orientation) so the
   reader can verify the local-support containment `[i, i+N−1] ⊆ marg∪boundary` and the
   boundary-dimension count (30).
4. Lead the contribution with the **diagnostic symptom-pattern (VII-C)** and the
   **re-centering-instead-of-FEJ** choice, which are independently novel.

### #4 — NEES under-powered + statistic-switching  →  ADAPT
**Verified.** `results.tex:344-347`: orientation NEES reported as **mean** (0.7–0.9×, flattering),
velocity as **median** (1.4–5.7), with the velocity *mean* explained away as MoCap-reference noise.
Backflips: only 6/50 windows survive the mask. Per-window samples on one trajectory are temporally
correlated, so χ²₃ "mean 3" is heuristic, not a formal test.

**Plan**
1. Report the **same statistic** for both channels (median + IQR or the full distribution), no
   per-channel switching.
2. **Caveat temporal correlation** explicitly (consecutive windows correlated → calibration check,
   not a powered hypothesis test) and the 6/50 backflip windows (drop or heavily caveat backflip NEES).
3. **Soften the abstract**: "NEES-consistent orientation covariance" → "the orientation covariance
   is approximately calibrated on the racing flights."

---

## MODERATE

### #6 — "Position parity" overstates  →  ADAPT
**Verified.** ekf-yrio beats Ours on absolute position in **all four** ICINS flights
(0.39<0.92, 0.20<0.51, 0.48<1.92, 0.22<0.92; flight 3 ~4×). "Same order of magnitude" is fine;
**"parity" is not.** Used at `results.tex:217,232` and `conclusion.tex:42,46`.

**Plan:** replace "parity" with "the same order of magnitude" / "baseline-competitive" where it
means *RANSAC closes the order-of-magnitude vertical-drift gap*; state plainly we **trail the
yaw-aided baseline on absolute position**; foreground the real wins (RANSAC vertical-bias
robustness; heading-aided full-3-DoF orientation; comparable roll/pitch).

### #9 — Magnetometer surrogate = noise-free MoCap yaw  →  ADAPT (caveat)
**True.** MoCap yaw has no bias/disturbance/noise; a real magnetometer does, so heading-aided
orientation is optimistic. **Plan:** add one sentence that the surrogate is an *idealized*
heading reference; explicitly lean on the **roll/pitch (heading-independent) decomposition**
(Table III(b)) and the **unaided-yaw check** as the honest mitigations; have the abstract's "<3°"
point to "heading-aided" rather than imply it's reference-free.

### #7 — Real-time is conditional  →  ADAPT (qualify)
**True.** 0.35–0.70 s/window vs 0.3 s nominal stride ⇒ not real-time at nominal stride; only
fast-racing at a **0.4 s** stride qualifies. (Good news: the **report abstract does not claim
real-time** — that was the conference version.) **Plan:** qualify the intro contribution
(`introduction.tex:46`) and `results.tex:76` to "real time for aggressive racing **at a 0.4 s
stride**"; note the nominal 0.3 s stride is not met.

### #8 — Settled backflip velocity 3.47 m/s artifact  →  ADAPT (drop the cell)
**True and self-inflicted.** A >3 m/s resampling artifact in one settled cell invites distrust of
the whole settled column. **Plan:** drop the **settled-velocity** entry for backflips (report the
**live** column only, which is the deployment-relevant one) with a one-line note; keep settled
pos/ori. Cleaner than defending a starred artifact.

### #10 — Scattered hyperparameters / "which number is real"  →  ADAPT (config table)
**True** — multiple slow-racing batch-position numbers (0.157/0.169/0.183/0.198/0.20/0.314) are
reconciled only by footnotes. **Plan:** add **one consolidated per-regime configuration table**
(Δt_p, Δt_o, λ_accel/λ_gyro/λ_heading, ω-gates, anchor λ, b, marg-prior scale, window/stride),
referenced from §IV and §VI. Defuses the "which number is the real one" reflex directly.

### #5 — Statistical thinness (N=1 per regime)  →  PARTIAL
**True and the residual reviewer risk.** **ADAPT:** make the limitation a **prominent dedicated
paragraph** in Limitations (not a clause in VI-E); reframe re-runs as **reproducibility, not
statistical power**. **DECLINE (for now, with reason):** "add 2–3 recordings per regime" — we have
no new flight data; collecting it is a data-collection effort, not an edit. *If the user can record
more flights, that is the single strongest response;* otherwise we state N=1 honestly and rely on
held-out + cross-platform breadth.

### #11 — Compute comparison + reverse-port value  →  PARTIAL
**ADAPT:** relabel the reverse port explicitly as a **sensor/stack-coupling characterization**
(we already lean this way) so the 99.7%-rejection result isn't read as a head-to-head verdict.
**CONSIDER (lighter):** a one-line compute note (EKF near sensor-rate vs our 0.35–0.70 s/window);
a full accuracy-vs-compute table is probably more than the point needs — propose the one-liner,
not the table.

---

## REJECT / misreadings

### #12 — "Timing inconsistency: 10 ms vs dominant"  →  REJECT (misreading) + small clarify
**These are two different operations.** `methodology.tex:347` "the marginalization **step**
costs ~10 ms" = **forming** the marginal prior (Schur complement) **once per window**.
`experimental_setup.tex:67` "marginalization-prior **recompute** (`compute_prior`) … dominant"
= **evaluating** that prior's cost/Hessian contribution **every LM iteration** (up to 400×). No
contradiction. **Small ADAPT:** add a half-sentence distinguishing *form-once* (~10 ms) from
*evaluate-per-iteration* (`compute_prior`, dominant) so no reader repeats the reviewer's conflation.

### "Abstract is very long"  →  REJECT (likely stale / already at cap)
The **report abstract is 196 words**, at the T-RO 200-word cap, and dense by necessity. The
"very long" note most likely reflects the conference version (354 words). Minor tightening is
possible but further cuts lose content. **No action** beyond what's already done.

---

## MINOR (mostly ADAPT)

- **Title "Sliding Window" → "Sliding-Window"** (adjectival): ADAPT (trivial).
  Title "6-DOF" vs non-observable absolute position: **REJECT** — defensible (we estimate a full
  6-DoF trajectory; the abstract already states position drifts).
- **Define tool/system names at first mention** (RVE, *reve*, basalt, x-rio, ekf-rio, ekf-yrio)
  with consistent typography: ADAPT.
- **"Doer et al. [1]" → "Doer and Trommer":** the report **does not contain "Doer et al."**
  (grep clean) — appears **already handled**; VERIFY the citation reads "Doer and Trommer."
- **[15] differentiation:** covered by #1.
- **Metric-ordering consistency** (abstract tumbling "2.9%/6.3°" vs velocity/orientation/position
  elsewhere): ADAPT — pick one order document-wide.
- **Doppler alias unwrapping:** clarify whether unwrapped points reach the **solver** Doppler
  factors or only seed P2: ADAPT (one sentence in IV-B/IV-D).
- **Fig. 4 backflip error-vs-time panel:** CONSIDER — would strengthen the tumbling claims; doable
  if the backflip `--save-arrays` npz is on disk. Propose, low priority.
- **Table I 10 rad/s = 95th-pct with "MoCap artifacts dominate peaks":** ADAPT — state whether
  10 rad/s is the typical or absolute peak.
- **Table II layout** (Drift%/t_win run together): ADAPT — already tightened recently; re-check
  camera-ready.
- **arXiv/venue year for [8] Ng et al.:** ADAPT-VERIFY against the source.

---

## Suggested execution order
1. **#2 sensor-only** (re-run + reconcile) — top honesty exposure.
2. **#3 adaptive-knot** (add study + fix VII-B) and **#4 NEES** (statistic + abstract) — contradictions/overclaims.
3. **#6 parity**, **#7 real-time**, **#9 mag-surrogate**, **#8 settled-velocity** — overclaim tempering.
4. **#1 marginalization reframe** + **#10 config table** — strengthen framing/clarity.
5. **#12 clarify**, **#11 relabel**, minors.
6. **#5** — write the prominent limitation; flag new-data decision to the user.

Most are edit-only. The only items needing a run are **#2** (sensor-only re-run; I can do it) and
the optional **Fig. 4** backflip panel. **#5's "more flights"** is the one thing that may need new
data and is the user's call.
