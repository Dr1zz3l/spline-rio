# paper/ — RANSAC-default rewrite (2026-06-14)

The RANSAC ego-velocity prefilter is now the **sole default** radar front-end
(re-benchmark verdict: fast_racing −22% live pos, ICINS order-of-magnitude ATE,
slow/backflips neutral, config unchanged → universality preserved). Applied to **both**
`report/` (master) and `paper/`:
- **Sec. III-B:** new "RANSAC ego-velocity prefilter (default front-end)" paragraph
  (reve-style 3D-LSQ, inlier 0.15, seed `default_rng(0)`, frame-load not per-window).
  Pipeline caption notes the prefilter. Radar measurement-model bridge clause removed.
- **Removed batch Table II**, demoted to a prose sentence (offline ceiling 0.20/1.0° slow,
  0.73/2.5° fast + pitch self-cal 27.0/27.2°). Kills the slow-batch +17% and
  backflips-batch bistability wrinkles; frees float space.
- **Table III (SW):** RANSAC numbers (fast live 0.39/0.32/2.84°/0.86%; slow 0.30/0.46/1.88°;
  backflips 1.55/2.35/6.26°). Abstract + Conclusion live numbers updated.
- **VI-F rewritten** to a clean baseline *match*: position parity (0.5–1.9m), the
  duration/portability hedge + "retain Huber, RANSAC future work" + own-bag bias-scaling
  para deleted. Tables V & VI updated (Ours = RANSAC; Table VI footnote flipped so
  Huber-disabled is the contrast). Conclusion limitation (3) → bidirectional design-split.
- **Backflips:** batch reframed as bistable (no single figure); SW numbers + figure
  (SW-only, `gen_paper_traj.py` HIDE_BATCH) + captions updated. Held-out + old-firmware
  numbers refreshed (kept 46–75%, no starvation; old-fw backflips ori 10.7→8.1/9.1°).
- Table VII footnote: absolute figures predate RANSAC, orderings intact.

**Page count after the RANSAC rewrite: 11** (float-bound). A subsequent **float-reduction
pass** (below) brought `paper/` to **10**.  Both papers build clean, 0 undefined refs.
Re-benchmark logs in `baselines/results/ransac_default/`.

**Float-reduction pass (paper/ only, → 10 pp).** Applied the reviewer's prioritized menu:
dropped the radar-contribution table (also audit item 1), Fig. RPE (`fig:rpe`, prose keeps
the 1.12%/0.43%/3.2% numbers), Table I datasets (→ one sentence), Table VIII prior-scale
(`tab:prior_scale`, VII-C prose keeps 0.825→1.279 + the 2.21° local max + the
predates-RANSAC caveat), the sliding-window schematic (`fig:sw_diagram`, unreferenced;
Eq. + IV-E carry it), and the error-timeseries figure (`fig:error_time`; RMSE in Table II,
trajectories in Fig. 3).  **Floor at 10 pp:** remaining floats are 3 figures (pipeline,
racing-traj, backflips-traj) + 4 tables (SW, baselines, decomp, ablations) + 24 refs
(~1.2 pp). All three final pages are full. **Reaching 8 requires cutting protected
content** (a trajectory figure, Table IV/V, the ablations table, or the reference list) —
stopped and reported per instruction rather than cut protected material.

**Audit pass (post-rewrite, same day) — pre-RANSAC absolutes that survived in prose:**
- **Dropped the radar-contribution table** (IMU-only vs RIO) → one sentence: it quoted the
  old Huber/pre-revision RIO endpoints (0.70/1.85% drift) that contradicted the headline.
  Now states IMU-only anchors (6.4/15.3% drift, 2.0/1.7 m/s) → the Table III figures
  (0.6–0.9%, 0.32–0.46 m/s), order-of-magnitude. Removes a float too.
- **VII-B b-sweep re-run under RANSAC:** 0.50→3.3→6.7 (Huber) replaced with measured
  0.39→3.2→6.6 m (b=0,−0.5,−1.0); b=0 now matches the headline.
- **Table V decomp footnote** was overrunning the table* width and clipping mid-word; shortened
  (dropped the trailing clause now in the caption). Renders fully.
- **P2 description:** "from all Doppler measurements" → "from the RANSAC inlier returns"
  (prefilter runs before P2 per Fig. 1).
- **Huber δ=1.0 rationale** restored (defense-in-depth on the surviving inliers) after the
  z-bias clause was cut.
- **Backflips:** added a clause that RANSAC is neutral there (sparse frames bypass the
  five-return floor; 1.51→1.55 m is run-to-run noise, not a regression).
- **Prior-scale table:** added a caveat that its absolutes are the inconsistent-prior
  diagnostic (predate RANSAC); the sensitivity pattern is the result. (`tab:ablations`
  already carried the analogous footnote.)

---

# paper/ — two-version split + 8-page cut (2026-06-13)

`paper/` is a copy of `report/` (the full-detail 12-page reference version, left untouched —
verified byte-for-byte identical, all 24 files). All edits here are in `paper/` only.

**Page count: 12 → 11.** Target was 8; see "Remaining overflow" below — items 1–7 are
exhausted and further reduction requires cutting protected content or floats beyond the
sanctioned list, so per instruction I stopped and report rather than cut protected content.

## STEP 1 — angle-convention clarification (III-A)
**Decision: branch 1 (Euler-pitch == physical downtilt), now numerically justified.**
Checked the extrinsic construction (`analysis/lib/radar_velocity_utils.py:814`,
`rotation_matrix_from_euler`, ZYX): `R_bs = Rz(yaw)Ry(pitch)Rx(roll)` maps sensor→body. Measured
the radar boresight (mean point direction) in the sensor frame = `[+0.99, +0.01, -0.11]` on our
bag → boresight along body forward, and computed that with boresight = sensor +x the Euler-pitch
**equals the physical boresight downtilt exactly** (27.5°→27.5°; +z would give 62.5°). So the
"27.5° as-built, a few degrees under the 30° CAD nominal" framing is numerically valid.
III-A now states the Euler order + frames explicitly and that the 25.5° nominal is an off-CAD
init the calibration overrides. Init text left as-is ("converges identically from 25.5° and
30° … data-driven, not init-dependent"); removed the "independent check" tail I'd added in
`report/` so it matches the requested plain wording.

## STEP 2 — cuts (12 → 11 pages)
1. **Dropped Fig. 8** (prior-scale sensitivity); kept Table VIII (same sweep). Repointed the
   VII-C reference from "Fig. 8" to "Table VIII".
2. **Merged Figs. 3 + 4** (slow + fast trajectory) into one float, single caption (`fig:traj`);
   updated the VI-B reference. Fig. 7 (backflips) kept separate.
3. **Consolidated IV-E ↔ VII-C overlap**: VII-C no longer re-derives the naive-marginalization
   mechanism (shared bias block / double-counting / interior conditioning) — that stays in IV-E;
   VII-C now points to IV-E and keeps only the symptom pattern + sweep (Table VIII) + resolution.
4. **Tightened VI-F**: reverse-port paragraph compressed ~27→12 lines to the design-space-split
   conclusion (kept the 99.7% gate rejection, >2 km, 64%, ~13 points; dropped ½gt², the PSD
   figure, "half of scans"). Trimmed two explanatory clauses in the forward paragraph (yaw
   surrogate; axis-redistribution parenthetical). **All numbers kept** (Huber −0.06/−0.16,
   RANSAC ≤0.10, dynamics scaling −0.019→−0.054, duration projection, Table VI RANSAC row) and
   the protective duration caveat.
5. **Tightened VII-B**: compressed the bistability and "what b really is" prose; kept the
   conclusions (b is a flight-regime proxy; pitch is a plateau; capped at −1.5) and the key
   numbers (0.50→3.3→6.7 m for b=0,−0.5,−1.0).
6. **Shrank Fig. 5** (error-over-time): regenerated as a compact **4-panel** (position +
   orientation × 2 bags) **single-column** figure (was full-width `figure*` 6-panel); velocity
   over time is fully covered by the RMSE tables. This reclaimed ~1 page (the only single full
   page recovered — text cuts alone didn't drop pages, see below). Compressed VI-D and VI-E prose.
7. **Lightly trimmed** Related Work CT-trajectory + marginalization paragraphs.

No numbers were changed; every number remains consistent with `report/`. Build clean, 0 broken
references/citations.

## Remaining overflow — why 11, not 8 (needs your decision)
The paper is **float-bound**, not prose-bound: after the cuts it carries **6 figures + 8 tables
+ 24 references** (~1 page of refs). In IEEE two-column, `[t]` floats cluster at page tops and
leave column bottoms underfull; **removing prose creates whitespace the floats don't vacate**, so
prose compression alone barely moves the page count (items 1–5 + 7 together dropped 0 pages; the
single page recovered came from de-floating Fig. 5 in item 6). Per-page fill is even
(~90–190 text-lines), so there is no near-empty page to reclaim.

Reaching 8 requires reducing **floats**, which goes beyond the sanctioned items and/or touches
protected content — your call. Options (none applied):
- Merge the three ablation-style tables — IV (IMU-only vs RIO), VII (selected ablations),
  VIII (prior-scale sweep) — into one or two, or move some rows to text (~1–1.5 pp).
- Drop or merge a figure: RPE (Fig. 6) folds into a sentence + the drift numbers already in
  text; or the backflips trajectory (Fig. 7) could be dropped (the result stays in Table III).
- Demote Table I (dataset summary) to a text sentence (~0.3 pp).
- Each ~1 pp; two or three of these would reach 8. Protected (NOT touched): IV-E, IV-B, VI-F,
  Tables V & VI, the duration caveat, VII-D negatives, all 24 references.

## report/ integrity
Verified byte-for-byte unchanged (sha256 of all 24 files identical before/after).
