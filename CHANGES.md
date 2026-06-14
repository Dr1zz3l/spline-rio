# CHANGES

## Post-RANSAC audit fixes + paper/ float trim (2026-06-14)

External audit of the 10-page `paper/` found cross-table inconsistencies left by the
front-end swap + table renumber. Verified each against both dirs; fixed all valid points.

### Correctness (BOTH report/ + paper/)
- **A1** per-window solve time "0.37--0.86 s" → **0.35--0.70 s** (match SW table + abstract).
- **A2 (biggest)** ablation table showed Schur "(default)" fast settled **0.726 m** vs headline
  **0.285 m**. Re-ran the SW rows under RANSAC: marginalization None **0.412**/Schur **0.285**
  (31% benefit, was 48% Huber); window-dur slow now 1.5s 0.284/0.292, 3.0s 0.286/0.303, 5.0s
  0.307/0.336 (settled/live). Footnote rewritten to split batch (fixed pre-RANSAC operating
  point) vs SW (RANSAC default); dropped the bare "(default)" tag. Window-dur prose: slow is
  now window-insensitive, so the 3 s choice is justified by fast racing (re-ran fast@2 s →
  0.44 m live vs 0.39 m at 3 s — a modest gain, NOT the old "2 s→11°" cliff, which was a
  pre-RANSAC config).
- **A3** ori-degradation "0.3--0.4°" → **0.4--0.5°** (slow Δ0.39°, fast Δ0.53°).
- **A4** Related-Work `\ref{sec:setup}` → `\ref{sec:heldout}` (12× quantization is in held-out).
- **A5** `fig:traj_backflips` was uncited → added the `\ref` to the VII-B SW-result sentence.
- **B3** substantiate "22%" with "(0.50→0.39 m live)".
- **B4** label the prior-scale sweep numbers "live".
- **B6** IV-A clause: pitch "optimized in batch for self-calibration, locked at 27.5° for SW".
- **B2** dropped the unverified 0.30° (vs table 0.25°) → "essentially unchanged".

### paper/ float trim (paper/ only; report/ keeps all floats)
- Cut **Fig. 1 (pipeline)** — non-protected; III-B + P1–P3 prose carry it.
- Trimmed the **IMU-preintegration** and **extrinsic optimize/lock** ablation rows (restated in
  VII-D and IV-A); tightened the "Key observations" prose.
- **Outcome: still 10 printed pages.** With 24 references (~1 page) + the protected float set
  (2 trajectory figures, SW/baselines/decomp tables), the document is ~9.5 effective pages and
  rounds up to 10. A printed 9 needs a protected decision (consolidate the two ICINS tables into
  one `table*`, or trim references) — out of the approved non-protected scope.

### paper/ further reduction attempts (paper/ only unless noted)
- **Merged the two ICINS tables** (headline + decomposition) into one full-width `table*`
  with panels (a)/(b), one caption/label; cross-refs repointed. Cleaner, but did not drop a page.
- **Tightened the abstract** (BOTH dirs): dropped the inline weighting-law formula and collapsed
  the headline number-dump to bounds (vel <0.5 m/s, ori <3°, drift <1% on racing; backflips
  2.9%/6.3%); NEES → "near-calibrated". Less quantitative.
- **Trimmed 4 references** 24→20 (BOTH dirs): `sola2018micro` (notation courtesy cite),
  `lv2022ctrlvio` (redundant with CLIC, same authors), `huang2024multiradar` (multi-radar tangent),
  `anderson2015steam` (GP/STEAM, an unused approach). IEEEtran auto-drops uncited entries.
- **Outcome: paper still prints 10 pages.** Reference trim freed ~0.9k chars (page 10: 6469→5591),
  but eliminating page 10 needs ~4.8k chars (~0.6 page). It is a genuine 10-page document; a
  printed 9/8 requires dropping a protected float (a trajectory figure or the ablations table) or
  moving content to an appendix.

Both build clean, 0 undefined refs (report 12pp, paper 10pp). Re-run logs:
`baselines/results/ransac_default/{fast_nomarg,fast_win20}_ransac.log`.
Prior changelog cleared this date; full history in git.
