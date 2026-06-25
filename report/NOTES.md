# report/ — IEEE Transactions on Robotics (T-RO) submission version

This is the **full journal version** (formerly `t-ro/`; the old conference-format
`report/` was retired into git history — the two were the same content, so they were
consolidated here). The depth is an asset at journal length, not a liability.
The lean 10-page conference cut lives in `paper/`.

Build: `latexmk -pdf main.tex` → `main.pdf` (12 pages, within T-RO's free-to-12 limit).

## Current milestone: supervisor-review draft (T-RO submission comes later)

The paper is content-complete and builds clean at 12 pages — this is the version
for **supervisor review now**. Everything in "TODO before the T-RO submission"
below (bios, ORCID, manuscript dates, supplementary video + code archive) is
**deferred to the actual T-RO submission** and does **not** block the supervisor
handoff. (Flight videos exist; they get packaged for the T-RO supplementary archive
at that point.)

## T-RO norms applied (source: ieee-ras.org T-RO Information for Authors, fetched 2026-06)
- **Class:** `\documentclass[journal]{IEEEtran}`, 10 pt, double-column Transactions layout.
- **Length:** max 20 pages; **free up to 12**, then $175/page for 13–20.
  This version is **12 pages → no page charges** (trimmed to the limit 2026-06-25).
- **Abstract:** ≤ 200 words (regular paper). Ours is **186**.
- **Index terms / keywords:** on page 1 (`\begin{IEEEkeywords}`) — present.
- **Author biography:** ≤ 100 words each; **required for the final accepted version,
  optional for initial submission**. A `IEEEbiographynophoto` placeholder is in
  `main.tex` but **commented out** for the 12-page budget — uncomment and fill in
  degrees/dates (and add a photo via `IEEEbiography`) for the camera-ready.
- **Self-contained:** the paper must be fully readable without the multimedia/code.
  A code+data availability note is in a `\thanks` footnote on page 1.
- **Supplementary material:** ≤ 50 MB zip, with `ReadMe.txt` + `Summary.txt`; submitted
  as a *separate* archive (video/dataset/code) — not part of `main.pdf`.

## What changed vs. report/
- `\documentclass[conference]` → `[journal]`.
- Conference `\IEEEauthorblockN/A` title block → journal `\author{...\thanks{...}}` with
  affiliation, manuscript-date, and code-availability footnotes, plus a `\markboth`
  running head. (`\url` inside `\thanks` must be `\protect\url` — a moving argument.)
- Abstract trimmed 270→186 words (well under the 200-word cap). All numbers/claims unchanged.
- Added an `IEEEbiographynophoto` placeholder before `\end{document}`.
- Trimmed to the 12-page limit (2026-06-25): **23 refs**, Table V single-column,
  RPE and prior-scale plots removed (data kept in-text/in-repo). See CLAUDE.md
  doc-map row for the current figure/table set.

## TODO before the T-RO submission (deferred — not needed for supervisor review)
- Fill the author biography (currently commented out in `main.tex`; optional for
  initial submission, required for camera-ready — add a photo then too).
- Set the real manuscript received/revised dates in the first `\thanks` (and the
  `\markboth` Vol/No placeholders; normally the editor fills these).
- Add ORCID iDs for both authors.
- Prepare the supplementary archive (flight video + code/dataset pointer,
  `ReadMe.txt`/`Summary.txt`, ≤50 MB zip, separate from `main.pdf`).
- Optional: T-RO accepted papers may be invited to present at ICRA/IROS/CASE.
