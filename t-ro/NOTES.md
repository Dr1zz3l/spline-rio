# t-ro/ — IEEE Transactions on Robotics (T-RO) submission version

A journal-formatted version of the paper, built from `report/` (the full-detail
version — its depth is an asset at journal length, not a liability).

Build: `latexmk -pdf main.tex` → `main.pdf` (11 pages).

## T-RO norms applied (source: ieee-ras.org T-RO Information for Authors, fetched 2026-06)
- **Class:** `\documentclass[journal]{IEEEtran}`, 10 pt, double-column Transactions layout.
- **Length:** max 20 pages; **free up to 12**, then $175/page for 13–20.
  This version is **11 pages → no page charges**.
- **Abstract:** ≤ 200 words (regular paper). Ours is **197**.
- **Index terms / keywords:** on page 1 (`\begin{IEEEkeywords}`) — present.
- **Author biography:** ≤ 100 words each; **required for the final accepted version,
  optional for initial submission**. A `IEEEbiographynophoto` placeholder is included —
  fill in degrees/dates (and add a photo via `IEEEbiography`) for the camera-ready.
- **Self-contained:** the paper must be fully readable without the multimedia/code.
  A code+data availability note is in a `\thanks` footnote on page 1.
- **Supplementary material:** ≤ 50 MB zip, with `ReadMe.txt` + `Summary.txt`; submitted
  as a *separate* archive (video/dataset/code) — not part of `main.pdf`.

## What changed vs. report/
- `\documentclass[conference]` → `[journal]`.
- Conference `\IEEEauthorblockN/A` title block → journal `\author{...\thanks{...}}` with
  affiliation, manuscript-date, and code-availability footnotes, plus a `\markboth`
  running head. (`\url` inside `\thanks` must be `\protect\url` — a moving argument.)
- Abstract trimmed 270→197 words for the 200-word cap. All numbers/claims unchanged.
- Added an `IEEEbiographynophoto` placeholder before `\end{document}`.
- Body, figures, tables, references identical to `report/` (20 refs).

## TODO before submitting
- Fill the author biography (and add a photo for camera-ready).
- Set the real manuscript received/revised dates in the first `\thanks`.
- Prepare the supplementary archive (flight video + code/dataset pointer, ReadMe/Summary).
- Optional: T-RO accepted papers may be invited to present at ICRA/IROS/CASE.
