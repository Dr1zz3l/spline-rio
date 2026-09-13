# report/ — the guided-research project report (10 ECTS)

The single maintained report. It grew out of the T-RO journal draft that
Lukas reviewed (rev0, 2026-07-14; his annotated PDF is on branch
`feedback_v0`) and was rebuilt on 2026-09-03 as the project report: same
IEEEtran two-column format (kept on purpose, Lukas reviewed it that way), a
report title block instead of the journal front matter, an AI-use
declaration on page 1 (required by the chair), no page limit, no
"negative results" section, and the deployed per-return radar model
throughout. The plan and its decision record:
`documentation/report_notes/REPORT_SYNC_TODO.md`.

Build: `latexmk -pdf main.tex` -> `main.pdf`.

## Structure

`main.tex` is a thin root: preamble + `\input{acronyms}` +
`\input{sections/*.tex}` (one file per section, numbered in reading order) + bibliography.

| section | file | carries |
|---|---|---|
| I | `00_introduction.tex` | problem, research question, three contributions |
| II | `01_related_work.tex` | RIO lines: EKF, continuous time, marginalization, single-chip aerial |
| III | `02_preliminaries.tex` | B-splines in R3 and on SO(3), MAP as weighted least squares (the math intro Lukas asked for) |
| IV | `03_system_overview.tex` | state, factor graph, MAP objective, hardware, pipeline |
| V | `04_characterization.tex` | the sensor study: timing, residual distribution ladder, outliers and the two defenses, the per-return noise law, frame-shared error, aliasing, stream error budgets. Each subsection ends in its design consequence. Mirrors `analysis/notebooks/12_report_characterization_brief.ipynb` |
| VI | `05_methodology.tex` | state, measurement models, radar front end and the complete weighting (eq. comp_g/a/r), regularization, P1-P3, sliding window |
| VII | `06_experimental_setup.tex` | datasets, the one configuration, evaluation protocol |
| VIII | `07_results.tex` | live-edge results (velocity first), sensor ablation, held-out, ICINS cross-validation, NEES, estimator ablations incl. the sigma_0 ladder |
| IX | `08_limitations.tex` | statistical scope, heading reference, backflips (the torus flight), aliasing beyond v_max, fixed noise-model constants, knot density |
| X | `09_conclusion.tex` | answer to the research question, what it took, future work |

## Figures (`figures/`)

| figure | generator | input |
|---|---|---|
| `ladder.pdf` (fig:ladder) | `gen_characterization.py` | cached per-return tables of `analysis/characterize_shape_contradiction.py` |
| `law.pdf` (fig:law) | same script | same |
| `traj_combined.pdf` (fig:traj) | `gen_combined_traj.py` | `plots/<bag>/live_solver/traj_arrays_*_sw.npz` from headline runs with `--save-arrays` |
| `error_over_time.pdf` (fig:error_time) | `gen_error_time.py` | same arrays |
| factor graph, pipeline | TikZ inline in `03_system_overview.tex` | - |

Numbers: every result in section VIII comes from the 2026-09-03 report
battery at the deployed flag-free configuration
(`worklog/2026-09-03_report-battery.md`); the characterization numbers
from the brief notebook and its worklog entries.

## Conventions

- No em-dashes in prose. Velocity first (m/s), then orientation (deg),
  then position drift (% of path); absolute position only where a
  published baseline needs it.
- `\rev{}` and `revblock` are no-ops (leftovers of the review markup).
- Nissov & Alexis (arXiv 2605.01773) is deliberately not cited
  (`worklog/2026-09-02_nissov-alexis-4d-radar-characterization.md`).
- `COVER_LETTER.md` and `supplementary/` are T-RO leftovers, kept only
  for a possible future journal submission.
