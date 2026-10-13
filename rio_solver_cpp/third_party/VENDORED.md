# Vendored header-only dependencies (2026-10-13)

Copied so the solver builds on the Jetson companion computer (and in the pixi
env) without the untracked `lie-spline-experiments/` checkout. Header-only;
licenses kept next to the sources. Do not edit; to update, copy again from
upstream and change the commits below.

| library | path | upstream commit | license |
|---|---|---|---|
| basalt-headers | `basalt-headers/include` | 86ebcf48df350d9332103971299009fc7a66d0ef (2022-11-12) | BSD-3-Clause (`basalt-headers/LICENSE`) |
| Sophus | `Sophus/sophus` | 8e1941677ef214aab0397608f42b18fbe1a19885 (2022-09-13) | MIT (`Sophus/LICENSE.txt`) |

Source on the dev host: `lie-spline-experiments/thirdparty/basalt/thirdparty/basalt-headers`
(+ its `thirdparty/Sophus`). CMake prefers these copies when the old path is absent
or `RIO_VENDORED_HEADERS=ON` (default).
