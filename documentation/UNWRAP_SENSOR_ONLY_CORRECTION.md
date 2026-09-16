# Doppler-unwrap correction: the published numbers used MoCap-aided alias selection

**Date: 2026-07-06. Action required from the main-branch / paper agent.**

> **RESOLVED 2026-07-07 (main branch).** Chose the **single-pass sensor-only** (IMU-path)
> variant, not two-pass, to keep the live-edge metric fully causal (two-pass is offline
> post-processing that re-unwraps against the full pass-1 trajectory). Sensor-only is now
> the **code default** in `preunwrap_radar_frames`; the legacy MoCap-aided prediction is
> opt-in via `--mocap-unwrap` / `RIO_MOCAP_UNWRAP=1`. Paper (`report/`) + `CLAUDE.md`
> updated: fast-racing live pos 0.40→0.47 m (0.88→1.04% drift), settled 0.31→0.38 m, batch
> 0.64→0.74 m, ablation rows (window-dur, marginalization) re-run, RANSAC live-pos benefit
> 20→8%, aliased-% 20.9→23.1 (fast) / 1.7→2.0 (slow), abstract "under 1%"→"around 1%",
> RPE beyond-20 m racing 1.0–1.7%→1.0–2.1%. Slow / backflips / held-out (0.37/0.90) / ICINS
> bit-identical (ICINS verified: 0/103608 pts alias). Two-pass (`--two-pass-unwrap`) retained
> as an offline option.

## The error

`report/sections/methodology.tex` states that Doppler alias unwrapping "uses
IMU-integrated world-frame velocity to select the correct alias once at frame
load". The implementation (`preunwrap_radar_frames` in
`analysis/validate_live_solver.py`) has, since 2026-03-27 (commit 3a9ba95),
preferred a **MoCap-differentiated velocity** for that prediction whenever
MoCap is available — which it is on every evaluation bag. All published
numbers therefore ran a MoCap-aided front-end step that the methodology
describes as sensor-only.

## Measured impact on the paper's (Ceres SW) numbers

Deployed per-bag configs, naive sensor-only (IMU-path) unwrapping vs
published (measured 2026-07-06 on gp-backend, identical driver code paths):

| bag | published (MoCap-aided) | naive sensor-only | delta |
|---|---|---|---|
| fast_racing live pos | 0.402 m (0.88%) | 0.469 m (~1.03%) | +17% |
| fast_racing live ori / vel | 2.89 deg / 0.319 | 2.90 deg / 0.318 | unchanged |
| slow_racing (all metrics) | — | bit-identical | none |
| backflips (all metrics) | — | bit-identical | none |
| ICINS flights | — | no unwraps fire | none |

Only the fast-racing position cells are affected.

## The fix that is now in this branch: `--two-pass-unwrap`

Pass 1 solves fully sensor-only (IMU-path alias prediction). Pass 2
re-unwraps the RAW Doppler against pass-1's own solved trajectory (velocity
+ lever arm; nearest alias) and re-solves. Rationale: pass-1's sensor-only
VELOCITY is unharmed by imperfect unwrapping (SW fast: 0.3184 vs 0.3186
m/s), and alias selection needs only velocity to +-v_max tolerance. The
pass-2 RANSAC prefilter uses a fresh seeded rng (a continued rng stream
changes borderline consensus membership; measured 2.6 cm on slow).

Validated on the GP backend (gp-backend branch, GP-18 v4): sensor-only
equals MoCap-aided to 4 digits on backflips, within 1.7% on fast, equals
pass-1 on slow.

Caveat: as implemented this is an OFFLINE/post-processing method (pass 2
re-unwraps against the full pass-1 trajectory). An online variant would
unwrap stride N against the stride N-1 live state.

## Measured two-pass numbers on this branch (2026-07-06, deployed configs)

| bag | metric | published | two-pass sensor-only | 2-digit |
|---|---|---|---|---|
| fast | live pos | 0.402 | **0.4147** | 0.40 -> **0.41** |
| fast | live ori / vel | 2.89 / 0.319 | 2.90 / 0.318 | unchanged |
| slow | all | — | bit-identical (live 0.3100) | unchanged |
| backflips | live pos / ori / vel | 1.545 / 6.35 / 2.35 | 1.569 / **6.30** / 2.45 | 1.55->1.57 / 6.35->6.30 / 2.35->2.45 |

backflips moves because pass-2 selects a different (trajectory-consistent)
alias set than the IMU pre-unwrap (604 vs 782 points); the shifts are
within that bag's measured RANSAC-draw noise floor (~12%, see gp-backend
GP-18). slow is untouched. ICINS should be re-run for completeness (no
unwraps fire there; expected unchanged).

## What the paper agent must do

1. Re-run the three deployed SW configs and the ICINS sweep with
   `--two-pass-unwrap` (it forces sensor-only pass 1 automatically; the
   exact deployed invocations are in CLAUDE.md "Running Analysis Scripts").
2. Update the affected cells per the table above (fast live pos 0.41;
   backflips small shifts) OR argue rounding/noise-floor equivalence —
   the paper agent's call with the user.
3. Update the methodology sentence to: alias selection at frame load uses
   the velocity of a first sensor-only solver pass (two-pass,
   post-processing), replacing the "IMU-integrated velocity" description.
4. Re-measure the quoted aliased-point percentages (currently 20.9% fast /
   1.7% slow — counted under the MoCap-aided path).
5. Note: the headline bags' radar config is the best_velocity chirp with
   v_max = 3.136 m/s and 0.049 m/s Doppler bins (empirically verified;
   the visualizer header's 3.84 is wrong). Any text claiming 0.63 m/s
   bins for THESE bags refers to the Dec-2025 old-firmware bags instead.

## Env for A/B testing

Sensor-only (IMU) prediction is now the **default** (single-pass). To reproduce the OLD
MoCap-aided column for A/B testing, set `RIO_MOCAP_UNWRAP=1` (or pass `--mocap-unwrap`).
The former `RIO_NO_MOCAP_UNWRAP` toggle is retired: its behavior is now the default.
