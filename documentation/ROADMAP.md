# ROADMAP — Speed & Performance Improvement Plan (2026-06-11)

> Outcome of a full audit of the SW solver (`sliding_window_solver.cpp`), driver
> (`validate_live_solver.py`), docs, and a survey of 2024–2026 literature.
> Goal: real-time-capable continuous-time B-spline RIO + publishable contribution.
>
> **Headline: the three worst mysteries — constant 28 LM iterations, no-universal
> marg_prior_scale, and the 0.65s "other" time — all have specific, testable,
> mundane explanations. None require new math, iSAM2, or a rewrite.**

---

## Part 1 — Suspected bugs / correctness issues (test first)

### 1.1 Marginalization prior is *conditioning*, not *marginalizing* + double-counts
`compute_prior()` builds `res_set` from **all residuals touching marg OR boundary
blocks**, then evaluates with `parameter_blocks = eval_blocks` only → interior
window blocks are treated as **perfectly known constants** (conditioning), and
boundary-only factors with timestamps ≥ `t_start + stride` are baked into S
*and* re-added next window (double counting). The prior is therefore
systematically overconfident — which is exactly why `marg_prior_scale` must be
tiny (2e-4 applied as scale² → info ×4e-8), why there's a harmful intermediate
regime, and why no universal scale exists.

**Fix (structurally clean for B-splines):** the Markov blanket of the marg
stride-zone blocks is *exactly* the boundary set (a factor touching marg knot i
touches at most knots i..i+3; bound_ori = next N_ORI−1 = 3 knots; same for pos
with N_POS−1 = 5). Correct rule: **include only residuals touching ≥ 1 marg
block, plus the previous MargPrior residual. Nothing else.** Support is then
automatically ⊂ marg ∪ boundary ∪ bias.

**Success criterion:** one `marg_prior_scale` near **1.0** works for both racing
bags; per-bag tuning disappears; paper §"Prior Scaling Has No Universal Optimum"
becomes a fixed bug + consistency finding.

### 1.2 Live edge warm-started from stale t=0 dead-reckoning → iter ≈ 28
`solver.initialize(all_pos_cps, …)` is called once with the full P1–P3 init;
each new stride's ~60 pos CPs / ~37 ori knots enter the problem holding **P2
dead-reckoned values integrated from t=0** (drifts over the flight) adjacent to
accurate solved boundary CPs. The seam discontinuity → huge accel/snap residuals
→ LM drags the new segment across the gap **every window**. Predicts the
observed iter≈28 regardless of warm/cold start.

**Fix:** drift-correct the new segment on entry: rigid shift (pos) / left-multiply
ΔR (ori) so the init segment aligns with the solved boundary, preserving the
locally-accurate P1–P3 shape. Expected: iter 28 → ~5–10 ⇒ ~2.5–3.5× on jac+linear
solve (the dominant ~70% of wall time).

### 1.3 compute_prior() costs ~20× more than it should
Jac eval is ~29 ms per evaluation (0.81s/28), yet compute_prior costs
~0.65–0.79s — overhead is `ceres::Problem::Evaluate()` itself. After 1.1 the
res_set shrinks to ~stride-zone factors; re-measure. If still slow, assemble the
restricted dense H by calling the analytic factors' `Evaluate()` directly.
Expected: → ~10–30 ms ⇒ flat −25–35% wall time.

### 1.4 Bias prior weight bug
`w = sqrt(λ_ba·λ_bg)` applies the geometric mean to all 6 components. Harmless
while both are 10000; wrong the moment they're tuned separately. Per-component
weights needed.

### 1.5 No noise-based whitening → cond(H) ≈ 5.5e10 → slow LM
λ's are hand-tuned; gyro factors contribute ~62 500 information vs ~1e4 for
position — that ratio *is* the condition number, and ill-conditioning *is* why
LM needs many small steps. Test noise-derived weights (gyro σ≈0.005–0.01 rad/s,
accel σ≈0.05–0.1 m/s², radar σ from 0.63 m/s bin). Caution: current weights
encode unmodeled error (vibration, radar systematics) — pure whitening may shift
the MAP solution; treat as an experiment, verify accuracy on racing bags.

---

## Part 2 — Speed roadmap to real-time (0.3s/window target; currently 1.7–2.6s)

| # | Lever | Expected gain | Effort | Risk |
|---|---|---|---|---|
| S1 | compute_prior direct evaluation (1.3) | −0.6s flat (~1.4×) | 1–2 d | none |
| S2 | Live-edge warm start fix (1.2) | iter 28→~8 ⇒ 2.5–3.5× | 1–2 d | low |
| S3 | Whitening (1.5) | fewer iters, sane tolerances | 1 d | low |
| S4 | dt_pos sweep {5,10,20,40 ms} (never swept!) | n 2900→~1600 ⇒ 1.5–2.5× lin solve | hours | medium |
| S5 | Accel decimation →200 Hz (keep gyro full-rate), weights rescaled | cheaper jac | hours | medium |
| S6 | Window 3.0→2.0s *after* S1–S3 | 1.3× | free | re-test (old result used inconsistent prior) |

Compounded conservatively: ~2.0s → ~0.25–0.4s. Real-time criterion is
**throughput** (pipelined), so 0.4s solve @ 0.4s stride is also a valid claim.

**Fallbacks (only if Gate A fails):**
- GTSAM/iSAM2 — proven for this problem: LC-RIO-ET (2026, IWR6843AOP, B-spline
  IMU model, 1s window, real-time). ~3–6 weeks.
- GP motion prior (Burnett et al., TRO 2024, steam_icp): 74–139 ms/window, ≤5 GN
  iters; WNOA prior gives block-tridiagonal structure *by formulation*; gyro as
  direct ω measurement + accel preintegrated → sidesteps both the 1 kHz factor
  count and gyro-dominance conditioning. Big rewrite; last resort.

---

## Part 3 — Backflips

Own data clue: **ω-gate=4.0 at dt_ori=0.008 already beats the dt_ori=0.0008
"cliff" on orientation (1.904m/6.98° vs 1.82m/8.31°)** → radar during the flip
is poison; the dense-knot cliff is an optimizer artifact. Mechanisms, in test
order:

1. **Radar–IMU time offset** (hours, top candidate): at ω≈10 rad/s a 5 ms offset
   = 2.9° orientation error in every Doppler residual — quadratic in dynamics,
   negligible in racing. Continuous-time spline makes ∂r/∂t_d trivial. Add a
   single time-offset parameter to backflips batch.
2. **Intra-frame motion distortion**: ~30 ms frame at 10 rad/s smears 17°. Check
   if per-chirp timing is recoverable; else ω-dependent noise inflation
   σ_eff² = σ₀² + (k·|ω|)² as a *soft* gate (deployable, no per-bag config).
3. **Lever-arm calibration**: ω×r ≈ 0.8 m/s at 10 rad/s; 2 cm error → 0.2 m/s
   systematic only during flips. Optimize t_bs (3 DOF + prior) on backflips batch.
4. **Non-uniform/adaptive knot density** (bigger effort, paper centerpiece):
   dense ori knots only where |ω| high (Coco-LIC style). Kills "no universal
   dt_ori" + per-window underdetermination. LC-RIO-ET names this as open future
   work for RIO — literature gap.

Re-test SW backflips after 1.1/1.2 — the "2.56m is the fixed-lag ceiling"
verdict was reached with the inconsistent prior and stale warm starts.

---

## Part 3a — Backflips bags investigation (2026-06-12, fifth session)

**Q (user): does the Dec-2025 `backflips` bag need a yaw flip?
A: NO — definitively.** `validate_physics` (mocap GT through the forward
model, no optimizer): no-flip Doppler corr **+0.957** / RMSE 0.549 m/s;
flipped corr **−0.84** / RMSE 3.0 m/s. The Dec bag uses the same frame as the
Mar-2026 bags; `bags.yaml` stays unflipped. (The flipped SW run's "better
position" was an SE3-alignment artifact — its orientation was 32°.)

**Why the Dec bag is far worse anyway** (SW: 5.8 m/14.8° vs Mar 2.06 m/11.1°):
1. **12× coarser Doppler quantization** — measured directly from the bags:
   Dec bins 0.604 m/s (old radar fw config, v_max 4.99 ✓ correctly assigned),
   Mar bins 0.049 m/s (best_velocity config). WLS ego-velocity per frame is
   ~12× noisier → P1-P3 init 6.5 m vs 2.25 m, radar factors much weaker.
   This is exactly why the Mar "best_velocity" recordings were made.
   *Paper angle: same maneuver, two radar configs → quantization ablation.*
2. **Accel X/Y unusable**: corr vs mocap ≈ −0.07 (Z is fine, 0.91).
   Unexplained (mount/vibration that day?).
3. **Real 0.152 rad/s gyro z-bias** — verified against mocap flight data
   (residual mean 0.147 ✓); correctly handled by stationary detection.
4. **Per-day time offset**: cross-corr fits radar_imu_offset ≈ 0.073 s
   (vs global 0.135 s). Fixed via the NEW `extrinsics_overrides` per-bag
   mechanism in `bags.yaml` (+`validate_live_solver.py`); modest gain
   (6.37→5.83 m settled).

**Verdict**: Dec bag = robustness/stress dataset; Mar bag remains the
benchmark. No flip, no config error — it is simply categorically
lower-quality data.

**Time-offset config experiment on the Mar bag — negative, instructive.**
Its own cross-corr best fit (0.1067 s vs global 0.135) made position WORSE
(2.06→2.79 m) with ori ~par (9.56→9.35 live) → reverted. Cross-corr offset
estimates scatter ±20–30 ms on aggressive content (slow_racing fits 0.148,
backflips 0.107, recorded the same day). **Conclusion: offline cross-corr
cannot settle the offset for aggressive bags — the in-solver t_d parameter
(joint MAP estimation, ROADMAP Part 3.1) is the right tool, and the
per-recording variability makes it more valuable, not less.**

## Part 3b — Time-offset sweep on backflips batch (2026-06-12) — RESOLVED-NEGATIVE

Hypothesis (Part 3.1): radar–IMU time offset is the top candidate for the
backflips ori gap (28 ms × 10 rad/s ≈ 16°). Tested via batch sweeps using the
new `--set-ext radar_imu_offset_sec=X` mechanism.

**Sweep 1 — batch config dt_ori=0.0008: UNUSABLE (bistable).** Ori oscillates
8.4↔24.8° across offsets (basin-of-attraction luck), and **final cost
anti-correlates with accuracy** (2.8e4 @ 24.8° vs 2.6e5 @ 8.4°): the
1.26 DOF/constraint regime overfits with wrong global orientation. Important
negative result: at dt_ori=0.0008 neither cost nor single-run RMSE is a valid
selection signal for ANY hyperparameter.

**Sweep 2 — well-conditioned dt_ori=0.008, λ_ori_accel=0.01:**

| offset s | pos m | ori ° | final cost |
|---|---|---|---|
| 0.105 | 6.31 | 7.851 | 1.3083e4 |
| 0.115 | 6.47 | 7.843 | 1.3046e4 |
| 0.125 | 4.09 | 7.900 | 1.3100e4 |
| 0.135 | 4.94 | 7.854 | 1.3043e4 |
| 0.145 | 3.85 | 7.883 | 1.3105e4 |
| 0.155 | 3.84 | 7.864 | 1.3072e4 |

1. **Orientation is FLAT across ±30 ms (spread 0.06°)** — the offset is NOT
   the backflips orientation lever. At 1 kHz gyro + heading anchoring, radar
   timing errors are absorbed by position/velocity, not orientation (per-axis
   @0.135: roll 6.7 / pitch 6.1 / yaw 5.0 — uniform, not yaw-specific).
2. **Batch cost is flat (±0.3%) and uncorrelated with pos RMSE** → an
   in-solver t_d parameter would be weakly observable; joint estimation would
   wander. The t_d-parameter idea is shelved.
3. Position varies 3.8–6.5 m non-monotonically — mild bistability persists
   even at the well-conditioned config; single batch runs on backflips are
   noisy measurements of any config change.
4. ⚠ Position level at this config (~4–6 m) is far worse than the historical
   §6 record (1.98 m, λ_ori_accel=0.01) — possible regression or stale doc;
   flag for follow-up.

**Remaining ori-gap candidates** (Part 3): intra-frame motion distortion /
ω-dependent noise inflation (soft gate), Doppler quantization (cf. Dec-bag
ablation), adaptive/non-uniform knots.

## Part 3c — ω-dependent radar noise inflation ("soft gate") — VALIDATED (2026-06-12)

New `omega_soft_sigma` (ω₀, rad/s; both solvers): per-frame radar weight
w = 1/(1+(|ω|/ω₀)²)  ⇔  σ_eff² = σ₀²(1+(|ω|/ω₀)²), |ω| from the warm-start
spline at build time (like the hard gate), applied as ScaledLoss around Huber.

**Batch sweep** (dt_ori=0.008, λ_ori_accel=0.01; pos is bistable-noisy at this
config — ori is the stable metric):

| Config | pos m | ori ° | yaw ° |
|---|---|---|---|
| no gate | 4.94 | 7.85 | 5.03 |
| soft ω₀=2 | 3.41 | 7.11 | 4.28 |
| **soft ω₀=4** | 7.34 | **6.97** | 4.26 |
| soft ω₀=8 | 5.66 | 7.23 | 4.44 |
| hard gate 4 | **2.20** | 7.26 | 4.42 |

Soft gate beats no-gate at every ω₀ (broad optimum) and edges the hard gate on
ori WITHOUT discarding data. Hard-gate pos 2.20 ≈ the historical 1.90 → the
"pos regression" flagged in Part 3b was bistability noise that gating
suppresses (radar-during-flips is the bistability source).

**SW backflips** (BF1 config + ω₀=4): settled **2.04 m / 10.07°** (was
2.06/11.10), live **2.20 / 9.28°** (was 2.22/9.56), all axes improved
(roll 10.8→9.1, pitch 9.2→8.4, yaw 8.9→7.3), dt 0.66 s unchanged.
**New SW backflips best.** Deployable: smooth, no per-bag threshold cliff —
candidate to become a default for aggressive dynamics.

Remaining ori gap (10° vs batch 8.3°): quantization ablation (Dec bag),
adaptive knots, intra-frame per-chirp timestamps (if recoverable from TI fw).

## Part 4a — χ² consistency & the weight-trade-off ablation (documented 2026-06-12)

> Explicit record of the whitening/χ² analysis so it survives context loss.

**The χ² consistency argument (and how ours actually comes out).**
If every residual is divided by its true noise σ ("absolute whitening"), then at
the MAP solution Σ(r/σ)² ~ χ²(dof), dof = N_residuals − N_params, so the Ceres
cost (= ½Σr²) should be ≈ ½·dof and the *reduced* χ² ≈ 1. This is the standard
NEES/NIS-style model-validation tool. Caveats: Huber-robustified residuals are
not exactly χ² (evaluate on inliers); regularizers/priors enter the dof budget;
our W-runs used radar-relative weights (radar ≡ 1 instead of 1/σ_r² = 4), so
expected cost_rel = ½·dof/4.

**Measured (W2, slow_racing, λ_g=7000=datasheet, per 3 s window):**
dof ≈ 17,000 → expected cost_rel ≈ 2,100. Observed ≈ 3.5–3.8 **million**
→ reduced χ² ≈ **1,700**. The excess is gyro-dominated: per-axis gyro residuals
≈ 0.25 rad/s vs datasheet σ = 0.006 — i.e. **effective in-flight gyro noise is
~40× datasheet**, caused by vibration + spline discretization error (dt_ori=8 ms
model bandwidth), not sensor noise.

**Implications:**
1. A naive "we whiten by datasheet noise → consistent estimator" paper claim is
   FALSE on this data and a reviewer running the χ² check would catch it.
2. The hand-tuned λ_g = 4 corresponds to σ_eff = 0.5 rad/s ≈ the measured
   effective noise → the production config is *approximately whitened w.r.t.
   effective noise*. This converts "hand-tuned weights" (a paper weakness) into
   "effective-noise whitening, validated by reduced-χ²" (a strength).
3. **Data-driven weighting (new roadmap item):** estimate σ_eff per sensor from
   residual statistics at the solution (1–2 reweighting iterations) instead of
   hand-tuning. Cheap (script + reruns), principled, and generalizes per-bag.
4. **The ori-vs-pos trade-off ablation** (D→W3→W1: gyro weight ↑ ⇒ ori/vel ↑,
   pos ↓ monotonically, §"Third session results") is a publishable figure: it
   shows weight allocation controls how much orientation flexes to absorb
   radar/accel systematics (z-bias, vibration) — connect to RESEARCH_NOTES §4.

## Part 4b — Radar z-bias: attributed, modeled, validated (2026-06-12)

Investigation chain (user-driven visual check → causal attribution → model):

1. **Visual check** (multi-view plot): backflips estimate captures the gross
   donut sweep but z drops episodically; loop circles attenuated.
2. **Tether attribution**: with `lambda_pos_init_prior=0` (tether off), the
   raw solver z error is a near-constant **−0.57 m/s sink → −8 m** (the
   episodic pattern in the tethered run = tether fighting the sink,
   corr 0.89 between profiles). Tether also degrades ori (10.07° vs 6.93°).
3. **Model**: per-point antenna-fixed elevation bias, new config
   `radar_zbias_fixed` (both solvers): `v_corr = v_meas − b·u_sensor.z()`.
4. **Discrimination sweep** (tether off): b ∈ {+0.5, 0, −0.5, −1.0} produces a
   clean LINEAR family of z-error curves; **b = −1.0 flattens the sink**
   (z err final +0.20 m vs −5.47 uncorrected). Antenna-fixed model confirmed
   for backflips; note |b|≈2× the WLS-level −0.5 figure (soft-gate
   down-weighting during flips means the correction acts through a fraction
   of the radar influence).
5. **Full config result (backflips SW: tether 10 + soft gate 4 + b=−1.0)**:
   settled **1.989 m / 9.22°**, live **2.138 m / 8.67°** — new best
   (was 2.04/10.07, 2.20/9.28); settled pos now ≈ the old batch ceiling
   (1.82), live ori below 9° for the first time.
6. **Racing interaction — do NOT apply b there**: fast batch pos degrades
   monotonically 0.758 → 2.56 → 5.39 m for b = 0 → −0.5 → −1.0. Cause: racing
   runs with `optimize_pitch_only` ON and the solver-calibrated pitch
   (25.5°→27.6°, the long-standing "+2° mystery") **already absorbs the
   z-bias for level flight** — explicit b double-corrects. Backflips locks
   extrinsics and tumbles through attitudes where pitch cannot absorb it →
   explicit correction needed. **The +2° pitch calibration and the radar
   z-bias are the same physical error in two parameterizations.** Paper
   material: entanglement of extrinsic pitch, elevation bias, and flight
   regime; principled fix = joint b+pitch calibration on level segments.

Config guidance: `radar_zbias_fixed=-1.0` for backflips-style (locked
extrinsics, aggressive attitude); 0.0 for racing-style (pitch self-calibrates).

## Part 4c — σ_eff residual statistics (2026-06-12, `--residual-stats`)

New driver flag: per-sensor residuals vs the SOLVED trajectory; robust σ
(1.4826·MAD) vs full std; residual-vs-time npz saved to
`plots/<bag>/residual_stats_<bag>.npz` (= adaptive-knot placement criterion).

| sensor | slow σ_core / std | backflips σ_core / std | current λ ⇔ σ |
|---|---|---|---|
| gyro (rad/s) | 0.131 / 0.467 | 0.156 / 4.19 | 4 ⇔ 0.5 |
| accel (m/s²) | 0.66 / 8.40 | 9.48 / 91.7 | 0.01 ⇔ 10 |
| radar (m/s) | 0.161 / 0.612 | 2.47 / 2.07 | 1 ⇔ 1 |

**Findings:**
1. **Hand-tuned λ = full-std whitening.** Both λ_gyro (σ=0.5 ≈ slow full std
   0.467) and λ_accel (σ=10 ≈ slow full std 8.4 / bf core 9.5) match the
   outlier-inclusive std — the manual tuning implicitly calibrated to
   heavy-tailed effective noise. This explains the whitening experiments
   (W1–W3): raising λ toward the 3–15× tighter robust cores hurts because
   gyro/accel tails then do quadratic damage — **gyro has NO robust loss
   (pure L2)**. Actionable: Huber on gyro (δ ≈ 2–3σ_core ≈ 0.3–0.5 rad/s)
   may unlock core-level weighting safely. Untested.
2. **corr(|r_gyro|, |ω|) = +0.94 on backflips (+0.60 even on slow racing).**
   The orientation-spline model error is almost perfectly rate-correlated —
   the strongest quantitative motivation for ADAPTIVE KNOT SPACING (knots
   where |ω| is high), and the npz time series is the placement signal.
3. Radar residual core on backflips is 15× slow racing (2.47 vs 0.16 m/s)
   even with the soft gate — validates rate-dependent down-weighting and
   bounds what radar can contribute during flips.

## Part 4 — Accuracy + publication

- **Radar z-bias parameter**: v_corr = v_meas + b_z·u_z (or
  elevation-proportional) — attacks the known −0.5 m/s elevation systematic that
  costs fast_racing ~7.5° roll.
- **Per-point noise from intensity/SNR** — fields already loaded; standard in
  recent RIO.
- **Publication gaps**: (a) Doppler-only continuous-time RIO under aggressive/
  acrobatic flight (10 rad/s backflips + Vicon GT — unpublished regime);
  (b) marginalization consistency for spline knots (1.1 as a finding);
  (c) adaptive knot density for Doppler RIO. **Must-have**: external baselines
  (EKF-RIO Doer & Trommer; x-RIO; steam_icp GP-RIO) on own dataset. Dataset
  itself is a workshop/dataset-track candidate.

---

## Sequence & gates

```
Week 1   D1 diagnose live-edge staleness; D2 compute_prior fix; D3 warm-start fix
Week 2   M1 Markov-blanket marginalization fix → scale sweep at ~1.0; M2 whitening
         GATE A: full SW benchmark. ≤0.5s/window → stay on Ceres; else iSAM2/GP spike.
Week 3   B1 time-offset (backflips); B2 ω-noise inflation; B3 lever arm;
         S4/S5 dt_pos + accel-rate sweeps
Week 4+  GATE B: backflips batch ≈ ceiling at dt_ori=0.008 → adaptive-knot design.
         EKF-RIO baseline; z-bias parameter; paper restructure.
```

## Part 1 — EXPERIMENTAL RESULTS (2026-06-12)

All fixes implemented in `sliding_window_solver.cpp` behind flags (default ON):
`marg_markov_blanket`, `warm_start_align` (`--set ...=0` for legacy A/B).
Plus: per-component bias prior weights (1.4, both solvers), PSD-projection
eigendecomposition square root of S (replaces fragile LLT), `cost0` (initial
cost) printed per window.

### Confirmed root causes

1. **The old prior pulled in EVERY IMU factor of the window via the bias
   block** (restricted Jacobian: 24,651 rows ≈ full window) — conditioning on
   interior + double counting. With Markov-blanket rule: ~3,100 rows
   (stride-zone only). `compute_prior` "other" time: 0.74 → **0.41 s**.
2. **Live-edge seam**: initial cost ≈ 5e13 every window (min-snap amplifies a
   CP kink by 1/dt⁴). Warm-start alignment reduces cost0 to ~3e8 (the residual
   is WLS-velocity mismatch at the seam — collapses in 1 LM iteration).
3. **iter≈28 is NOT the seam**: LM trace shows iterations 15–28 are a
   curved-valley traversal of soft modes (|step| grows 0.02→0.58, Δcost→0) —
   per-window yaw re-derivation against stiff gyro chain + weakly-observable
   absolute position. Conditioning/valley issue (1.5), not warm start.
4. **PSD failures**: consistent prior S is genuinely low-rank (rank 9/30 at
   1e-6·λmax threshold — gyro-stiff ori directions); plain LLT intermittently
   failed; eigendecomposition square root fixes it. ‖r‖²prior ≈ 0–25 at
   solution → prior consistent with new data (was 2M–5.5T before!).

### slow_racing (settled / live, 76 windows)

| Run | Config | pos m | ori ° | live pos | live ori | yaw° | dt s |
|---|---|---|---|---|---|---|---|
| baseline | legacy, scale 1e-7 (tuned) | **0.226** | 1.57 | 0.332 | 2.08 | 0.93 | 2.28 |
| A2 | fixes, scale 1e-7 | 0.837 | 1.56 | 0.995 | 2.08 | 0.91 | 2.02 |
| B | fixes, **scale 1.0** | 0.435 | **1.22** | 0.606 | 2.14 | 0.79 | 1.93 |
| E0 | fixes − align, scale 1.0 | 0.324 | 1.25 | 0.443 | 2.21 | 0.83 | 2.00 |
| E | B + pos tether λ=0.5 | 0.435 | 1.23 | 0.607 | 2.14 | 0.79 | — |
| C | B + max_iter 12 | 0.307 | 4.21 | 0.325 | 4.08 | 4.22 | 0.89 |
| G | C + λ_heading 5 | 0.288 | 2.16 | 0.305 | 2.35 | 2.01 | **0.82** |
| H | B + iter 16, λh 5 | 0.290 | 2.27 | 0.308 | 2.48 | 2.12 | 1.18 |
| I | **B + iter 12, λh 10** | 0.287 | 1.63 | **0.303** | **1.92** | 1.37 | 0.86 |

H vs G: more iterations do NOT improve yaw at λh=5 (plateau) — yaw is
heading-weight-limited, not iteration-limited. I (λh=10) drops yaw 2.0→1.37°
and brings live ori below baseline (1.92 vs 2.08).

Settled velocity: baseline 0.890 (stride-jump artifact) → B/E0 **0.205** —
the consistent prior eliminates the stride-boundary position jumps entirely
(accel RMSE 190 → 28 m/s²). G roll/pitch (0.76/0.58) beat baseline (1.04/0.59).

### fast_racing (settled / live, 50 windows)

| Run | Config | pos m | ori ° | live pos | live ori | dt s |
|---|---|---|---|---|---|---|
| baseline | legacy, scale 2e-4 (tuned) | 0.726 | **3.19** | 0.829 | **3.65** | ~1.7 |
| D | fixes, **scale 1.0** | **0.596** | 3.35 | **0.697** | 4.00 | 1.37 |
| J | D + iter 12, λh 5 | 1.551 | 3.47 | 1.709 | 4.86 | 0.63 |
| J2 | D + iter 12, λh 10 | 1.553 | 3.08 | 1.711 | 4.34 | 0.61 |

**Iteration cuts are dynamics-dependent**: fast_racing at iter12 fails
(roll/yaw 7–9°) regardless of heading weight — its convergence valley is
longer. Gentle dynamics tolerate iter12; aggressive dynamics need full
convergence (or a structural fix: yaw/roll gauge pre-alignment per window).

### Conclusions

- **marg_prior_scale=1.0 now works universally** (was catastrophic): per-bag
  scale tuning eliminated; the "no universal optimum" negative result in the
  paper is explained as a fixable inconsistency (conditioning + double count).
- **Speed**: 2.28 → 0.82 s/window (2.8×) at iter12+λh5 with better live pos.
  Remaining gap to 0.3 s: ~2.7×.
- **Trade-off**: settled position slow_racing 0.226→0.29–0.44. The legacy
  baseline benefited from an *accidental* absolute-position tether (new CPs
  entered holding raw P2 dead-reckoned world positions, re-injected every
  window). E0 shows align-off recovers some settled pos (0.32) at full iter.
  An explicit tether (`lambda_pos_init_prior=0.5`) had no effect — needs a
  proper sweep (λ ∈ {5, 50}) if settled pos matters.
- **Yaw is the remaining iteration bottleneck**: at iter12, yaw 4.2°→2.0°→1.37°
  via λ_heading 0.6→5→10 (weight-limited, not iteration-limited; H=G at iter16).
  Caveat: λh=10 is calibrated against clean MoCap yaw — with a real
  magnetometer this would inject noise; the principled fix is closed-form
  per-window yaw gauge pre-alignment (rotate window state by mean heading
  residual before solve — kills the dominant curved-valley mode and should
  make iteration cuts safe for aggressive dynamics too).
- compute_prior is now ~0.1–0.2 s of "other"; full direct evaluation (1.3)
  still available but no longer dominant.

### Recommended configs (current state)

```bash
# slow/gentle:  0.86 s/window, live 0.303 m / 1.92° (beats legacy live on both)
--set marg_prior_scale=1.0 --set max_iterations=12 --set lambda_heading=10.0
# fast/aggressive:  1.37 s/window, settled 0.596 m (beats tuned legacy)
--set marg_prior_scale=1.0
```

### Next steps — RESULTS (2026-06-12, second session)

**1. Yaw gauge pre-alignment — implemented (`yaw_prealign`, default off).**
Closed-form Rz(Δψ̄) rotation of the window state about the boundary-anchor
position; exact gauge direction (accel/gyro/radar invariant). Results:

| Run | Config | settled pos/ori | live pos/ori | yaw° | dt |
|---|---|---|---|---|---|
| P2 (slow) | iter12, λh **0.6** + prealign | 0.551/1.67 | 0.615/2.66 | **1.54** | 0.87 |
| C (ref) | iter12, λh 0.6, no prealign | 0.307/4.21 | 0.325/4.08 | 4.22 | 0.89 |
| P1 (fast) | iter12, λh 10 + prealign | 1.486/2.72 | 1.634/3.97 | 7.25 | 0.61 |

P2: prealign fixes yaw at the *default* heading weight (4.22→1.54° — important
for magnetometer deployment where λh=10 would inject noise), but costs settled
position (0.31→0.55): the per-window Δψ̄ estimate is noisy and the rigid
rotation jitter random-walks position. Needs damping (apply k·Δψ̄, k≈0.5, or
only when |Δψ̄|>threshold) before production use.
P1: prealign does NOT rescue fast_racing at iter12 (roll 7.6° — the failure is
roll/overall convergence under high dynamics, not the yaw gauge).

**2. function_tolerance knob — implemented.** P3 (fast, ftol=1e-5, full cap):
0.601/3.38, dt 1.32, iter 26.5 — identical accuracy to D at ~equal cost. Fast's
iterations are genuine progress, not tail grind; adaptive stopping saves
nothing. Fast speedup requires conditioning work (whitening, 1.5).

**3. Settled-pos tether — plumbing bug found + fixed, then found inert.**
`cfg.lambda_pos_init_prior` was NEVER plumbed from Python into C++ (all
historical SW runs, incl. documented backflips Phase 3, ran with tether=0;
fixed at `validate_live_solver.py:382`; correction note added to
SW_DEVELOPMENT §7). With the working tether, λ ∈ {5, 50} on the I-config is
still inert — measured: the solved trajectory stays within ~2–3 cm of
`init_pos_cps_` on slow_racing (tether cost ≈ +20 of ~3000 total), so there is
nothing for the tether to pull on. The "P2 init drifts meters" premise was
wrong for slow_racing; the legacy 0.226 vs new ~0.29–0.31 settled-pos gap
(~7 cm) is NOT a tether effect and is currently unexplained — candidates:
legacy's overconfident prior acting as accidental extra smoothing, or
iter/convergence differences. Re-examine only if mapping accuracy below 0.3 m
matters.

**4. Mapping config found (M1)**: align=0, scale=1.0, λh=10, full iterations →
settled **0.314 m / 1.116°, yaw 0.556°** — best orientation of ANY run
(legacy: 0.226 m / 1.57°, yaw 0.93°), settled vel 0.198. dt 2.06 s (offline OK).

### Recommended configs (updated)

```bash
# LIVE / real-time (slow-to-moderate dynamics): 0.86 s/window
#   live 0.303 m / 1.92° (beats legacy live on both axes)
--set marg_prior_scale=1.0 --set max_iterations=12 --set lambda_heading=10.0

# LIVE aggressive dynamics: 0.45 s/window (dt_pos sweep 2026-06-12; iteration
# caps unsafe here — leave max_iterations at default, natural count is ~11)
--set marg_prior_scale=1.0 --set dt_pos=0.04

# MAPPING / settled (offline): 2.06 s/window
#   settled 0.314 m / 1.12° (yaw 0.56° — best of all runs)
--set marg_prior_scale=1.0 --set lambda_heading=10.0 --set warm_start_align=0
```

### Third session results (2026-06-12): whitening, prealign damping, backflips

**Whitening (1.5) — RESOLVED: it's a trade-off dial, not a speed lever.**
Datasheet-derived weights relative to radar (σ_r=0.5 m/s → λ_r≡4 baseline):
gyro σ=0.006 rad/s → λ_g≈7000, accel σ(vibration)≈1.5 m/s² → λ_a≈0.11.

| Run | Config | settled pos/ori | live pos/ori | iter | dt |
|---|---|---|---|---|---|
| D (ref) | fast, λg=4, λa=0.01 | 0.596/3.35 | 0.697/4.00 | 28.4 | 1.37 |
| W3 | fast, λg=400, λa=0.11 | 0.730/2.61 | 0.808/**2.96** | 23.4 | 1.21 |
| W1 | fast, λg=7000, λa=0.11, λh=200 | 1.323/2.28 | 1.459/2.58 | 20.5 | 0.87 |
| B (ref) | slow, λg=4 | 0.435/1.22 | 0.606/2.14 | 27.4 | 1.93 |
| W2 | slow, λg=7000, λa=0.11, λh=200 | 1.442/1.36 | 1.557/2.12 | 18.6 | 1.25 |

Raising gyro weight toward its statistically-correct value monotonically
improves orientation/velocity (W3 live ori −26%, live vel −28%) and cuts
iterations modestly (28→23→19) — but monotonically degrades position (the
stiffer gyro stops absorbing radar/accel inconsistencies; ftol also fires
earlier, before the position valley is traversed). Full whitening is wrong in
practice because quadrotor accel "noise" is structured vibration, not white.
**Verdict:** keep λg=4 for position-priority; λg≈400 is a legitimate
orientation-priority operating point; iteration count is NOT primarily a
weighting artifact → fast-racing real-time needs the Part-2 structural levers
(dt_pos sweep, window size, compute_prior direct eval) instead.

**Prealign damping (gain=0.5) — no position recovery.** PD: 0.539/1.85
(yaw 1.72) vs undamped P2 0.551/1.67 (yaw 1.54), vs no-prealign C 0.307.
The ~0.24 m position tax comes from the rotation mechanism itself, not the
Δψ̄ noise level. Prealign remains a niche tool (yaw fix when the heading
weight cannot be raised, e.g. cold-start with yaw-gauge init error); not a
default. Possible future variant: rotate ori knots only.

**Backflips SW re-test (consistent prior + working tether λ=10) — MAJOR WIN.**
```
--set dt_ori=0.008 --set lambda_ori_accel=0.001 --set lock_gyro_bias=0 \
--set marg_prior_scale=1.0 --set lambda_pos_init_prior=10
```
| | Phase 3 (doc) | New (BF1) | batch ceiling |
|---|---|---|---|
| Settled pos | 2.56 m | **2.06 m** (−20%) | 1.82 m |
| Live pos | 3.33 m | **2.22 m** (−33%) | — |
| Settled ori | 10.87° | 11.10° | 8.31° |
| Live ori | 9.33° | **9.56°** | — |
| dt | — | **0.66 s** (iter 20) | — |

The old "2.56 m is the fixed-lag ceiling" verdict was an artifact of the
broken (double-counting) prior + dead tether. SW backflips now sits 13% above
the batch position ceiling, at 2× real-time headroom vs the 0.3 s stride.

### Plan audit (2026-06-12, post-whitening) — what changed

- ~~S3 whitening as a speed lever~~ → RESOLVED-NEGATIVE: iteration count on
  fast is intrinsic (real convergence), not a weighting artifact. Fast
  real-time must come from **per-iteration cost reduction**, i.e. smaller n
  (dt_pos), smaller window, cheaper compute_prior — not fewer iterations.
- The Part-2 window-size blocker ("fast needs 3.0 s, all shorter windows give
  9–15° yaw", §9b) was measured with the BROKEN prior — must be re-tested with
  the consistent scale=1.0 prior, which now actually transfers inter-window
  information. Same for the dt_pos question (never swept at all).
- NEW item from χ² analysis: data-driven effective-noise weighting (σ_eff from
  residual statistics, 1–2 reweighting passes) — replaces hand-tuning with a
  defensible procedure; likely lands near current weights for gyro, may
  *improve* accel/radar allocation.

### dt_pos / window sweep results (2026-06-12, fourth session)

fast_racing, scale=1.0, full iteration cap (vs reference D: dt_pos=5 ms, win 3.0 s):

| Run | dt_pos | win | settled pos/ori | live pos/ori | iter | dt s |
|---|---|---|---|---|---|---|
| D | 5 ms | 3.0 | **0.596**/3.35 | **0.697**/4.00 | 28.4 | 1.37 |
| F1 | 10 ms | 3.0 | 0.708/3.06 | 0.765/3.68 | 22.2 | 0.88 |
| F2 | 20 ms | 3.0 | 0.700/3.01 | 0.765/3.55 | 16.6 | 0.61 |
| **F6** | **40 ms** | 3.0 | 0.701/**3.13** | 0.803/**3.55** | **11.0** | **0.45** |
| F3 | 5 ms | 2.0 | 1.026/3.98 (roll/yaw ~11°!) | 1.157/4.39 | 27.8 | 0.78 |
| F4 | 10 ms | 2.0 | 0.933/2.96 (roll/yaw ~8°) | 1.048/3.46 | 22.0 | 0.54 |

slow_racing (λh=10):

| Run | dt_pos | iter cap | settled pos/ori | live pos/ori | dt s |
|---|---|---|---|---|---|
| I | 5 ms | 12 | **0.287**/1.63 | **0.303**/1.92 | 0.86 |
| F5 | 20 ms | 12 | 1.305/1.01 ⚠ pos blown | 1.451/1.73 | 0.80 |
| F7 | 20 ms | full (~16) | 0.334/**1.17** | 0.424/1.98 | 1.03 |

**Findings:**
1. **dt_pos was massively over-dense for fast_racing.** 5→40 ms: position
   plateaus at ~0.70 m (+17%, all incurred between 5→10 ms), orientation
   *improves* (3.35→3.13), iterations drop 28→11 (smaller, better-conditioned
   problem), wall time 1.37→**0.45 s** (3×). F6 breakdown: jac 0.12, lin 0.20,
   other 0.12 → compute_prior direct evaluation (next item) brings ~0.35 s,
   i.e. real-time at stride 0.35–0.4 s, or 1.5× off the 0.3 s stride.
2. **The §9b "fast needs 3.0 s window" blocker is CONFIRMED even with the
   consistent prior** (F3: roll/yaw ≈ 11°). It is a genuine
   observability/dynamics limit, not a prior artifact. Window stays 3.0 s.
3. **Iteration caps interact with grid density**: slow@20 ms needs its natural
   ~16 iterations; capping at 12 explodes position (F5 vs F7). Caps must be
   set ≥ the natural count for the chosen grid (or left at 40 since the
   natural count already drops with coarser grids).
4. slow_racing live recommendation stays I-config (5 ms, iter12, λh10);
   fast_racing switches to dt_pos=40 ms.

### compute_prior direct evaluation — DONE (2026-06-12, `marg_fast_prior`)

Replaced `ceres::Problem::Evaluate()` (rebuilds Program + Evaluator per call)
with direct `CostFunction::Evaluate()` over the res_set, accumulating the
dense GN Hessian with exact `Problem::Evaluate` semantics: SO(3) tangent
columns via manifold PlusJacobian, Triggs corrector for robust losses
(α-term only when ρ″>0), out-of-set blocks fixed. Default ON
(`--set marg_fast_prior=0` for the legacy path).

**Verification: bit-identical RMSE** to the legacy path on both configs
(V1 ≡ run I, V2 ≡ F6 to 4 decimals). Timing:

| Config | before | after | components after |
|---|---|---|---|
| fast, dt_pos=40 ms | 0.447 s | **0.374 s** | jac 0.10, lin 0.18, other 0.087 |
| slow, 5 ms, iter12, λh10 | 0.86 s | **0.82 s** | jac 0.31, lin 0.32, other 0.18 |

fast_racing is now within 25% of the 0.3 s stride → **real-time at a 0.4 s
stride on a laptop, today**. Remaining "other" is Ceres problem construction
(~6 000 AddResidualBlock per window) + Python-side slicing/snapshots — next
lever would be incremental problem reuse (diminishing returns).

### Next up (in execution order)

1. Radar–IMU time-offset parameter on backflips batch — top accuracy candidate
   for the 11° vs 8.3° ori gap; trivial Jacobian via continuous-time spline.
2. Data-driven effective-noise weighting script (residual-stats σ_eff).
3. Backflips: dt_pos sweep (never swept there; likely similar win);
   tether sweep λ_pos_init ∈ {3, 30}; z-bias radar parameter; settled-pos 7 cm
   gap (low priority).
4. (speed, only if needed) incremental problem construction / window 2.5 s
   probe for fast.

## Part 5 — Adaptive knot spacing: Phase-0 validation KILLED the hypothesis (2026-06-12, branch `adaptive-knots`)

Plan: generalize the orientation spline to non-uniform knots (Coco-LIC style),
dense during flips. Phase 0 (`analysis/adaptive_knots/`) validated assumptions
BEFORE any C++ work — and the go/no-go gate fired **NO-GO**.

### V0 — basis math (PASSED, library kept)
`nonuniform_bspline.py` + `v0_basis_check.py`: non-uniform cumulative SO(3)
basis (Cox–de Boor, ghost-knot boundaries) is machine-exact vs scipy AND
reproduces basalt's `cumulative_blending_matrix_` bit-exactly for uniform
grids. Two transferable findings:
1. **Greville abscissae**: CP j corresponds to time (τ_{j+1}+τ_{j+2}+τ_{j+3})/3
   ≈ τ_j + 2dt, NOT τ_j. Sampling CPs at knot times lags the whole curve by
   ~2dt (10.5° on a synthetic 7 rad/s signal). Any init/fitting code must
   sample at Greville times. ACTION ITEM: check whether the solver's Python
   init (`_base_rotations`) has this lag — if yes, fixing it improves
   warm-start quality during high-ω at zero cost.
2. **Hard density transitions create angular-accel transients**
   ~|δ_coarse|/dt_dense² (13,000 rad/s² for 32→4 ms; geometric ramp cuts 7×).
   ω stays C¹ (verified by h-scaling), but any future non-uniform grid must
   ramp density geometrically.

### V1 — representation-error go/no-go (FAILED the GO gate)
Fitting the SO(3) spline directly to backflips MoCap (`v1_representation_error.py`):
uniform **8 ms already represents the flips to 1.28° RMSE** (vs the 8.3–9.2°
solver error); adaptive equal-budget improves only 1.26× (gate required ≥2×).
The flips peak at ~11 rad/s with 0.7 s period — slow relative to 8 ms knots.

### V1b–d — what the rate-correlated residual actually is
(`v1b_gyro_analysis.py`, `v1c_followup.py`, `v1d_gyro_only.py`)
- **ω-tracking is grid-independent**: spline fit to the gyro stream gives ω
  residual RMS ≈ 0.295 rad/s FLAT across dt = 16/8/4 ms with **flip ≈ quiet**
  (0.294 vs 0.292), near the measured >25 Hz vibration floor (0.17 rad/s,
  rate-INDEPENDENT). The Part-4c corr(|r_gyro|,|ω|)=+0.94 was measured against
  the SOLVED trajectory → it is the **optimizer failing to rotate, not the
  spline failing to represent**. (Supporting: solved-residual full std 4.19
  rad/s ≈ the <25 Hz flip-segment signal RMS — the solution barely tracks the
  flips at all in the worst segments.)
- **Open-loop gyro dead-reckoning beats the solver**: zero-bias integration
  over the clean 17 s window = **7.80° RMSE vs solver 9.2°** → the fusion
  (radar+accel+tether) is net-NEGATIVE for orientation on this bag. The
  orientation lever is weighting/robustness during flips, not knot density.
- **MoCap GT is degraded exactly during flips**: broken tail t>17.2 s
  (48,758 rad/s FD spikes, 116–126 ms dropout gaps; reported RMSE unaffected —
  eval window ends before it), plus 12% occlusion-masked samples concentrated
  at flip peaks. Interpolation through them clips FD-ω peaks ~13%, which
  masquerades as a gyro scale error (regression says −13%, but applying it to
  dead-reckoning explodes 7.8°→118° → artifact). Implication: (a) gyro scale
  cannot be calibrated against this mocap; (b) during-flip mocap orientation
  itself likely carries few-degree smoothing/lag errors — part of the "9°"
  may be GT error. Paper caveat material.

### Verdict & pivot
**Adaptive/non-uniform knots: NO-GO for the backflips ori gap** (representation
is not the limiter; racing bags never needed it). C++ Phases 1–3 cancelled.
Phase-0 cost: one day, zero solver changes — the validate-before-code gate
worked exactly as designed. Keep: `analysis/adaptive_knots/` library + V0
findings + `placement.py` (if a future budget-reduction use-case appears, e.g.
coarse knots on long quiet segments for compute, not accuracy).

**Pivot candidates for the ori gap (in suggested order):**
1. **λ_gyro sweep on backflips** — DONE, see below.
2. **Huber on gyro** (currently pure L2; σ_core 0.156 vs std 4.19 — open
   thread from Part 4c, small C++ change).
3. **Accel soft-gate during flips** (mirror omega_soft_sigma onto the accel
   factor: thrust transients + lever-arm/vibration make accel poison during
   flips; the gravity-direction information it provides is ~nil mid-flip anyway).
4. Greville init check (V0 finding 1) — CONFIRMED in code:
   `validate_live_solver.py` samples init rotations at knot times
   (line ~1018) and `from_rotation_samples` treats them as knots → init curve
   lags ~2·dt_ori (≈9° during flips at dt_ori=8 ms); the SW position tether
   similarly pulls toward a ~3·dt_pos-lagged position init. Untested fix.
5. Paper: mocap-GT-degradation caveat for the backflips table.

### Pivot result: λ_gyro sweep on backflips SW (2026-06-12) — VALIDATED, new best

Full SW config (tether 10 + soft gate 4 + z-bias −1.0 + scale 1.0) + λ_gyro:

| λ_gyro | settled pos/ori | live pos/ori | per-axis r/p/y | dt/window |
|---|---|---|---|---|
| 4 (ref) | 1.989 / 9.22° | 2.138 / 8.67° | ~9.1/8.4/7.3 | 0.66 s |
| 40 | 1.982 / 7.45° | 2.130 / 7.78° | 5.96/5.85/4.17 | 0.70 s |
| **400** | **1.988 / 7.15°** | **2.137 / 7.62°** | 5.63/5.46/3.75 | 0.61 s |
| 1000 | 1.988 / 7.12° | 2.135 / 7.60° | 5.61/5.43/3.73 | — |

- **Monotone-saturating ori improvement, −2.1° settled / −1.05° live,
  position EXACTLY flat (1.98–1.99 m), timing flat** — unlike the racing
  W-runs where λ_gyro=400 cost ~0.13 m position. With the soft gate already
  down-weighting radar during flips, stiffening the gyro has no position
  cost on backflips. λ_gyro=400 = knee; new backflips SW operating point.
- Settled ori 7.15° beats the old batch ceiling (8.31°) and approaches the
  Part-3c batch best (6.97°) while holding 1.99 m position (batch position
  is 2.2–7.3 m bistable at that config).
- Batch + λg=400 single run landed in a bad basin (8.13°/7.1 m) — consistent
  with Part 3b: backflips batch single runs are not a valid comparison;
  the SW result (deterministic, 4-point monotone, all axes) is the evidence.
- Interpretation chain (V1b–d → here): the optimizer was under-rotating
  during flips because accel/radar residuals outweighed gyro fidelity;
  open-loop gyro = 7.8° bounded what pure integration gives; λg=400 fusion
  (7.12° live-capable with heading anchors) now slightly beats open-loop.
  Remaining ~7° floor candidates: accel distortion mid-flip (pivot #3),
  gyro-vs-mocap GT error during flips (V1d: mocap is degraded exactly there
  — the floor may be partly GT artifact), Greville init lag (#4).

## Key references

- LC-RIO-ET: Radar-Inertial Odometry with Online Spatio-Temporal Calibration via
  Continuous-Time IMU Modeling — arxiv.org/abs/2603.19958
- Burnett, Schoellig, Barfoot: Continuous-Time Radar-/Lidar-Inertial Odometry
  using a GP Motion Prior, TRO 2024 — arxiv.org/abs/2402.06174,
  github.com/utiasASRL/steam_icp
- Coco-LIC (non-uniform B-spline, adaptive knots) — arxiv.org/abs/2309.09808
- LIO-MARS (non-uniform CT trajectories, real-time) — arxiv.org/abs/2511.13985
- CT-UIO (adaptive knot span) — arxiv.org/abs/2502.06287
- Ctrl-VIO (FEJ for spline knots in SW) — arxiv.org/abs/2208.12008
- Chen et al., IROS 2023: Optimization-based VINS: Consistency, Marginalization,
  FEJ — pgeneva.com/downloads/papers/Chen2023IROS.pdf
- Impact of Temporal Delay on RIO — arxiv.org/abs/2503.02509
