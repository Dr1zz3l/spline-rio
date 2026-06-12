# Paper replan (2026-06-12) — audit + new structure

Guiding question (supervisor): *"Theorie plus Durchführungsaspekte vollumfänglich;
primär verstehen wie und ob man aus Radar eine State Estimation bekommt — ist es
überhaupt möglich, und wenn ja, wie kommen wir zu guten Resultaten."*
→ The paper is a question-driven investigation, fully reproducible, no overclaims.

## Audit of the current tex

**Factual errors / overclaims found:**
1. ❌ `hug2022continuous` described as "CT sliding-window + Schur marginalization on
   2D FMCW radar, most closely related" — it is **stereo-inertial** (IROS 2022).
   Fix positioning: closest radar CT works are Burnett TRO 2024 (GP CT
   radar/lidar-inertial), LC-RIO-ET (CT IMU-modeling RIO); Hug stays as the
   closest *CT-SW-marginalization* reference (different modality).
2. ❌ The elevation-bias narrative ("b and extrinsic pitch are the same physical
   error in two parameterizations") is DISPROVED by the 4.2 joint calibration:
   racing explodes under any b even with locked pitch; backflips improves
   monotonically past the WLS-measured −0.5 → b is a flip-regime radar-error
   proxy (likely intra-frame motion), NOT an antenna-fixed bias. Pitch 27.5° is
   simply the correct mount angle (plateau 26.5–28.5; in-solver calibration
   retired). The "+2° mystery" section must be rewritten.
3. ❌ "online extrinsic pitch self-calibration" listed as contribution — retired
   (locked 27.5 beats it; fast pos −21%). Reframe as a calibration *finding*.
4. Stale numbers everywhere (tab:sw, abstract, conclusion, backflips §):
   superseded by universal config + 4.2 (see Final numbers below).
5. λ_ori_accel ablation row claims 0.1 best — under the stiff (ω-adaptive) gyro
   the regularizer is inert; row must be re-scoped to the old config or dropped.
6. Radar-contribution table + RPE figure still from pre-revision runs (footnoted).
   Regenerate with final configs.
7. Missing entirely: ω-adaptive gyro weighting (the central mechanism!), accel
   soft gate, SNR weighting, the universal weighting law, the vel↔ori gate
   Pareto, held-out validation, NEES covariance calibration, adaptive-knots
   negative result + diagnosis chain, mocap-GT-degradation caveat.

## New structure

1. **Introduction** — the question itself: what state information does a
   single-chip mmWave radar physically provide (per-point Doppler ⇒ ego-velocity;
   no bearing-stable features at 10–60 pts/frame ⇒ no position fixing), and can
   it, with an IMU, yield usable 6-DOF state estimation? Answer preview:
   yes for velocity+orientation (0.39–0.46 m/s, 2–3° live on racing); position
   is their integral and drifts 0.6–2.8 %/m — by construction, not by defect.
   Contributions (rewritten): (a) reproducible CT B-spline RIO system + full
   methodology; (b) consistent marginalization (Markov-blanket rule) finding;
   (c) dynamics-adaptive measurement weighting law (ω-adaptive gyro + ω-gates +
   SNR), one config hover→10 rad/s, validated on held-out flights; (d) NEES-
   calibrated output covariance (racing ori ≈ perfectly consistent); (e) honest
   negative results incl. the validate-before-build adaptive-knots kill and the
   b/pitch entanglement resolution.
2. **Related work** — fix Hug; add: Burnett TRO 2024 (burnett2024gp,
   arXiv 2402.06174), LC-RIO-ET (lcrioet2026, arXiv 2603.19958), Ctrl-VIO
   (ctrlvio2022, arXiv 2208.12008), multi-radar RIO (girod2023multi, arXiv
   2311.08608), 4D radar-aided INS (zhu2025robust, arXiv 2502.15452),
   Coco-LIC (lang2023coco, arXiv 2309.09808 — cited in the adaptive-knots
   negative result). Honest note: no like-for-like external baseline numbers
   (different hardware/firmware); planned future work.
3. **Sensor characterization & observability (NEW section)** — what the radar
   measures: FMCW Doppler physics, 0.049 m/s quantization (firmware-dependent:
   0.604 in the old config — quantization ablation across Dec/Mar bags), 10 Hz,
   10–60 points/frame, intensity field; elevation weakness of the 2-TX array;
   measured noise: σ_core 0.16 m/s (racing) vs 2.47 m/s (flips).
   Observability from Doppler+IMU: yaw = gauge freedom (proof sketch);
   absolute position unobservable (velocity sensor ⇒ drift); roll/pitch via
   gravity; ⇒ the system is a velocity+orientation factor source; magnetometer
   required for yaw; drift-% is the honest position metric (velocity-bias
   witness). THIS section answers "ist es überhaupt möglich" rigorously.
4. **Method** — state rep (unchanged math), measurement models + the
   **dynamics-adaptive weighting law** (new unified subsection):
   λ_gyro·(1+(|z_g|/4)⁴), radar/accel gates w=1/(1+(|ω|/ω₀)²) ω₀=4/8, SNR
   w=clip((I/I_med)^1, .25, 4), flip-regime z-proxy b=−1.5 (with the honest
   "proxy not physics" discussion); initialization P1–P3; SW + consistent
   marginalization (keep, already good); implementation details + FULL
   hyperparameter table (reproducibility) incl. solver opts, all λ, grids
   per dynamics class, analytic Jacobians note, determinism statement.
5. **Experimental methodology** — platform, datasets incl. data-health screening
   (UART crashes; windows chosen on healthy spans) and GT QUALITY: mocap
   degradation during flips (12% occlusion-masked at peaks, FD-ω clipping ~13%,
   broken tail outside eval window) ⇒ flip-time GT carries few-degree errors;
   metrics (live vs settled, drift-%, RPE; settled-vel artifact note);
   **tuning protocol + held-out split** (tuned on 3 bags, validated untouched
   on 2 Mar held-out + 5 Dec stress bags); evaluation alignment; determinism.
6. **Results** —
   - Main table (universal config; live vel/ori/pos+drift, settled, t_win):
     fast 0.41 / 3.24° / 0.50 m (1.1%) live, 0.447/2.66° settled, 0.35 s;
     slow 0.46 / 1.97° / 0.30 m (0.63%), 0.286/1.58°, 0.70 s
     (mapping variant 0.281/1.22°, yaw 0.71°);
     backflips 2.29 / 6.29° / 1.51 m (2.8%), 1.64/4.98°, ~0.6 s.
   - Held-out table (fast_nc 0.45/2.90° live beats tuning bag; circle_bv;
     Dec stress incl. Dec-backflips 10.7° untouched; circle_fast flip finding).
   - Trajectories (3 bags, regenerated), error-over-time, RPE (regen).
   - Radar contribution (regen with final config).
   - **NEES consistency (NEW)**: racing ori inflation ×0.7–0.9 (consistent!),
     vel median-calibrated w/ GT tails, backflips ×3σ overconfident; per-regime
     inflation recipe for downstream fusion.
7. **Analysis / how we got there (NEW framing for ablations)** — the diagnosis
   chain as narrative: rate-correlated residual → representation analysis
   (spline tracks flips at 16 ms to noise floor) → open-loop gyro beats solver
   → weighting, not knots → λ_gyro sweep (zero pos cost) → gates Pareto
   (vel↔ori exchange measured) → SNR weighting (fast −0.4° ori) → universal
   law + bisection lessons (p=2 too early; λh breaks fast pos; tether =
   backflips-only). Ablation tables: per-component on/off; gate Pareto table;
   b/pitch grid + racing verification.
8. **Negative results & limitations** — adaptive knots (validate-first kill:
   V0/V1 representation study), asymmetric split (locates the information
   limit; ragged petals), preintegration, GNC, full extrinsics, prior-scale
   symptom (keep — strong), bistability caution (keep), loop-geometry gap
   (radar data quality during flips), heading requirement.
9. **Conclusion** — direct answer to the supervisor's question: yes —
   velocity+orientation estimation from single-chip radar+IMU is possible and
   robust (numbers); what it took (consistency, dynamics-adaptive trust,
   characterized sensor systematics); what it cannot do (absolute position,
   yaw); calibrated uncertainty for downstream fusion; future work (baselines,
   per-chirp timestamps, intra-frame model, online placement).

## Figures (regen list)
- traj_{slow,fast,backflips} from final-config runs (gen_paper_traj.py; npz fresh)
- error_over_time (regen), rpe_drift (regen), prior_scale_sensitivity (keep)
- NEW: NEES consistency plot (per-window NEES vs χ² bands) — from nees_*.npz
- NEW (optional): gate Pareto scatter (vel vs ori, backflips)
- pipeline + SW TikZ diagrams: keep, add weighting-law block to pipeline

## Final numbers (single source of truth = CLAUDE.md tables, 2026-06-12 evening)
See CLAUDE.md "UNIVERSAL weighting config results" + ROADMAP Part 5 subsections:
universal = scale 1.0, λg 4·(1+(|ω|/4)⁴), gates 4/8, SNR α=1; per-regime extras:
grids, λh, tether/b (backflips), locked pitch 27.5 global.

## Finalization (user instruction 2026-06-12 17:02)
1. After the rewrite compiles: `latexmk -pdf IEEE-conference-template-062824.tex`
   (or pdflatex+bibtex x2), COMMIT THE PDF (it is normally untracked? check —
   commit it explicitly so it is viewable on github.com).
2. Merge all branches to main: adaptive-knots → radar-pos-split are stacked;
   merge radar-pos-split (contains everything) into main, push main.
   adaptive-knots is an ancestor — fast-forward covered.

## Revision pass 2 — Opus literature check (2026-06-12 18:03, VETTED)

User ran the compiled paper through an independent Opus literature review
(uploads/17cfe9b2-RIO_paper_revision_notes.md). My vetting against project
context: items 1,2,4,5 VALID; item 3 (external baseline) correct but DEFERRED
by user to a later phase. Actions for the next paper pass:

1. **CRITICAL — cite Ng et al., "Continuous-time Radar-inertial Odometry for
   Automotive Radars", IROS 2021, arXiv:2201.02437.** Closest prior art,
   currently uncited. REWRITE the "Continuous-time radar-inertial estimation
   is rare" sentence in Related Work (it is indefensible). Positioning
   differences to state (VERIFY against the PDF before asserting: number of
   radars used, SE(3)-vs-split spline): automotive/ground + gentle motion vs
   our single-chip IWR6843AOP + aggressive/tumbling flight; batch-over-window
   vs our consistency-analyzed Schur marginalization.
2. **HIGH — reposition the marginalization contribution.** Add lineage:
   Dong-Si & Mourikis ICRA 2012 (SW marginalization inconsistency analysis,
   intra.ece.ucr.edu/~mourikis/papers/DongSi2012-ICRA.pdf), FEJ/observability-
   constrained estimators (cite via ETH survey "Continuous-Time State
   Estimation Methods in Robotics: A Survey",
   research-collection.ethz.ch/handle/20.500.11850/637309), Ctrl-VIO (already
   cited), LIO-MARS arXiv:2511.13985 (CT knot marginalization + FEJ), Lv et
   al. CT fixed-lag smoothing T-Mech 2023 (arXiv:2302.07456). Make the precise
   claim: ours is *correct marginal construction via factor-SET closure from
   B-spline local support* (no double counting, no conditioning on interior)
   — a DIFFERENT mechanism from FEJ (which fixes linearization points to
   preserve the observability nullspace; we instead re-center the prior and
   keep only curvature). State both mechanisms and the distinction explicitly.
   Keep the prior-scale-symptom section framed as the transferable diagnostic.
3. **DEFERRED (user): external baseline.** When resumed: Doer/Trommer toolbox
   github.com/christopherdoer/rio (ekf_rio, x_rio + public datasets;
   x-RIO = natural choice), or run ours on their bags; STEAM-ICP code
   github.com/utiasASRL/steam_icp. Until then keep the honest no-comparison
   caveat but expect this objection.
4. **MEDIUM — demote to rigor/validation (not contributions):** NEES analysis
   (standard Bar-Shalom consistency checking — present as evidence the
   covariance is usable, move out of contribution bullets); adaptive
   weighting (defensible specifics only: ONE config hover→tumbling, the
   explicit vel/ori Pareto knee, held-out validation — not "adaptive
   weighting" in the abstract).
5. **Lead with the defensible core:** single-chip radar under
   aggressive/tumbling flight + the characterization of where radar breaks
   (intra-frame smearing; b = flight-regime proxy not antenna constant) +
   honest negative results + the prior-scale diagnostic.

### Bib entries to add (verify metadata at write time)
- ng2021ctrio: Ng, Choi, Tan, Heng, IROS 2021, arXiv:2201.02437
- dongsi2012consistency: Dong-Si, Mourikis, ICRA 2012
- lv2023clic: Lv et al., CT fixed-lag smoothing LIC SLAM, IEEE/ASME T-Mech 2023, arXiv:2302.07456
- liomars2025: LIO-MARS, arXiv:2511.13985
- ethsurvey: CT State Estimation Survey (ETH research collection 20.500.11850/637309)
(burnett2024gp, stironja2026rio, lv2022ctrlvio, lang2023cocolic already added.)
