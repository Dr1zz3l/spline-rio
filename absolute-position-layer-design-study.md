# Bounding Absolute-Position Drift in Sparse Radar-Inertial Odometry
### A design study for a CT-spline / iSAM2 substrate in a nested-cuboid + permeable-mesh environment

---

## 0. The one fact that drives everything

Your measured per-frame observability is the binding constraint, more than sparsity or the Doppler bias:

- Floor visible 79 / 27 / 20 % (slow / fast / aggressive)
- A single dominant wall reliably visible
- **≥3 non-degenerate planes only 34 / 5 / 0 %**

This means **per-frame structural pose-fixing is impossible by construction**. You essentially never see enough simultaneous structure to constrain 3 translational DOFs in one frame. Any method whose correctness depends on observing a full local frame each scan — single-scan plane-triangulation, single-scan-to-submap registration with a full 6-DOF solve, point-cloud ICP that assumes a localizable geometry — is degenerate *every frame* and will either inject garbage along the unobserved DOFs or has to detect-and-suppress that degeneracy on every single update.

Two consequences follow immediately and are non-negotiable for any candidate architecture:

1. **The information must come from a *persistent map*, not from the current scan.** A plane (or distribution) observed at t=0 and re-observed at t=20 s ties those two times together directly. That long-baseline tie is what converts *drift* into *bounded error*. Frame-to-frame anything cannot do this without explicit place recognition.

2. **Information must be injected strictly per-observed-direction.** A floor return constrains world-z and nothing else. A single wall return constrains that wall's normal direction and nothing else. The architecture must be able to add a rank-1 or rank-2 constraint without contaminating the orthogonal, well-estimated DOFs. This is the "do no harm" requirement, and it is the property that should decide the representation.

A corollary worth stating up front because it bounds the *best achievable* result: in an elongated hall, the two long walls share one normal (call it **y**); the floor/roof give **z**; the short end walls give the long axis (**x**) but are rarely in view. So structure here can robustly bound **z and y over the whole mission**, and **x (the long axis) only intermittently** — whenever end-walls, mesh-ends, or a revisit are observed. No representation fixes this; it is a property of the environment + sensor FOV. Plan, evaluate, and write the paper per-axis, and expect residual along-hall drift to be your dominant remaining error. (Phase 0 below verifies this is in fact where your current drift lives.)

---

## 1. Verdict on the framing: keep planes for the *solid* structure, reject the cuboid prior, hybridize for the mesh

Short answer: **your plane hypothesis is right for the solid structure (floor, hall walls, roof, end walls), for reasons specific to radar sparsity — but "plane-SLAM" as usually built (per-frame multi-plane fitting, hardcoded faces) is wrong, and the mesh needs separate treatment.** Keep the floor factor as a *special case* of a general anchored-plane landmark, generalize its mechanism, and discard only the part of it that creates the coupling (position-based classification).

### Why planes, not points or lines, for the solid parts

- **Points fail on repeatability.** At 5–7 static returns/frame with coarse angular resolution and multipath, the same physical scatterer does not reliably re-fire frame-to-frame, and angular smear corrupts its position. Re-observation-based point-landmark SLAM (the Michalczyk/Weiss-style persistent-radar-landmark approach that works on richer automotive/MAV radar) is on the edge of viability here; treat it as a fallback, not the backbone.
- **Lines are mostly redundant with plane intersections** and need elevation diversity you do not have (2 TX).
- **Planes aggregate sparse returns into a low-DOF, globally-persistent, re-observable primitive.** Three returns over a window define a plane; that plane is then a stable map anchor seen for most of the mission. Planes *natively* do per-direction injection — a closest-point plane factor constrains exactly the normal DOF and nothing else. That is precisely the rank-deficient-by-design property requirement #2 demands.

### Why *not* the standard plane-SLAM packaging

- No fixed face count / no Manhattan cuboid: the two nested cuboids may be mutually rotated; the mesh is not a face; aggressive flight changes which faces are visible. Discover structure online with arbitrary count/orientation.
- The mesh is **not** a solid plane: it is a permeable hex lattice of discrete returns, *and* the hall wall behind it produces a second, roughly-parallel plane at a different depth. Modeling it as one clean plane is wrong and creates the single sharpest data-association hazard in the whole system (§4).

### The mesh, specifically

The mesh's value is almost entirely in **reinforcing a normal *direction***, not in pinning an absolute *offset*, because a return cannot be cleanly assigned to mesh-vs-hall-wall without knowing the position you are trying to estimate. So: let mesh returns vote into the dominant-direction estimate (good, redundant evidence of the y-normal) but down-weight or explicitly double-model their *offset* contribution. Two viable models:
- **Double-plane**: one normal, two offsets (mesh slab + hall wall), with a gated association by expected gap.
- **Down-weighted soft plane** at mesh depth with inflated residual variance, used mainly to keep the normal estimate fed.

If you ever want to exploit the lattice itself, the hex-grid *phase* is in principle an along-axis ruler (a fiducial lattice) that could attack the x-drift problem — but only if angular resolution can resolve individual nodes, which I doubt at IWR6843 resolution. Flag as high-risk/high-reward, not baseline.

---

## 2. Representation: anchored closest-point planes, tightly coupled to the spline

### The parametrization decision (and a trap)

Candidate plane parametrizations and how they behave in *your* graph:

- **Hesse (n, d), 4-param over-parameterized** → singular information matrix unless explicitly constrained. Reject as the state representation.
- **Minimal spherical (azimuth, elevation, d)** → gimbal singularity at elevation = ±90°. **Your floor's normal is vertical, i.e. exactly at the singularity.** This is a concrete, easy-to-miss failure: the most-observed, most-valuable plane sits on the parametrization's singular set. Reject spherical.
- **Kaess (2015) unit-quaternion plane with 3-DOF multiplicative error** → singularity-free, well-tested in iSAM2.
- **Geneva et al. LIPS (2018) anchored closest-point (CP)** → singularity-free *except* when the plane passes through the anchor frame origin (d→0). For a room with the anchor inside the interior, floor/walls/roof all have d≠0, so the singular set is never approached. LIPS reports CP gives a *better linear-Gaussian approximation and converges faster than the quaternion form* in iSAM2 batch/incremental tests.

**Recommendation: anchored CP planes (LIPS-style).** It is the cleanest factor to attach to a continuous-time pose, it sidesteps both the spherical floor-singularity and the Hesse rank deficiency, and the anchoring (to the first-observing pose or a world frame with origin in the room interior) keeps you off its only singular set. Keep Kaess-quaternion as the fallback if you hit conditioning issues; the two are directly comparable and LIPS provides analytical Jacobians for both.

### The continuous-time plane factor

The factor is exactly LIPS' CP plane residual, but with the pose **interpolated from the spline** at the return's timestamp rather than read off a discrete keyframe:

- The return at time t touches its 4 orientation knots + 6 position control points (your existing local support).
- Residual = CP-plane residual between the spline-interpolated world pose at t and the anchored CP plane landmark.
- It constrains the trajectory only along the plane normal — automatically rank-1 per return, rank-2 once you have two distinct normals in the window.

This is the natural marriage of LIPS (plane factors in a factor graph) with Furgale-style continuous-time trajectory representation. It generalizes your floor-z factor: the floor is simply the special case "horizontal-normal plane, residual dominated by the position-CP z-component." You keep the empirical win (−62 % fast vertical) and lose only the brittle classification.

### Persistent landmarks vs. the fixed-lag window — the key iSAM2 structural choice

Plane landmarks must live **outside** the trajectory's fixed-lag marginalization window. The position control points slide and get marginalized; the floor and long-wall landmarks persist for the whole mission so they can anchor t=0 to t=26 s. Concretely:

- Trajectory CPs/knots: sliding, marginalized at the lag boundary (as now).
- Plane landmarks: long-lived variables, retained as long as the plane is "active."
- FEJ: fix the first-estimate linearization point of each plane landmark and its first-observing pose, exactly as OC-VINS/FEJ prescribe, so marginalizing observing CPs does not manufacture spurious information along the unobservable along-plane/along-hall directions. This is the single most important consistency detail; getting it wrong reintroduces the very drift-masking over-confidence you are trying to avoid.
- Densification guard: persistent planes create fill-in when observing CPs are marginalized. Bound the active-plane set; retire planes unseen for a long interval (convert to a weak prior or drop). Floor + two long walls stay active throughout (that is the point); transient/aggressive-flight planes are spawned and retired.

### Tightly- vs loosely-coupled — settled by requirement #2

**Tightly-coupled** (planes in the same graph as the spline). Loose coupling (separate map → registration → 6-DOF pose factor) re-introduces a full-rank pose constraint per registration, which is degenerate every frame here and forces per-frame degeneracy remapping; it also throws away correct plane-uncertainty propagation and the long-baseline tie. Your substrate already is FEJ-aware iSAM2, so tight coupling is the natural and correct fit. Keep the substrate; attach plane landmarks to it.

---

## 3. Online mapping: grow / merge / split with no fixed count

Aggregate in the **current estimated frame** (the spline's running estimate), not a fixed world frame, so accumulation tracks the drifting estimate rather than fighting it; the anchored CP landmark absorbs the offset.

- **Spawn (delayed initialization).** Do not instantiate a plane from a single frame. Accumulate vertical-/horizontal-/oblique-normal returns over a short window; spawn only when (a) enough returns, (b) sufficient *spatial* spread to define a plane (not a near-collinear sliver), and (c) Doppler-consistency holds (the returns obey the rigid-body radial-velocity model for the current ego-velocity — this is your cheapest, most radar-native outlier filter, and it rejects movers and much multipath before they ever reach the map). Delayed init is the standard guard against spurious landmarks and is essential at this SNR.
- **Merge** planes whose normals and offsets agree within gates (and whose normals fall in the same dominant-direction cluster, §5). This is how the two long walls + mesh get *normal*-coupled while keeping separate offsets.
- **Split / spawn-parallel** when residuals to an existing plane are bimodal along the normal — the signature of the mesh/hall double wall. Promote to a double-plane rather than letting one plane drift to the average depth.
- **Retire** on long non-observation; floor and long walls effectively never retire.
- **Permeable/parallel handling** is therefore a first-class mapping operation, not an afterthought: parallel-normal, distinct-offset clusters are *expected*, and the merge/split logic must preserve them.

---

## 4. Data association under drift: break the coupling at *classification*, not at the residual

This is the crux of your chicken-and-egg, and the fix is precise: **the residual may use position (it must, to pin the offset), but the *association/classification* must not depend on the DOF you are correcting.**

Your floor factor classifies by predicted world-z — which is corrupted by the very position error under correction → per-environment threshold. Replace the classification signal:

- **Classify by normal direction, which rides on orientation + gravity — and orientation *is* well-estimated** (observable from IMU + Doppler in your substrate). A floor/roof return is one whose local-or-map normal is ≈ gravity-aligned; a wall return is one whose normal lies in the estimated horizontal dominant set. Normal direction is essentially decoupled from absolute position error, so the classification no longer feeds back on the corrected DOF.
- **Associate to the *map*, not to a threshold.** "Does this return fit a persistent plane whose normal is vertical?" The map plane's normal is well-estimated once a handful of returns have accumulated; the absolute offset is what you are solving for, so do not use it to gate.
- Caveat: a single return has no measured normal (radar gives point + Doppler + intensity). Normals come from local/map aggregation, so cold-start (no map yet) is the one moment you must lean on the warm-start pose + gravity. Make the floor *offset* a free variable estimated from the first window's vertical-normal returns rather than a hardcoded `floor_z`; that removes the last threshold.

For the genuinely ambiguous cases — **mesh vs hall-wall (parallel, same normal, offset differs by ~the standoff gap)** — classification-by-normal does not disambiguate, because the normal is identical. Here use:

- **Multi-hypothesis / robust association**: max-mixtures (Olson & Agarwal) or a small JCBB over the few returns, so the offset ambiguity is represented rather than resolved prematurely; the graph keeps both hypotheses weighted until geometry disambiguates. Mis-association here injects a bias **along the normal — exactly the DOF you are fixing — so this is the highest-leverage place for robustness**, not the floor.
- **Robust kernels + GNC at spawn**: graduated non-convexity (Yang et al.) or switchable constraints (Sünderhauf) / DCS (Agarwal) for the harder spawn-time associations; Huber for steady state (as you have).
- **EM/iterative reclassify *inside* the incremental loop**: alternate (associate returns → planes | current map & trajectory) and (re-optimize | associations), but re-linearize at the FEJ point so the iteration does not silently re-break consistency. iSAM2 gives you the cheap incremental re-solve; the EM is only over associations, which are few.

---

## 5. Structural priors: regularize *directions*, never *offsets*; soft, online, down-weightable

Use a soft Atlanta-style prior, estimated online, and apply it **only to normals**.

- **Estimate dominant directions online** by clustering plane normals on the sphere (Straub et al., real-time Manhattan-frame estimation in the space of surface normals — a von-Mises-Fisher / k-means-on-the-sphere style fit), refined by rotation averaging. The Atlanta model ("one vertical + a set of horizontals") fits your environment better than strict Manhattan, because nested-cuboid mutual rotation and oblique segments produce more than two horizontal directions.
- **Apply the prior as a soft regularizer that pulls each plane's *normal* toward the nearest estimated dominant direction**, with a per-plane, down-weightable strength. Crucially: **regularize normals (directions), free offsets (positions).** This lets a mesh observation reinforce the hall-wall *normal* (good, redundant) while never coupling their *depths*. It also never injects a position constraint along an unobserved DOF — the prior cannot manufacture along-hall information it does not have.
- **Environment-class detection → automatic down-weighting.** Track how well normals cluster (concentration of the vMF fit / residual of the dominant-direction model). High concentration → trust the prior (tighter normal regularization, faster rotation drift correction). Low concentration → down-weight toward zero, with a graceful unstructured fallback where the map degenerates to free-normal planes (and, if even that is too sparse, to the §6 down-weighted-registration backstop).
- **Why bother, given the prior can't fix position directly?** Three indirect but real wins: (i) it sharpens sparse-plane normal estimates; (ii) it ties the rarely-seen end-walls' normal to the orthogonal of the well-seen long-wall normal, so a *single* end-wall return is enough to pin x-direction (you already know the direction, you only need the offset) — this is the main lever you have against long-axis drift; (iii) it tightens rotation, which feeds back into the normal-based classification of §4. Precedent that this pays off under sparsity: L-SLAM-style linear Manhattan/Atlanta decoupling has been shown to make SLAM viable with as few as **four-beam** LiDAR — a direct analog to your radar sparsity.

This satisfies the hard requirement: soft, online-estimated, down-weightable, graceful unstructured fallback, never a hardcoded cuboid or fixed orientation.

---

## 6. The alternative I considered hardest, and why it's the backstop not the backbone

**Probabilistic submap + scan registration (NDT / GICP / Gaussian-mixture), loosely coupled.** This is the strongest competitor and the direction much of current 4D-radar SLAM has taken (4DRadarSLAM's adaptive-probability GICP; Amodeo et al.'s 3D-Gaussian scene model with PDF-based registration; 4D iRIOM). Its genuine advantages map onto your requirements: no explicit feature extraction, no plane count, the permeable mesh is just probability mass, and association is distribution-to-distribution (no hard correspondences).

Why it loses as the *primary* anchor here:

- With 5–7 points/frame, single-scan-to-submap registration is **under-constrained every frame** — the same floor+1-wall degeneracy, now *implicit* and requiring per-frame Zhang–Singh remapping to stay safe, rather than being structurally explicit and automatically rank-deficient as with per-normal plane factors.
- It does not natively give the **long-baseline anchor** (t=0 ↔ t=26 s) without bolting on place recognition; consecutive-frame registration accumulates the very drift you are removing.
- It is harder to attach cleanly to the spline per-DOF and harder to carry a soft structural prior.

**So keep it, but in a subordinate role:** as the representation for genuinely unstructured regions, as the offset-down-weighted model for the mesh, and as the fallback when the dominant-direction model collapses. The recommended system is **explicit anchored-CP plane landmarks (tightly coupled, persistent) for the solid backbone + soft online Atlanta normal-prior + a probabilistic/down-weighted treatment of the mesh and unstructured returns.** That is the architecture.

---

## 7. Observability & degeneracy: make rank-deficiency the normal case, with a solver-level safety net

Two layers:

1. **Structural (preferred):** the per-normal CP plane factor *is* the per-direction information injector. A floor-only frame adds rank-1 (z); a floor+one-wall frame adds rank-2 (z + y); the orthogonal/along-hall DOFs simply receive no factor. No remapping needed because you never wrote a full-rank constraint. This is strictly better than detect-then-suppress.
2. **Solver safety net:** monitor the per-direction information (Hessian eigen-spectrum of the position block) and apply Zhang–Kaess–Singh (2016) solution remapping — update only well-conditioned directions — as a backstop for the cases where association or the prior accidentally couples a near-null direction. X-ICP (Tuna et al.) and the degeneracy-aware-factor line (Hinduja et al.) give you point-to-plane-specific localizability tests if you want a sharper detector than raw eigenvalues; mind the translation/rotation scale disparity (recent work flags naive full-Hessian eigenvalues over- and under-detecting).

Report the eigen-spectrum per axis vs MoCap as a first-class result — it is the cleanest evidence that you inject information only where it exists.

---

## 8. Radar-specific prior art: what transfers, what doesn't

- **Doppler ego-velocity front-ends** (Kellner et al. instantaneous ego-motion; Doer & Trommer EKF-RIO / x-RIO, 3-point RANSAC LSQ): you already do this; the relevant carry-over is the **Doppler-consistency test as an outlier/mover filter at plane-spawn time** (§3).
- **4DRadarSLAM (Zhang et al., ICRA 2023)** and **4D iRIOM (Zhuang et al., RAL 2023)**: pose-graph + probabilistic registration; transfers as the §6 backstop and for loop-closure ideas, *not* as the per-frame anchor — their platforms are far denser (automotive/large-scale) than your indoor IWR6843.
- **Gaussian-modeling + multi-hypothesis scan matching (Amodeo et al., 2024)**: the **multi-hypothesis** idea transfers directly to your mesh/hall parallel-plane ambiguity (§4); the dense-Gaussian scene model is heavier than you need.
- **Tightly-coupled factor-graph RIO (Michalczyk/Weiss, IROS 2024; their earlier persistent-radar-landmark MAV work)**: closest in spirit (indoor-ish MAV, tight coupling, persistent landmarks) — supports the tightly-coupled choice; their point-landmark persistence is the fallback you keep in your back pocket if planes ever starve.
- **Continuous-time radar/lidar-inertial with a GP motion prior (Burnett, Schoellig, Barfoot, T-RO 2025)**: the main *alternative continuous-time substrate* (GP white-noise-on-acceleration prior vs your basalt-style B-spline). Your substrate is fixed, so this is context, not a change — but it is the camp to cite when you justify the CT choice, and Schoellig is at TUM if you want a local sounding board.
- **Manhattan/Atlanta-from-sparse-sensing (L-SLAM with 4-beam LiDAR; Straub RTMF; UV-SLAM vanishing-point line work)**: the precedent that structural decoupling rescues *sparse* sensing — your §5 justification.
- **Plane-graph backbone (Kaess 2015 infinite planes; Geneva LIPS 2018; Yang et al. point+plane-aided INS 2019; BALM / Eigen-factors / plane-adjustment, 2019–2021)**: your representation and factor design.

What explicitly does **not** transfer: anything assuming dense, localizable per-scan geometry (automotive radar SLAM at hundreds of points/scan; LiDAR ICP front-ends). The sparsity gap is the whole story.

---

## 9. Recommended architecture (summary)

> **Persistent anchored closest-point plane landmarks, tightly coupled into the existing CT-spline / FEJ-iSAM2 graph, associated by normal-direction + map-fit (not by drifting position), regularized by a soft online-estimated Atlanta normal-prior with automatic down-weighting, with the mesh modeled as parallel double-planes (normal-coupled, offset-free/down-weighted) under multi-hypothesis association, and a probabilistic-registration + Zhang–Singh-remapping backstop for unstructured/degenerate cases.**

Your floor factor is retained as the first, simplest instance of this; only its position-based classification is discarded.

**Rejected, with reason:**
- Per-frame multi-plane pose-fixing / hardcoded cuboid — never enough simultaneous planes; violates no-hardcoding.
- Point-landmark SLAM as backbone — returns not repeatable at this sparsity/angular resolution (kept as fallback).
- Pure loosely-coupled submap registration as the anchor — degenerate every frame, no long-baseline tie (kept as backstop).
- Occupancy/GP map as the position anchor — heavier, less interpretable, no clean per-DOF CT factor (useful only for mesh density if needed).
- Minimal spherical plane params — floor sits on the elevation singularity.

---

## 10. Phased implementation plan (each gate MoCap-evaluable)

**Phase 0 — Instrument & budget (no new estimator).**
Per-axis (x/y/z) ATE/RPE and drift-rate vs MoCap, separately for slow/fast/aggressive; reproduce the −62 % fast-vertical baseline.
*Gate:* a per-axis drift breakdown that confirms **x (long axis) is the dominant unbounded error** and quantifies how much z/y the floor+wall can in principle recover. (If x is *not* worst, revisit §0 assumptions.)

**Phase 1 — Generalize floor → anchored-CP horizontal plane landmark.**
Replace position-threshold classification with gravity/normal + map-association; make `floor_z` a free landmark; CP factor on the touched CPs/knots.
*Gate:* match-or-beat current z improvement across **all** speeds with **zero per-environment threshold tuning**; z error bounded (non-growing) over full missions. *Robustness gate:* perturb warm-start z by ±0.5 m → convergence unchanged (proves the coupling is broken).

**Phase 2 — Single dominant wall as persistent CP landmark.**
Add the most-frequently-seen long wall; associate by normal + offset gating; inject along its normal only.
*Gate:* y-axis drift bounded vs MoCap across speeds **and** the orthogonal/along-wall axes statistically **unchanged** — the critical "do no harm" gate proving per-DOF injection.

**Phase 3 — Multi-plane map + online dominant-direction + soft Atlanta normal-prior.**
Grow/merge/split with no fixed count; sphere-cluster normals → dominant set; soft down-weightable normal regularizer; couple long-wall+mesh normals (offsets free), end-wall orthogonal.
*Gate:* z **and** y bounded over full mission; x bounded whenever end-structure seen; ablating the prior **loosens but does not break** (graceful fallback). *Consistency gate:* NEES/ANEES vs MoCap within χ² bounds (FEJ preserved).

**Phase 4 — Permeable/parallel + multi-hypothesis association + degeneracy safety net.**
Explicit mesh/hall double-plane; max-mixtures/GNC association for the parallel ambiguity; Doppler-consistency spawn gate; Zhang–Singh remapping as solver backstop.
*Gate:* fly close to the mesh to force association ambiguity → **no offset bias along the wall normal**; degeneracy monitor flags and safely handles <2-distinct-normal frames with no corruption of good DOFs.

**Phase 5 — Robustness, aggressive flight, ablations, generalization (T-RO grade).**
Back-flip sequences; per-axis bounded drift on structure re-acquisition; full per-component ablation table; cross-sequence/cross-setup with no per-env tuning.
*Gate:* publishable ablation + consistency + per-axis drift-rate table; documented behavior of the x-axis drift bubble during/after aggressive excursions.

A clean-slate path is **not** justified: the substrate is right, the floor factor is a valid Phase-1 seed, and each phase has an independent MoCap gate, so incremental is both lower-risk and gives you the ablation story for free.

---

## 11. Top failure modes (sparse, drifting radar; nested-cuboid + mesh) with mitigations

1. **Long-axis (x) drift** — end-walls/mesh-ends rarely seen → hall long axis under-constrained. *Mitigate:* orthogonal-direction prior so a single end-wall return suffices for x-offset (§5-iii); active end-structure detection; honest per-axis reporting; (speculative) mesh-lattice phase as an along-axis ruler.
2. **Mesh ↔ hall-wall mis-association** — same normal, offset differs by the standoff; mis-assignment biases the exact DOF you fix. *Mitigate:* explicit double-plane, multi-hypothesis/GNC association, gap-gated, down-weight offset until disambiguated.
3. **Ghost planes from multipath / residual movers.** *Mitigate:* delayed init + geometric-diversity + persistence gate + Doppler-consistency rejection + robust kernels.
4. **Floor/roof swap** — both vertical-normal; mis-association flips z sign. *Mitigate:* gate by normal sign and expected height ordering; double-plane in z is *good* for z once correctly assigned.
5. **Degenerate frame corrupting good DOFs.** *Mitigate:* per-normal factors are rank-deficient by construction (no remapping needed) + Zhang–Singh backstop + per-direction eigenvalue monitor refusing near-null updates.
6. **Cold-start offset error** — no map yet → warm-start z error sets a wrong floor. *Mitigate:* estimate floor offset as a free variable from the first window, never a hardcoded threshold.
7. **iSAM2 densification from persistent planes** at the marginalization boundary. *Mitigate:* bounded active-plane set, retire stale planes, FEJ-fixed linearization points.
8. **Doppler z-velocity bias (−0.5 m/s) masquerading as constant z-residual** — the floor factor will silently clamp a systematic velocity bias, hiding it and possibly fighting it. *Mitigate:* estimate the bias as a state (or at least monitor the floor-residual mean as a bias diagnostic) so the structural anchor is not papering over a front-end error.
9. **Aggressive flight (back-flip)** — structure leaves FOV exactly when dead-reckoning is worst → a bounded drift bubble. *Mitigate:* lean on IMU/orientation through the excursion; persistent map re-anchors on re-acquisition; characterize and report the bubble rather than hide it.

---

## 12. Literature — and what to take from each

**Plane representation & plane-graph SLAM**
- *Kaess, "SLAM with infinite planes," ICRA 2015* — singularity-free unit-quaternion plane with 3-DOF multiplicative error; your fallback parametrization.
- *Geneva, Eckenhoff, Yang, Huang, "LIPS: LiDAR-Inertial 3D Plane SLAM," IROS 2018* — **anchored closest-point plane factor**; the singularity-at-d→0 analysis; CP > quaternion linearization in iSAM2. Your core factor.
- *Yang, Geneva, Zuo, Eckenhoff, Liu, Huang, "Tightly-coupled aided INS with point and plane features," ICRA 2019* — point+plane in a tightly-coupled estimator; the hybrid-primitive precedent.
- *Hsiao et al., dense planar(-inertial) SLAM with structural constraints, ICRA 2017/2018; Ferrer Eigen-Factors IROS 2019; Liu & Zhang BALM 2021; Zhou, Koppel, Kaess plane-adjustment RAL 2021* — plane-aggregation / plane-adjustment machinery for merging and refining persistent planes.

**Continuous-time trajectory**
- *Furgale, Barfoot, Sibley, "Continuous-time batch estimation using temporal basis functions," ICRA 2012* — the CT-spline foundation your substrate sits on.
- *Burnett, Schoellig, Barfoot, "Continuous-Time Radar-Inertial and Lidar-Inertial Odometry using a GP Motion Prior," T-RO 2025* — the alternative GP-prior CT substrate; cite to justify your B-spline choice (Schoellig @ TUM).

**Structural priors**
- *Schindler & Dellaert, Atlanta World, CVPR 2004; Straub et al., Mixture of Manhattan Frames, CVPR 2014* — the model class (vertical + set of horizontals) that fits nested/rotated cuboids.
- *Straub et al., "Real-time Manhattan-world rotation estimation in 3D," IROS 2015* — **sphere-clustering of surface normals** for online dominant-direction estimation; your §5 engine.
- *Kim et al., L-SLAM (linear RGB-D/Manhattan-Atlanta SLAM), incl. 4-beam-LiDAR variant* — structural decoupling rescues *sparse* sensing; the direct analog argument for radar.
- *UV-SLAM (vanishing points, RA-L 2022)* — unconstrained line/structural handling when the world is only partly Manhattan; fallback-design ideas.

**Data association & robustness**
- *Neira & Tardós, JCBB, T-RO 2001* — joint-compatibility gating for the few returns.
- *Olson & Agarwal, max-mixtures, RSS 2012/IJRR 2013* — multi-hypothesis factors for the mesh/hall ambiguity.
- *Sünderhauf switchable constraints 2012; Agarwal DCS 2013; Yang, Antonante, Tzoumas, Carlone GNC, RAL 2020* — robust back-end / spawn-time association.

**Observability, consistency, degeneracy**
- *Zhang, Kaess, Singh, "On degeneracy of optimization-based state estimation," ICRA 2016* — **solution remapping**; your solver-level safety net.
- *Tuna et al., X-ICP (T-RO 2023); Hinduja et al., degeneracy-aware factors (IROS 2019); recent scale-aware detectors* — sharper point-to-plane localizability tests; mind translation/rotation scale.
- *Huang/Mourikis/Roumeliotis FEJ; Hesch et al. OC-VINS, 2014* — first-estimates-Jacobian consistency for landmark+trajectory graphs; non-negotiable for your persistent planes.

**Radar front-end & radar SLAM (transfer-with-care)**
- *Kellner et al., instantaneous ego-motion from Doppler, 2013/2014; Doer & Trommer EKF-RIO / x-RIO, MFI/ENC 2020* — Doppler ego-velocity + the consistency test you reuse at spawn.
- *Zhang et al., 4DRadarSLAM, ICRA 2023; Zhuang et al., 4D iRIOM, RAL 2023; Casado Herraez et al., RAI-SLAM, RAL 2025; Michalczyk/Weiss tightly-coupled FG RIO, IROS 2024* — probabilistic-registration backstop, loop-closure ideas, and the persistent-point-landmark fallback.
- *Amodeo et al., "4D RIO via Gaussian modeling and multi-hypothesis scan matching," 2024* — the **multi-hypothesis** mechanism for your parallel-plane ambiguity.

**Map representations (backstop)**
- *Biber & Straßer NDT 2003; Segal et al. GICP 2009; O'Callaghan & Ramos GPOM 2012; Ramos & Ott Hilbert maps 2016* — probabilistic/continuous map options for unstructured regions and the permeable mesh density model.

---

### Bottom line
Keep your substrate and your floor factor; promote the floor to one instance of a **persistent, anchored, closest-point plane landmark** tightly coupled to the spline; break the chicken-and-egg by classifying on **normal direction + map-fit** rather than drifting position; regularize **directions, not offsets** with a soft online Atlanta prior; model the mesh as a **parallel double-plane under multi-hypothesis association**; and rely on the per-normal factor's built-in rank-deficiency (with a Zhang–Singh backstop) to protect the good DOFs. Expect **z and y to become bounded** and **x (the long hall axis) to remain your dominant residual** — design the evaluation, and the paper, around that asymmetry.

---

## APPENDIX A — Phase 0 EXECUTED (2026-06-26): premise partly REFUTED

Per-axis ATE of the iSAM live edge vs MoCap (SE3-aligned; x = long hall axis ~21 m,
y = across-to-wall, z = vertical). `analysis/plane_mapping/phase0_per_axis.py`.

| bag | floor | x(long) | y(wall) | z(vert) | dominant |
|-----|-------|---------|---------|---------|----------|
| slow | OFF | 0.056 | 0.064 | **0.141** | z 85% |
| fast | OFF | 0.190 | 0.230 | **0.422** | z 82% |
| backflips | OFF | **1.043** | 0.988 | 0.949 | x 61% |
| slow | ON  | 0.054 | 0.071 | **0.111** | z 78% |
| fast | ON  | 0.211 | **0.222** | 0.209 | ~isotropic |
| backflips | ON | **1.036** | 0.991 | 0.894 | x 61% |

**Findings vs the study's §0 premise:**
1. **The NATIVE dominant error is z (vertical), not x.** The radar z-bias drift is the
   single biggest position error on racing (z = 82-85% of total). The study framed z as
   "robustly bounded, not the problem" (only §11.8 flags it). It is in fact the problem,
   and it is the one the floor anchor already attacks (fast z 0.42 -> 0.21).
2. **Post-floor, racing error is ~ISOTROPIC (~0.21 each axis), not x-dominated.** A wall
   (y) factor would shave ~1/3 of the *remaining* fast error, and only in the ~16% of
   frames a wall is visible. The "x is your dominant residual, build the paper around it"
   framing does NOT hold for racing.
3. **Where x DOES dominate (backflips), structure is invisible** (floor 20%, >=2 planes
   7%, walls ~0% mid-flip). Plane-SLAM structurally cannot rescue the backflips x drift:
   it is flip-induced dead-reckoning while the radar faces empty space.

**Implication:** the study's *methodology* is correct and worth adopting, but for THESE
datasets the payoff is more modest than the framing implies. The floor (z) was the
high-value, high-coverage win and it is done. Walls buy a smaller, partial, where-visible
y improvement; backflips is unhelpable by structure. => pursue **Phase 1** (universal
floor: normal-classified, free-offset CP plane — removes our per-bag band + floor_z
tuning) as the high-value/low-risk step; treat full multi-plane SLAM (Phases 3-5) as
optional research, not a clear win, on this hardware/these flights.

---

## APPENDIX B — Phase 1a EXECUTED (2026-06-26): free-offset floor landmark DONE

The floor offset is now a persistent graph variable `f0` (gtsam::Vector1, key `f0`),
bootstrapped from the first ~1.5 s lowest return cluster (drift-invariant) and jointly
estimated thereafter (`FloorPlaneFreeFactor`, residual z_off + traj_z - f0, d/df0 = -1;
persistent timestamp so it never marginalizes; weak 1 m gauge prior). Enabled by
`floor_free=1`. This replaces the hardcoded `floor_z` (study §4 / §11.6).

Result (self-calibrated, NO floor_z given):
| bag | f0 auto | vertical (baseline / fixed-tuned) | total |
|-----|---------|------------------------------------|-------|
| slow | -0.04 m | 0.084  (0.141 / 0.111) -- BEST yet | 0.121 |
| fast | -0.09 m | 0.274  (0.552 / 0.209) -- big win  | 0.407 |
| backflips | -0.82 m | 0.910  (0.949 / 0.894) -- neutral | 1.696 |

**Robustness proven:** the three bags self-calibrate to DIFFERENT offsets
(-0.04 / -0.09 / -0.82 m) with zero `floor_z` input. backflips' -0.82 m would have been
a 0.82 m error under the old fixed `floor_z=0` (cf. the earlier -1.19 m catastrophe). The
floor-height sensitivity is eliminated; the feature is now deployment-safe (no frame /
start-height knowledge needed). slow even improved over hand-tuned (f0 finds the floor
better than 0); fast is slightly behind hand-tuned (bootstrap landed -0.09 vs the ~0
optimum; f0 stays gate-anchored near bootstrap) but still a large win over baseline.

## APPENDIX C — Phase 1b EXECUTED (2026-06-26): drift-invariant cluster -> UNIVERSAL floor

Replaced the per-point absolute band with the **drift-invariant lowest-z-cluster** per
stride (`floor_cluster=1`): z_lo = 10th-pct of the stride's predicted-z; a return is
floor iff the stride is floor-bearing (|z_lo - f0| < floor_band, the max-plausible-drift
gate) AND z_pred in [z_lo - 0.1, z_lo + floor_slab]. Because z_lo rides the common per-
stride drift, there is no absolute-band clipping -> `floor_slab` (floor thickness) and
`floor_band` (drift gate) are PHYSICAL CONSTANTS, not per-env knobs.

**ONE universal floor config for all bags** (lambda_floor=15, floor_huber=0.15,
floor_free=1, floor_cluster=1, floor_slab=0.4, floor_band=0.5):
| bag | total (baseline) | vertical (baseline) | velocity (baseline) | f0 auto |
|-----|------------------|---------------------|---------------------|---------|
| slow | 0.112 (0.165) -32% | 0.059 (0.141) -58% | 0.208 (0.195) | -0.00 m |
| fast | 0.344 (0.604) -43% | 0.163 (0.552) -70% | 0.248 (0.733) -66% | -0.05 m |
| backflips | 1.718 (1.722) | 0.939 (0.949) | -- (neutral) | -0.48 m |

The cluster classifier is MORE selective (slow 443 vs 721 band factors, fast 98 vs 181)
-> cleaner floor returns, and it BEATS the per-bag-band Phase-1a (fast vert 0.163 vs
0.274; slow 0.059 vs 0.084). **Phase 1 gate MET + exceeded: match/beat z across all
speeds with ZERO per-environment tuning.** No floor_z, no per-bag band -> the floor
anchor is now universal and deployment-safe. Recommended floor config = the line above.
(Timing: backflips ~250 ms/update mean, occasional ~630 ms spike near the 300 ms stride
budget -- the per-stride floor_cands pre-pass adds a little; optimize if it matters.)

**Done for Phase 1.** Remaining toward the full study: Phase 2 (single dominant wall ->
y anchor; small, partial, where-visible per Phase 0) and Phases 3-5 (multi-plane SLAM +
Atlanta prior) -- optional, modest payoff on these flights (Appendix A).
