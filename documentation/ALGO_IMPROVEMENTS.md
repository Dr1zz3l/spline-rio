# Algorithm Improvement Ideas (brainstorm, 2026-06-25)

Forward-looking ideas for the spline-RIO system beyond the iSAM2 backend
(see ISAM2_MIGRATION.md). Grounded in this project's own findings (the
adaptive-knot NO-GO, the cond(H)~5.5e10, the "absolute position unobservable"
limit, heavy-tailed residuals, the firmware capabilities). Each item: what it is,
why it helps HERE, value/effort, and grounding.

---

## A. Knot spacing
**Accuracy = proven dead end; compute = real win.**
- Adaptive knots for ACCURACY: NO-GO, already validated (ROADMAP Part 5). Uniform
  8ms represents 10 rad/s flips to 1.28 deg; the backflips ori gap is
  optimization/weighting, not spline bandwidth. Do not re-chase.
- Adaptive knots for COMPUTE: open + now more valuable with iSAM2 (per-update cost
  scales with knots-in-window). Fine during |omega| spikes, coarse during hover ->
  maneuver-accuracy + hover-speed in ONE config, eliminating per-bag dt_pos/dt_ori
  tuning (fast_racing proved coarsening 236->88ms). basalt-exact non-uniform basis
  exists (analysis/adaptive_knots/, V0 validated). Effort moderate (geometric
  density ramping required).

## B. Automatic hyperparameter tuning  [TOP-3 #1]
1. **Data-driven NIS-adaptive noise (TOP PICK).** Estimate each sensor's effective
   noise from residual statistics at the solution (reduced-chi^2 ~ 1) instead of
   hand-tuning lambdas. The hand-tuned lambdas ALREADY approximate this
   (ROADMAP 4a/4c), so formalizing removes tuning with no accuracy loss + gives a
   consistency (NEES) story. The incremental smoother produces the innovation
   covariance every update -> classic adaptive-Kalman (Mehra) covariance estimation
   ONLINE/real-time; noise self-tunes per-flight. Moderate effort.
   **PROTOTYPED (2026-06-25, `adapt_noise_stride` in IsamSolver): partial success.**
   Sets each sensor sigma = std of its residuals at the solution (EMA). From a
   deliberately-WRONG start (lambda_gyro=1, lambda_accel=1) on slow_racing it adapts
   directionally -- pos 1.59->0.68m, vel 2.03->1.48 -- but does NOT recover the
   hand-tuned 0.165m/1.39deg (ori 1.87->2.80). Confirms ROADMAP 4a/4c: hand-tuned
   ~= std-whitening but the velocity<->orientation BALANCE matters, plus a
   chicken-and-egg (residuals at a bad solution -> loose sigma -> bad solution).
   Naive whitening insufficient. Promising directions to make it competitive:
   (a) iterate-to-fixed-point (re-solve + re-estimate until sigma stabilizes, not
   one EMA pass); (b) per-sensor robust-vs-std choice (gyro full-std, radar core);
   (c) the Mehra innovation-covariance form (uses the smoother's predicted cov,
   avoids the bad-solution feedback); (d) keep a fixed relative prior on the
   trade-off and only adapt the global scale.
2. Type-II ML / evidence maximization (empirical Bayes): maximize the factor-graph
   marginal likelihood over the hyperparameters; GTSAM gives the info matrix so the
   evidence gradient w.r.t. log-noise is computable. Higher effort.
3. Black-box (BO/CMA-ES) over a self-supervised proxy (reduced-chi^2 / held-out
   consistency, NOT mocap -> avoid overfit). Uses eval_bags.py infra. Least elegant.

## C. Theoretically-optimal formulations  [TOP-3 #3]
1. **GP / WNOA motion prior instead of B-spline + min-snap (DEFERRED -- do NOT
   undo splines yet; user 2026-06-25).** A white-noise-on-acceleration/jerk
   Gaussian-process trajectory is MAP-optimal continuous-time under a stated
   stochastic motion model (Barfoot/Anderson; steam_icp, cited in the report).
   STILL continuous-time (query any t via exact GP interpolation Lambda(t)x_k +
   Psi(t)x_{k+1}); REPLACES the spline -- variables become physical (pose,
   body-velocity) "GP states" at sparse times instead of abstract control points.
   KEY INSIGHT: a min-snap smoothing spline IS, up to details, the MAP of a
   white-noise-on-jerk GP -- so our quintic+lambda_snap is already an IMPLICIT GP
   with the smoothness added as a hand-tuned penalty. Making it explicit:
   (i) replaces lambda_snap_pos + lambda_ori_accel with ONE physical, estimable Q
   (serves auto-tuning); (ii) block-tridiagonal info matrix BY CONSTRUCTION
   (sparser/faster); (iii) directly estimates velocity (our good metric). Large
   effort (re-derive every factor in GP-state terms; SO(3) GP per Anderson&Barfoot
   2015 is involved -- keep the R^3 pos + SO(3) ori split). Burnett: 74-139 ms/win,
   <=5 GN iters. Tractable first half: the R^3 POSITION GP alone. **PARKED for later.**
2. **Optimal robust loss from the estimated residual distribution.** Radar residuals
   are heavy-tailed (documented). ML-optimal loss is -log p(r): fit a Student-t to
   the residuals, use its NLL (DOF is estimable). Replaces hand-picked huber_delta;
   Student-t is the principled generalization of Huber/Welsch. Moderate.
3. **Observability Gramian / Fisher info as a diagnostic.** Derive which params are
   estimable vs frozen (pitch is weakly observable empirically -> the Gramian says
   it principledly; flags the unobservable abs-position/yaw modes quantitatively).
   Low-moderate; diagnostic, not a solver change.

## D. Sensor model
1. **Per-chirp radar timestamps / deskew -- DEAD END (firmware-verified 2026-06-25).**
   Config: frameCfg numLoops=16 x 3 TX, profileCfg idle 43us/rampEnd 40us
   (PRI ~83us) -> active chirp burst 48 x 83us ~= 4ms inside a 33.3ms (30Hz) frame.
   The point cloud carries ONE timestamp/frame (ros::Time::now at data arrival),
   NO per-point time; detections come from COHERENT range-Doppler processing over
   all chirps (Doppler FFT over the 16-loop dimension). Array order = detection-
   matrix raster (rangeIdx/dopplerIdx), encodes range/velocity NOT time. Sub-frame
   timing is not recoverable; deskew would need raw ADC reprocessing. AND the smear
   ceiling at peak ~12-15 rad/s over 4ms is only ~3 deg, below the ~5.7 deg backflips
   ori gap -> not the lever even if recoverable. The TDM-MIMO rotation angle-bias is
   real but also unfixable from the computed cloud (needs virtual-array phase).
   FIRMWARE NOTE (user asked): the launch file does NOT flash -- mmWaveQuickConfig
   sends the .cfg over the serial CLI to the already-flashed TI SDK demo at runtime.
   Reflashing custom firmware can't help either: a coherent Doppler detection has no
   per-point time (the Doppler FFT integrates all 48 chirps); per-chirp range-only
   detection would recover timing but DESTROY the velocity measurement the system
   relies on. Only a frame-level DSP hardware timestamp is achievable (removes host
   USB-latency jitter) -- low value (offset already calibrated, not the ori lever).
2. **Radar landmark / PLANE mapping for absolute position (HIGH value, feasible).**
   Doppler+IMU CANNOT observe absolute position (documented hard limit -> the drift).
   The environment is highly structured: a cuboid hall (one side ~20% longer) inside
   a cuboid rope-mesh flight arena (walls/ceiling offset from the hall, shared
   foam floor). User has already SEEN the mesh walls/roof, hall walls/roof, and
   floor in accumulated radar returns (mocap-posed, slow bag). So instead of fragile
   radar POINT-feature tracking, fit PLANES (the 6+ cuboid faces, mostly
   axis-aligned -> a Manhattan/Atlanta-world prior a la Doer x-RIO) and constrain
   the drone position relative to them. Planes are stable where radar points
   fluctuate. Bounds the 0.3-1.7m drift using known geometry. Two nested cuboids
   (hall + mesh) give redundancy; the closer mesh could give fine position. Large
   effort but it attacks the ONE thing the system structurally cannot do.

## E. Metrics / diagnostics (sparked by user, 2026-06-25)
**Our metrics may be too simple for backflips degradation.** GT flies many
backflips while drifting sideways into a full circle. If the estimate flies the
CIRCLE but skips/attenuates the FLIPS, position RMSE looks "fairly good" (~flip
radius, SE3-aligned) while orientation is poor -- i.e. position RMSE UNDERSTATES
the failure and the two metrics decouple. Actions:
  - Diagnostic FIRST: is the iSAM2 backflips estimate actually flipping (going
    up-and-over each loop), or smoothing through into a flat circle? Plot estimate
    vs GT per-flip (the vertical loop excursion + the roll/pitch sweep). This may
    EXPLAIN the 10.7 deg ori + decent 1.72m pos and tell us whether it's a real
    failure or a metric artifact.
  - Better backflips metric: per-flip loop-tracking (does each vertical excursion
    appear?) + joint pos/ori; decouple "circle drift" (low-freq) from "flip
    tracking". Connects to the report's NEES/decomposition.
  - Cheap, high diagnostic value; should precede further backflips-ori tuning.
  - **DONE (2026-06-25, plot diagnostic):** the iSAM2 backflips estimate DOES flip
    (trajectory traces the petal/flower loops; estimate omega spikes match mocap;
    pitch RMSE 8deg is incompatible with flattening, which would give ~150deg).
    The "flies the circle without the flips" hypothesis is DISPROVEN. BUT the 10.7deg
    ori RMSE is dominated by TRANSIENT SPIKES at the flip peaks (orientation-error-
    per-axis plot) -- exactly where rotation is fastest AND mocap GT is most degraded
    (occlusion/FD spikes, CLAUDE.md). => Next: recompute ori RMSE EXCLUDING the
    occlusion-masked / degraded-GT flip-peak samples; the "real" estimator error is
    likely well below 10.7deg. This is the right backflips metric refinement (the
    measurement model/metric, not estimator tuning, is the lever -- consistent with
    the project's own negative results).
  - **GT-aware metric DONE + result (2026-06-25):** confirmed the mocap degradation
    is REAL (52 dropout gaps <=126ms; FD-omega spikes to 88000 rad/s vs ~14 physical).
    Added a clean-GT orientation RMSE to validate_live_solver.py (masks samples
    adjacent to unphysical FD rate / gap). BUT it changed NOTHING: 0.5% excluded,
    10.74 -> 10.75 deg. The degradation does NOT inflate the orientation metric,
    because the FD spikes are quaternion SIGN FLIPS (q == -q -> same rotation
    matrix, and the RMSE is matrix-based/sign-invariant) and gaps remove samples
    rather than corrupt them. => the backflips 10.7deg is GENUINE estimator error
    (also DISPROVEN: flattening, via the trajectory plot). Residual mocap
    smoothing-LAG during flips may add a few deg (undetectable), but the bulk is
    real. The remaining lever is the estimator (spline-omega gate vs gyro-mag
    proxy) or accepting that single-chip 10 rad/s orientation is just hard.

## E. Other
- Learned radar front-end (static/dynamic + ground/structure classification) to
  beat RANSAC's crude filtering, esp. elevation-biased single-chip returns.
- extra_iters -> trust-region / early-stop (iterate to a convergence criterion,
  not a fixed count) -- reclaim compute on easy strides.

---

## Top 3 to pursue (value-per-effort), as directed
1. **Data-driven NIS-adaptive noise (B1)** -- removes most hand-tuning, real-time-
   natural with the smoother, principled, moderate effort.
2. ~~Per-chirp radar timestamps (D1)~~ -- DEAD END (firmware-verified; see D1).
   SUBSTITUTE: the backflips metrics diagnostic (E) + plane-mapping (D2), since the
   real backflips lever is the measurement geometry/metric, not deskew.
3. **GP/WNOA motion prior (C1)** -- theoretically-optimal substrate; collapses the
   regularizer hyperparameters into one estimable Q + sparsity for free. Largest
   effort / highest ceiling.
