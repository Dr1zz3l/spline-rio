#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <vector>
#include <string>

namespace rio {

// ============================================================================
// Input data structures (passed from Python via pybind11)
// ============================================================================

struct RadarPoint {
    double x, y, z;   // position in sensor frame (m)
    double v;         // Doppler velocity (m/s)
    double intensity{0.0};  // signal intensity/SNR (loader units; 0 = absent)
    double w{1.0};    // extra per-point weight multiplier (1 = neutral)
};

struct RadarFrame {
    double timestamp;               // absolute time (s)
    std::vector<RadarPoint> points;
};

struct ImuSample {
    double timestamp;   // absolute time (s)
    double ax, ay, az;  // accelerometer (m/s²)
    double gx, gy, gz;  // gyroscope (rad/s)
};

// ============================================================================
// PreintFactor — preintegrated IMU measurement (Forster TRO-2017)
// ============================================================================
// Holds ΔR, Δv, Δp integrated from t_i to t_j plus first-order Jacobians
// for on-the-fly bias correction during optimization.
struct PreintFactor {
    double t_i{0.0}, t_j{0.0}, dt{0.0};

    // Preintegrated measurements (at linearization biases b_a0, b_g0)
    Eigen::Matrix3d delta_R{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d delta_v{Eigen::Vector3d::Zero()};
    Eigen::Vector3d delta_p{Eigen::Vector3d::Zero()};

    // Linearization biases
    Eigen::Vector3d b_a0{Eigen::Vector3d::Zero()};
    Eigen::Vector3d b_g0{Eigen::Vector3d::Zero()};

    // First-order bias Jacobians (3×3 each)
    Eigen::Matrix3d d_R_d_bg{Eigen::Matrix3d::Zero()};
    Eigen::Matrix3d d_v_d_ba{Eigen::Matrix3d::Zero()};
    Eigen::Matrix3d d_v_d_bg{Eigen::Matrix3d::Zero()};
    Eigen::Matrix3d d_p_d_ba{Eigen::Matrix3d::Zero()};
    Eigen::Matrix3d d_p_d_bg{Eigen::Matrix3d::Zero()};
};

// ============================================================================
// Config (subset of solver.yaml)
// ============================================================================
struct SolverConfig {
    // Spline
    double dt_pos{0.005};
    double dt_ori{0.008};

    // Radar
    double huber_delta{1.0};
    double min_range{0.2};
    double radar_weight{1.0};   // global per-point radar weight (1.0 = legacy numeraire)

    // IMU weights
    double lambda_accel{0.01};
    double lambda_gyro{1.0};
    double huber_delta_accel{2.0};

    // Regularization
    double lambda_snap_pos{0.0001};
    double lambda_ori_reg{0.001};   // angular vel reg: penalizes ||log(q_i^{-1}*q_{i+1})||² (legacy)
    double lambda_ori_accel{0.0};  // angular accel reg: penalizes ||Δω||² per knot triplet (preferred)

    // Gravity direction factor
    double lambda_gravity{0.001};
    double gravity_accel_threshold{3.0};

    // Heading prior
    double lambda_heading{3.0};

    // Bias priors
    double lambda_bias_prior_accel{1.0};
    double lambda_bias_prior_gyro{1.0};

    // Boundary priors
    double lambda_boundary_pos{1000.0};
    double lambda_boundary_vel{1000.0};
    double lambda_boundary_ori{1000.0};
    double lambda_boundary_ori_yaw{0.0};

    // Extrinsics
    bool lock_extrinsics{false};
    bool optimize_pitch_only{true};
    double lambda_extrinsic_prior{10.0};

    // Gyro bias locking (sliding window only).
    // When true, the gyro components of the bias block are clamped back to the
    // initial stationary estimate after each window solve, before compute_prior()
    // bakes the corrected state into the Schur complement prior.
    // Prevents runaway gyro bias in rank-deficient windows (e.g. backflips at
    // dt_ori=0.0008 where orientation DoF exceed gyro constraints per window).
    bool lock_gyro_bias{false};

    // Position-to-init prior (sliding window only).
    // When > 0, every position CP in the active window is softly anchored to its
    // P1-P3 initialisation value via a direct CP-level penalty.  Prevents radar-
    // sparse windows (e.g. backflips at dt_ori=0.008, ~10 Hz radar) from drifting
    // position freely while the orientation solver refines the gyro spline.
    // Suggested sweep: {0.1, 1, 10, 100}; start at 10 for backflips.
    // 0.0 (default): disabled; no effect on racing bags.
    double lambda_pos_init_prior{0.0};

    // ω-gated radar: skip radar frames when |ω_body| exceeds this threshold (rad/s).
    // During rapid maneuvers (backflips, ~10 rad/s peak), even a small orientation
    // spline error creates a large Doppler residual via the lever-arm term (ω × r).
    // Dropping radar at those instants lets accel+gyro carry position through the
    // flip without corruption from bad radar projections.
    // Gate is evaluated from the initial spline at problem-build time (not per-iteration).
    // 0.0 (default): disabled. Suggested starting point for backflips: 5.0 rad/s.
    double omega_gate_threshold{0.0};

    // ω-dependent radar noise inflation ("soft gate", ROADMAP Part 3 item 2).
    // Physical motivation: a radar frame integrates over ~tens of ms; at body
    // rate ω the scene smears by ω·T_frame (≈17° at 10 rad/s), so the
    // single-timestamp Doppler model error grows with |ω|.  Instead of
    // discarding frames (hard gate), down-weight them continuously:
    //   σ_eff²(ω) = σ₀²·(1 + (|ω|/ω₀)²)  ⇒  weight w = 1/(1 + (|ω|/ω₀)²)
    // applied as a ScaledLoss around the Huber loss, with |ω| evaluated from
    // the initial/warm-start spline at problem-build time (like the gate).
    // omega_soft_sigma = ω₀ in rad/s; weight halves at |ω| = ω₀.
    // 0.0 (default): disabled. Suggested sweep for backflips: {2, 4, 8}.
    double omega_soft_sigma{0.0};

    // ω-dependent ACCEL down-weighting (ROADMAP Part 5 Tier-1 #6): same
    // rationale and form as omega_soft_sigma but applied to the accelerometer
    // factor — during flips the accel carries thrust transients, vibration and
    // lever-arm error (residual std 91.7 m/s² on backflips), and its
    // gravity-direction information is ~nil mid-flip; it mainly distorts
    // orientation there.  w = 1/(1 + (|ω|/ω₀)²), |ω| from the warm-start
    // spline at build time.  0.0 (default): disabled.
    double accel_soft_sigma{0.0};

    // Asymmetric ω-gate split (ROADMAP Part 5, position-during-flip
    // information): when the radar soft gate down-weights the full factor to
    // w, ALSO add a position-only radar factor (orientation/ω frozen at
    // warm-start values → linear in pos CPs) with weight
    // (1−w)·radar_pos_split.  Radar velocity information then keeps flowing
    // into position during flips without dragging orientation.
    // Requires omega_soft_sigma > 0.  0.0 (default): disabled; 1.0 =
    // complementary split.  SW solver only.
    double radar_pos_split{0.0};

    // ω-adaptive gyro weight (ROADMAP Part 5, universal-config unification of
    // the backflips λ_gyro=400 finding): per-sample effective weight
    //   λ_eff = lambda_gyro · (1 + (|z_gyro|/ω₀)²)
    // using the MEASURED rate (direct, causal, no spline eval needed).
    // Rationale: the gyro is the only clean sensor during fast rotation —
    // trust it more exactly there; in gentle flight λ_eff ≈ lambda_gyro so
    // racing behavior is unchanged.  With ω₀ = 1 rad/s: 4→~400 at 10 rad/s
    // (matches the backflips sweep optimum).  0.0 (default): disabled.
    double lambda_gyro_omega_sigma{0.0};
    // Exponent p in w = 1 + (|z_gyro|/ω₀)^p.  p=2 ramps early (hurts racing:
    // stiffens the gyro at 2–5 rad/s where radar is still informative);
    // p=4 stays ≈flat below ~0.7·ω₀ and ramps hard above — kicks in only for
    // genuinely aggressive rotation.
    double lambda_gyro_omega_pow{2.0};

    // Per-point radar weighting from signal intensity/SNR (ROADMAP Part 4,
    // "per-point noise"; the Python WLS already weights by intensity).
    // Weight w_i = (I_i / I_med)^α with I_med the per-frame median —
    // median-relative so the global radar weight scale is unchanged (pure
    // redistribution within a frame).  α = radar_intensity_weight:
    // 0.0 (default) disabled, 1.0 = linear-relative (matches WLS precedent).
    // No clamp (removed 2026-07-19): measured per-point I/I_med spans
    // 0.37-3.3 on all own-platform bags and <=4.3 on ICINS, so the old
    // [0.25, 4] guardrail never engaged own-platform and clipped <0.005%
    // of ICINS points. SW solver only.
    double radar_intensity_weight{0.0};

    // Fixed radar elevation (z) bias correction (ROADMAP Part 4 / z-bias).
    // The IWR6843's 2-TX elevation diversity produces a systematic per-point
    // Doppler error proportional to the ray's sensor-frame z component:
    //   v_corr = v_meas - radar_zbias_fixed * u_sensor.z()
    // FINDINGS: WLS z-velocity biased -0.5..-0.65 m/s. Applied at problem
    // build (no new parameter). Sign/magnitude empirical — sweep ±0.5.
    // Discriminates antenna-fixed vs scene-driven bias: antenna-fixed should
    // stop the backflips z-sink; if the sink persists the bias is scene-driven.
    double radar_zbias_fixed{0.0};

    // Optimizer
    int max_iterations{40};
    int num_threads{0};   // 0 = auto (std::thread::hardware_concurrency())

    // Preintegration
    bool   use_preintegration{false}; // replace raw IMU with preintegrated factors
    double lambda_preint{1.0};        // weight for r_R (3 residuals)
    double lambda_preint_v{0.0};      // weight for r_v (3 residuals); 0 = disabled (safe default)
    double lambda_preint_p{0.0};      // weight for r_p (3 residuals); 0 = disabled (safe default)
    double preint_hz{100.0};          // informational; Python uses to set dt_ori = 1/preint_hz

    // Sliding window: fix leading knots constant (previously solved, trusted)
    int n_fix_leading_pos{0};   // number of leading pos CPs to freeze
    int n_fix_leading_ori{0};   // number of leading ori knots to freeze

    // Marginalization prior scale: multiplies sqrt_info before attaching prior.
    // The raw Schur complement encodes the full information from all previous
    // measurements, which can be orders of magnitude tighter than the boundary
    // priors (lambda ~1000).  Scale < 1 softens the prior to prevent it from
    // over-constraining boundary CPs and blocking adaptation to new data.
    double marg_prior_scale{1.0};

    // Adaptive marginalization prior scale (data-driven, opt-in).
    // When true, final_scale = marg_prior_scale * adaptive_scale where
    //   adaptive_scale = sqrt(lambda_boundary_pos / max_eigenvalue_of_S)
    // This normalises the max eigenvalue of the Schur complement to match
    // lambda_boundary_pos, making marg_prior_scale a relative fine-tuning
    // multiplier instead of an absolute hack.
    // Typical effect: adaptive_scale ≈ 1.3e-6 (for slow/fast_racing).
    // NOTE: this makes the prior ~150× softer than marg_prior_scale=2e-4 alone.
    //       Test carefully before enabling on tuned configs.
    bool use_adaptive_marg_scale{false};

    // Cauchy robust loss on the marginalization prior residual (||r||² space).
    // delta = 0 (default): disabled, retains current quadratic behaviour.
    // Note: requires care with delta choice — see marg_prior_residual_norm diagnostic.
    double marg_prior_cauchy_delta{0.0};

    // ---- Linear system dump (for banded-solver prototyping) ------------------
    // When true, evaluates the full problem Jacobian at the warm-start (before
    // the LM solve) and stores it in SolverResult.  Also evaluates a second
    // snapshot after the solve (converged point).
    // Slows down the solve by one extra problem.Evaluate() call per window.
    // Default: false (no overhead in production).
    bool dump_system{false};

    // Compute the live-edge joint covariance per window via ceres::Covariance
    // (SolverResult::nees_cov; offline NEES consistency study — adds
    // ~0.1-0.5 s/window, not for real-time use).
    bool nees_covariance{false};

    // [Session 2] Use BandedSchurSolver (dense Eigen LDLT skeleton) instead of
    // SPARSE_NORMAL_CHOLESKY.  Slower but verifies the custom solver hook.
    // Default: false (use SuiteSparse/EigenSparse as before).
    bool use_banded_schur{false};

    // Eigenvalue clipping for the Schur complement S before forming sqrt_info.
    // S is extremely ill-conditioned (cond≈5.5e10): gyro-dominated orientation DOF
    // have eigenvalues ~5.5e14 while position DOF are ~1e4, ratio 5.5e10.
    // With uniform scale, the prior over-constrains orientation DOF, preventing
    // the boundary from adapting to new data (root cause of per-bag marg_prior_scale need).
    //
    // Clipping caps eigenvalues of S at eig_clip before Cholesky, removing the
    // directional dominance.  Increase marg_prior_scale proportionally to compensate.
    //
    // Suggested starting point:
    //   marg_prior_eig_clip = 1e6   (≈ position-DOF magnitude, 8 decades below max)
    //   marg_prior_scale    = 0.05  (compensates for reduced max eigenvalue)
    //   → effective max eigenvalue = (0.05)² × 1e6 = 2500  ≈ λ_boundary (1000)
    //
    // 0.0 (default): disabled, S used as-is.
    double marg_prior_eig_clip{0.0};

    // ---- Markov-blanket marginalization (consistency fix, ROADMAP §1.1) ------
    // When true, compute_prior() marginalizes the FULL out-of-next-window set
    // (previous-boundary "leading" blocks + stride-zone blocks) and includes
    // ONLY residuals that touch a marginalized block.  By the B-spline locality
    // property, such residuals' support is automatically contained in
    // marg ∪ boundary, so the Schur complement is a true marginal — no
    // conditioning on interior states, no double counting of factors that are
    // re-added in the next window (previously: ALL residuals touching boundary
    // or bias — i.e. every IMU factor in the window — were baked into S and
    // then re-added, making the prior overconfident by orders of magnitude;
    // that is why marg_prior_scale had to be ~2e-4).
    // Residuals whose support is not closed (touch a free block outside
    // marg ∪ boundary ∪ bias, e.g. the ~1 ms pos/ori grid-mismatch sliver, or
    // pitch_delta) are handled per marg_cond_pitch below / dropped.
    // false: legacy behaviour (conditioning + double counting + tiny scale).
    bool marg_markov_blanket{true};

    // ---- Live-edge warm-start alignment (ROADMAP §1.2) -----------------------
    // New CPs/knots entering the window each stride hold P1-P3 init values
    // dead-reckoned from t=0, which drift away from the solved trajectory over
    // the flight.  The resulting seam discontinuity forces LM to drag the new
    // segment across the gap every window (observed iter≈28 regardless of warm
    // start).  When true, the entering segment is rigidly aligned to the solved
    // boundary: pos shifted by (solved - init) at the seam CP, ori left-
    // multiplied by ΔR = R_solved(seam) · R_init(seam)^T.  This preserves the
    // locally-accurate P1-P3 shape while removing accumulated drift.
    // false: legacy behaviour (stale absolute init values).
    bool warm_start_align{true};

    // ---- Yaw gauge pre-alignment (ROADMAP next-steps) -------------------------
    // Before each window solve, compute the circular-mean yaw residual of the
    // window's heading samples against the warm-start spline and rigidly rotate
    // the whole window state by Rz(Δψ) about the position at the window-start
    // boundary.  Rotation about the gravity axis is exactly the gauge direction
    // of the radar+IMU cost (accel/gyro/radar residuals are invariant; only the
    // heading priors respond), so this closed-form step removes the dominant
    // curved-valley mode that otherwise consumes most LM iterations — without
    // changing the converged solution.  The marg prior and boundary anchors are
    // re-centered/evaluated AFTER the rotation, so they do not fight it.
    bool yaw_prealign{false};

    // Damping gain for yaw_prealign: the applied rotation is gain·Δψ̄.
    // gain=1.0 collapses the full mean heading residual but injects the
    // per-window noise of the Δψ̄ estimate into position (rigid rotation about
    // the boundary pivot), which random-walks settled position.  gain≈0.5
    // halves the injected noise while still keeping the optimizer near the
    // valley bottom (the remaining yaw error decays geometrically across
    // windows).
    double yaw_prealign_gain{1.0};

    // Ceres function_tolerance (relative cost-change stopping criterion).
    // Default matches Ceres (1e-6).  Looser values (1e-5) stop the LM tail
    // adaptively — unlike a hard max_iterations cap, hard windows still get
    // their iterations while easy windows exit early.
    double function_tolerance{1e-6};

    // ---- Stochastic noise model (branch stochastic-noise-model) --------------
    // Bias random walk ACROSS strides: before storing the marginalization
    // prior, inflate its bias-block covariance by Q = diag(qa²I₃, qg²I₃) via an
    // information-space Woodbury downdate of the Schur complement S
    //   S ← S − S P (Q⁻¹ + Pᵀ S P)⁻¹ Pᵀ S
    // (mean-preserving: the prior is re-centered at each window's warm start).
    // Lets the bias adapt at stride granularity so high gyro weights no longer
    // conflict with slow bias drift.  0.0 disables that coordinate group.
    double bias_rw_sigma_accel{0.0};   // per-stride accel-bias σ (m/s²)
    double bias_rw_sigma_gyro{0.0};    // per-stride gyro-bias σ (rad/s)

    // Per-frame correlated radar noise: measured per-point residuals share a
    // frame-level error (attitude/timing/smearing; intra-frame correlation
    // 0.06–0.22, see analysis/characterize_frame_correlation.py).  Model
    // Σ = I + σ_c²·11ᵀ per frame (per-point σ ≡ 1 nominal) and whiten exactly
    // with W = I − (γ/n)·11ᵀ, γ = 1 − 1/√(1 + n σ_c²), implemented as ONE
    // n-dim residual block per frame with the SAME parameter-block support as
    // the per-point factors (marginalization logic unaffected).
    // Per-point intensity weights are folded in before whitening
    // (approximation); the frame-level ω weight wraps the block as ScaledLoss.
    // Per-point robustness via warm-start Huber-IRLS weights (RANSAC
    // prefilter remains the primary outlier stage).  Works with locked and
    // free extrinsics (WithPitch frame variant).
    // 0.0 (default): per-point factors as before.
    double radar_frame_sigma_c{0.0};
    // Second whitening direction: frame-varying elevation-correlated error
    // (ratio to sigma_0, like sigma_c; measured ~0.33 across flights,
    // 2026-08-11).  0 = off (bit-identical to before).
    double radar_frame_sigma_z{0.0};
    // Frame whitening with a HETEROSCEDASTIC diagonal.  The whitening solves
    // Sigma = sigma_i^2 I + sigma_c^2 v v^T; with 0 the shared term is taken
    // along 1 (correct only when every per-point weight is equal), with 1 it is
    // taken along the actual v_j = sqrt(w_j).  Default 0 = bit-identical to the
    // deployed behaviour; 1 is the derived form and is a gate-battery candidate
    // (it moves the DEPLOYED config too, because the SNR weight is already
    // per-point).  See FEEDBACK_PLAN 2026-08-06.
    int radar_frame_hetero{0};

    // ---- Estimated radar z-bias STATE (SW only) ------------------------------
    // v_corr = v_meas - b_z * u_z(BODY) with b_z one persistent 1-dof block:
    // absolute prior N(0, prior_sigma^2) anchored at zero + random-walk
    // inflation zbias_rw_sigma per stride inside the marg prior (same
    // Woodbury mechanism as the IMU-bias RW).  The principled replacement for
    // the retired radar_zbias_fixed constant (which was poison because the
    // bias is regime-dependent).  Measured basis + derived constants (prior
    // 0.5, rw 0.10, body frame): worklog/2026-08-12_zbias-state-measurement.md
    // (backflips body-frame -0.29/-0.35 persistent; racing/ICINS ~0).
    // SUPPORTED PATH: frame-whitened radar factors (radar_frame_sigma_c > 0)
    // with lock_extrinsics=1.  Guarded errors otherwise, and against the two
    // competing instruments: radar_frame_sigma_z > 0 (whitens exactly the
    // direction this state needs to observe -- the b_z Jacobian column IS d2)
    // and radar_zbias_fixed != 0.  The system-dump/banded paths do not carry
    // the extra column.  Default 0: bit-identical when off.
    int    radar_zbias_state{0};
    double radar_zbias_prior_sigma{0.5};   // m/s, absolute anchor at 0
    double radar_zbias_rw_sigma{0.10};     // m/s per stride (window-to-window)

    // ---- GP-backend ports (opt-in probes, 2026-08-12; both 0 = off) ---------
    // huber_omega_sigma: ω-scaled Huber knee in the frame-whitened radar path.
    // The GP applies Huber to the WHITENED residual, so its raw-m/s inlier
    // band widens with σ_eff(ω); here delta_eff = huber_delta *
    // sqrt(1+(|ω_ws|/X)²) in the warm-start IRLS weights (frame path only).
    double huber_omega_sigma{0.0};
    // huber_model_knee: the ZERO-parameter founded knee.  The knee sits at a
    // fixed number of MODEL sigmas per point: delta_i = huber_delta / sqrt(w_i)
    // with w_i the per-point weight ratio the solver already assigns
    // (speed-law frame weight for clean returns, w_al for aliased,
    // x intensity).  Mutually exclusive with huber_omega_sigma.
    int huber_model_knee{0};
    // marg_stale_phi: rotation-swept staleness inflation of the marg prior's
    // boundary-ORIENTATION rows, sigma_add = marg_stale_phi * theta_swept
    // (theta = accumulated knot-to-knot rotation over the marginalized zone),
    // via the same Woodbury downdate as the bias random walk.  The only
    // mechanism whose ablation moves backflips position on the GP branch
    // (its default there: 0.01).
    double marg_stale_phi{0.0};
    // marg_stale_pos: the position-row analogue (2026-09-04 vertical-drift
    // investigation): the marg prior's boundary-POSITION rows were built with
    // radar-factor Jacobians linearized at the attitude of the marginalized
    // window (R_bw enters every Doppler row), so they stale with the same
    // swept rotation; sigma_add = marg_stale_pos * theta_swept (m per rad),
    // k measured against GT (live position-error increment per swept radian).
    double marg_stale_pos{0.0};
    // boundary_first_window_only (2026-09-07 Fig. 1 audit): the P3 window-start
    // pins (lambda_boundary_pos/vel/ori) were re-added in EVERY window, anchored
    // to the previous window's estimate at t_start and then marginalized into
    // the next prior, i.e. a hand-set pseudo-measurement re-asserting the
    // previous estimate beside the marginalization prior.  1 = add the pins
    // only while no marginalization prior exists (first window, or after a
    // dropped prior); 0 = legacy behaviour (bit-identical).
    int boundary_first_window_only{0};

    // ---- Fast marginalization prior (direct factor evaluation) ---------------
    // ceres::Problem::Evaluate() rebuilds an internal Program + Evaluator on
    // every call (~0.1–0.4 s per window, the "other" wall-time component).
    // When true, compute_prior() instead calls each res_set cost function's
    // Evaluate() directly and accumulates the dense Gauss-Newton Hessian
    // itself (~ms).  Semantics match Problem::Evaluate with
    // apply_loss_function=true: tangent-space Jacobian columns for SO(3)
    // blocks (manifold PlusJacobian), Triggs corrector for robust losses,
    // out-of-set blocks (pitch_delta, window-1 constants) treated as fixed.
    // false: legacy Problem::Evaluate path (for A/B verification).
    bool marg_fast_prior{true};
};

// ============================================================================
// Extrinsic config
// ============================================================================
struct ExtrinsicConfig {
    double roll_deg{180.0};
    double pitch_deg{25.5};
    double yaw_deg{0.0};
    double tx{0.08}, ty{0.02}, tz{-0.01};  // translation_body_m

    Sophus::SO3d R_radar_to_body() const;
};

// ============================================================================
// Block map entry — describes one parameter block's position in the
// linearized Jacobian tangent-space columns (used by dump_system).
// ============================================================================
struct BlockMapEntry {
    // Variable type: "ori_knot" | "pos_cp" | "bias" | "pitch_delta"
    std::string type_id;
    // Index within that type (0-based; 0 for bias / pitch_delta which are singletons)
    int index{0};
    // First column in the tangent-space Jacobian for this block
    int col_offset{0};
    // Tangent-space dimension (3 for ori/pos via SO3 manifold, 6 for bias, 1 for pitch)
    int tangent_size{0};
};

// ============================================================================
// Solver output
// ============================================================================
struct SolverResult {
    // Optimised trajectory state (same size as input)
    std::vector<std::array<double, 3>> pos_cps;
    std::vector<std::array<double, 4>> ori_knots;
    std::array<double, 6> biases{};
    std::array<double, 3> extrinsic_euler_deg{};
    double radar_zbias{0.0};   // estimated z-bias state (radar_zbias_state)

    // Diagnostics
    std::vector<double> cost_history;   // one entry per accepted iteration
    double solve_time_s{0.0};
    int    num_iterations{0};           // successful LM steps (summary.num_successful_steps)
    std::string solver_summary;

    // Ceres timing breakdown (seconds)
    double time_residual_eval_s{0.0};   // evaluating residuals (r)
    double time_jacobian_eval_s{0.0};   // autodiff Jacobians (J)
    double time_linear_solver_s{0.0};   // sparse Cholesky  (J'J + λI)Δx = -J'r

    // Marginalization prior diagnostics (set by compute_prior each window)
    bool        marg_prior_valid{false};
    int         marg_prior_dim{0};
    double      marg_cond_number{0.0};
    double      marg_min_eigenvalue{0.0};
    double      marg_max_eigenvalue{0.0};
    int         marg_numerical_rank{0};
    std::string marg_drop_reason;

    // Boundary state covariance (S^{-1}) from Schur complement
    // trace_cov = tr(S^{-1}): total posterior variance of boundary DOF
    // adaptive_scale = sqrt(lambda_boundary_pos / max_eig_S): data-driven normaliser
    // applied_scale: the actual final_scale used when attaching the prior
    double marg_trace_cov{0.0};
    double marg_adaptive_scale{0.0};
    double marg_applied_scale{0.0};

    // Prior residual norm after solve: ||r||² = local_x^T * S * local_x
    // This is the squared Mahalanobis distance between the solved boundary state
    // and the prior mean.  Measures how much the current window's data disagreed
    // with the prior.  Used to calibrate marg_prior_cauchy_delta:
    //   small (~d_b=30)  → prior was consistent; delta should be above sqrt(30)≈5.5
    //   large (>>30)     → prior was over-constraining; delta should be below this
    // -1.0 if no prior was active this window.
    double marg_prior_residual_norm{-1.0};  // ||r||² at solution

    // Staleness diagnostics (marg_stale_phi derivation): theta_swept (rad)
    // recorded when the consumed prior was built; marg_bound_dphi = mean
    // boundary-ori update warm-start -> solution (rad).  k = slope dphi/theta.
    double marg_theta_swept{0.0};
    double marg_bound_dphi{0.0};

    // Post-solve boundary state covariance (set by compute_prior)
    // Two complementary views:
    //   boundary_cov_trace (S^{-1}):      accumulated prior covariance — how well
    //                                     previous windows constrain boundary state
    //   window_cov_trace   (H_bb^{-1}):   current-window sensor-only covariance —
    //                                     how well THIS window's data constrains it
    // Ratio window_cov_trace / boundary_cov_trace < 1: window more informative than prior.
    bool            boundary_cov_valid{false};
    double          boundary_cov_trace{0.0};    // tr(S^{-1}), same as marg_trace_cov
    double          window_cov_trace{0.0};      // tr(H_bb^{-1}): window-only
    Eigen::MatrixXd boundary_covariance;        // S^{-1},  d_b × d_b
    Eigen::MatrixXd window_covariance;          // H_bb^{-1}, d_b × d_b

    // ---- Live-edge covariance for the NEES study (SolverConfig::nees_covariance)
    // Joint tangent-space covariance (ceres::Covariance, SPARSE_QR) of the
    // trailing N_POS position CPs + N_ORI orientation knots of the window —
    // the blocks that determine v(t_live) and R(t_live).
    // Layout: [pos CP (nees_pos_idx0..+N_POS-1) ×3 | ori knot (nees_ori_idx0..+N_ORI-1) ×3]
    bool            nees_cov_valid{false};
    int             nees_pos_idx0{-1};          // global index of first included pos CP
    int             nees_ori_idx0{-1};          // global index of first included ori knot
    Eigen::MatrixXd nees_cov;                   // (3·N_POS+3·N_ORI)² tangent covariance

    // ---- Linear system dump (populated when SolverConfig::dump_system = true) --
    // Two snapshots: warm-start (pre-solve) and converged (post-solve).
    // Each snapshot stores the tangent-space Jacobian J in CRS format, the
    // residual vector r, the negative gradient g = -JᵀWr, and the block map
    // describing which tangent columns belong to which parameter block.
    //
    // To reconstruct the normal-equation system in Python:
    //   J = scipy.sparse.csr_matrix((values, cols, row_ptr), shape=(nrows, ncols))
    //   H = J.T @ J          # normal equations (apply_loss_function=True → Huber baked in)
    //   g = J.T @ r          # gradient (NB: Ceres minimizes 0.5*||r||², gradient = JᵀWr)
    //
    // Variable ordering in tangent columns:
    //   ori knots (3 DoF each, SO3 manifold) → pos CPs (3 DoF each) →
    //   bias (6 DoF) → pitch_delta (1 DoF, only when lock_extrinsics=false)
    // This ordering is enforced via eval_opts.parameter_blocks in the dump call.

    // Pre-solve (warm-start) snapshot
    struct SystemDump {
        std::vector<double> jac_values;        // CRS non-zero values
        std::vector<int>    jac_cols;          // CRS column indices
        std::vector<int>    jac_row_ptr;       // CRS row pointers (size = nrows+1)
        int                 jac_num_rows{0};
        int                 jac_num_cols{0};   // tangent-space column count
        std::vector<double> residuals;         // r  (nrows)
        std::vector<double> gradient;          // -Jᵀr  (ncols)
        std::vector<BlockMapEntry> block_map;  // one entry per parameter block
        bool valid{false};
    };

    SystemDump dump_pre;   // evaluated at warm-start, before ceres::Solve()
    SystemDump dump_post;  // evaluated at converged point, after  ceres::Solve()
};

// ============================================================================
// Main solve function
// ============================================================================
SolverResult solve(
    const std::vector<RadarFrame>& radar_frames,
    const std::vector<ImuSample>& imu_samples,
    const std::vector<PreintFactor>& preint_factors,
    const SolverConfig& cfg,
    const ExtrinsicConfig& extrinsic,
    // Initial state (from Python P1-P3 initialisation)
    const std::vector<std::array<double, 3>>& init_pos_cps,
    const std::vector<std::array<double, 4>>& init_ori_knots,
    const std::array<double, 6>& init_biases,
    double t_ref,
    // Optional: MoCap heading samples [timestamp, yaw_rad]
    const std::vector<std::pair<double, double>>& heading_samples = {}
);

}  // namespace rio
