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
// Solver output
// ============================================================================
struct SolverResult {
    // Optimised trajectory state (same size as input)
    std::vector<std::array<double, 3>> pos_cps;
    std::vector<std::array<double, 4>> ori_knots;
    std::array<double, 6> biases{};
    std::array<double, 3> extrinsic_euler_deg{};

    // Diagnostics
    std::vector<double> cost_history;   // one entry per accepted iteration
    double solve_time_s{0.0};
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
