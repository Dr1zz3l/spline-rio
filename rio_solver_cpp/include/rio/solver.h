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

    // Optimizer
    int max_iterations{40};

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
