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
    double lambda_ori_reg{0.001};

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
};

// ============================================================================
// Main solve function
// ============================================================================
SolverResult solve(
    const std::vector<RadarFrame>& radar_frames,
    const std::vector<ImuSample>& imu_samples,
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
