#pragma once

// IsamSolver — incremental fixed-lag smoother backend (Phase 2).
// GTSAM IncrementalFixedLagSmoother over the continuous-time B-spline factor
// graph (factors.h), with random-walk bias (Phase-0c decision).  Sibling of the
// Ceres SlidingWindowSolver; fed NON-overlapping strides (the lag handles the
// window).  Validated Python prototype: analysis/isam_spike/test_phase0d.py.

#include <map>
#include <set>
#include <vector>
#include <array>
#include <utility>
#include <Eigen/Dense>

#include <rio/solver.h>          // RadarFrame, ImuSample, ExtrinsicConfig
#include <gtsam/nonlinear/Values.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

namespace rio {

struct IsamConfig {
    double dt_pos{0.005}, dt_ori{0.008};
    double lambda_accel{0.01}, lambda_gyro{4.0};
    double huber_delta{1.0};               // radar Huber (raw m/s)
    double lambda_snap_pos{2e-5}, lambda_ori_accel{0.1};
    double lambda_heading{0.6};
    double lambda_bias_prior{10000.0};
    double bias_rw_sigma{1e-3};
    double boundary_sigma{1e-3};           // strong gauge anchor on first knots
    double min_range{0.2};
    double lag{1.5};
    double relinearize_threshold{0.01};
    int    relinearize_skip{1};
    bool   use_qr{true};                    // QR vs Cholesky (conditioning)
    bool   fej{false};                      // pin marginalized-coupled linpoints
};

class IsamSolver {
public:
    IsamSolver(const IsamConfig& cfg, const ExtrinsicConfig& ext);

    // Store the full P1-P3 init (knots are pulled from here as they enter).
    void initialize(const std::vector<std::array<double, 3>>& pos_cps,
                    const std::vector<std::array<double, 4>>& ori_quats,
                    const std::array<double, 6>& biases, double t_ref);

    // Feed the NEW measurements in (t_prev, t_now]; t_now drives marginalization.
    // Returns the update wall-time (s).
    double update(const std::vector<RadarFrame>& radar,
                  const std::vector<ImuSample>& imu,
                  const std::vector<std::pair<double, double>>& heading,
                  double t_now);

    // Live-edge trajectory record (per-knot last estimate before marginalization).
    std::vector<std::pair<int, std::array<double, 4>>> ori_knots() const;
    std::vector<std::pair<int, std::array<double, 3>>> pos_cps() const;
    std::array<double, 6> biases() const { return last_bias_; }
    int num_active() const { return num_active_; }

private:
    bool ori_active(double t_abs, int& k, double& u) const;
    bool pos_active(double t_abs, int& k, double& u) const;
    bool in_domain(double t_abs) const;

    IsamConfig cfg_;
    ExtrinsicConfig ext_;
    Eigen::Matrix3d R_bs_;
    Eigen::Vector3d t_bs_;
    double inv_dt_ori_, inv_dt_pos_;

    std::vector<std::array<double, 3>> init_pos_;
    std::vector<std::array<double, 4>> init_ori_;
    std::array<double, 6> init_bias_{}, last_bias_{};
    double t_ref_{0.0};
    int n_ori_{0}, n_pos_{0};

    std::unique_ptr<gtsam::IncrementalFixedLagSmoother> smoother_;
    std::set<int> added_ori_, added_pos_, added_snap_, added_aacc_;
    int bias_k_{0};
    bool first_{true};

    std::map<int, std::array<double, 4>> live_ori_;
    std::map<int, std::array<double, 3>> live_pos_;
    int num_active_{0};

    // noise models
    gtsam::SharedNoiseModel n_acc_, n_gyr_, n_radar_, n_snap_, n_aacc_, n_heading_,
                            n_bias_prior_, n_bias_rw_, n_boundary_r_, n_boundary_p_;
};

}  // namespace rio
