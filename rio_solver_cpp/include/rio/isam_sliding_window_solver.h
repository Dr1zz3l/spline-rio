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
    // --- universal weighting (aggressive dynamics / backflips); 0 = off ---
    double lambda_gyro_omega_sigma{0.0};  // gyro stiffen: w_g = 1+(|z_gyro|/sigma)^pow
    double lambda_gyro_omega_pow{2.0};
    double omega_soft_sigma{0.0};         // radar down-weight: w=1/(1+(|w|/sigma)^2)
    double accel_soft_sigma{0.0};         // accel down-weight (same form)
    bool   radar_zbias_estimate{false};   // estimate the radar elevation Doppler bias as
                                          // a self-calibrating state b_z (v_corr = v -
                                          // b_z*u_sz, init 0). Needs the floor anchor on
                                          // for observability (RadarBiasFactor). The old
                                          // hardcoded radar_zbias_fixed was retired here:
                                          // the floor anchor makes b=0 optimal (the -1.5
                                          // backflips value was a non-physical proxy for
                                          // the vertical drift the floor now fixes). The
                                          // Ceres SW backend keeps it (no floor anchor;
                                          // reproduces the frozen paper). ALGO_IMPROVEMENTS D6.
    double radar_intensity_weight{0.0};   // per-point w_int=clamp((I/I_med)^a,0.25,4)
    double lambda_pos_init_prior{0.0};    // per-CP tether to the (aligned) init
    double radar_pos_split{0.0};          // gated radar's (1-w) weight -> position-
                                          // only factor (frozen R,w): keep radar
                                          // velocity for position during flips
    // --- floor-plane absolute-z anchor (plane mapping); 0 = off ---
    double lambda_floor{0.0};             // weight of the floor point-to-plane factor
    double floor_z{0.0};                  // world height of the floor plane (init frame)
    double floor_band{0.5};               // |z_pred_world - floor_ref| < band => floor
    double floor_huber{0.3};              // Huber delta on the floor residual (m)
    bool   floor_cluster{false};          // Phase 1b: classify floor returns by the
                                          // drift-invariant LOWEST z-cluster per stride
                                          // ([z_lo, z_lo+floor_slab], gated |z_lo-f0| <
                                          // floor_band) instead of the per-point absolute
                                          // band -> removes the per-bag band (the band &
                                          // slab become universal physical constants).
    double floor_slab{0.4};               // cluster width above z_lo (floor thickness, m)
    bool   floor_free{false};             // Phase 1: estimate the floor offset as a
                                          // persistent landmark f0 (FK(0)) instead of
                                          // hardcoding floor_z; bootstrapped from the
                                          // first stride's lowest return cluster, then
                                          // the band gates on the current f0 estimate
                                          // (self-calibrating: no floor_z / start-height
                                          // knowledge, robust to warm-start z error).
    double boundary_sigma{1e-3};           // strong gauge anchor on first knots
    double min_range{0.2};
    double lag{1.5};
    double relinearize_threshold{0.01};
    int    relinearize_skip{1};
    int    extra_iters{0};        // extra empty ISAM2 updates/stride to converge
                                  // (roll recovery from the P1-P3 init; the Ceres
                                  // SW re-solves each window to convergence)
    double extra_iters_rtol{0.0}; // early-stop the extra updates when the relative
                                  // error reduction < rtol (0 = always do all
                                  // extra_iters; >0 reclaims compute on easy strides)
                                  // NOTE: cost-based -> FAILS (orientation hidden under
                                  // the dominant gyro/radar residuals); use _dnorm.
    double extra_iters_dnorm{0.0};// early-stop when the ISAM2 step (max-abs getDelta())
                                  // < dnorm (0 = off). Step-norm catches the slow-mode
                                  // (roll/pitch) convergence the cost criterion misses,
                                  // so easy strides stop at ~1 iter, hard ones run full.
    int    adapt_noise_stride{0}; // NIS-adaptive noise: every N strides, set each
                                  // sensor's sigma = std of its residuals at the
                                  // solution (data-driven whitening; 0 = off)
    double adapt_noise_alpha{0.3};// EMA smoothing for the sigma estimates
    bool   use_qr{true};                    // QR vs Cholesky (conditioning)
    bool   fej{false};                      // pin marginalized-coupled linpoints
    // NOTE: "selective FEJ" (exclude ori knots from the marginal FEJ-freeze) was
    // tried and REJECTED (2026-06-26, ISAM2_MIGRATION.md "Selective FEJ"): worse
    // everywhere (backflips 10.7->13.6 deg, slow_racing 1.39->6.66 deg). The freeze
    // is load-bearing for marginal validity; letting the boundary drift from its
    // linpoint makes the stale linear marginal mis-constrain. No flag retained.
    bool   warm_start_align{true};          // align entering knots to solved
                                            // boundary; OFF -> enter at raw init
                                            // (better for flips: gyro init beats
                                            // the propagated solver orientation)
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
    int num_fixed() const;   // FEJ: variables whose linearization is frozen by marg
    int num_floor() const { return n_floor_; }  // cumulative floor factors added
    double floor_offset() const { return floor_off_est_; }  // estimated f0 (free floor)
    double zbias() const { return bz_est_; }     // estimated radar z-velocity bias b_z

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
    int n_floor_{0};
    bool floor_init_{false};       // FK(0) inserted (Phase-1 free-offset floor)
    double floor_off_est_{0.0};    // current estimate of the floor offset f0
    std::vector<double> floor_cand_;  // bootstrap buffer (lowest-cluster -> f0 init)
    int floor_boot_strides_{0};
    double floor_zlo_{0.0};           // this stride's floor level (Phase-1b cluster)
    bool   bz_init_{false};           // BZK(0) inserted (estimated radar z-bias)
    double bz_est_{0.0};              // current estimate of b_z

    std::map<int, std::array<double, 4>> live_ori_;
    std::map<int, std::array<double, 3>> live_pos_;
    int num_active_{0};

    // NIS-adaptive noise: current per-sensor effective sigma (EMA), data-driven.
    double sigma_g_{0.0}, sigma_a_{0.0}, sigma_r_{0.0};
    int stride_count_{0};
    gtsam::Key bkey_last_{0};
    void adapt_noise(const gtsam::Values& est,
                     const std::vector<RadarFrame>& radar,
                     const std::vector<ImuSample>& imu);

    // noise models
    gtsam::SharedNoiseModel n_acc_, n_gyr_, n_radar_, n_snap_, n_aacc_, n_heading_,
                            n_bias_prior_, n_bias_rw_, n_boundary_r_, n_boundary_p_;
};

}  // namespace rio
