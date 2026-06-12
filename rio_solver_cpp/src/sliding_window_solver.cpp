#include <rio/sliding_window_solver.h>
#include <rio/marginalization.h>
#include <rio/factors/radar_doppler.h>
#include <rio/factors/imu_accel.h>
#include <rio/factors/imu_gyro.h>
#include <rio/factors/analytic/gyro_analytic.h>
#include <rio/factors/analytic/accel_analytic.h>
#include <rio/factors/analytic/radar_analytic.h>
#include <rio/factors/gravity_direction.h>
#include <rio/factors/heading_prior.h>
#include <rio/factors/bias_prior.h>
#include <rio/factors/regularization.h>
#include <rio/factors/imu_preintegration.h>

#include <sophus/ceres_manifold.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <sstream>
#include <thread>

namespace rio {

// ============================================================================
// dump_linear_system (SW) — same contract as the batch version in solver.cpp
// ============================================================================
static void dump_linear_system(
    ceres::Problem& problem,
    const std::vector<double*>& eval_blocks,
    const std::vector<BlockMapEntry>& block_map,
    SolverResult::SystemDump& out)
{
    out.valid = false;
    ceres::Problem::EvaluateOptions opts;
    opts.apply_loss_function = true;
    opts.parameter_blocks    = eval_blocks;

    double cost;
    std::vector<double> residuals, gradient;
    ceres::CRSMatrix J_crs;
    if (!problem.Evaluate(opts, &cost, &residuals, &gradient, &J_crs))
        return;

    out.jac_values   = std::move(J_crs.values);
    out.jac_cols     = std::move(J_crs.cols);
    out.jac_row_ptr  = std::move(J_crs.rows);
    out.jac_num_rows = J_crs.num_rows;
    out.jac_num_cols = J_crs.num_cols;
    out.residuals    = std::move(residuals);
    out.gradient     = std::move(gradient);
    out.block_map    = block_map;
    out.valid        = true;
}

// ============================================================================
// build_dump_blocks (SW) — builds eval_blocks and block_map for one window.
// Unlike the batch version, variable indices are relative to the window
// (oi0, pi0 are the first active knots/CPs in this window).
// Constant blocks (leading knots from the prior window) are excluded.
// ============================================================================
static void build_dump_blocks_window(
    Trajectory& traj,
    int oi0, int oi1,           // ori knot global index range [oi0, oi1]
    int pi0, int pi1,           // pos CP  global index range [pi0, pi1]
    bool optimize_ext,
    const std::unordered_set<double*>& constant_blocks, // blocks set constant by problem
    std::vector<double*>& eval_blocks,
    std::vector<BlockMapEntry>& block_map)
{
    eval_blocks.clear();
    block_map.clear();
    int col = 0;
    int local_ori_idx = 0;
    for (int i = oi0; i <= oi1; ++i) {
        double* ptr = traj.ori_knot_data(i);
        if (constant_blocks.count(ptr)) { ++local_ori_idx; continue; }
        eval_blocks.push_back(ptr);
        block_map.push_back({"ori_knot", i, col, 3});
        col += 3;
        ++local_ori_idx;
    }
    for (int i = pi0; i <= pi1; ++i) {
        double* ptr = traj.pos_cp_data(i);
        if (constant_blocks.count(ptr)) continue;
        eval_blocks.push_back(ptr);
        block_map.push_back({"pos_cp", i, col, 3});
        col += 3;
    }
    eval_blocks.push_back(traj.bias_data());
    block_map.push_back({"bias", 0, col, 6});
    col += 6;
    if (optimize_ext) {
        eval_blocks.push_back(traj.pitch_delta_data());
        block_map.push_back({"pitch_delta", 0, col, 1});
        col += 1;
    }
}

// ============================================================================
// Helpers (copied from solver.cpp to avoid header exposure)
// ============================================================================

static Sophus::SO3d extrinsic_R(const ExtrinsicConfig& ext) {
    double r = ext.roll_deg  * M_PI / 180.0;
    double p = ext.pitch_deg * M_PI / 180.0;
    double y = ext.yaw_deg   * M_PI / 180.0;
    Eigen::Matrix3d Rz, Ry, Rx;
    Rz << std::cos(y), -std::sin(y), 0,
          std::sin(y),  std::cos(y), 0,
          0,            0,           1;
    Ry << std::cos(p),  0, std::sin(p),
          0,            1, 0,
         -std::sin(p),  0, std::cos(p);
    Rx << 1, 0,           0,
          0, std::cos(r), -std::sin(r),
          0, std::sin(r),  std::cos(r);
    return Sophus::SO3d(Rz * Ry * Rx);
}

template <typename Functor>
static ceres::CostFunction* make_auto_cost_sw(Functor* f, int num_residuals,
                                               const std::vector<int>& param_sizes) {
    auto* cost = new ceres::DynamicAutoDiffCostFunction<Functor, 4>(f);
    cost->SetNumResiduals(num_residuals);
    for (int s : param_sizes) cost->AddParameterBlock(s);
    return cost;
}

// Analytical cost functions for high-frequency IMU factors (mirrors solver.cpp).
static inline ceres::CostFunction* make_gyro_cost(
    const Eigen::Vector3d& z_gyro, double u_ori, double inv_dt_ori) {
    return new analytic::GyroAnalyticFactor(z_gyro, u_ori, inv_dt_ori);
}
static inline ceres::CostFunction* make_accel_cost(
    const Eigen::Vector3d& z_acc, double u_ori, double inv_dt_ori,
    double u_pos, double inv_dt_pos) {
    return new analytic::AccelAnalyticFactor(z_acc, u_ori, inv_dt_ori, u_pos, inv_dt_pos);
}
static inline ceres::CostFunction* make_radar_cost(
    const Eigen::Vector3d& u_sensor, double v_meas,
    double u_ori, double inv_dt_ori, double u_pos, double inv_dt_pos,
    const Sophus::SO3d& R_rb, const Eigen::Vector3d& t_bs) {
    return new analytic::RadarAnalyticFactor(
        u_sensor, v_meas, u_ori, inv_dt_ori, u_pos, inv_dt_pos, R_rb, t_bs);
}
static inline ceres::CostFunction* make_radar_with_pitch_cost(
    const Eigen::Vector3d& u_sensor, double v_meas,
    double u_ori, double inv_dt_ori, double u_pos, double inv_dt_pos,
    const Sophus::SO3d& R_rb, const Eigen::Vector3d& t_bs) {
    return new analytic::RadarAnalyticWithPitchFactor(
        u_sensor, v_meas, u_ori, inv_dt_ori, u_pos, inv_dt_pos, R_rb, t_bs);
}

// ============================================================================
// SlidingWindowSolver
// ============================================================================

SlidingWindowSolver::SlidingWindowSolver(SolverConfig cfg, ExtrinsicConfig ext)
    : cfg_(cfg), ext_(ext) {}

void SlidingWindowSolver::initialize(
    const std::vector<std::array<double, 3>>& pos_cps,
    const std::vector<std::array<double, 4>>& ori_knots,
    const std::array<double, 6>& biases,
    double t_ref)
{
    traj_.t_ref     = t_ref;
    traj_.dt_pos    = cfg_.dt_pos;
    traj_.dt_ori    = cfg_.dt_ori;
    traj_.pos_cps   = pos_cps;
    traj_.ori_knots = ori_knots;
    traj_.biases    = biases;
    init_biases_    = biases;    // absolute anchor: never updated after initialization
    init_pos_cps_   = pos_cps;  // absolute anchor for position-init prior
    init_ori_knots_ = ori_knots; // reference shape for warm-start alignment
    prev_pi1_       = -1;
    prev_oi1_       = -1;
    traj_.extrinsic_euler_deg = {ext_.roll_deg, ext_.pitch_deg, ext_.yaw_deg};
    traj_.pitch_delta = 0.0;
    prior_.valid    = false;
    initialized_    = true;
}

// ============================================================================
// solve_window
// ============================================================================
SolverResult SlidingWindowSolver::solve_window(
    const std::vector<RadarFrame>& radar_frames,
    const std::vector<ImuSample>& imu_samples,
    const std::vector<PreintFactor>& preint_factors,
    const std::vector<std::pair<double, double>>& heading_samples,
    double t_start, double t_end, double stride)
{
    auto wall_start = std::chrono::high_resolution_clock::now();

    const int n_pos_total = traj_.n_pos_cps();
    const int n_ori_total = traj_.n_ori_knots();

    // ---- Compute window index range ----------------------------------------
    // "raw" start: the first CP index at or after t_start
    int pi0_raw = std::max(0, static_cast<int>(std::round(
        (t_start - traj_.t_ref) / traj_.dt_pos)));
    int oi0_raw = std::max(0, static_cast<int>(std::round(
        (t_start - traj_.t_ref) / traj_.dt_ori)));

    // "extended" start: include N_POS-1 / N_ORI-1 leading CPs for boundary prior
    int pi0 = std::max(0, pi0_raw - (N_POS - 1));
    int oi0 = std::max(0, oi0_raw - (N_ORI - 1));

    // End: include N_POS-1 / N_ORI-1 trailing CPs for spline support
    int pi1 = std::min(static_cast<int>(std::round(
        (t_end - traj_.t_ref) / traj_.dt_pos)) + N_POS - 1, n_pos_total - 1);
    int oi1 = std::min(static_cast<int>(std::round(
        (t_end - traj_.t_ref) / traj_.dt_ori)) + N_ORI - 1, n_ori_total - 1);

    const int n_win_pos = pi1 - pi0 + 1;
    const int n_win_ori = oi1 - oi0 + 1;

    // ---- Live-edge warm-start alignment (ROADMAP §1.2) ----------------------
    // CPs/knots entering the window for the first time still hold P1-P3 init
    // values dead-reckoned from t=0, which drift away from the solved
    // trajectory.  Additionally, the previous window's TRAILING spline-support
    // CPs (last N-1 indices) were only constrained by ~one knot interval of
    // data + regularization and can sit far off the trajectory.  Any CP-level
    // kink at this seam explodes through the min-snap term (~Δ/dt_pos⁴ →
    // initial cost ~1e13) and forces LM to spend ~half its iterations
    // recovering.
    //
    // Fix: rigidly align the init segment to the last *data-supported* CP of
    // the previous window (seam = prev_pi1_ - (N_POS-1)), and overwrite
    // everything past the seam — both the old weakly-constrained trailing
    // CPs and the newly entering CPs — with the aligned init shape.  This
    // preserves the locally-accurate P1-P3 shape (WLS-velocity / gyro
    // integration over the last stride) while removing absolute drift.
    // Note: init_pos_cps_ / init_ori_knots_ themselves are NOT modified — the
    // lambda_pos_init_prior anchor semantics are unchanged.
    if (cfg_.warm_start_align) {
        if (prev_pi1_ >= 0 && pi1 > prev_pi1_) {
            const int seam_p = std::max(pi0, prev_pi1_ - (N_POS - 1));
            std::array<double, 3> off;
            for (int c = 0; c < 3; ++c)
                off[c] = traj_.pos_cps[seam_p][c] - init_pos_cps_[seam_p][c];
            for (int i = seam_p + 1; i <= pi1; ++i)
                for (int c = 0; c < 3; ++c)
                    traj_.pos_cps[i][c] = init_pos_cps_[i][c] + off[c];
        }
        if (prev_oi1_ >= 0 && oi1 > prev_oi1_) {
            const int seam_o = std::max(oi0, prev_oi1_ - (N_ORI - 1));
            const auto& qs = traj_.ori_knots[seam_o];          // solved seam (xyzw)
            const auto& qi = init_ori_knots_[seam_o];          // init seam (xyzw)
            Sophus::SO3d R_solved(Eigen::Quaterniond(qs[3], qs[0], qs[1], qs[2]));
            Sophus::SO3d R_init  (Eigen::Quaterniond(qi[3], qi[0], qi[1], qi[2]));
            Sophus::SO3d R_off = R_solved * R_init.inverse();
            for (int i = seam_o + 1; i <= oi1; ++i) {
                const auto& q0 = init_ori_knots_[i];
                Sophus::SO3d R(Eigen::Quaterniond(q0[3], q0[0], q0[1], q0[2]));
                Eigen::Quaterniond q = (R_off * R).unit_quaternion();
                traj_.ori_knots[i] = {q.x(), q.y(), q.z(), q.w()};
            }
        }
    }

    // Effective t_ref for this window (for passing to the boundary prior logic)
    // The spline with pos_cps starting at pi0 has t_ref = global_t_ref + pi0*dt_pos,
    // and its first valid time is t_ref + (N_POS-1)*dt_pos = t_start.
    // We keep the GLOBAL t_ref in traj_, so pos_index / ori_index give global indices.

    // ---- Yaw gauge pre-alignment (closed-form, see SolverConfig docs) -------
    // Rotation about gravity is the exact gauge direction of the radar+IMU
    // cost; only heading priors respond.  Collapse the mean heading residual
    // analytically instead of letting LM traverse the curved valley.
    if (cfg_.yaw_prealign && !heading_samples.empty()) {
        const double inv_dt_o = 1.0 / cfg_.dt_ori;
        double sum_sin = 0.0, sum_cos = 0.0;
        int n_h = 0;
        for (const auto& [t_h, yaw_ref] : heading_samples) {
            double u; int i0;
            if (!traj_.ori_index(t_h, u, i0)) continue;
            if (i0 < oi0 || i0 + N_ORI - 1 > oi1) continue;
            const double* kp[N_ORI];
            for (int k = 0; k < N_ORI; ++k) kp[k] = traj_.ori_knot_data(i0 + k);
            Sophus::SO3d R;
            CeresSplineHelper<N_ORI>::template evaluate_lie<double, Sophus::SO3>(
                kp, u, inv_dt_o, &R, nullptr, nullptr);
            const Eigen::Matrix3d Rm = R.matrix();
            const double yaw_est = std::atan2(Rm(1, 0), Rm(0, 0));
            const double d = yaw_ref - yaw_est;
            sum_sin += std::sin(d);
            sum_cos += std::cos(d);
            ++n_h;
        }
        if (n_h > 0) {
            const double dpsi = cfg_.yaw_prealign_gain
                                * std::atan2(sum_sin / n_h, sum_cos / n_h);
            // Pivot: warm-start position at the window-start boundary time —
            // the same point the BoundaryPos anchor pins, so the anchor value
            // is unchanged by the rotation.
            const double t_pos_bnd = traj_.t_ref + pi0_raw * traj_.dt_pos;
            double u_c; int i0_c;
            Eigen::Vector3d pivot = Eigen::Vector3d::Zero();
            bool have_pivot = false;
            if (traj_.pos_index(t_pos_bnd, u_c, i0_c) &&
                i0_c >= pi0 && i0_c + N_POS - 1 <= pi1) {
                const double* cps[N_POS];
                for (int k = 0; k < N_POS; ++k) cps[k] = traj_.pos_cp_data(i0_c + k);
                CeresSplineHelper<N_POS>::template evaluate<double, 3, 0>(
                    cps, u_c, 1.0 / cfg_.dt_pos, &pivot);
                have_pivot = true;
            }
            if (have_pivot && std::abs(dpsi) > 1e-6) {
                const Sophus::SO3d Rz = Sophus::SO3d::rotZ(dpsi);
                for (int i = oi0; i <= oi1; ++i) {
                    const auto& q0 = traj_.ori_knots[i];
                    Sophus::SO3d R(Eigen::Quaterniond(q0[3], q0[0], q0[1], q0[2]));
                    Eigen::Quaterniond q = (Rz * R).unit_quaternion();
                    traj_.ori_knots[i] = {q.x(), q.y(), q.z(), q.w()};
                }
                const Eigen::Matrix3d Rzm = Rz.matrix();
                for (int i = pi0; i <= pi1; ++i) {
                    Eigen::Vector3d p(traj_.pos_cps[i][0], traj_.pos_cps[i][1],
                                      traj_.pos_cps[i][2]);
                    p = Rzm * (p - pivot) + pivot;
                    traj_.pos_cps[i] = {p.x(), p.y(), p.z()};
                }
            }
        }
    }

    // ---- Build Ceres problem ------------------------------------------------
    ceres::Problem problem;
    auto* so3_manifold = new Sophus::Manifold<Sophus::SO3>();

    for (int i = oi0; i <= oi1; ++i)
        problem.AddParameterBlock(traj_.ori_knot_data(i), 4, so3_manifold);
    for (int i = pi0; i <= pi1; ++i)
        problem.AddParameterBlock(traj_.pos_cp_data(i), 3);
    problem.AddParameterBlock(traj_.bias_data(), 6);

    // Pitch extrinsic parameter — persists across windows (slow calibration, like bias)
    const bool optimize_ext = !cfg_.lock_extrinsics;
    if (optimize_ext)
        problem.AddParameterBlock(traj_.pitch_delta_data(), 1);

    Sophus::SO3d R_radar_to_body = extrinsic_R(ext_);
    Eigen::Vector3d t_body_sensor(ext_.tx, ext_.ty, ext_.tz);
    const double inv_dt_pos = 1.0 / cfg_.dt_pos;
    const double inv_dt_ori = 1.0 / cfg_.dt_ori;

    // ---- Boundary / marginalization prior -----------------------------------
    const bool had_prior = prior_.valid;   // captured for compute_prior()
    if (prior_.valid) {
        // Re-linearize: shift the prior's center to the current warm-start so
        // it contributes curvature information without pulling toward a stale
        // historical estimate.  sqrt_info (the curvature shape) is unchanged.
        for (int i = 0; i < (int)prior_.bound_pos.size(); ++i)
            prior_.bound_pos[i] = traj_.pos_cps[prior_.pos_start + i];
        for (int i = 0; i < (int)prior_.bound_ori.size(); ++i)
            prior_.bound_ori[i] = traj_.ori_knots[prior_.ori_start + i];
        prior_.biases = traj_.biases;

        add_prior_to_problem(problem);
    } else {
        // First window: fix the leading CPs/knots constant (nothing to marginalize)
        for (int i = pi0; i < pi0_raw; ++i) problem.SetParameterBlockConstant(traj_.pos_cp_data(i));
        for (int i = oi0; i < oi0_raw; ++i) problem.SetParameterBlockConstant(traj_.ori_knot_data(i));
    }
    // Track which blocks are constant for the dump (used to exclude them from
    // eval_blocks so the Jacobian column count matches block_map).
    std::unordered_set<double*> constant_blocks;
    if (!prior_.valid) {
        for (int i = pi0; i < pi0_raw; ++i) constant_blocks.insert(traj_.pos_cp_data(i));
        for (int i = oi0; i < oi0_raw; ++i) constant_blocks.insert(traj_.ori_knot_data(i));
    }

    // ---- Radar Doppler factors ----------------------------------------------
    auto* huber_loss = new ceres::HuberLoss(cfg_.huber_delta);

    for (const auto& frame : radar_frames) {
        double u_ori; int ori0;
        if (!traj_.ori_index(frame.timestamp, u_ori, ori0)) continue;
        double u_pos; int pos0;
        if (!traj_.pos_index(frame.timestamp, u_pos, pos0)) continue;
        // Skip if any referenced block is outside our window
        if (ori0 < oi0 || ori0 + N_ORI - 1 > oi1) continue;
        if (pos0 < pi0 || pos0 + N_POS - 1 > pi1) continue;

        // ω-gate (hard skip) and/or ω-dependent noise inflation (soft gate).
        // Both evaluate |ω| from the warm-start spline at problem-build time.
        double w_omega = 1.0;
        Sophus::SO3d R_ws;            // warm-start rotation at frame time
        Eigen::Vector3d omega_ws = Eigen::Vector3d::Zero();
        if (cfg_.omega_gate_threshold > 0.0 || cfg_.omega_soft_sigma > 0.0) {
            const double* kp[N_ORI];
            for (int k = 0; k < N_ORI; ++k) kp[k] = traj_.ori_knot_data(ori0 + k);
            CeresSplineHelper<N_ORI>::template evaluate_lie<double, Sophus::SO3>(
                kp, u_ori, inv_dt_ori, &R_ws, &omega_ws, nullptr);
            const double w_norm = omega_ws.norm();
            if (cfg_.omega_gate_threshold > 0.0 && w_norm > cfg_.omega_gate_threshold)
                continue;
            if (cfg_.omega_soft_sigma > 0.0) {
                const double q = w_norm / cfg_.omega_soft_sigma;
                w_omega = 1.0 / (1.0 + q * q);   // σ_eff² = σ₀²(1 + q²)
            }
        }

        // Per-frame median intensity for median-relative per-point weighting
        // (radar_intensity_weight; median ⇒ global radar scale unchanged).
        double inv_med_intensity = 0.0;
        if (cfg_.radar_intensity_weight > 0.0) {
            std::vector<double> ints;
            ints.reserve(frame.points.size());
            for (const auto& pt : frame.points)
                if (pt.intensity > 0.0) ints.push_back(pt.intensity);
            if (ints.size() >= 3) {
                std::nth_element(ints.begin(), ints.begin() + ints.size()/2, ints.end());
                const double med = ints[ints.size()/2];
                if (med > 0.0) inv_med_intensity = 1.0 / med;
            }
        }

        for (const auto& pt : frame.points) {
            double range = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
            if (range < cfg_.min_range) continue;
            Eigen::Vector3d u_sensor(pt.x/range, pt.y/range, pt.z/range);
            const double v_meas_c = pt.v - cfg_.radar_zbias_fixed * u_sensor.z();

            // Per-point intensity weight w = clamp((I/I_med)^α, 0.25, 4)
            double w_int = 1.0;
            if (inv_med_intensity > 0.0 && pt.intensity > 0.0) {
                w_int = std::pow(pt.intensity * inv_med_intensity,
                                 cfg_.radar_intensity_weight);
                w_int = std::max(0.25, std::min(w_int, 4.0));
            }

            std::vector<double*> params;
            for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(ori0 + k));
            for (int k = 0; k < N_POS; ++k) params.push_back(traj_.pos_cp_data(pos0 + k));
            params.push_back(traj_.bias_data());

            ceres::CostFunction* cost;
            if (optimize_ext) {
                params.push_back(traj_.pitch_delta_data());
                cost = make_radar_with_pitch_cost(
                    u_sensor, v_meas_c,
                    u_ori, inv_dt_ori,
                    u_pos, inv_dt_pos,
                    R_radar_to_body, t_body_sensor);
            } else {
                cost = make_radar_cost(
                    u_sensor, v_meas_c,
                    u_ori, inv_dt_ori,
                    u_pos, inv_dt_pos,
                    R_radar_to_body, t_body_sensor);
            }
            // Soft gate and/or per-point intensity weight: wrap a per-point
            // ScaledLoss around an owned Huber (the shared huber_loss cannot
            // be wrapped — ownership).
            ceres::LossFunction* loss = huber_loss;
            if (w_omega < 1.0 || w_int != 1.0)
                loss = new ceres::ScaledLoss(
                    new ceres::HuberLoss(cfg_.huber_delta), w_omega * w_int,
                    ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost, loss, params);

            // Asymmetric split: complementary position-only radar factor with
            // orientation/ω frozen at warm-start (see radar_pos_split docs).
            if (cfg_.radar_pos_split > 0.0 && w_omega < 1.0) {
                const double w_split = (1.0 - w_omega) * cfg_.radar_pos_split * w_int;
                if (w_split > 1e-6) {
                    std::vector<double*> pparams;
                    for (int k = 0; k < N_POS; ++k)
                        pparams.push_back(traj_.pos_cp_data(pos0 + k));
                    auto* pcost = new analytic::RadarPosOnlyAnalyticFactor(
                        u_sensor, v_meas_c, u_pos, inv_dt_pos,
                        R_ws, omega_ws, R_radar_to_body, t_body_sensor);
                    problem.AddResidualBlock(pcost,
                        new ceres::ScaledLoss(
                            new ceres::HuberLoss(cfg_.huber_delta), w_split,
                            ceres::TAKE_OWNERSHIP),
                        pparams);
                }
            }
        }
    }

    // ---- IMU factors --------------------------------------------------------
    if (cfg_.use_preintegration && !preint_factors.empty()) {
        // ---- Preintegrated IMU factors (Forster TRO-2017) -------------------
        for (const auto& pf : preint_factors) {
            double u_ori_i; int oi0_i;
            if (!traj_.ori_index(pf.t_i, u_ori_i, oi0_i)) continue;
            double u_ori_j; int oi0_j;
            if (!traj_.ori_index(pf.t_j, u_ori_j, oi0_j)) continue;
            if (oi0_j != oi0_i + 1) continue;

            double u_pos_i; int pi0_i;
            if (!traj_.pos_index(pf.t_i, u_pos_i, pi0_i)) continue;
            double u_pos_j; int pi0_j;
            if (!traj_.pos_index(pf.t_j, u_pos_j, pi0_j)) continue;
            const int k_pos_stride = pi0_j - pi0_i;
            if (k_pos_stride <= 0) continue;

            // Window bounds check: all 5 ori knots + all pos CPs must be within window
            if (oi0_i < oi0 || oi0_i + 4 > oi1) continue;
            if (pi0_i < pi0 || pi0_i + N_POS + k_pos_stride - 1 > pi1) continue;

            const double scale_v = (cfg_.lambda_preint > 0.0)
                ? std::sqrt(cfg_.lambda_preint_v / cfg_.lambda_preint) : 0.0;
            const double scale_p = (cfg_.lambda_preint > 0.0)
                ? std::sqrt(cfg_.lambda_preint_p / cfg_.lambda_preint) : 0.0;
            auto* f = new IMUPreintegrationFunctor(
                pf, u_ori_i, u_ori_j, inv_dt_ori,
                    u_pos_i, u_pos_j, inv_dt_pos, k_pos_stride,
                    scale_v, scale_p);

            std::vector<int> sizes;
            for (int k = 0; k < 5; ++k) sizes.push_back(4);
            for (int k = 0; k < N_POS + k_pos_stride; ++k) sizes.push_back(3);
            sizes.push_back(6);

            auto* cost = make_auto_cost_sw(f, 9, sizes);

            std::vector<double*> params;
            for (int k = 0; k < 5; ++k)
                params.push_back(traj_.ori_knot_data(oi0_i + k));
            for (int k = 0; k < N_POS + k_pos_stride; ++k)
                params.push_back(traj_.pos_cp_data(pi0_i + k));
            params.push_back(traj_.bias_data());

            problem.AddResidualBlock(
                cost,
                new ceres::ScaledLoss(nullptr, cfg_.lambda_preint, ceres::TAKE_OWNERSHIP),
                params);
        }

        // Raw accel + gravity direction (position-orientation coupling;
        // preint r_R only replaces gyro, not accel).
        for (const auto& imu : imu_samples) {
            double u_ori; int ori0;
            if (!traj_.ori_index(imu.timestamp, u_ori, ori0)) continue;
            if (ori0 < oi0 || ori0 + N_ORI - 1 > oi1) continue;

            double u_pos; int pos0;
            bool has_pos = traj_.pos_index(imu.timestamp, u_pos, pos0);
            if (has_pos && (pos0 < pi0 || pos0 + N_POS - 1 > pi1)) has_pos = false;

            // Accel factor
            if (has_pos) {
                // ω-dependent accel down-weighting (mirrors the radar soft gate):
                // |ω| from the warm-start spline at build time.
                double w_acc = 1.0;
                if (cfg_.accel_soft_sigma > 0.0) {
                    const double* kp[N_ORI];
                    for (int k = 0; k < N_ORI; ++k) kp[k] = traj_.ori_knot_data(ori0 + k);
                    Sophus::SO3d dummy_R;
                    Eigen::Vector3d omega;
                    CeresSplineHelper<N_ORI>::template evaluate_lie<double, Sophus::SO3>(
                        kp, u_ori, inv_dt_ori, &dummy_R, &omega, nullptr);
                    const double q = omega.norm() / cfg_.accel_soft_sigma;
                    w_acc = 1.0 / (1.0 + q * q);   // σ_eff² = σ₀²(1 + q²)
                }
                Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
                auto* cost = make_accel_cost(z_acc, u_ori, inv_dt_ori, u_pos, inv_dt_pos);
                std::vector<double*> params;
                for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(ori0 + k));
                for (int k = 0; k < N_POS; ++k) params.push_back(traj_.pos_cp_data(pos0 + k));
                params.push_back(traj_.bias_data());
                auto* huber_a = new ceres::HuberLoss(cfg_.huber_delta_accel);
                problem.AddResidualBlock(cost,
                    new ceres::ScaledLoss(huber_a, cfg_.lambda_accel * w_acc, ceres::TAKE_OWNERSHIP), params);
            }

            // Gravity direction (when not accelerating hard)
            if (cfg_.lambda_gravity > 0.0 && has_pos) {
                Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
                if (std::abs(z_acc.norm() - 9.81) < cfg_.gravity_accel_threshold) {
                    auto* f = new GravityDirectionFunctor(z_acc, u_ori, inv_dt_ori);
                    std::vector<int> sizes;
                    for (int k = 0; k < N_ORI; ++k) sizes.push_back(4);
                    sizes.push_back(6);
                    auto* cost = make_auto_cost_sw(f, 3, sizes);
                    std::vector<double*> params;
                    for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(ori0 + k));
                    params.push_back(traj_.bias_data());
                    problem.AddResidualBlock(cost,
                        new ceres::ScaledLoss(nullptr, cfg_.lambda_gravity, ceres::TAKE_OWNERSHIP), params);
                }
            }
        }
    } else {
        // ---- Raw IMU factors (gyro + accel + gravity direction) -------------
        for (const auto& imu : imu_samples) {
            double u_ori; int ori0;
            if (!traj_.ori_index(imu.timestamp, u_ori, ori0)) continue;
            if (ori0 < oi0 || ori0 + N_ORI - 1 > oi1) continue;

            double u_pos; int pos0;
            bool has_pos = traj_.pos_index(imu.timestamp, u_pos, pos0);
            if (has_pos && (pos0 < pi0 || pos0 + N_POS - 1 > pi1)) has_pos = false;

            // Gyro
            {
                Eigen::Vector3d z_gyro(imu.gx, imu.gy, imu.gz);
                // ω-adaptive weight: trust the gyro more during fast rotation
                // (measured rate — direct and causal; see lambda_gyro_omega_sigma).
                double w_g = 1.0;
                if (cfg_.lambda_gyro_omega_sigma > 0.0) {
                    const double q = z_gyro.norm() / cfg_.lambda_gyro_omega_sigma;
                    w_g = 1.0 + std::pow(q, cfg_.lambda_gyro_omega_pow);
                }
                auto* cost = make_gyro_cost(z_gyro, u_ori, inv_dt_ori);
                std::vector<double*> params;
                for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(ori0 + k));
                params.push_back(traj_.bias_data());
                problem.AddResidualBlock(cost,
                    new ceres::ScaledLoss(nullptr, cfg_.lambda_gyro * w_g, ceres::TAKE_OWNERSHIP), params);
            }

            // Accel
            if (has_pos) {
                // ω-dependent accel down-weighting (mirrors the radar soft gate).
                double w_acc = 1.0;
                if (cfg_.accel_soft_sigma > 0.0) {
                    const double* kp[N_ORI];
                    for (int k = 0; k < N_ORI; ++k) kp[k] = traj_.ori_knot_data(ori0 + k);
                    Sophus::SO3d dummy_R;
                    Eigen::Vector3d omega;
                    CeresSplineHelper<N_ORI>::template evaluate_lie<double, Sophus::SO3>(
                        kp, u_ori, inv_dt_ori, &dummy_R, &omega, nullptr);
                    const double q = omega.norm() / cfg_.accel_soft_sigma;
                    w_acc = 1.0 / (1.0 + q * q);   // σ_eff² = σ₀²(1 + q²)
                }
                Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
                auto* cost = make_accel_cost(z_acc, u_ori, inv_dt_ori, u_pos, inv_dt_pos);
                std::vector<double*> params;
                for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(ori0 + k));
                for (int k = 0; k < N_POS; ++k) params.push_back(traj_.pos_cp_data(pos0 + k));
                params.push_back(traj_.bias_data());
                auto* huber_a = new ceres::HuberLoss(cfg_.huber_delta_accel);
                problem.AddResidualBlock(cost,
                    new ceres::ScaledLoss(huber_a, cfg_.lambda_accel * w_acc, ceres::TAKE_OWNERSHIP), params);
            }

            // Gravity direction (when not accelerating hard)
            if (cfg_.lambda_gravity > 0.0 && has_pos) {
                Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
                if (std::abs(z_acc.norm() - 9.81) < cfg_.gravity_accel_threshold) {
                    auto* f = new GravityDirectionFunctor(z_acc, u_ori, inv_dt_ori);
                    std::vector<int> sizes;
                    for (int k = 0; k < N_ORI; ++k) sizes.push_back(4);
                    sizes.push_back(6);
                    auto* cost = make_auto_cost_sw(f, 3, sizes);
                    std::vector<double*> params;
                    for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(ori0 + k));
                    params.push_back(traj_.bias_data());
                    problem.AddResidualBlock(cost,
                        new ceres::ScaledLoss(nullptr, cfg_.lambda_gravity, ceres::TAKE_OWNERSHIP), params);
                }
            }
        }
    }

    // ---- Min-snap regularization -------------------------------------------
    if (cfg_.lambda_snap_pos > 0.0) {
        for (int i = pi0; i + N_POS <= pi1; ++i) {
            auto* f = new MinSnapFunctor(0.5, inv_dt_pos);
            std::vector<int> sizes(N_POS, 3);
            auto* cost = make_auto_cost_sw(f, 3, sizes);
            std::vector<double*> params;
            for (int k = 0; k < N_POS; ++k) params.push_back(traj_.pos_cp_data(i + k));
            problem.AddResidualBlock(cost,
                new ceres::ScaledLoss(nullptr, cfg_.lambda_snap_pos, ceres::TAKE_OWNERSHIP), params);
        }
    }

    // ---- Orientation increment regularization ------------------------------
    if (cfg_.lambda_ori_reg > 0.0) {
        for (int i = oi0; i + 1 <= oi1; ++i) {
            auto* f = new OrientationRegFunctor();
            auto* cost = make_auto_cost_sw(f, 3, {4, 4});
            problem.AddResidualBlock(cost,
                new ceres::ScaledLoss(nullptr, cfg_.lambda_ori_reg, ceres::TAKE_OWNERSHIP),
                {traj_.ori_knot_data(i), traj_.ori_knot_data(i + 1)});
        }
    }

    // ---- Angular acceleration regularization --------------------------------
    if (cfg_.lambda_ori_accel > 0.0) {
        for (int i = oi0; i + 2 <= oi1; ++i) {
            auto* f = new AngularAccelRegFunctor();
            auto* cost = make_auto_cost_sw(f, 3, {4, 4, 4});
            problem.AddResidualBlock(cost,
                new ceres::ScaledLoss(nullptr, cfg_.lambda_ori_accel, ceres::TAKE_OWNERSHIP),
                {traj_.ori_knot_data(i),
                 traj_.ori_knot_data(i + 1),
                 traj_.ori_knot_data(i + 2)});
        }
    }

    // ---- Bias prior --------------------------------------------------------
    // Anchored to init_biases_ (stationary calibration estimate), NOT to the
    // current warm-start.  The marg prior is correctly re-centered each window
    // (it is curvature-only); the bias prior is an absolute sensor measurement
    // and must not drift with the warm-start.
    if (cfg_.lambda_bias_prior_accel > 0.0 || cfg_.lambda_bias_prior_gyro > 0.0) {
        Eigen::Matrix<double, 6, 1> b0;
        for (int j = 0; j < 6; ++j) b0[j] = init_biases_[j];
        // Per-component weights baked into the residual (no ScaledLoss):
        // accel components get sqrt(lambda_ba), gyro components sqrt(lambda_bg).
        auto* f = new BiasPriorFunctor(b0, cfg_.lambda_bias_prior_accel,
                                           cfg_.lambda_bias_prior_gyro);
        auto* cost = make_auto_cost_sw(f, 6, {6});
        problem.AddResidualBlock(cost, nullptr, {traj_.bias_data()});
    }

    // ---- Extrinsic pitch prior ---------------------------------------------
    if (optimize_ext && cfg_.lambda_extrinsic_prior > 0.0) {
        auto* f = new PitchDeltaPriorFunctor();
        auto* cost = make_auto_cost_sw(f, 1, {1});
        problem.AddResidualBlock(cost,
            new ceres::ScaledLoss(nullptr, cfg_.lambda_extrinsic_prior, ceres::TAKE_OWNERSHIP),
            {traj_.pitch_delta_data()});
    }

    // ---- Heading priors ----------------------------------------------------
    if (cfg_.lambda_heading > 0.0) {
        for (const auto& [t_h, yaw_ref] : heading_samples) {
            double u_ori; int ori0;
            if (!traj_.ori_index(t_h, u_ori, ori0)) continue;
            if (ori0 < oi0 || ori0 + N_ORI - 1 > oi1) continue;
            auto* f = new HeadingPriorFunctor(yaw_ref, u_ori, inv_dt_ori);
            std::vector<int> sizes(N_ORI, 4);
            auto* cost = make_auto_cost_sw(f, 1, sizes);
            std::vector<double*> params;
            for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(ori0 + k));
            problem.AddResidualBlock(cost,
                new ceres::ScaledLoss(nullptr, cfg_.lambda_heading, ceres::TAKE_OWNERSHIP), params);
        }
    }

    // ---- Boundary priors at the start of the active (non-leading) region ---
    // Pin the first valid point of the window to the current trajectory state.
    // This prevents the optimizer from drifting the boundary freely.
    {
        // Position boundary at t = traj_.t_ref + pi0_raw * dt_pos
        double t_pos_bnd = traj_.t_ref + pi0_raw * traj_.dt_pos;
        double t_ori_bnd = traj_.t_ref + oi0_raw * traj_.dt_ori;

        // Evaluate initial spline for anchor values
        auto eval_pos_init = [&](double u, int i0) {
            std::vector<const double*> cps(N_POS);
            for (int k = 0; k < N_POS; ++k) cps[k] = traj_.pos_cp_data(i0 + k);
            Eigen::Vector3d p;
            CeresSplineHelper<N_POS>::template evaluate<double, 3, 0>(cps.data(), u, inv_dt_pos, &p);
            return p;
        };
        auto eval_vel_init = [&](double u, int i0) {
            std::vector<const double*> cps(N_POS);
            for (int k = 0; k < N_POS; ++k) cps[k] = traj_.pos_cp_data(i0 + k);
            Eigen::Vector3d v;
            CeresSplineHelper<N_POS>::template evaluate<double, 3, 1>(cps.data(), u, inv_dt_pos, &v);
            return v;
        };
        auto eval_ori_init = [&](double u, int i0) {
            std::vector<const double*> qs(N_ORI);
            for (int k = 0; k < N_ORI; ++k) qs[k] = traj_.ori_knot_data(i0 + k);
            Sophus::SO3d R;
            CeresSplineHelper<N_ORI>::template evaluate_lie<double, Sophus::SO3>(
                qs.data(), u, inv_dt_ori, &R, nullptr, nullptr);
            return R;
        };

        if (cfg_.lambda_boundary_pos > 0.0) {
            double u; int i0;
            if (traj_.pos_index(t_pos_bnd, u, i0) && i0 >= pi0 && i0 + N_POS - 1 <= pi1) {
                Eigen::Vector3d p_anchor = eval_pos_init(u, i0);
                auto* f = new BoundaryPosFunctor(p_anchor, u, inv_dt_pos);
                std::vector<int> sizes(N_POS, 3);
                auto* cost = make_auto_cost_sw(f, 3, sizes);
                std::vector<double*> params;
                for (int k = 0; k < N_POS; ++k) params.push_back(traj_.pos_cp_data(i0 + k));
                problem.AddResidualBlock(cost,
                    new ceres::ScaledLoss(nullptr, cfg_.lambda_boundary_pos, ceres::TAKE_OWNERSHIP), params);
            }
        }
        if (cfg_.lambda_boundary_vel > 0.0) {
            double u; int i0;
            if (traj_.pos_index(t_pos_bnd, u, i0) && i0 >= pi0 && i0 + N_POS - 1 <= pi1) {
                Eigen::Vector3d v_anchor = eval_vel_init(u, i0);
                auto* f = new BoundaryVelFunctor(v_anchor, u, inv_dt_pos);
                std::vector<int> sizes(N_POS, 3);
                auto* cost = make_auto_cost_sw(f, 3, sizes);
                std::vector<double*> params;
                for (int k = 0; k < N_POS; ++k) params.push_back(traj_.pos_cp_data(i0 + k));
                problem.AddResidualBlock(cost,
                    new ceres::ScaledLoss(nullptr, cfg_.lambda_boundary_vel, ceres::TAKE_OWNERSHIP), params);
            }
        }
        if (cfg_.lambda_boundary_ori > 0.0) {
            double u; int i0;
            if (traj_.ori_index(t_ori_bnd, u, i0) && i0 >= oi0 && i0 + N_ORI - 1 <= oi1) {
                Sophus::SO3d R_anchor = eval_ori_init(u, i0);
                auto* f = new BoundaryOriFunctor(R_anchor, u, inv_dt_ori);
                std::vector<int> sizes(N_ORI, 4);
                auto* cost = make_auto_cost_sw(f, 3, sizes);
                std::vector<double*> params;
                for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(i0 + k));
                problem.AddResidualBlock(cost,
                    new ceres::ScaledLoss(nullptr, cfg_.lambda_boundary_ori, ceres::TAKE_OWNERSHIP), params);
            }
        }
    }

    // ---- Position-to-init prior (SW only, lambda_pos_init_prior > 0) ----------
    // Anchors every position CP in the window to its P1-P3 radar-velocity-
    // integrated init value (init_pos_cps_, captured in initialize()).
    // Each CP gets one cheap 3-residual factor (no spline evaluation, just
    // a direct CP-level penalty): r = cp_i - init_cp_i.
    //
    // Purpose: at dt_ori=0.008 (well-conditioned orientation, ~10 Hz radar)
    // the optimizer freely adjusts position to minimise radar residuals, but
    // ~10 Hz radar is too sparse to constrain position well → 7.2m drift.
    // A soft prior prevents that drift while the gyro + heading factors drive
    // the orientation improvement (target ~2m / ~10° for backflips).
    //
    // Scale guidance: 10 → position allowed to move ~0.3m from init per
    // window; 100 → ~0.1m; 1 → ~1m.  Start at 10 for backflips.
    if (cfg_.lambda_pos_init_prior > 0.0) {
        for (int i = pi0; i <= pi1; ++i) {
            Eigen::Vector3d p_init(
                init_pos_cps_[i][0],
                init_pos_cps_[i][1],
                init_pos_cps_[i][2]);
            auto* f = new PosInitPriorFunctor(p_init);
            auto* cost = make_auto_cost_sw(f, 3, {3});
            problem.AddResidualBlock(cost,
                new ceres::ScaledLoss(nullptr, cfg_.lambda_pos_init_prior,
                                      ceres::TAKE_OWNERSHIP),
                {traj_.pos_cp_data(i)});
        }
    }

    // ---- Optional: dump warm-start linear system (pre-solve) ----------------
    std::vector<double*>      dump_blocks;
    std::vector<BlockMapEntry> dump_bmap;
    if (cfg_.dump_system) {
        build_dump_blocks_window(traj_, oi0, oi1, pi0, pi1, optimize_ext,
                                 constant_blocks, dump_blocks, dump_bmap);
    }

    // ---- Solve -------------------------------------------------------------
    ceres::Solver::Options options;
    if (cfg_.use_banded_schur) {
        options.linear_solver_type = ceres::BANDED_SCHUR;
        // Globals = bias (6 DoF) + pitch_delta (1 DoF if extrinsics unlocked)
        options.banded_n_global_cols = 6 + (optimize_ext ? 1 : 0);
    } else {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    }
    options.minimizer_type                  = ceres::TRUST_REGION;
    options.trust_region_strategy_type      = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations              = cfg_.max_iterations;
    options.function_tolerance              = cfg_.function_tolerance;
    {
        int n = cfg_.num_threads;
        if (n <= 0) n = static_cast<int>(std::thread::hardware_concurrency());
        options.num_threads = std::max(1, n);
    }
    options.minimizer_progress_to_stdout    = false;

    // Declare result early so dump fields can be populated before and after solve.
    SolverResult result;
    if (cfg_.dump_system)
        dump_linear_system(problem, dump_blocks, dump_bmap, result.dump_pre);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (cfg_.dump_system)
        dump_linear_system(problem, dump_blocks, dump_bmap, result.dump_post);

    // ---- Live-edge joint covariance (NEES study, cfg.nees_covariance) ------
    // ceres::Covariance over the trailing N_POS pos CPs + N_ORI ori knots —
    // the blocks determining v(t_live), R(t_live).  Tangent-space blocks
    // (manifold-aware).  Offline use only (~0.1-0.5 s/window).
    if (cfg_.nees_covariance) {
        // Active support blocks at the live edge (t_end-eps): pos_index/ori_index
        // return i0 = first active CP/knot for that time.
        double u_dummy;
        int p0 = -1, o0 = -1;
        traj_.pos_index(t_end - 1e-6, u_dummy, p0);
        traj_.ori_index(t_end - 1e-6, u_dummy, o0);
        if (p0 >= pi0 && o0 >= oi0 &&
            p0 + N_POS - 1 <= pi1 && o0 + N_ORI - 1 <= oi1) {
            std::vector<const double*> blocks;
            for (int i = 0; i < N_POS; ++i) blocks.push_back(traj_.pos_cp_data(p0 + i));
            for (int i = 0; i < N_ORI; ++i) blocks.push_back(traj_.ori_knot_data(o0 + i));
            std::vector<std::pair<const double*, const double*>> pairs;
            for (size_t a = 0; a < blocks.size(); ++a)
                for (size_t b = a; b < blocks.size(); ++b)
                    pairs.emplace_back(blocks[a], blocks[b]);
            ceres::Covariance::Options cov_opts;
            cov_opts.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
            cov_opts.apply_loss_function = true;
            ceres::Covariance cov(cov_opts);
            if (cov.Compute(pairs, &problem)) {
                const int D = 3 * (int)blocks.size();
                result.nees_cov.setZero(D, D);
                double buf[9];
                bool ok = true;
                for (size_t a = 0; a < blocks.size() && ok; ++a)
                    for (size_t b = a; b < blocks.size() && ok; ++b) {
                        ok = cov.GetCovarianceBlockInTangentSpace(blocks[a], blocks[b], buf);
                        if (!ok) break;
                        for (int r = 0; r < 3; ++r)
                            for (int c2 = 0; c2 < 3; ++c2) {
                                result.nees_cov(3 * a + r, 3 * b + c2) = buf[r * 3 + c2];
                                result.nees_cov(3 * b + c2, 3 * a + r) = buf[r * 3 + c2];
                            }
                    }
                if (ok) {
                    result.nees_cov_valid = true;
                    result.nees_pos_idx0 = p0;
                    result.nees_ori_idx0 = o0;
                }
            }
        }
    }

    // ---- lock_gyro_bias: post-solve gyro clamp ----------------------------
    // Rank-deficient orientation windows (e.g. backflips at dt_ori=0.0008:
    // 1.25 DoF/gyro-constraint per window) can still move the gyro bias
    // despite the absolute prior, because the null-space of the underdetermined
    // orientation system allows simultaneous adjustments to Ω knots and b_g.
    // Clamping here resets gyro bias before compute_prior() bakes it into the
    // Schur complement boundary state, so the inter-window prior correctly
    // encodes the calibrated gyro estimate.
    if (cfg_.lock_gyro_bias) {
        for (int j = 3; j < 6; ++j)
            traj_.biases[j] = init_biases_[j];
    }

    // ---- Evaluate prior residual norm at solution ---------------------------
    // ||r||² = local_x^T * S * local_x: squared Mahalanobis distance between
    // the solved boundary state and the prior mean.  Useful for calibrating
    // marg_prior_cauchy_delta.  Evaluated BEFORE compute_prior() overwrites prior_.
    if (prior_.valid) {
        const int n_pos = static_cast<int>(prior_.bound_pos.size());
        const int n_ori = static_cast<int>(prior_.bound_ori.size());
        std::vector<double*> pparams;
        for (int i = 0; i < n_pos; ++i)
            pparams.push_back(traj_.pos_cp_data(prior_.pos_start + i));
        for (int i = 0; i < n_ori; ++i)
            pparams.push_back(traj_.ori_knot_data(prior_.ori_start + i));
        pparams.push_back(traj_.bias_data());
        Eigen::VectorXd r(prior_.d_b);
        MargPriorFunctor functor(prior_);
        std::vector<const double*> cpparams(pparams.begin(), pparams.end());
        functor(cpparams.data(), r.data());
        result.marg_prior_residual_norm = r.squaredNorm();
    }

    // ---- Compute marginalization prior for next window ----------------------
    // Note: compute_prior() also populates prior_.covariance = S^{-1} (the Schur
    // complement boundary covariance), which is used as boundary_covariance below.
    int k_stride_pos = std::max(1, static_cast<int>(std::round(stride / cfg_.dt_pos)));
    int k_stride_ori = std::max(1, static_cast<int>(std::round(stride / cfg_.dt_ori)));
    compute_prior(problem, pi0, oi0, pi0_raw, oi0_raw, k_stride_pos, k_stride_ori,
                  had_prior);

    // Record the window extent for the next call's warm-start alignment.
    prev_pi1_ = pi1;
    prev_oi1_ = oi1;

    // ---- Package result ----------------------------------------------------
    auto wall_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(wall_end - wall_start).count();

    result.pos_cps    = traj_.pos_cps;
    result.ori_knots  = traj_.ori_knots;
    result.biases     = traj_.biases;
    double final_pitch = ext_.pitch_deg + traj_.pitch_delta * (180.0 / M_PI);
    result.extrinsic_euler_deg = {ext_.roll_deg, final_pitch, ext_.yaw_deg};
    result.solve_time_s  = elapsed;
    result.num_iterations = summary.num_successful_steps;
    result.cost_history.push_back(summary.initial_cost);
    result.cost_history.push_back(summary.final_cost);
    result.time_residual_eval_s = summary.residual_evaluation_time_in_seconds;
    result.time_jacobian_eval_s = summary.jacobian_evaluation_time_in_seconds;
    result.time_linear_solver_s = summary.linear_solver_time_in_seconds;
    std::ostringstream oss;
    oss << summary.BriefReport();
    result.solver_summary = oss.str();

    // Marginalization prior diagnostics
    result.marg_prior_valid    = prior_.valid;
    result.marg_prior_dim      = prior_.d_b;
    result.marg_cond_number    = prior_.cond_number;
    result.marg_min_eigenvalue = prior_.min_eigenvalue;
    result.marg_max_eigenvalue = prior_.max_eigenvalue;
    result.marg_numerical_rank = prior_.numerical_rank;
    result.marg_drop_reason    = prior_.drop_reason;

    // Boundary state covariance diagnostics
    result.marg_trace_cov      = prior_.trace_cov;
    result.marg_adaptive_scale = prior_.adaptive_scale;
    result.marg_applied_scale  = prior_.last_adaptive_scale;

    // Two covariance views of boundary state (from compute_prior's restricted Jacobian)
    result.boundary_cov_valid   = prior_.valid && prior_.covariance.size() > 0;
    result.boundary_cov_trace   = prior_.trace_cov;          // S^{-1} trace
    result.window_cov_trace     = prior_.window_cov_trace;   // H_bb^{-1} trace
    if (result.boundary_cov_valid) {
        result.boundary_covariance = prior_.covariance;       // S^{-1}
        result.window_covariance   = prior_.window_covariance; // H_bb^{-1}
    }

    return result;
}

// ============================================================================
// add_prior_to_problem
// ============================================================================
void SlidingWindowSolver::add_prior_to_problem(ceres::Problem& problem) {
    if (!prior_.valid) return;

    const int n_pos = static_cast<int>(prior_.bound_pos.size());
    const int n_ori = static_cast<int>(prior_.bound_ori.size());

    auto* functor = new MargPriorFunctor(prior_);
    auto* cost = new ceres::DynamicAutoDiffCostFunction<MargPriorFunctor, 4>(functor);
    cost->SetNumResiduals(prior_.d_b);
    for (int i = 0; i < n_pos; ++i) cost->AddParameterBlock(3);
    for (int i = 0; i < n_ori; ++i) cost->AddParameterBlock(4);
    cost->AddParameterBlock(6);

    std::vector<double*> params;
    for (int i = 0; i < n_pos; ++i)
        params.push_back(traj_.pos_cp_data(prior_.pos_start + i));
    for (int i = 0; i < n_ori; ++i)
        params.push_back(traj_.ori_knot_data(prior_.ori_start + i));
    params.push_back(traj_.bias_data());

    // Compute final scale: optionally multiply by data-driven adaptive_scale.
    // adaptive_scale normalises max eigenvalue of S to lambda_boundary_pos,
    // making marg_prior_scale a relative fine-tuner instead of an absolute hack.
    double final_scale = cfg_.marg_prior_scale;
    if (cfg_.use_adaptive_marg_scale && prior_.adaptive_scale > 0.0)
        final_scale *= prior_.adaptive_scale;
    prior_.last_adaptive_scale = final_scale;

    ceres::LossFunction* inner = nullptr;
    if (cfg_.marg_prior_cauchy_delta > 0.0)
        inner = new ceres::CauchyLoss(cfg_.marg_prior_cauchy_delta);

    ceres::LossFunction* loss = nullptr;
    if (std::abs(final_scale - 1.0) > 1e-12 || inner != nullptr) {
        loss = new ceres::ScaledLoss(inner, final_scale * final_scale,
                                     ceres::TAKE_OWNERSHIP);
    }
    problem.AddResidualBlock(cost, loss, params);
}

// ============================================================================
// compute_prior  — Schur complement marginalization
// ============================================================================
void SlidingWindowSolver::compute_prior(
    ceres::Problem& problem,
    int pi0, int oi0,
    int pi0_raw, int oi0_raw,
    int k_stride_pos, int k_stride_ori,
    bool had_prior)
{
    // Indices of marginalized variables.
    //   These CPs/knots go out of support in the next window.
    //
    // Markov-blanket mode (cfg_.marg_markov_blanket): the marginalized set is
    // EVERYTHING that exists in this window but not in the next one — i.e. the
    // leading blocks [pi0, pi0_raw) (= previous boundary, free params when a
    // prior was attached) plus the stride zone.  The old behaviour started at
    // pi0_raw, silently *conditioning* on the leading blocks (treating them as
    // perfectly known) instead of marginalizing them.
    // In window 1 (no prior) the leading blocks are SetParameterBlockConstant,
    // so the marg set starts at pi0_raw as before (conditioning on genuinely
    // fixed blocks is exact).
    const bool mb = cfg_.marg_markov_blanket;
    int marg_pos_start = (mb && had_prior) ? pi0 : pi0_raw;
    int marg_pos_end   = pi0_raw + k_stride_pos - N_POS;   // inclusive
    int marg_ori_start = (mb && had_prior) ? oi0 : oi0_raw;
    int marg_ori_end   = oi0_raw + k_stride_ori - N_ORI;   // inclusive

    prior_.drop_reason = "";  // clear from previous call
    if (marg_pos_end < marg_pos_start || marg_ori_end < marg_ori_start) {
        prior_.valid = false;
        prior_.drop_reason = "stride too small";
        return;
    }

    int n_marg_pos = marg_pos_end - marg_pos_start + 1;  // e.g. 55
    int n_marg_ori = marg_ori_end - marg_ori_start + 1;  // e.g. 34

    // Indices of boundary variables (last N_POS-1 / N_ORI-1 in stride zone)
    int bound_pos_start = marg_pos_end + 1;               // e.g. pi0_raw + 55
    int bound_pos_end   = pi0_raw + k_stride_pos - 1;     // e.g. pi0_raw + 59
    int bound_ori_start = marg_ori_end + 1;
    int bound_ori_end   = oi0_raw + k_stride_ori - 1;

    int n_bound_pos = N_POS - 1;                          // = 5
    int n_bound_ori = N_ORI - 1;                          // = 3

    // Check bounds
    if (bound_pos_end >= traj_.n_pos_cps() || bound_ori_end >= traj_.n_ori_knots()) {
        prior_.valid = false;
        prior_.drop_reason = "boundary index out of range";
        return;
    }

    // d_a: local-param dimension of marginalized set (3 per block, all Euclidean or SO3)
    int d_a = 3 * n_marg_pos + 3 * n_marg_ori;
    // d_b: local-param dimension of boundary + bias
    int d_b = 3 * n_bound_pos + 3 * n_bound_ori + 6;

    // ---- Explicit parameter block ordering for Evaluate --------------------
    // Order: [marg_pos..., marg_ori..., bound_pos..., bound_ori..., bias]
    std::vector<double*> eval_blocks;
    for (int i = marg_pos_start; i <= marg_pos_end; ++i)
        eval_blocks.push_back(traj_.pos_cp_data(i));
    for (int i = marg_ori_start; i <= marg_ori_end; ++i)
        eval_blocks.push_back(traj_.ori_knot_data(i));
    for (int i = bound_pos_start; i <= bound_pos_end; ++i)
        eval_blocks.push_back(traj_.pos_cp_data(i));
    for (int i = bound_ori_start; i <= bound_ori_end; ++i)
        eval_blocks.push_back(traj_.ori_knot_data(i));
    eval_blocks.push_back(traj_.bias_data());

    // ---- Collect residuals for the Schur complement --------------------------
    std::unordered_set<ceres::ResidualBlockId> res_set;
    if (mb) {
        // Markov-blanket rule: include ONLY residuals touching >= 1 marginalized
        // block.  By B-spline locality, a factor touching marg block i has
        // support [i, i+N-1] ⊆ marg ∪ boundary, so the Schur complement is a
        // true marginal.  The previous MargPrior residual touches the leading
        // blocks and is picked up automatically.
        //
        // Residuals touching boundary/bias but NOT a marg block are excluded:
        // they are re-added to the next window's problem, and including them
        // here would double-count their information (the old behaviour pulled
        // in EVERY IMU factor of the window via the bias block — the root cause
        // of the overconfident prior that required marg_prior_scale ≈ 2e-4).
        std::vector<double*> marg_blocks;
        for (int i = marg_pos_start; i <= marg_pos_end; ++i)
            marg_blocks.push_back(traj_.pos_cp_data(i));
        for (int i = marg_ori_start; i <= marg_ori_end; ++i)
            marg_blocks.push_back(traj_.ori_knot_data(i));
        for (auto* ptr : marg_blocks) {
            std::vector<ceres::ResidualBlockId> rids;
            problem.GetResidualBlocksForParameterBlock(ptr, &rids);
            for (auto id : rids) res_set.insert(id);
        }

        // Support-closure filter: drop residuals touching any FREE block outside
        // marg ∪ boundary ∪ bias.  Two known cases:
        //  - pitch_delta (persistent global, not part of the prior): allowed as
        //    a documented conditioning approximation (slow-varying, strong prior).
        //  - pos/ori grid mismatch sliver: when (N_ORI-1)*dt_ori is within one
        //    knot of (N_POS-1)*dt_pos, a ~1 ms band of factors touches an ori
        //    marg block while its pos support reaches one CP past the boundary.
        //    Dropping them loses a sliver of information but keeps the prior
        //    consistent (standard treatment, cf. DSO/OKVIS).
        std::unordered_set<double*> allowed(eval_blocks.begin(), eval_blocks.end());
        if (!cfg_.lock_extrinsics)
            allowed.insert(traj_.pitch_delta_data());
        for (auto it = res_set.begin(); it != res_set.end(); ) {
            std::vector<double*> pbs;
            problem.GetParameterBlocksForResidualBlock(*it, &pbs);
            bool ok = true;
            for (auto* pb : pbs) {
                if (allowed.count(pb)) continue;
                if (problem.IsParameterBlockConstant(pb)) continue;  // exact
                ok = false;
                break;
            }
            it = ok ? std::next(it) : res_set.erase(it);
        }
    } else {
        // Legacy: residuals touching marginalized OR boundary params.
        // Conditions on interior states and double-counts boundary factors;
        // kept for A/B comparison (--set marg_markov_blanket=0).
        for (auto* ptr : eval_blocks) {
            std::vector<ceres::ResidualBlockId> rids;
            problem.GetResidualBlocksForParameterBlock(ptr, &rids);
            for (auto id : rids) res_set.insert(id);
        }
    }
    if (res_set.empty()) {
        prior_.valid = false;
        prior_.drop_reason = "no residuals touch marg/boundary blocks";
        return;
    }

    // ---- Form the dense Gauss-Newton Hessian over [a | b] columns -----------
    const int d_total = d_a + d_b;
    Eigen::MatrixXd H_full;

    if (cfg_.marg_fast_prior) {
        // Direct factor evaluation (ROADMAP: compute_prior direct evaluation).
        // Problem::Evaluate() rebuilds an internal Program + Evaluator on every
        // call (~0.1–0.4 s/window); instead, evaluate each res_set cost
        // function directly and accumulate H = Σ J̃ᵀJ̃ ourselves (~ms).
        // Semantics matched to Problem::Evaluate(apply_loss_function=true):
        //   - SO(3) blocks: ambient (N×4) Jacobian × manifold PlusJacobian
        //     (4×3) → tangent-space columns
        //   - robust losses: Triggs corrector (alpha term only when ρ'' > 0,
        //     same convention as ceres::internal::Corrector / VINS-Mono)
        //   - blocks outside [a|b] (pitch_delta, window-1 constants): fixed,
        //     their Jacobians are simply not accumulated.
        H_full = Eigen::MatrixXd::Zero(d_total, d_total);

        // Column layout, same order as eval_blocks: [marg_pos, marg_ori,
        // bound_pos, bound_ori, bias], 3 tangent columns per CP/knot, 6 bias.
        struct ColInfo { int col; bool is_so3; };
        std::unordered_map<const double*, ColInfo> colmap;
        {
            int col = 0;
            for (int i = marg_pos_start; i <= marg_pos_end; ++i, col += 3)
                colmap[traj_.pos_cp_data(i)] = {col, false};
            for (int i = marg_ori_start; i <= marg_ori_end; ++i, col += 3)
                colmap[traj_.ori_knot_data(i)] = {col, true};
            for (int i = bound_pos_start; i <= bound_pos_end; ++i, col += 3)
                colmap[traj_.pos_cp_data(i)] = {col, false};
            for (int i = bound_ori_start; i <= bound_ori_end; ++i, col += 3)
                colmap[traj_.ori_knot_data(i)] = {col, true};
            colmap[traj_.bias_data()] = {col, false};
        }

        Sophus::Manifold<Sophus::SO3> so3_manifold_local;
        using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>;

        std::vector<double*>         pbs;
        std::vector<const double*>   cpbs;
        std::vector<RowMat>          jac_bufs;
        std::vector<double*>         jac_ptrs;
        std::vector<Eigen::MatrixXd> jac_local;   // tangent-space, corrected
        std::vector<int>             jac_col;     // column offset (-1 = skip)
        Eigen::VectorXd r;

        for (const auto& rid : res_set) {
            const ceres::CostFunction* cf =
                problem.GetCostFunctionForResidualBlock(rid);
            const ceres::LossFunction* lf =
                problem.GetLossFunctionForResidualBlock(rid);
            problem.GetParameterBlocksForResidualBlock(rid, &pbs);
            const auto& psz = cf->parameter_block_sizes();
            const int nres  = cf->num_residuals();
            const int npb   = static_cast<int>(pbs.size());

            r.resize(nres);
            cpbs.assign(pbs.begin(), pbs.end());
            jac_bufs.resize(npb);
            jac_ptrs.assign(npb, nullptr);
            for (int i = 0; i < npb; ++i) {
                jac_bufs[i].resize(nres, psz[i]);
                jac_ptrs[i] = jac_bufs[i].data();
            }
            if (!cf->Evaluate(cpbs.data(), r.data(), jac_ptrs.data()))
                continue;

            // Robust-loss correction
            double sqrt_rho1 = 1.0, alpha_sq_norm = 0.0;
            if (lf) {
                const double sq_norm = r.squaredNorm();
                double rho[3];
                lf->Evaluate(sq_norm, rho);
                sqrt_rho1 = std::sqrt(rho[1]);
                if (rho[2] > 0.0 && sq_norm > 0.0) {
                    const double alpha =
                        1.0 - std::sqrt(1.0 + 2.0 * sq_norm * rho[2] / rho[1]);
                    alpha_sq_norm = alpha / sq_norm;
                }
            }

            // Tangent-space conversion + loss scaling, per block
            jac_local.assign(npb, Eigen::MatrixXd());
            jac_col.assign(npb, -1);
            for (int i = 0; i < npb; ++i) {
                auto it = colmap.find(pbs[i]);
                if (it == colmap.end()) continue;   // fixed: skip
                Eigen::MatrixXd Jl;
                if (it->second.is_so3) {
                    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> plus;
                    so3_manifold_local.PlusJacobian(pbs[i], plus.data());
                    Jl.noalias() = jac_bufs[i] * plus;
                } else {
                    Jl = jac_bufs[i];
                }
                if (lf) {
                    if (alpha_sq_norm != 0.0)
                        Jl -= alpha_sq_norm * (r * (r.transpose() * Jl));
                    Jl *= sqrt_rho1;
                }
                jac_local[i] = std::move(Jl);
                jac_col[i]   = it->second.col;
            }

            // Accumulate upper-triangular block products
            for (int i = 0; i < npb; ++i) {
                if (jac_col[i] < 0) continue;
                for (int j = i; j < npb; ++j) {
                    if (jac_col[j] < 0) continue;
                    int ci = jac_col[i], cj = jac_col[j];
                    const Eigen::MatrixXd* Ji = &jac_local[i];
                    const Eigen::MatrixXd* Jj = &jac_local[j];
                    if (ci > cj) { std::swap(ci, cj); std::swap(Ji, Jj); }
                    H_full.block(ci, cj, Ji->cols(), Jj->cols()).noalias() +=
                        Ji->transpose() * (*Jj);
                }
            }
        }
        // Mirror upper triangle to lower (diagonal blocks of repeated
        // parameter pairs were accumulated once with ci<=cj ordering).
        H_full.triangularView<Eigen::StrictlyLower>() =
            H_full.triangularView<Eigen::StrictlyUpper>().transpose();
    } else {
        // ---- Legacy: restricted Problem::Evaluate ---------------------------
        ceres::Problem::EvaluateOptions eval_opts;
        eval_opts.apply_loss_function = true;
        eval_opts.parameter_blocks = eval_blocks;
        eval_opts.residual_blocks = std::vector<ceres::ResidualBlockId>(
            res_set.begin(), res_set.end());

        double cost;
        ceres::CRSMatrix J_crs;
        if (!problem.Evaluate(eval_opts, &cost, nullptr, nullptr, &J_crs)) {
            prior_.valid = false;
            prior_.drop_reason = "problem.Evaluate() failed";
            return;
        }
        if (J_crs.num_cols != d_total) {
            prior_.valid = false;
            prior_.drop_reason = "J column count mismatch ("
                                 + std::to_string(J_crs.num_cols)
                                 + " vs " + std::to_string(d_total) + ")";
            return;
        }
        const int nr = J_crs.num_rows;
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nr, d_total);
        for (int row = 0; row < nr; ++row)
            for (int idx = J_crs.rows[row]; idx < J_crs.rows[row + 1]; ++idx)
                J(row, J_crs.cols[idx]) = J_crs.values[idx];
        H_full.noalias() = J.transpose() * J;
    }

    // ---- Split H into blocks ------------------------------------------------
    Eigen::MatrixXd H_aa = H_full.topLeftCorner(d_a, d_a);
    Eigen::MatrixXd H_ab = H_full.topRightCorner(d_a, d_b);
    Eigen::MatrixXd H_bb = H_full.bottomRightCorner(d_b, d_b);

    // ---- Current-window boundary covariance = H_bb^{-1} ---------------------
    // This is the covariance of the boundary state from the current window's
    // sensor data only (no marginalization of stride zone, no prior history).
    // Complement to S^{-1} which encodes the accumulated prior information.
    {
        Eigen::MatrixXd H_bb_reg = H_bb + 1e-6 * Eigen::MatrixXd::Identity(d_b, d_b);
        Eigen::LDLT<Eigen::MatrixXd> ldlt_bb(H_bb_reg);
        if (ldlt_bb.info() == Eigen::Success) {
            prior_.window_covariance = ldlt_bb.solve(Eigen::MatrixXd::Identity(d_b, d_b));
            prior_.window_cov_trace  = prior_.window_covariance.trace();
        }
    }

    // ---- Schur complement: S = H_bb - H_ab^T * H_aa^{-1} * H_ab -----------
    const double reg_a = 1e-6;
    H_aa += reg_a * Eigen::MatrixXd::Identity(d_a, d_a);

    Eigen::LDLT<Eigen::MatrixXd> ldlt(H_aa);
    if (ldlt.info() != Eigen::Success) {
        prior_.valid = false;
        prior_.drop_reason = "LDLT of H_aa failed (rank-deficient marginalized block)";
        return;
    }
    Eigen::MatrixXd S = H_bb - H_ab.transpose() * ldlt.solve(H_ab);

    // ---- PSD projection square root of S ------------------------------------
    // S = J^T J Schur complement is mathematically PSD, but floating-point
    // round-off produces small negative eigenvalues — especially with the
    // Markov-blanket res_set, where S is genuinely low-rank (boundary pos DOF
    // get little direct stride-zone information).  A plain LLT then fails
    // ("Schur complement not PSD") and the prior is dropped entirely.
    //
    // Instead: eigendecompose once, clamp eigenvalues to [0, eig_clip], and use
    //   sqrt_info = V · diag(sqrt(λ))      (so sqrt_info · sqrt_infoᵀ = S⁺)
    // MargPriorFunctor only requires sqrt_info · sqrt_infoᵀ = S (no triangular
    // structure), so this is a drop-in PSD-projection square root.  Negative
    // directions carry no information and are zeroed rather than failing.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(S);
    if (eig.info() != Eigen::Success) {
        prior_.valid = false;
        prior_.drop_reason = "eigendecomposition of S failed";
        return;
    }
    Eigen::VectorXd lam = eig.eigenvalues();
    // Optional max-eigenvalue clipping to balance prior across DOF (gyro-
    // dominated orientation eigenvalues ~5.5e14 vs position ~1e4).
    if (cfg_.marg_prior_eig_clip > 0.0)
        lam = lam.array().min(cfg_.marg_prior_eig_clip);
    lam = lam.array().max(0.0);   // PSD projection

    const double lmax = lam.maxCoeff();
    {
        const double lmin = lam.minCoeff();
        prior_.min_eigenvalue = lmin;
        prior_.max_eigenvalue = lmax;
        prior_.cond_number    = (lmin > 0.0) ? (lmax / lmin)
                                              : std::numeric_limits<double>::infinity();
        prior_.numerical_rank = static_cast<int>(
            (lam.array() > 1e-6 * lmax).count());

        // Adaptive scale: normalises max eigenvalue of S to lambda_boundary_pos.
        // With use_adaptive_marg_scale=true, final_scale = marg_prior_scale * adaptive_scale.
        if (lmax > 0.0)
            prior_.adaptive_scale = std::sqrt(cfg_.lambda_boundary_pos / lmax);
        else
            prior_.adaptive_scale = 1e-4;   // fallback
        prior_.adaptive_scale = std::max(prior_.adaptive_scale, 1e-8);
    }

    // sqrt_info = V · diag(sqrt(λ_clamped))
    prior_.sqrt_info = eig.eigenvectors() * lam.cwiseSqrt().asDiagonal();

    // Covariance: pseudo-inverse over the informative subspace (diagnostics only)
    {
        Eigen::VectorXd inv_lam = Eigen::VectorXd::Zero(d_b);
        const double floor_ev = std::max(1e-12 * lmax, 1e-300);
        for (int i = 0; i < d_b; ++i)
            if (lam[i] > floor_ev) inv_lam[i] = 1.0 / lam[i];
        prior_.covariance = eig.eigenvectors() * inv_lam.asDiagonal()
                            * eig.eigenvectors().transpose();
        prior_.trace_cov  = prior_.covariance.trace();
    }

    // ---- Store prior --------------------------------------------------------
    prior_.valid     = true;
    prior_.d_b       = d_b;

    prior_.bound_pos.resize(n_bound_pos);
    prior_.bound_ori.resize(n_bound_ori);
    for (int i = 0; i < n_bound_pos; ++i)
        prior_.bound_pos[i] = traj_.pos_cps[bound_pos_start + i];
    for (int i = 0; i < n_bound_ori; ++i)
        prior_.bound_ori[i] = traj_.ori_knots[bound_ori_start + i];
    prior_.biases     = traj_.biases;
    prior_.pos_start  = bound_pos_start;
    prior_.ori_start  = bound_ori_start;
}

}  // namespace rio
