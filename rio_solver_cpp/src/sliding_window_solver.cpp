#include <rio/sliding_window_solver.h>
#include <rio/marginalization.h>
#include <rio/factors/radar_doppler.h>
#include <rio/factors/imu_accel.h>
#include <rio/factors/imu_gyro.h>
#include <rio/factors/gravity_direction.h>
#include <rio/factors/heading_prior.h>
#include <rio/factors/bias_prior.h>
#include <rio/factors/regularization.h>

#include <sophus/ceres_manifold.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <sstream>

namespace rio {

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
    traj_.extrinsic_euler_deg = {ext_.roll_deg, ext_.pitch_deg, ext_.yaw_deg};
    prior_.valid    = false;
    initialized_    = true;
}

// ============================================================================
// solve_window
// ============================================================================
SolverResult SlidingWindowSolver::solve_window(
    const std::vector<RadarFrame>& radar_frames,
    const std::vector<ImuSample>& imu_samples,
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

    // Effective t_ref for this window (for passing to the boundary prior logic)
    // The spline with pos_cps starting at pi0 has t_ref = global_t_ref + pi0*dt_pos,
    // and its first valid time is t_ref + (N_POS-1)*dt_pos = t_start.
    // We keep the GLOBAL t_ref in traj_, so pos_index / ori_index give global indices.

    // ---- Build Ceres problem ------------------------------------------------
    ceres::Problem problem;
    auto* so3_manifold = new Sophus::Manifold<Sophus::SO3>();

    for (int i = oi0; i <= oi1; ++i)
        problem.AddParameterBlock(traj_.ori_knot_data(i), 4, so3_manifold);
    for (int i = pi0; i <= pi1; ++i)
        problem.AddParameterBlock(traj_.pos_cp_data(i), 3);
    problem.AddParameterBlock(traj_.bias_data(), 6);

    // Extrinsics are kept fixed (same as solver.cpp — not part of the problem)
    Sophus::SO3d R_radar_to_body = extrinsic_R(ext_);
    Eigen::Vector3d t_body_sensor(ext_.tx, ext_.ty, ext_.tz);
    const double inv_dt_pos = 1.0 / cfg_.dt_pos;
    const double inv_dt_ori = 1.0 / cfg_.dt_ori;

    // ---- Boundary / marginalization prior -----------------------------------
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

        for (const auto& pt : frame.points) {
            double range = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
            if (range < cfg_.min_range) continue;
            Eigen::Vector3d u_sensor(pt.x/range, pt.y/range, pt.z/range);

            auto* f = new RadarDopplerFunctor(u_sensor, pt.v,
                                               u_ori, inv_dt_ori,
                                               u_pos, inv_dt_pos,
                                               R_radar_to_body, t_body_sensor);
            std::vector<int> sizes;
            for (int k = 0; k < N_ORI; ++k) sizes.push_back(4);
            for (int k = 0; k < N_POS; ++k) sizes.push_back(3);
            sizes.push_back(6);
            auto* cost = make_auto_cost_sw(f, 1, sizes);

            std::vector<double*> params;
            for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(ori0 + k));
            for (int k = 0; k < N_POS; ++k) params.push_back(traj_.pos_cp_data(pos0 + k));
            params.push_back(traj_.bias_data());
            problem.AddResidualBlock(cost, huber_loss, params);
        }
    }

    // ---- IMU factors --------------------------------------------------------
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
            auto* f = new GyroFunctor(z_gyro, u_ori, inv_dt_ori);
            std::vector<int> sizes;
            for (int k = 0; k < N_ORI; ++k) sizes.push_back(4);
            sizes.push_back(6);
            auto* cost = make_auto_cost_sw(f, 3, sizes);
            std::vector<double*> params;
            for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(ori0 + k));
            params.push_back(traj_.bias_data());
            problem.AddResidualBlock(cost,
                new ceres::ScaledLoss(nullptr, cfg_.lambda_gyro, ceres::TAKE_OWNERSHIP), params);
        }

        // Accel
        if (has_pos) {
            Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
            auto* f = new AccelFunctor(z_acc, u_ori, inv_dt_ori, u_pos, inv_dt_pos);
            std::vector<int> sizes;
            for (int k = 0; k < N_ORI; ++k) sizes.push_back(4);
            for (int k = 0; k < N_POS; ++k) sizes.push_back(3);
            sizes.push_back(6);
            auto* cost = make_auto_cost_sw(f, 3, sizes);
            std::vector<double*> params;
            for (int k = 0; k < N_ORI; ++k) params.push_back(traj_.ori_knot_data(ori0 + k));
            for (int k = 0; k < N_POS; ++k) params.push_back(traj_.pos_cp_data(pos0 + k));
            params.push_back(traj_.bias_data());
            auto* huber_a = new ceres::HuberLoss(cfg_.huber_delta_accel);
            problem.AddResidualBlock(cost,
                new ceres::ScaledLoss(huber_a, cfg_.lambda_accel, ceres::TAKE_OWNERSHIP), params);
        }

        // Gravity direction (when not accelerating hard)
        if (cfg_.lambda_gravity > 0.0 && has_pos) {
            Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
            double acc_norm = z_acc.norm();
            if (std::abs(acc_norm - 9.81) < cfg_.gravity_accel_threshold) {
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
    if (cfg_.lambda_bias_prior_accel > 0.0 || cfg_.lambda_bias_prior_gyro > 0.0) {
        double w = std::sqrt(cfg_.lambda_bias_prior_accel * cfg_.lambda_bias_prior_gyro);
        Eigen::Matrix<double, 6, 1> b0;
        for (int j = 0; j < 6; ++j) b0[j] = traj_.biases[j];
        auto* f = new BiasPriorFunctor(b0);
        auto* cost = make_auto_cost_sw(f, 6, {6});
        problem.AddResidualBlock(cost,
            new ceres::ScaledLoss(nullptr, w, ceres::TAKE_OWNERSHIP),
            {traj_.bias_data()});
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

    // ---- Solve -------------------------------------------------------------
    ceres::Solver::Options options;
    options.linear_solver_type              = ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.minimizer_type                  = ceres::TRUST_REGION;
    options.trust_region_strategy_type      = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations              = cfg_.max_iterations;
    options.num_threads                     = 4;
    options.minimizer_progress_to_stdout    = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // ---- Compute marginalization prior for next window ----------------------
    int k_stride_pos = std::max(1, static_cast<int>(std::round(stride / cfg_.dt_pos)));
    int k_stride_ori = std::max(1, static_cast<int>(std::round(stride / cfg_.dt_ori)));
    compute_prior(problem, pi0_raw, oi0_raw, k_stride_pos, k_stride_ori);

    // ---- Package result ----------------------------------------------------
    auto wall_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(wall_end - wall_start).count();

    SolverResult result;
    result.pos_cps    = traj_.pos_cps;
    result.ori_knots  = traj_.ori_knots;
    result.biases     = traj_.biases;
    result.extrinsic_euler_deg = traj_.extrinsic_euler_deg;
    result.solve_time_s = elapsed;
    result.cost_history.push_back(summary.initial_cost);
    result.cost_history.push_back(summary.final_cost);
    result.time_residual_eval_s = summary.residual_evaluation_time_in_seconds;
    result.time_jacobian_eval_s = summary.jacobian_evaluation_time_in_seconds;
    result.time_linear_solver_s = summary.linear_solver_time_in_seconds;
    std::ostringstream oss;
    oss << summary.BriefReport();
    result.solver_summary = oss.str();
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

    ceres::LossFunction* loss = nullptr;
    if (cfg_.marg_prior_scale != 1.0) {
        loss = new ceres::ScaledLoss(nullptr, cfg_.marg_prior_scale * cfg_.marg_prior_scale,
                                     ceres::TAKE_OWNERSHIP);
    }
    problem.AddResidualBlock(cost, loss, params);
}

// ============================================================================
// compute_prior  — Schur complement marginalization
// ============================================================================
void SlidingWindowSolver::compute_prior(
    ceres::Problem& problem,
    int pi0_raw, int oi0_raw,
    int k_stride_pos, int k_stride_ori)
{
    // Indices of marginalized variables (in the stride zone)
    //   These CPs/knots go out of support in the next window.
    int marg_pos_start = pi0_raw;
    int marg_pos_end   = pi0_raw + k_stride_pos - N_POS;   // inclusive
    int marg_ori_start = oi0_raw;
    int marg_ori_end   = oi0_raw + k_stride_ori - N_ORI;   // inclusive

    if (marg_pos_end < marg_pos_start || marg_ori_end < marg_ori_start) {
        // Stride too small to marginalize anything
        prior_.valid = false;
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

    // ---- Collect residuals touching marginalized OR boundary params ---------
    // Include boundary residuals so that prior propagated from window k-1
    // contributes correctly to H_bb in the Schur complement.
    std::unordered_set<ceres::ResidualBlockId> res_set;
    for (auto* ptr : eval_blocks) {
        std::vector<ceres::ResidualBlockId> rids;
        problem.GetResidualBlocksForParameterBlock(ptr, &rids);
        for (auto id : rids) res_set.insert(id);
    }
    if (res_set.empty()) {
        prior_.valid = false;
        return;
    }

    // ---- Evaluate restricted Jacobian ---------------------------------------
    ceres::Problem::EvaluateOptions eval_opts;
    eval_opts.apply_loss_function = true;
    eval_opts.parameter_blocks = eval_blocks;
    eval_opts.residual_blocks = std::vector<ceres::ResidualBlockId>(
        res_set.begin(), res_set.end());

    double cost;
    ceres::CRSMatrix J_crs;
    if (!problem.Evaluate(eval_opts, &cost, nullptr, nullptr, &J_crs)) {
        prior_.valid = false;
        return;
    }

    if (J_crs.num_cols != d_a + d_b) {
        // Column count mismatch (can happen if some blocks are constant)
        prior_.valid = false;
        return;
    }

    // ---- Convert CRS → dense (restricted Jacobian is small) ----------------
    const int nr = J_crs.num_rows;
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nr, d_a + d_b);
    for (int row = 0; row < nr; ++row)
        for (int idx = J_crs.rows[row]; idx < J_crs.rows[row + 1]; ++idx)
            J(row, J_crs.cols[idx]) = J_crs.values[idx];

    // ---- H = J^T J, split into blocks --------------------------------------
    Eigen::MatrixXd J_a = J.leftCols(d_a);
    Eigen::MatrixXd J_b = J.rightCols(d_b);

    Eigen::MatrixXd H_aa = J_a.transpose() * J_a;
    Eigen::MatrixXd H_ab = J_a.transpose() * J_b;
    Eigen::MatrixXd H_bb = J_b.transpose() * J_b;

    // ---- Schur complement: S = H_bb - H_ab^T * H_aa^{-1} * H_ab -----------
    const double reg_a = 1e-6;
    H_aa += reg_a * Eigen::MatrixXd::Identity(d_a, d_a);

    Eigen::LDLT<Eigen::MatrixXd> ldlt(H_aa);
    if (ldlt.info() != Eigen::Success) {
        prior_.valid = false;
        return;
    }
    Eigen::MatrixXd S = H_bb - H_ab.transpose() * ldlt.solve(H_ab);

    // ---- LLT Cholesky of S (PSD regularization) ----------------------------
    S += 1e-6 * Eigen::MatrixXd::Identity(d_b, d_b);
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    if (llt.info() != Eigen::Success) {
        prior_.valid = false;
        return;
    }

    // ---- Store prior --------------------------------------------------------
    prior_.valid     = true;
    prior_.sqrt_info = llt.matrixL();   // lower Cholesky L, S = L*L^T
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
