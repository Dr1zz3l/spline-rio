#include <rio/solver.h>
#include <rio/trajectory.h>
#include <rio/factors/radar_doppler.h>
#include <rio/factors/imu_accel.h>
#include <rio/factors/imu_gyro.h>
#include <rio/factors/gravity_direction.h>
#include <rio/factors/heading_prior.h>
#include <rio/factors/bias_prior.h>
#include <rio/factors/regularization.h>

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <sophus/ceres_manifold.hpp>
#include <ceres/ceres.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

namespace rio {

// ============================================================================
// ExtrinsicConfig::R_radar_to_body
// ============================================================================
Sophus::SO3d ExtrinsicConfig::R_radar_to_body() const {
    // Build from Euler angles [roll, pitch, yaw] in degrees (ZYX convention)
    double r = roll_deg  * M_PI / 180.0;
    double p = pitch_deg * M_PI / 180.0;
    double y = yaw_deg   * M_PI / 180.0;
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

// ============================================================================
// Helper: build DynamicAutoDiffCostFunction from a functor
// ============================================================================
template <typename Functor>
ceres::CostFunction* make_auto_cost(Functor* f,
                                    int num_residuals,
                                    const std::vector<int>& param_sizes) {
    auto* cost = new ceres::DynamicAutoDiffCostFunction<Functor, 4>(f);
    cost->SetNumResiduals(num_residuals);
    for (int s : param_sizes)
        cost->AddParameterBlock(s);
    return cost;
}

// ============================================================================
// solve()
// ============================================================================
SolverResult solve(
    const std::vector<RadarFrame>& radar_frames,
    const std::vector<ImuSample>& imu_samples,
    const SolverConfig& cfg,
    const ExtrinsicConfig& extrinsic,
    const std::vector<std::array<double, 3>>& init_pos_cps,
    const std::vector<std::array<double, 4>>& init_ori_knots,
    const std::array<double, 6>& init_biases,
    double t_ref,
    const std::vector<std::pair<double, double>>& heading_samples)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    // ---- Copy initial state into Trajectory ---------------------------------
    Trajectory traj;
    traj.t_ref = t_ref;
    traj.dt_pos = cfg.dt_pos;
    traj.dt_ori = cfg.dt_ori;
    traj.pos_cps = init_pos_cps;
    traj.ori_knots = init_ori_knots;
    traj.biases = init_biases;
    traj.extrinsic_euler_deg = {extrinsic.roll_deg, extrinsic.pitch_deg, extrinsic.yaw_deg};

    const int n_pos = traj.n_pos_cps();
    const int n_ori = traj.n_ori_knots();

    Sophus::SO3d R_radar_to_body = extrinsic.R_radar_to_body();
    Eigen::Vector3d t_body_sensor(extrinsic.tx, extrinsic.ty, extrinsic.tz);

    // ---- Build Ceres problem ------------------------------------------------
    ceres::Problem problem;

    // Manifold for SO3 quaternion knots
    auto* so3_manifold = new Sophus::Manifold<Sophus::SO3>();

    // Add orientation parameter blocks with SO3 manifold
    for (int i = 0; i < n_ori; ++i) {
        problem.AddParameterBlock(traj.ori_knot_data(i), 4, so3_manifold);
    }

    // Add position parameter blocks
    for (int i = 0; i < n_pos; ++i) {
        problem.AddParameterBlock(traj.pos_cp_data(i), 3);
    }

    // Add bias block
    problem.AddParameterBlock(traj.bias_data(), 6);

    // Sliding window: freeze leading knots (trusted from previous window solve)
    for (int i = 0; i < cfg.n_fix_leading_pos && i < n_pos; ++i)
        problem.SetParameterBlockConstant(traj.pos_cp_data(i));
    for (int i = 0; i < cfg.n_fix_leading_ori && i < n_ori; ++i)
        problem.SetParameterBlockConstant(traj.ori_knot_data(i));

    const double inv_dt_pos = 1.0 / cfg.dt_pos;
    const double inv_dt_ori = 1.0 / cfg.dt_ori;

    // ---- Radar Doppler factors -----------------------------------------------
    auto* huber_loss = new ceres::HuberLoss(cfg.huber_delta);

    for (const auto& frame : radar_frames) {
        double u_ori;
        int ori0;
        if (!traj.ori_index(frame.timestamp, u_ori, ori0)) continue;

        double u_pos;
        int pos0;
        if (!traj.pos_index(frame.timestamp, u_pos, pos0)) continue;

        for (const auto& pt : frame.points) {
            // Range filter
            double range = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
            if (range < cfg.min_range) continue;

            Eigen::Vector3d u_sensor(pt.x / range, pt.y / range, pt.z / range);

            auto* f = new RadarDopplerFunctor(
                u_sensor, pt.v,
                u_ori, inv_dt_ori,
                u_pos, inv_dt_pos,
                R_radar_to_body, t_body_sensor);

            // Parameter block sizes: N_ORI * 4, then N_POS * 3, then 6 (bias)
            std::vector<int> sizes;
            for (int i = 0; i < N_ORI; ++i) sizes.push_back(4);
            for (int i = 0; i < N_POS; ++i) sizes.push_back(3);
            sizes.push_back(6);

            auto* cost = make_auto_cost(f, 1, sizes);

            // Build parameter block list
            std::vector<double*> param_blocks;
            for (int i = 0; i < N_ORI; ++i)
                param_blocks.push_back(traj.ori_knot_data(ori0 + i));
            for (int i = 0; i < N_POS; ++i)
                param_blocks.push_back(traj.pos_cp_data(pos0 + i));
            param_blocks.push_back(traj.bias_data());

            problem.AddResidualBlock(cost, huber_loss, param_blocks);
        }
    }

    // ---- IMU factors ---------------------------------------------------------
    auto* huber_accel = new ceres::HuberLoss(cfg.huber_delta_accel);

    for (const auto& imu : imu_samples) {
        double u_ori;
        int ori0;
        if (!traj.ori_index(imu.timestamp, u_ori, ori0)) continue;

        double u_pos;
        int pos0;
        bool has_pos = traj.pos_index(imu.timestamp, u_pos, pos0);

        // Gyro factor (orientation only)
        {
            Eigen::Vector3d z_gyro(imu.gx, imu.gy, imu.gz);
            auto* f = new GyroFunctor(z_gyro, u_ori, inv_dt_ori);

            std::vector<int> sizes;
            for (int i = 0; i < N_ORI; ++i) sizes.push_back(4);
            sizes.push_back(6);

            auto* cost = make_auto_cost(f, 3, sizes);

            std::vector<double*> params;
            for (int i = 0; i < N_ORI; ++i)
                params.push_back(traj.ori_knot_data(ori0 + i));
            params.push_back(traj.bias_data());

            // Apply lambda_gyro as inverse-sigma (sqrt(lambda) * residual)
            auto* scaled = new ceres::ScaledLoss(nullptr, cfg.lambda_gyro,
                                                  ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost, scaled, params);
        }

        // Accel factor (needs position too)
        if (has_pos) {
            Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
            auto* f = new AccelFunctor(z_acc, u_ori, inv_dt_ori, u_pos, inv_dt_pos);

            std::vector<int> sizes;
            for (int i = 0; i < N_ORI; ++i) sizes.push_back(4);
            for (int i = 0; i < N_POS; ++i) sizes.push_back(3);
            sizes.push_back(6);

            auto* cost = make_auto_cost(f, 3, sizes);

            std::vector<double*> params;
            for (int i = 0; i < N_ORI; ++i)
                params.push_back(traj.ori_knot_data(ori0 + i));
            for (int i = 0; i < N_POS; ++i)
                params.push_back(traj.pos_cp_data(pos0 + i));
            params.push_back(traj.bias_data());

            // lambda_accel via ScaledLoss(HuberLoss)
            auto* loss = new ceres::ScaledLoss(
                new ceres::HuberLoss(cfg.huber_delta_accel),
                cfg.lambda_accel,
                ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost, loss, params);
        }

        // Gravity direction factor
        if (cfg.lambda_gravity > 0.0 && has_pos) {
            Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
            // Dynamic trust: skip if accelerometer shows strong dynamics
            double a_norm = z_acc.norm();
            double dev = std::abs(a_norm - 9.81);
            if (dev < cfg.gravity_accel_threshold) {
                auto* f = new GravityDirectionFunctor(z_acc, u_ori, inv_dt_ori);

                std::vector<int> sizes;
                for (int i = 0; i < N_ORI; ++i) sizes.push_back(4);
                sizes.push_back(6);

                auto* cost = make_auto_cost(f, 3, sizes);

                std::vector<double*> params;
                for (int i = 0; i < N_ORI; ++i)
                    params.push_back(traj.ori_knot_data(ori0 + i));
                params.push_back(traj.bias_data());

                auto* loss = new ceres::ScaledLoss(nullptr, cfg.lambda_gravity,
                                                    ceres::TAKE_OWNERSHIP);
                problem.AddResidualBlock(cost, loss, params);
            }
        }
    }

    // ---- Min-snap regularization -------------------------------------------
    if (cfg.lambda_snap_pos > 0.0) {
        // Sample at interior of each segment
        int n_segs = n_pos - N_POS;
        for (int seg = 0; seg < n_segs; ++seg) {
            double u = 0.5;
            int pos0 = seg;
            auto* f = new MinSnapFunctor(u, inv_dt_pos);

            std::vector<int> sizes;
            for (int i = 0; i < N_POS; ++i) sizes.push_back(3);

            auto* cost = make_auto_cost(f, 3, sizes);

            std::vector<double*> params;
            for (int i = 0; i < N_POS; ++i)
                params.push_back(traj.pos_cp_data(pos0 + i));

            auto* loss = new ceres::ScaledLoss(nullptr, cfg.lambda_snap_pos,
                                                ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost, loss, params);
        }
    }

    // ---- Orientation increment regularization --------------------------------
    if (cfg.lambda_ori_reg > 0.0) {
        for (int i = 0; i < n_ori - 1; ++i) {
            auto* f = new OrientationRegFunctor();
            auto* cost = make_auto_cost(f, 3, {4, 4});
            auto* loss = new ceres::ScaledLoss(nullptr, cfg.lambda_ori_reg,
                                                ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost, loss,
                traj.ori_knot_data(i), traj.ori_knot_data(i+1));
        }
    }

    // ---- Bias prior ---------------------------------------------------------
    {
        Eigen::Matrix<double, 6, 1> b0;
        for (int i = 0; i < 6; ++i) b0[i] = init_biases[i];

        auto* f_b = new BiasPriorFunctor(b0);
        std::vector<int> sizes = {6};

        // Accel bias prior (first 3)
        // Gyro bias prior (last 3)
        // We apply them together with separate weights by adding 2 factors
        // using a selection. For simplicity, add one combined factor with
        // geometric mean weight.
        // TODO: split into separate accel/gyro blocks if weight difference matters.
        auto* cost = make_auto_cost(f_b, 6, {6});

        // Use lambda_bias_prior_accel for now (gyro usually same)
        double w = std::sqrt(cfg.lambda_bias_prior_accel * cfg.lambda_bias_prior_gyro);
        auto* loss = new ceres::ScaledLoss(nullptr, w, ceres::TAKE_OWNERSHIP);
        problem.AddResidualBlock(cost, loss, traj.bias_data());
    }

    // ---- Heading prior -------------------------------------------------------
    if (cfg.lambda_heading > 0.0 && !heading_samples.empty()) {
        for (const auto& [t_abs, yaw_ref] : heading_samples) {
            double u_ori;
            int ori0;
            if (!traj.ori_index(t_abs, u_ori, ori0)) continue;

            auto* f = new HeadingPriorFunctor(yaw_ref, u_ori, inv_dt_ori);

            std::vector<int> sizes;
            for (int i = 0; i < N_ORI; ++i) sizes.push_back(4);

            auto* cost = make_auto_cost(f, 1, sizes);

            std::vector<double*> params;
            for (int i = 0; i < N_ORI; ++i)
                params.push_back(traj.ori_knot_data(ori0 + i));

            auto* loss = new ceres::ScaledLoss(nullptr, cfg.lambda_heading,
                                                ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost, loss, params);
        }
    }

    // ---- Boundary priors (pin start of trajectory) --------------------------
    // Match Python solver: pin position, velocity, and orientation at t_start.
    // Anchor values are evaluated from the INITIAL spline state.
    {
        const double t_pos_bnd = traj.t_pos_start();
        const double t_ori_bnd = traj.t_ori_start();

        // Helpers to evaluate initial spline at boundary (double only, non-autodiff)
        auto eval_pos_init = [&](double u, int i0) -> Eigen::Vector3d {
            std::vector<const double*> ptrs(N_POS);
            for (int i = 0; i < N_POS; ++i) ptrs[i] = traj.pos_cp_data(i0 + i);
            Eigen::Vector3d out;
            CeresSplineHelper<N_POS>::template evaluate<double, 3, 0>(
                ptrs.data(), u, inv_dt_pos, &out);
            return out;
        };
        auto eval_vel_init = [&](double u, int i0) -> Eigen::Vector3d {
            std::vector<const double*> ptrs(N_POS);
            for (int i = 0; i < N_POS; ++i) ptrs[i] = traj.pos_cp_data(i0 + i);
            Eigen::Vector3d out;
            CeresSplineHelper<N_POS>::template evaluate<double, 3, 1>(
                ptrs.data(), u, inv_dt_pos, &out);
            return out;
        };
        auto eval_ori_init = [&](double u, int i0) -> Sophus::SO3d {
            std::vector<const double*> ptrs(N_ORI);
            for (int i = 0; i < N_ORI; ++i) ptrs[i] = traj.ori_knot_data(i0 + i);
            Sophus::SO3d out;
            CeresSplineHelper<N_ORI>::template evaluate_lie<double, Sophus::SO3>(
                ptrs.data(), u, inv_dt_ori, &out, nullptr, nullptr);
            return out;
        };

        // Position boundary
        if (cfg.lambda_boundary_pos > 0.0) {
            double u; int i0;
            if (traj.pos_index(t_pos_bnd, u, i0)) {
                Eigen::Vector3d p_anchor = eval_pos_init(u, i0);
                auto* f = new BoundaryPosFunctor(p_anchor, u, inv_dt_pos);
                std::vector<int> sizes(N_POS, 3);
                auto* cost = make_auto_cost(f, 3, sizes);
                std::vector<double*> params;
                for (int i = 0; i < N_POS; ++i) params.push_back(traj.pos_cp_data(i0 + i));
                problem.AddResidualBlock(
                    cost,
                    new ceres::ScaledLoss(nullptr, cfg.lambda_boundary_pos, ceres::TAKE_OWNERSHIP),
                    params);
            }
        }

        // Velocity boundary
        if (cfg.lambda_boundary_vel > 0.0) {
            double u; int i0;
            if (traj.pos_index(t_pos_bnd, u, i0)) {
                Eigen::Vector3d v_anchor = eval_vel_init(u, i0);
                auto* f = new BoundaryVelFunctor(v_anchor, u, inv_dt_pos);
                std::vector<int> sizes(N_POS, 3);
                auto* cost = make_auto_cost(f, 3, sizes);
                std::vector<double*> params;
                for (int i = 0; i < N_POS; ++i) params.push_back(traj.pos_cp_data(i0 + i));
                problem.AddResidualBlock(
                    cost,
                    new ceres::ScaledLoss(nullptr, cfg.lambda_boundary_vel, ceres::TAKE_OWNERSHIP),
                    params);
            }
        }

        // Orientation boundary
        if (cfg.lambda_boundary_ori > 0.0) {
            double u; int i0;
            if (traj.ori_index(t_ori_bnd, u, i0)) {
                Sophus::SO3d R_anchor = eval_ori_init(u, i0);
                auto* f = new BoundaryOriFunctor(R_anchor, u, inv_dt_ori);
                std::vector<int> sizes(N_ORI, 4);
                auto* cost = make_auto_cost(f, 3, sizes);
                std::vector<double*> params;
                for (int i = 0; i < N_ORI; ++i) params.push_back(traj.ori_knot_data(i0 + i));
                problem.AddResidualBlock(
                    cost,
                    new ceres::ScaledLoss(nullptr, cfg.lambda_boundary_ori, ceres::TAKE_OWNERSHIP),
                    params);
            }
        }
    }

    // ---- Solve --------------------------------------------------------------
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = cfg.max_iterations;
    options.num_threads = 4;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // ---- Package result -----------------------------------------------------
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    SolverResult result;
    result.pos_cps    = traj.pos_cps;
    result.ori_knots  = traj.ori_knots;
    result.biases     = traj.biases;
    result.extrinsic_euler_deg = traj.extrinsic_euler_deg;
    result.solve_time_s = elapsed;

    // Cost history from iteration callbacks (simplification: just final cost)
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

}  // namespace rio
