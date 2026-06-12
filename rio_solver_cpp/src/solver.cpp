#include <rio/solver.h>
#include <rio/trajectory.h>
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

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <sophus/ceres_manifold.hpp>
#include <ceres/ceres.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>

namespace rio {

// ============================================================================
// dump_linear_system — evaluate J, r, grad from an assembled ceres::Problem
// and store them in a SolverResult::SystemDump.
//
// eval_blocks: ordered list of parameter block pointers; Ceres uses this
//   order to assign tangent-space columns in J.  Must NOT contain constant
//   (SetParameterBlockConstant) blocks — Ceres skips those silently but the
//   resulting column count will mismatch the block_map.
// ============================================================================
static void dump_linear_system(
    ceres::Problem& problem,
    const std::vector<double*>& eval_blocks,       // ordered: ori → pos → bias → pitch
    const std::vector<BlockMapEntry>& block_map,   // pre-built from eval_blocks
    SolverResult::SystemDump& out)
{
    out.valid = false;

    ceres::Problem::EvaluateOptions opts;
    opts.apply_loss_function = true;   // bake Huber √ρ′ into J so H = JᵀJ is exact
    opts.parameter_blocks    = eval_blocks;
    // Leave residual_blocks empty → evaluate all residuals

    double cost;
    std::vector<double> residuals, gradient;
    ceres::CRSMatrix J_crs;

    if (!problem.Evaluate(opts, &cost, &residuals, &gradient, &J_crs))
        return;

    out.jac_values  = std::move(J_crs.values);
    out.jac_cols    = std::move(J_crs.cols);
    out.jac_row_ptr = std::move(J_crs.rows);
    out.jac_num_rows = J_crs.num_rows;
    out.jac_num_cols = J_crs.num_cols;
    out.residuals   = std::move(residuals);
    out.gradient    = std::move(gradient);
    out.block_map   = block_map;
    out.valid       = true;
}

// ============================================================================
// build_dump_blocks — build the ordered eval_blocks and block_map for a
// batch solve (full trajectory, not windowed).
// ============================================================================
static void build_dump_blocks(
    Trajectory& traj,
    bool optimize_ext,
    std::vector<double*>& eval_blocks,
    std::vector<BlockMapEntry>& block_map)
{
    eval_blocks.clear();
    block_map.clear();
    int col = 0;

    // Orientation knots first (tangent = 3 via SO3 manifold)
    for (int i = 0; i < traj.n_ori_knots(); ++i) {
        eval_blocks.push_back(traj.ori_knot_data(i));
        block_map.push_back({"ori_knot", i, col, 3});
        col += 3;
    }
    // Position control points
    for (int i = 0; i < traj.n_pos_cps(); ++i) {
        eval_blocks.push_back(traj.pos_cp_data(i));
        block_map.push_back({"pos_cp", i, col, 3});
        col += 3;
    }
    // Bias (6 DoF, Euclidean)
    eval_blocks.push_back(traj.bias_data());
    block_map.push_back({"bias", 0, col, 6});
    col += 6;
    // Extrinsic pitch (1 DoF, only when unlocked)
    if (optimize_ext) {
        eval_blocks.push_back(traj.pitch_delta_data());
        block_map.push_back({"pitch_delta", 0, col, 1});
        col += 1;
    }
}

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

// Analytical cost functions for high-frequency IMU factors.
inline ceres::CostFunction* make_gyro_cost(
    const Eigen::Vector3d& z_gyro, double u_ori, double inv_dt_ori) {
    return new analytic::GyroAnalyticFactor(z_gyro, u_ori, inv_dt_ori);
}
inline ceres::CostFunction* make_accel_cost(
    const Eigen::Vector3d& z_acc, double u_ori, double inv_dt_ori,
    double u_pos, double inv_dt_pos) {
    return new analytic::AccelAnalyticFactor(z_acc, u_ori, inv_dt_ori, u_pos, inv_dt_pos);
}
inline ceres::CostFunction* make_radar_cost(
    const Eigen::Vector3d& u_sensor, double v_meas,
    double u_ori, double inv_dt_ori, double u_pos, double inv_dt_pos,
    const Sophus::SO3d& R_rb, const Eigen::Vector3d& t_bs) {
    return new analytic::RadarAnalyticFactor(
        u_sensor, v_meas, u_ori, inv_dt_ori, u_pos, inv_dt_pos, R_rb, t_bs);
}
inline ceres::CostFunction* make_radar_with_pitch_cost(
    const Eigen::Vector3d& u_sensor, double v_meas,
    double u_ori, double inv_dt_ori, double u_pos, double inv_dt_pos,
    const Sophus::SO3d& R_rb, const Eigen::Vector3d& t_bs) {
    return new analytic::RadarAnalyticWithPitchFactor(
        u_sensor, v_meas, u_ori, inv_dt_ori, u_pos, inv_dt_pos, R_rb, t_bs);
}

// ============================================================================
// solve()
// ============================================================================
SolverResult solve(
    const std::vector<RadarFrame>& radar_frames,
    const std::vector<ImuSample>& imu_samples,
    const std::vector<PreintFactor>& preint_factors,
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

    // Add pitch_delta extrinsic parameter (1 scalar, radians)
    const bool optimize_ext = !cfg.lock_extrinsics;
    traj.pitch_delta = 0.0;
    if (optimize_ext)
        problem.AddParameterBlock(traj.pitch_delta_data(), 1);

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

        // ω-gate: skip this frame if body angular rate exceeds threshold.
        // Evaluated from the current spline state (not inside the functor),
        // so this is a fixed gate at problem-build time.
        double w_omega = 1.0;
        if (cfg.omega_gate_threshold > 0.0 || cfg.omega_soft_sigma > 0.0) {
            const double* kp[N_ORI];
            for (int i = 0; i < N_ORI; ++i) kp[i] = traj.ori_knot_data(ori0 + i);
            Sophus::SO3d dummy_R;
            Eigen::Vector3d omega;
            CeresSplineHelper<N_ORI>::template evaluate_lie<double, Sophus::SO3>(
                kp, u_ori, inv_dt_ori, &dummy_R, &omega, nullptr);
            const double w_norm = omega.norm();
            if (cfg.omega_gate_threshold > 0.0 && w_norm > cfg.omega_gate_threshold)
                continue;
            if (cfg.omega_soft_sigma > 0.0) {
                const double q = w_norm / cfg.omega_soft_sigma;
                w_omega = 1.0 / (1.0 + q * q);   // sigma_eff^2 = sigma0^2 (1 + q^2)
            }
        }

        for (const auto& pt : frame.points) {
            // Range filter
            double range = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
            if (range < cfg.min_range) continue;

            Eigen::Vector3d u_sensor(pt.x / range, pt.y / range, pt.z / range);
            const double v_meas_c = pt.v - cfg.radar_zbias_fixed * u_sensor.z();

            // Build parameter block list (shared by both functor variants)
            std::vector<double*> param_blocks;
            for (int i = 0; i < N_ORI; ++i)
                param_blocks.push_back(traj.ori_knot_data(ori0 + i));
            for (int i = 0; i < N_POS; ++i)
                param_blocks.push_back(traj.pos_cp_data(pos0 + i));
            param_blocks.push_back(traj.bias_data());

            ceres::CostFunction* cost;
            if (optimize_ext) {
                param_blocks.push_back(traj.pitch_delta_data());
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

            // Soft gate: per-frame ScaledLoss around an owned Huber.
            ceres::LossFunction* loss = huber_loss;
            if (w_omega < 1.0)
                loss = new ceres::ScaledLoss(
                    new ceres::HuberLoss(cfg.huber_delta), w_omega,
                    ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost, loss, param_blocks);
        }
    }

    // ---- IMU factors ---------------------------------------------------------
    if (cfg.use_preintegration && !preint_factors.empty()) {
        // ---- Preintegrated IMU factors (Forster TRO-2017) -------------------
        for (const auto& pf : preint_factors) {
            double u_ori_i; int oi0_i;
            if (!traj.ori_index(pf.t_i, u_ori_i, oi0_i)) continue;
            double u_ori_j; int oi0_j;
            if (!traj.ori_index(pf.t_j, u_ori_j, oi0_j)) continue;
            if (oi0_j != oi0_i + 1) continue;  // adjacency guard

            double u_pos_i; int pi0_i;
            if (!traj.pos_index(pf.t_i, u_pos_i, pi0_i)) continue;
            double u_pos_j; int pi0_j;
            if (!traj.pos_index(pf.t_j, u_pos_j, pi0_j)) continue;
            const int k_pos_stride = pi0_j - pi0_i;
            if (k_pos_stride <= 0) continue;

            const double scale_v = (cfg.lambda_preint > 0.0)
                ? std::sqrt(cfg.lambda_preint_v / cfg.lambda_preint) : 0.0;
            const double scale_p = (cfg.lambda_preint > 0.0)
                ? std::sqrt(cfg.lambda_preint_p / cfg.lambda_preint) : 0.0;
            auto* f = new IMUPreintegrationFunctor(
                pf, u_ori_i, u_ori_j, inv_dt_ori,
                    u_pos_i, u_pos_j, inv_dt_pos, k_pos_stride,
                    scale_v, scale_p);

            // 5 unique ori knots × 4, (N_POS+k_pos_stride) pos CPs × 3, bias × 6
            std::vector<int> sizes;
            for (int k = 0; k < 5; ++k) sizes.push_back(4);
            for (int k = 0; k < N_POS + k_pos_stride; ++k) sizes.push_back(3);
            sizes.push_back(6);

            auto* cost = make_auto_cost(f, 9, sizes);

            std::vector<double*> params;
            for (int k = 0; k < 5; ++k)
                params.push_back(traj.ori_knot_data(oi0_i + k));
            for (int k = 0; k < N_POS + k_pos_stride; ++k)
                params.push_back(traj.pos_cp_data(pi0_i + k));
            params.push_back(traj.bias_data());

            problem.AddResidualBlock(
                cost,
                new ceres::ScaledLoss(nullptr, cfg.lambda_preint, ceres::TAKE_OWNERSHIP),
                params);
        }

        // Raw accel + gravity direction factors (position-orientation coupling,
        // same as the raw IMU path — preint r_R only replaces gyro, not accel).
        for (const auto& imu : imu_samples) {
            double u_ori; int ori0;
            if (!traj.ori_index(imu.timestamp, u_ori, ori0)) continue;

            double u_pos; int pos0;
            bool has_pos = traj.pos_index(imu.timestamp, u_pos, pos0);

            // Accel factor (needs position too)
            if (has_pos) {
                Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
                auto* cost = make_accel_cost(z_acc, u_ori, inv_dt_ori, u_pos, inv_dt_pos);

                std::vector<double*> params;
                for (int i = 0; i < N_ORI; ++i)
                    params.push_back(traj.ori_knot_data(ori0 + i));
                for (int i = 0; i < N_POS; ++i)
                    params.push_back(traj.pos_cp_data(pos0 + i));
                params.push_back(traj.bias_data());

                auto* loss = new ceres::ScaledLoss(
                    new ceres::HuberLoss(cfg.huber_delta_accel),
                    cfg.lambda_accel,
                    ceres::TAKE_OWNERSHIP);
                problem.AddResidualBlock(cost, loss, params);
            }

            // Gravity direction factor
            if (cfg.lambda_gravity > 0.0 && has_pos) {
                Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
                double dev = std::abs(z_acc.norm() - 9.81);
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
                    problem.AddResidualBlock(
                        cost,
                        new ceres::ScaledLoss(nullptr, cfg.lambda_gravity, ceres::TAKE_OWNERSHIP),
                        params);
                }
            }
        }
    } else {
        // ---- Raw IMU factors (gyro + accel + gravity direction) -------------
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
                auto* cost = make_gyro_cost(z_gyro, u_ori, inv_dt_ori);

                std::vector<double*> params;
                for (int i = 0; i < N_ORI; ++i)
                    params.push_back(traj.ori_knot_data(ori0 + i));
                params.push_back(traj.bias_data());

                auto* scaled = new ceres::ScaledLoss(nullptr, cfg.lambda_gyro,
                                                      ceres::TAKE_OWNERSHIP);
                problem.AddResidualBlock(cost, scaled, params);
            }

            // Accel factor (needs position too)
            if (has_pos) {
                Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
                auto* cost = make_accel_cost(z_acc, u_ori, inv_dt_ori, u_pos, inv_dt_pos);

                std::vector<double*> params;
                for (int i = 0; i < N_ORI; ++i)
                    params.push_back(traj.ori_knot_data(ori0 + i));
                for (int i = 0; i < N_POS; ++i)
                    params.push_back(traj.pos_cp_data(pos0 + i));
                params.push_back(traj.bias_data());

                auto* loss = new ceres::ScaledLoss(
                    new ceres::HuberLoss(cfg.huber_delta_accel),
                    cfg.lambda_accel,
                    ceres::TAKE_OWNERSHIP);
                problem.AddResidualBlock(cost, loss, params);
            }

            // Gravity direction factor
            if (cfg.lambda_gravity > 0.0 && has_pos) {
                Eigen::Vector3d z_acc(imu.ax, imu.ay, imu.az);
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

    // ---- Angular acceleration regularization --------------------------------
    if (cfg.lambda_ori_accel > 0.0) {
        for (int i = 0; i < n_ori - 2; ++i) {
            auto* f = new AngularAccelRegFunctor();
            auto* cost = make_auto_cost(f, 3, {4, 4, 4});
            auto* loss = new ceres::ScaledLoss(nullptr, cfg.lambda_ori_accel,
                                                ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost, loss,
                traj.ori_knot_data(i),
                traj.ori_knot_data(i + 1),
                traj.ori_knot_data(i + 2));
        }
    }

    // ---- Bias prior ---------------------------------------------------------
    {
        Eigen::Matrix<double, 6, 1> b0;
        for (int i = 0; i < 6; ++i) b0[i] = init_biases[i];

        // Per-component weights baked into the residual (no ScaledLoss):
        // accel components get sqrt(lambda_ba), gyro components sqrt(lambda_bg).
        auto* f_b = new BiasPriorFunctor(b0, cfg.lambda_bias_prior_accel,
                                             cfg.lambda_bias_prior_gyro);
        auto* cost = make_auto_cost(f_b, 6, {6});
        problem.AddResidualBlock(cost, nullptr, traj.bias_data());
    }

    // ---- Extrinsic pitch prior -----------------------------------------------
    if (optimize_ext && cfg.lambda_extrinsic_prior > 0.0) {
        auto* f = new PitchDeltaPriorFunctor();
        auto* cost = make_auto_cost(f, 1, {1});
        auto* loss = new ceres::ScaledLoss(nullptr, cfg.lambda_extrinsic_prior,
                                            ceres::TAKE_OWNERSHIP);
        problem.AddResidualBlock(cost, loss, traj.pitch_delta_data());
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

    // ---- Optional: dump warm-start linear system (pre-solve) ----------------
    // Declare result early so dump_pre/dump_post can be populated here.
    SolverResult result;
    std::vector<double*>      dump_blocks;
    std::vector<BlockMapEntry> dump_bmap;
    if (cfg.dump_system) {
        build_dump_blocks(traj, optimize_ext, dump_blocks, dump_bmap);
        dump_linear_system(problem, dump_blocks, dump_bmap, result.dump_pre);
    }

    // ---- Solve --------------------------------------------------------------
    ceres::Solver::Options options;
    if (cfg.use_banded_schur) {
        options.linear_solver_type = ceres::BANDED_SCHUR;
        // Globals = bias (6 DoF) + pitch_delta (1 DoF if extrinsics unlocked)
        options.banded_n_global_cols = 6 + (optimize_ext ? 1 : 0);
    } else {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    }
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = cfg.max_iterations;
    {
        int n = cfg.num_threads;
        if (n <= 0) n = static_cast<int>(std::thread::hardware_concurrency());
        options.num_threads = std::max(1, n);
    }
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // ---- Optional: dump converged linear system (post-solve) ----------------
    if (cfg.dump_system) {
        dump_linear_system(problem, dump_blocks, dump_bmap, result.dump_post);
    }

    // ---- Package result -----------------------------------------------------
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    result.pos_cps    = traj.pos_cps;
    result.ori_knots  = traj.ori_knots;
    result.biases     = traj.biases;
    // Return optimized pitch (nominal + delta converted to degrees)
    double final_pitch = extrinsic.pitch_deg + traj.pitch_delta * (180.0 / M_PI);
    result.extrinsic_euler_deg = {extrinsic.roll_deg, final_pitch, extrinsic.yaw_deg};
    result.solve_time_s  = elapsed;
    result.num_iterations = summary.num_successful_steps;

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
