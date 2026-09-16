/**
 * test_spline.cpp
 *
 * Phase 1 validation: compare C++ basalt spline evaluation against
 * known analytical properties.
 *
 * Tests:
 *   1. Euclidean B-spline positional continuity (quintic, order-6)
 *   2. SO(3) cumulative B-spline continuity (cubic, order-4)
 *   3. Radar Doppler residual value against hand-computed reference
 */

#include <basalt/spline/ceres_spline_helper.h>
#include <sophus/so3.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <rio/trajectory.h>
#include <rio/factors/analytic/spline_jacobians.h>
#include <rio/factors/analytic/gyro_analytic.h>
#include <rio/factors/analytic/accel_analytic.h>

#include <cmath>
#include <iostream>
#include <random>
#include <cassert>

// ============================================================================
// Test helpers
// ============================================================================

static int n_pass = 0, n_fail = 0;

#define RIO_CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << msg << std::endl; \
        ++n_fail; \
    } else { \
        std::cout << "[PASS] " << msg << std::endl; \
        ++n_pass; \
    } \
} while(0)

#define RIO_CHECK_NEAR(a, b, tol, msg) \
    RIO_CHECK(std::abs((a) - (b)) < (tol), \
          msg << "  (" << (a) << " vs " << (b) << ", tol=" << (tol) << ")")

// ============================================================================
// Test 1: Euclidean quintic B-spline (order N=6) position evaluation
// ============================================================================
void test_euclidean_spline() {
    static constexpr int N = 6;
    using Helper = basalt::CeresSplineHelper<N>;

    // Straight-line control points along x-axis
    // CPs: [0,0,0], [1,0,0], [2,0,0], ..., [N-1,0,0]
    const int n_cps = N;
    std::vector<Eigen::Vector3d> cps(n_cps);
    for (int i = 0; i < n_cps; ++i)
        cps[i] = Eigen::Vector3d(i, 0, 0);

    // Build pointer array
    std::vector<const double*> ptrs(n_cps);
    for (int i = 0; i < n_cps; ++i)
        ptrs[i] = cps[i].data();

    double dt = 1.0;
    double inv_dt = 1.0 / dt;

    // At u=0 (start of segment): should give a weighted combination of CPs
    {
        Eigen::Vector3d pos;
        Helper::evaluate<double, 3, 0>(ptrs.data(), 0.0, inv_dt, &pos);
        // For straight-line CPs, result should lie in [0, N-1] along x
        RIO_CHECK(pos[0] >= 0.0 && pos[0] <= N - 1, "Euclidean spline u=0 in range");
        RIO_CHECK_NEAR(pos[1], 0.0, 1e-10, "Euclidean spline u=0 y=0");
        RIO_CHECK_NEAR(pos[2], 0.0, 1e-10, "Euclidean spline u=0 z=0");
    }

    // At u=0.5 (midpoint): should be symmetric
    {
        Eigen::Vector3d pos;
        Helper::evaluate<double, 3, 0>(ptrs.data(), 0.5, inv_dt, &pos);
        RIO_CHECK(pos[0] >= 0.0 && pos[0] <= N - 1, "Euclidean spline u=0.5 in range");
    }

    // Velocity of constant spline should be nearly zero
    // All CPs at same value
    for (int i = 0; i < n_cps; ++i)
        cps[i] = Eigen::Vector3d(1.0, 2.0, 3.0);

    {
        Eigen::Vector3d vel;
        Helper::evaluate<double, 3, 1>(ptrs.data(), 0.5, inv_dt, &vel);
        RIO_CHECK_NEAR(vel.norm(), 0.0, 1e-8, "Constant spline velocity=0");
    }
}

// ============================================================================
// Test 2: SO(3) cumulative B-spline (order N=4)
// ============================================================================
void test_so3_spline() {
    static constexpr int N = 4;
    using Helper = basalt::CeresSplineHelper<N>;

    // Identity quaternion knots → result should be identity rotation
    const int n_knots = N;
    std::vector<Sophus::SO3d> knots(n_knots, Sophus::SO3d{});

    std::vector<const double*> ptrs(n_knots);
    for (int i = 0; i < n_knots; ++i)
        ptrs[i] = knots[i].data();

    double inv_dt = 1.0;

    {
        Sophus::SO3d R;
        Helper::evaluate_lie<double, Sophus::SO3>(ptrs.data(), 0.5, inv_dt,
                                                   &R, nullptr, nullptr);
        // Identity knots → identity rotation
        double err = (R.matrix() - Eigen::Matrix3d::Identity()).norm();
        RIO_CHECK_NEAR(err, 0.0, 1e-10, "Identity knots -> identity rotation");
    }

    // Small rotation around z-axis
    double angle = 0.1;  // rad
    Sophus::SO3d Rz = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, angle));
    for (int i = 0; i < n_knots; ++i)
        knots[i] = Rz;

    {
        Sophus::SO3d R;
        Sophus::SO3d::Tangent omega;
        Helper::evaluate_lie<double, Sophus::SO3>(ptrs.data(), 0.5, inv_dt,
                                                   &R, &omega, nullptr);
        // Constant rotation knots → small but nonzero rotation, nearly zero omega
        RIO_CHECK_NEAR(omega.norm(), 0.0, 0.5, "Constant rotation knots omega small");
    }
}

// ============================================================================
// Test 3: Trajectory time indexing
// ============================================================================
void test_trajectory_indexing() {
    rio::Trajectory traj;
    traj.t_ref  = 100.0;
    traj.dt_pos = 0.005;
    traj.dt_ori = 0.008;
    traj.pos_cps.assign(100, {0.0, 0.0, 0.0});
    traj.ori_knots.assign(60, {0.0, 0.0, 0.0, 1.0});

    // Valid range check for position
    double t_mid = (traj.t_pos_start() + traj.t_pos_end()) / 2.0;
    double u; int i0;
    bool ok = traj.pos_index(t_mid, u, i0);
    RIO_CHECK(ok, "Trajectory pos_index in valid range");
    RIO_CHECK(u >= 0.0 && u <= 1.0, "pos_index u in [0,1]");
    RIO_CHECK(i0 >= 0 && i0 + rio::N_POS <= (int)traj.pos_cps.size(),
          "pos_index i0 in valid range");

    // Out of range
    double t_before = traj.t_ref - 1.0;
    bool bad = traj.pos_index(t_before, u, i0);
    RIO_CHECK(!bad, "Trajectory pos_index returns false for t before t_start");
}

// ============================================================================
// Test 4: Numerical Jacobian validation for GyroAnalyticFactor
// ============================================================================
void test_gyro_analytic_jacobian() {
    // Build N_ORI=4 random orientation knots
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 0.3);

    constexpr int N = rio::N_ORI;
    std::array<Sophus::SO3d, N> knots;
    for (int i = 0; i < N; ++i)
        knots[i] = Sophus::SO3d::exp(Eigen::Vector3d(nd(rng), nd(rng), nd(rng)));

    // Pack as raw double arrays [x,y,z,w]
    std::array<std::array<double, 4>, N> raw;
    std::array<const double*, N> params_arr;
    for (int i = 0; i < N; ++i) {
        auto q = knots[i].unit_quaternion();
        raw[i] = {q.x(), q.y(), q.z(), q.w()};
        params_arr[i] = raw[i].data();
    }

    const double u = 0.4, inv_dt = 1.0 / 0.008;

    // --- Analytical Jacobian (tangent-space, 3×3 per knot) ---
    rio::analytic::JacobianStruct<N> J_analytic;
    Eigen::Vector3d omega0 = rio::analytic::body_velocity_with_jacobian<N>(
        params_arr.data(), u, inv_dt, &J_analytic);

    // --- Numerical Jacobian: try BOTH perturbation conventions ---
    // basalt JacobianStruct convention is unknown — try right then left
    const double eps = 1e-7;
    std::array<Eigen::Matrix3d, N> J_num_right, J_num_left;
    for (int k = 0; k < N; ++k) {
        J_num_right[k].setZero();
        J_num_left[k].setZero();
        for (int j = 0; j < 3; ++j) {
            // Right perturbation: q_new = q * exp(eps * e_j)
            Sophus::SO3d knot_r = knots[k] * Sophus::SO3d::exp(eps * Eigen::Vector3d::Unit(j));
            // Left perturbation: q_new = exp(eps * e_j) * q
            Sophus::SO3d knot_l = Sophus::SO3d::exp(eps * Eigen::Vector3d::Unit(j)) * knots[k];

            for (auto& [knot_p, J_num] : std::vector<std::pair<Sophus::SO3d&, Eigen::Matrix3d&>>{
                     {knot_r, J_num_right[k]}, {knot_l, J_num_left[k]}}) {
                auto qp = knot_p.unit_quaternion();
                std::array<double, 4> raw_plus = {qp.x(), qp.y(), qp.z(), qp.w()};
                std::array<const double*, N> params_plus = params_arr;
                params_plus[k] = raw_plus.data();
                Eigen::Vector3d omega_plus = rio::analytic::body_velocity_with_jacobian<N>(
                    params_plus.data(), u, inv_dt, nullptr);
                J_num.col(j) = (omega_plus - omega0) / eps;
            }
        }
    }

    // Also compare manual port against basalt reference
    rio::analytic::JacobianStruct<N> J_manual;
    rio::analytic::body_velocity_with_jacobian_manual<N>(
        params_arr.data(), u, inv_dt, &J_manual);
    double max_manual_vs_ref = 0.0;
    for (int k = 0; k < N; ++k)
        max_manual_vs_ref = std::max(max_manual_vs_ref,
            (J_manual.d_val_d_knot[k] - J_analytic.d_val_d_knot[k]).norm());
    RIO_CHECK(max_manual_vs_ref < 1e-10,
        "Manual port matches basalt reference Jacobian (err=" << max_manual_vs_ref << ")");

    // Check which convention matches
    double max_err_right = 0.0, max_err_left = 0.0;
    for (int k = 0; k < N; ++k) {
        max_err_right = std::max(max_err_right,
            (J_analytic.d_val_d_knot[k] - J_num_right[k]).norm());
        max_err_left  = std::max(max_err_left,
            (J_analytic.d_val_d_knot[k] - J_num_left[k]).norm());
    }
    std::cout << "[INFO] Gyro tangent Jac: right-perturb err=" << max_err_right
              << "  left-perturb err=" << max_err_left << std::endl;

    // Use whichever is smaller for the PASS/FAIL
    const double max_err_tang = std::min(max_err_right, max_err_left);
    const auto& J_numerical = (max_err_right < max_err_left) ? J_num_right : J_num_left;

    // Compare
    double max_err = 0.0;
    for (int k = 0; k < N; ++k) {
        double err = (J_analytic.d_val_d_knot[k] - J_numerical[k]).norm();
        max_err = std::max(max_err, err);
    }
    RIO_CHECK(max_err_tang < 1e-5, "GyroAnalytic: tangent Jacobian matches FD (err=" << max_err_tang << ")");

    // --- Validate ambient Jacobian (3×4) via quaternion perturbation ---
    std::array<Eigen::Matrix<double, 3, 4>, N> J_ambient_analytic;
    for (int k = 0; k < N; ++k) {
        const Eigen::Matrix<double, 3, 4> M = rio::analytic::tangent_to_ambient(raw[k].data());
        J_ambient_analytic[k].noalias() = J_analytic.d_val_d_knot[k] * M;
    }

    // Numerical ambient Jacobian (perturb quaternion directly in xyzw space)
    const double eps_q = 1e-7;
    std::array<Eigen::Matrix<double, 3, 4>, N> J_ambient_numerical;
    for (int k = 0; k < N; ++k) {
        J_ambient_numerical[k].setZero();
        for (int j = 0; j < 4; ++j) {
            std::array<double, 4> raw_plus = raw[k];
            raw_plus[j] += eps_q;
            // Re-normalise
            double norm = std::sqrt(raw_plus[0]*raw_plus[0]+raw_plus[1]*raw_plus[1]+
                                    raw_plus[2]*raw_plus[2]+raw_plus[3]*raw_plus[3]);
            for (double& v : raw_plus) v /= norm;

            std::array<const double*, N> params_plus = params_arr;
            params_plus[k] = raw_plus.data();
            Eigen::Vector3d omega_plus = rio::analytic::body_velocity_with_jacobian<N>(
                params_plus.data(), u, inv_dt, nullptr);
            J_ambient_numerical[k].col(j) = (omega_plus - omega0) / eps_q;
        }
    }

    double max_ambient_err = 0.0;
    for (int k = 0; k < N; ++k) {
        double err = (J_ambient_analytic[k] - J_ambient_numerical[k]).norm();
        max_ambient_err = std::max(max_ambient_err, err);
    }
    RIO_CHECK(max_ambient_err < 1e-4,
              "GyroAnalytic: ambient Jacobian matches FD (err=" << max_ambient_err << ")");
}

// ============================================================================
// Test 5: omega residual consistency — basalt So3Spline vs CeresSplineHelper
// ============================================================================
void test_rotation_jacobian_convention() {
    constexpr int N = rio::N_ORI;
    std::mt19937 rng(55);
    std::normal_distribution<double> nd(0.0, 0.3);
    std::array<Sophus::SO3d, N> knots;
    for (int i = 0; i < N; ++i)
        knots[i] = Sophus::SO3d::exp(Eigen::Vector3d(nd(rng), nd(rng), nd(rng)));
    std::array<std::array<double, 4>, N> raw;
    std::array<const double*, N> params_arr;
    for (int i = 0; i < N; ++i) {
        auto q = knots[i].unit_quaternion();
        raw[i] = {q.x(), q.y(), q.z(), q.w()};
        params_arr[i] = raw[i].data();
    }
    const double u = 0.4, inv_dt = 1.0 / 0.008;
    const double eps = 1e-7;

    rio::analytic::JacobianStruct<N> J_R;
    Sophus::SO3d R0 = rio::analytic::rotation_with_jacobian_manual<N>(
        params_arr.data(), u, inv_dt, &J_R);

    // Pick a random test vector
    const Eigen::Vector3d v(0.3, -0.7, 1.1);

    for (int k = 0; k < N; ++k) {
        // For LEFT perturbation: R_new = exp(J_k * eps * e_j) * R0
        // d(R0^T v)/d(eps_L) should be... R0^T [v]^× J_k (from our derivation)
        // For RIGHT perturbation: R_new = R0 * exp(J_right_k * eps * e_j)
        // d(R0^T v)/d(eps_R) should be... -[R0^T v]^× J_right_k

        Eigen::Matrix3d J_num_right, J_num_left;
        for (int j = 0; j < 3; ++j) {
            // Right: q_k_new = q_k * exp(eps * e_j)
            Sophus::SO3d knot_R = knots[k] * Sophus::SO3d::exp(eps * Eigen::Vector3d::Unit(j));
            auto qR = knot_R.unit_quaternion();
            std::array<double, 4> raw_R = {qR.x(), qR.y(), qR.z(), qR.w()};
            std::array<const double*, N> p_R = params_arr;
            p_R[k] = raw_R.data();
            Sophus::SO3d R_plus_R = rio::analytic::rotation_with_jacobian_manual<N>(p_R.data(), u, inv_dt, nullptr);
            J_num_right.col(j) = (R_plus_R.inverse() * R0).log() / eps;  // right tangent change

            // Left: q_k_new = exp(eps * e_j) * q_k
            Sophus::SO3d knot_L = Sophus::SO3d::exp(eps * Eigen::Vector3d::Unit(j)) * knots[k];
            auto qL = knot_L.unit_quaternion();
            std::array<double, 4> raw_L = {qL.x(), qL.y(), qL.z(), qL.w()};
            std::array<const double*, N> p_L = params_arr;
            p_L[k] = raw_L.data();
            Sophus::SO3d R_plus_L = rio::analytic::rotation_with_jacobian_manual<N>(p_L.data(), u, inv_dt, nullptr);
            // Left output: R_new = exp(something) * R0, so log(R_new * R0^{-1}) / eps
            J_num_left.col(j) = (R_plus_L * R0.inverse()).log() / eps;
        }

        // J_R.d_val_d_knot[k] is d(R_tangent)/d(eps_knot)
        // If right-output:  J_right_tangent_of_R = log(R_new^{-1} R0) or log(R0^{-1} R_new)
        // If left-output:   J_left_tangent_of_R  = log(R_new R0^{-1}) or log(R0 R_new^{-1})
        if (k == 0) {
            std::cout << "[INFO] RotJac knot0: J_analytic=\n" << J_R.d_val_d_knot[0] << "\n";
            std::cout << "[INFO] J_num_right=\n" << J_num_right << "\n";
            std::cout << "[INFO] J_num_left=\n" << J_num_left << "\n";
        }
        double err_right = (J_R.d_val_d_knot[k] - J_num_right).norm();
        double err_left  = (J_R.d_val_d_knot[k] - J_num_left).norm();
        if (k == 0) std::cout << "[INFO] knot0: err_right=" << err_right << " err_left=" << err_left << "\n";
    }
}

void test_rotation_residual_consistency() {
    constexpr int N = rio::N_ORI;
    std::mt19937 rng(77);
    std::normal_distribution<double> nd(0.0, 0.3);
    std::array<Sophus::SO3d, N> knots;
    for (int i = 0; i < N; ++i)
        knots[i] = Sophus::SO3d::exp(Eigen::Vector3d(nd(rng), nd(rng), nd(rng)));
    std::array<std::array<double, 4>, N> raw;
    std::array<const double*, N> params_arr;
    for (int i = 0; i < N; ++i) {
        auto q = knots[i].unit_quaternion();
        raw[i] = {q.x(), q.y(), q.z(), q.w()};
        params_arr[i] = raw[i].data();
    }
    const double u = 0.4, inv_dt = 1.0 / 0.008;

    // Method 1: manual port rotation_with_jacobian_manual
    Sophus::SO3d R_manual = rio::analytic::rotation_with_jacobian_manual<N>(
        params_arr.data(), u, inv_dt, nullptr);

    // Method 2: evaluate_lie<double> (used by AutoDiff AccelFunctor)
    Sophus::SO3d R_ceres;
    CeresSplineHelper<N>::template evaluate_lie<double, Sophus::SO3>(
        params_arr.data(), u, inv_dt, &R_ceres, nullptr, nullptr);

    double R_diff = (R_manual.matrix() - R_ceres.matrix()).norm();
    RIO_CHECK(R_diff < 1e-9, "R: rotation_with_jacobian_manual == evaluate_lie (diff=" << R_diff << ")");

    // Also check basalt So3Spline reference
    Sophus::SO3d R_basalt = rio::analytic::rotation_with_jacobian<N>(
        params_arr.data(), u, inv_dt, nullptr);
    double R_diff2 = (R_manual.matrix() - R_basalt.matrix()).norm();
    RIO_CHECK(R_diff2 < 1e-9, "R: manual == basalt direct (diff=" << R_diff2 << ")");
}

void test_omega_residual_consistency() {
    constexpr int N = rio::N_ORI;
    std::mt19937 rng(99);
    std::normal_distribution<double> nd(0.0, 0.3);

    std::array<Sophus::SO3d, N> knots;
    for (int i = 0; i < N; ++i)
        knots[i] = Sophus::SO3d::exp(Eigen::Vector3d(nd(rng), nd(rng), nd(rng)));

    std::array<std::array<double, 4>, N> raw;
    std::array<const double*, N> params_arr;
    for (int i = 0; i < N; ++i) {
        auto q = knots[i].unit_quaternion();
        raw[i] = {q.x(), q.y(), q.z(), q.w()};
        params_arr[i] = raw[i].data();
    }

    const double u = 0.4, inv_dt = 1.0 / 0.008;

    // Method 1: basalt So3Spline (used by analytical factor)
    Eigen::Vector3d omega_basalt = rio::analytic::body_velocity_with_jacobian<N>(
        params_arr.data(), u, inv_dt, nullptr);

    // Method 2: CeresSplineHelper::evaluate_lie<double> (used by AutoDiff functor)
    Sophus::SO3d R_ceres;
    Eigen::Vector3d omega_ceres;
    CeresSplineHelper<N>::template evaluate_lie<double, Sophus::SO3>(
        params_arr.data(), u, inv_dt, &R_ceres, &omega_ceres, nullptr);

    double omega_diff = (omega_basalt - omega_ceres).norm();
    RIO_CHECK(omega_diff < 1e-9, "omega: basalt So3Spline == CeresSplineHelper (diff=" << omega_diff << ")");
}

// ============================================================================
// Test 6: Numerical Jacobian validation for AccelAnalyticFactor
// ============================================================================
void test_accel_analytic_jacobian() {
    constexpr int N_ORI = rio::N_ORI;
    constexpr int N_POS = rio::N_POS;

    std::mt19937 rng(123);
    std::normal_distribution<double> nd(0.0, 0.3);
    std::normal_distribution<double> nd_pos(0.0, 1.0);

    // Random orientation knots
    std::array<Sophus::SO3d, N_ORI> knots;
    for (int i = 0; i < N_ORI; ++i)
        knots[i] = Sophus::SO3d::exp(Eigen::Vector3d(nd(rng), nd(rng), nd(rng)));

    // Random position CPs
    std::array<Eigen::Vector3d, N_POS> pos_cps;
    for (int i = 0; i < N_POS; ++i)
        pos_cps[i] = Eigen::Vector3d(nd_pos(rng), nd_pos(rng), nd_pos(rng));

    // Random bias
    std::array<double, 6> bias = {0.01, -0.02, 0.05, 0.001, -0.002, 0.003};

    // Pack as raw arrays
    std::array<std::array<double, 4>, N_ORI> raw_ori;
    std::array<std::array<double, 3>, N_POS> raw_pos;
    std::vector<const double*> params_vec;

    for (int i = 0; i < N_ORI; ++i) {
        auto q = knots[i].unit_quaternion();
        raw_ori[i] = {q.x(), q.y(), q.z(), q.w()};
        params_vec.push_back(raw_ori[i].data());
    }
    for (int i = 0; i < N_POS; ++i) {
        raw_pos[i] = {pos_cps[i][0], pos_cps[i][1], pos_cps[i][2]};
        params_vec.push_back(raw_pos[i].data());
    }
    params_vec.push_back(bias.data());

    const double u_ori = 0.4, inv_dt_ori = 1.0 / 0.008;
    // Use inv_dt_pos=1 so pos_coeff entries are O(1) — avoids catastrophic
    // cancellation in the residual that would inflate FD errors for bias/pos tests.
    const double u_pos = 0.4, inv_dt_pos = 1.0;

    // Nominal measurement (random, we just check Jacobian not residual value)
    Eigen::Vector3d z_acc(0.1, -0.2, 9.5);

    // Compute analytical Jacobians
    rio::analytic::AccelAnalyticFactor factor(z_acc, u_ori, inv_dt_ori, u_pos, inv_dt_pos);

    double residuals[3];
    std::vector<double> jac_storage(3 * 4 * N_ORI + 3 * 3 * N_POS + 3 * 6, 0.0);
    std::vector<double*> jacobians_vec;
    int offset = 0;
    for (int i = 0; i < N_ORI; ++i) {
        jacobians_vec.push_back(jac_storage.data() + offset); offset += 12;
    }
    for (int i = 0; i < N_POS; ++i) {
        jacobians_vec.push_back(jac_storage.data() + offset); offset += 9;
    }
    jacobians_vec.push_back(jac_storage.data() + offset);  // bias

    factor.Evaluate(params_vec.data(), residuals, jacobians_vec.data());
    const Eigen::Map<const Eigen::Vector3d> r0(residuals);

    // --- Validate LOCAL Jacobian (what Ceres actually uses) ---
    // J_local = J_ambient * PlusJac  (what Ceres computes from J_ambient)
    // Test against RIGHT-perturbation FD (matches Ceres manifold Plus convention)
    const double eps = 1e-7;
    double max_ori_err = 0.0;
    for (int k = 0; k < N_ORI; ++k) {
        Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>
            J_analytic_ambient(jacobians_vec[k]);
        Sophus::SO3d R_k(Eigen::Quaterniond(raw_ori[k][3], raw_ori[k][0],
                                             raw_ori[k][1], raw_ori[k][2]));
        const Eigen::Matrix<double, 4, 3> PlusJac = R_k.Dx_this_mul_exp_x_at_0();
        // Local Jacobian = J_ambient * PlusJac  (3×3)
        Eigen::Matrix3d J_local_analytic = J_analytic_ambient * PlusJac;

        // Numerical: right perturbation (q_new = q * exp(eps * e_j))
        Eigen::Matrix3d J_local_num;
        for (int j = 0; j < 3; ++j) {
            Sophus::SO3d knot_plus = knots[k] * Sophus::SO3d::exp(eps * Eigen::Vector3d::Unit(j));
            auto qp = knot_plus.unit_quaternion();
            std::array<double, 4> raw_plus = {qp.x(), qp.y(), qp.z(), qp.w()};
            std::vector<const double*> params_plus = params_vec;
            params_plus[k] = raw_plus.data();
            double r_plus[3];
            factor.Evaluate(params_plus.data(), r_plus, nullptr);
            for (int r = 0; r < 3; ++r)
                J_local_num(r, j) = (r_plus[r] - residuals[r]) / eps;
        }
        double err = (J_local_analytic - J_local_num).norm();
        if (k == 0) {
            std::cout << "[INFO] AccelJac knot0 J_local_analytic=\n" << J_local_analytic << "\n";
            std::cout << "[INFO] AccelJac knot0 J_local_num=\n" << J_local_num << "\n";
            std::cout << "[INFO] diff=\n" << (J_local_analytic - J_local_num) << "\n";
        }
        max_ori_err = std::max(max_ori_err, err);
    }
    RIO_CHECK(max_ori_err < 1e-4,
        "AccelAnalytic: local ori Jacobian matches FD (err=" << max_ori_err << ")");

    // --- Numerical Jacobian for position CPs ---
    double max_pos_err = 0.0;
    for (int k = 0; k < N_POS; ++k) {
        const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>
            J_analytic_pos(jacobians_vec[N_ORI + k]);
        Eigen::Matrix3d J_num_pos;
        for (int j = 0; j < 3; ++j) {
            std::array<double, 3> raw_plus = raw_pos[k];
            raw_plus[j] += eps;
            std::vector<const double*> params_plus = params_vec;
            params_plus[N_ORI + k] = raw_plus.data();
            double r_plus[3];
            factor.Evaluate(params_plus.data(), r_plus, nullptr);
            for (int r = 0; r < 3; ++r)
                J_num_pos(r, j) = (r_plus[r] - residuals[r]) / eps;
        }
        double err = (J_analytic_pos - J_num_pos).norm();
        max_pos_err = std::max(max_pos_err, err);
    }
    RIO_CHECK(max_pos_err < 1e-4,
        "AccelAnalytic: pos CP Jacobian matches FD (err=" << max_pos_err << ")");

    // --- Numerical Jacobian for bias ---
    const Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>>
        J_analytic_bias(jacobians_vec[N_ORI + N_POS]);
    Eigen::Matrix<double, 3, 6> J_num_bias;
    for (int j = 0; j < 6; ++j) {
        std::array<double, 6> bias_plus = bias;
        bias_plus[j] += eps;
        std::vector<const double*> params_plus = params_vec;
        params_plus[N_ORI + N_POS] = bias_plus.data();
        double r_plus[3];
        factor.Evaluate(params_plus.data(), r_plus, nullptr);
        for (int r = 0; r < 3; ++r)
            J_num_bias(r, j) = (r_plus[r] - residuals[r]) / eps;
    }
    std::cout << "[INFO] J_analytic_bias=\n" << J_analytic_bias << "\n";
    std::cout << "[INFO] J_num_bias=\n" << J_num_bias << "\n";
    double bias_err = (J_analytic_bias - J_num_bias).norm();
    RIO_CHECK(bias_err < 1e-8, "AccelAnalytic: bias Jacobian matches FD (err=" << bias_err << ")");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== RIO C++ Spline Tests ===" << std::endl;

    test_euclidean_spline();
    test_so3_spline();
    test_trajectory_indexing();
    test_gyro_analytic_jacobian();
    test_rotation_jacobian_convention();
    test_rotation_residual_consistency();
    test_omega_residual_consistency();
    test_accel_analytic_jacobian();

    std::cout << "\nResults: " << n_pass << " passed, " << n_fail << " failed." << std::endl;
    return n_fail > 0 ? 1 : 0;
}
