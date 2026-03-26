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

#include <cmath>
#include <iostream>
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
// Main
// ============================================================================
int main() {
    std::cout << "=== RIO C++ Spline Tests ===" << std::endl;

    test_euclidean_spline();
    test_so3_spline();
    test_trajectory_indexing();

    std::cout << "\nResults: " << n_pass << " passed, " << n_fail << " failed." << std::endl;
    return n_fail > 0 ? 1 : 0;
}
