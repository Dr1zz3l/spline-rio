#pragma once

// Analytical spline Jacobians for the C++ Ceres solver.
//
// Provides two implementations:
//   1. body_velocity_with_jacobian — uses basalt's So3Spline directly (reference,
//      heap-allocates a temporary spline per call — use only for testing).
//   2. body_velocity_with_jacobian_manual — stack-only port of basalt's
//      velocityBody(t, J), zero heap allocation (production use).
//
// Both use basalt's LEFT-perturbation Jacobian convention:
//   d_val_d_knot[i] = d(f) / d(eps_L_i)  where  q_i_new = exp(eps_L_i) * q_i
//
// Use tangent_to_ambient() to convert to the 3×4 ambient Jacobian that
// Ceres expects (Ceres uses RIGHT perturbation via SophusManifold<SO3>).

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <basalt/spline/so3_spline.h>
#include <basalt/spline/ceres_spline_helper.h>
#include <basalt/utils/sophus_utils.hpp>
#include <rio/trajectory.h>
#include <array>

namespace rio {
namespace analytic {

// ============================================================================
// JacobianStruct — 3×3 blocks, one per orientation knot
// ============================================================================
template <int _N>
struct JacobianStruct {
    std::array<Eigen::Matrix3d, _N> d_val_d_knot;
};

// ============================================================================
// tangent_to_ambient
// ============================================================================
// Returns the (3×4) matrix M such that:
//   J_ambient (3×4) = J_basalt_left (3×3) · M
//
// Basalt uses LEFT perturbation; Ceres uses RIGHT. Conversion chain:
//   J_right = J_left · Ad(q) = J_left · R.matrix()         (adjoint)
//   J_ambient = J_right · PlusJac⁺ = J_right · 4·PlusJac^T
//   ⟹  M = R.matrix() · 4·PlusJac^T                       (3×4)
//
// q_xyzw = params[i] in Ceres/Sophus quaternion layout [x,y,z,w].
inline Eigen::Matrix<double, 3, 4> tangent_to_ambient(const double* q_xyzw) {
    Sophus::SO3d R(Eigen::Quaterniond(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]));
    return R.matrix() * (4.0 * R.Dx_this_mul_exp_x_at_0().transpose());  // (3×4)
}

// ============================================================================
// body_velocity_with_jacobian  (reference — uses basalt So3Spline, heap alloc)
// ============================================================================
// Correct Jacobians; slow due to deque allocation. Use for testing only.
template <int N>
Eigen::Vector3d body_velocity_with_jacobian(
    double const* const* params, double u, double inv_dt, JacobianStruct<N>* J)
{
    using SplineT = basalt::So3Spline<N>;
    using SO3 = Sophus::SO3d;

    const int64_t dt_ns = static_cast<int64_t>((1.0 / inv_dt) * 1e9);
    SplineT spline(dt_ns, 0);
    for (int i = 0; i < N; ++i) {
        const double* q = params[i];
        spline.knotsPushBack(SO3(Eigen::Quaterniond(q[3], q[0], q[1], q[2])));
    }
    const int64_t t_ns = static_cast<int64_t>(u * static_cast<double>(dt_ns));

    if (J) {
        typename SplineT::JacobianStruct jac;
        Eigen::Vector3d omega = spline.velocityBody(t_ns, &jac);
        for (int i = 0; i < N; ++i) J->d_val_d_knot[i] = jac.d_val_d_knot[i];
        return omega;
    }
    return spline.velocityBody(t_ns);
}

// ============================================================================
// rotation_with_jacobian  (reference — uses basalt So3Spline, heap alloc)
// ============================================================================
template <int N>
Sophus::SO3d rotation_with_jacobian(
    double const* const* params, double u, double inv_dt, JacobianStruct<N>* J)
{
    using SplineT = basalt::So3Spline<N>;
    using SO3 = Sophus::SO3d;

    const int64_t dt_ns = static_cast<int64_t>((1.0 / inv_dt) * 1e9);
    SplineT spline(dt_ns, 0);
    for (int i = 0; i < N; ++i) {
        const double* q = params[i];
        spline.knotsPushBack(SO3(Eigen::Quaterniond(q[3], q[0], q[1], q[2])));
    }
    const int64_t t_ns = static_cast<int64_t>(u * static_cast<double>(dt_ns));

    if (J) {
        typename SplineT::JacobianStruct jac;
        SO3 R = spline.evaluate(t_ns, &jac);
        for (int i = 0; i < N; ++i) J->d_val_d_knot[i] = jac.d_val_d_knot[i];
        return R;
    }
    return spline.evaluate(t_ns);
}

// ============================================================================
// body_velocity_with_jacobian_manual  (production — zero heap allocation)
// ============================================================================
// Stack-only port of basalt::So3Spline::velocityBody(t_ns, J).
// Uses CeresSplineHelper's blending matrices (same values as basalt's).
// Convention: LEFT perturbation (same as basalt reference above).
template <int N>
Eigen::Vector3d body_velocity_with_jacobian_manual(
    double const* const* params, double u, double inv_dt, JacobianStruct<N>* J)
{
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;
    using SO3  = Sophus::SO3d;
    constexpr int DEG = N - 1;

    using Helper = CeresSplineHelper<N>;
    using VecN   = Eigen::Matrix<double, N, 1>;

    VecN p;
    Helper::template baseCoeffsWithTime<0>(p, u);
    const VecN coeff  = Helper::cumulative_blending_matrix_ * p;

    Helper::template baseCoeffsWithTime<1>(p, u);
    const VecN dcoeff = inv_dt * (Helper::cumulative_blending_matrix_ * p);

    auto knot = [&](int i) -> SO3 {
        const double* q = params[i];
        return SO3(Eigen::Quaterniond(q[3], q[0], q[1], q[2]));
    };

    // Backward pass: accumulate products and intermediate matrices for Jacobian
    Vec3 delta_vec[DEG];
    Mat3 R_tmp[DEG];
    SO3  exp_k_delta[DEG];
    Mat3 Jr_delta_inv[DEG];
    Mat3 Jr_kdelta[DEG];

    SO3 accum;  // identity
    for (int i = DEG - 1; i >= 0; --i) {
        SO3  r01 = knot(i).inverse() * knot(i + 1);
        delta_vec[i] = r01.log();

        Sophus::rightJacobianInvSO3(delta_vec[i], Jr_delta_inv[i]);
        Jr_delta_inv[i] *= knot(i + 1).inverse().matrix();

        Vec3 k_delta = coeff[i + 1] * delta_vec[i];
        Sophus::rightJacobianSO3(-k_delta, Jr_kdelta[i]);

        R_tmp[i]       = accum.matrix();
        exp_k_delta[i] = SO3::exp(-k_delta);
        accum *= exp_k_delta[i];
    }

    // Forward pass: compute angular velocity and Jacobian
    Mat3 d_vel_d_delta[DEG];
    d_vel_d_delta[0] = dcoeff[1] * R_tmp[0] * Jr_delta_inv[0];
    Vec3 rot_vel     = delta_vec[0] * dcoeff[1];

    for (int i = 1; i < DEG; ++i) {
        d_vel_d_delta[i] =
            R_tmp[i - 1] * SO3::hat(rot_vel) * Jr_kdelta[i] * coeff[i + 1]
            + R_tmp[i] * dcoeff[i + 1];
        d_vel_d_delta[i] *= Jr_delta_inv[i];
        rot_vel = exp_k_delta[i] * rot_vel + delta_vec[i] * dcoeff[i + 1];
    }

    if (J) {
        for (int i = 0; i < N; ++i) J->d_val_d_knot[i].setZero();
        for (int i = 0; i < DEG; ++i) {
            J->d_val_d_knot[i]     -= d_vel_d_delta[i];
            J->d_val_d_knot[i + 1] += d_vel_d_delta[i];
        }
    }
    return rot_vel;
}

// ============================================================================
// rotation_with_jacobian_manual  (production — zero heap allocation)
// ============================================================================
// Port of basalt::So3Spline::evaluate(t_ns, J). LEFT perturbation convention.
template <int N>
Sophus::SO3d rotation_with_jacobian_manual(
    double const* const* params, double u, double inv_dt, JacobianStruct<N>* J)
{
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;
    using SO3  = Sophus::SO3d;
    constexpr int DEG = N - 1;

    using Helper = CeresSplineHelper<N>;
    using VecN   = Eigen::Matrix<double, N, 1>;

    VecN p;
    Helper::template baseCoeffsWithTime<0>(p, u);
    const VecN coeff = Helper::cumulative_blending_matrix_ * p;

    auto knot = [&](int i) -> SO3 {
        const double* q = params[i];
        return SO3(Eigen::Quaterniond(q[3], q[0], q[1], q[2]));
    };

    SO3 res = knot(0);
    Mat3 J_helper;
    if (J) {
        J->d_val_d_knot[0].setIdentity();
        J_helper.setIdentity();
    }

    for (int i = 0; i < DEG; ++i) {
        SO3  r01    = knot(i).inverse() * knot(i + 1);
        Vec3 delta  = r01.log();
        Vec3 kdelta = coeff[i + 1] * delta;

        if (J) {
            Mat3 Jl_inv_delta, Jl_k_delta;
            Sophus::leftJacobianInvSO3(delta,  Jl_inv_delta);
            Sophus::leftJacobianSO3   (kdelta, Jl_k_delta);
            J->d_val_d_knot[i] = J_helper;
            J_helper = coeff[i + 1] * res.matrix()
                       * Jl_k_delta * Jl_inv_delta
                       * knot(i).inverse().matrix();
            J->d_val_d_knot[i] -= J_helper;
        }
        res *= SO3::exp(kdelta);
    }
    if (J) J->d_val_d_knot[DEG] = J_helper;
    return res;
}

}  // namespace analytic
}  // namespace rio
