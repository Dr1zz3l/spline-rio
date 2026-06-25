#pragma once

// GTSAM-tangent factor math for the two regularizers (Phase 1).
//   MinSnap:   r = p^(4)(u_pos)  (linear in N_POS CPs)   -> d_r/d_cp[i] = coeff[i] I
//   AngAccel:  r = Log(q1^-1 q2) - Log(q0^-1 q1)  over 3 ori knots
//
// AngAccel GTSAM right-tangent Jacobians (qi -> qi Exp(delta_i)):
//   phi_a = Log(q0^-1 q1),  phi_b = Log(q1^-1 q2)
//   d(Log(qa^-1 qb))/d(delta_a) = -Jl^-1(phi),  /d(delta_b) = +Jr^-1(phi)
//   => d_r/d0 =  Jl^-1(phi_a)
//      d_r/d1 = -Jl^-1(phi_b) - Jr^-1(phi_a)
//      d_r/d2 =  Jr^-1(phi_b)

#include <Eigen/Dense>
#include <array>
#include <sophus/so3.hpp>
#include <basalt/utils/sophus_utils.hpp>
#include <rio/trajectory.h>
#include <basalt/spline/ceres_spline_helper.h>

namespace rio {
namespace gtsam_factors {

struct MinSnapResidual {
    Eigen::Vector3d residual;
    std::array<double, N_POS> coeff;     // d_r/d_cp[i] = coeff[i] * I3
};

inline MinSnapResidual minsnap_residual_gtsam(
    double const* const* pos_cps, double u_pos, double inv_dt_pos)
{
    using Vec3 = Eigen::Vector3d;
    using Helper = CeresSplineHelper<N_POS>;
    using VecNP = Eigen::Matrix<double, N_POS, 1>;
    VecNP p4;
    Helper::template baseCoeffsWithTime<4>(p4, u_pos);
    const double s = inv_dt_pos * inv_dt_pos * inv_dt_pos * inv_dt_pos;
    const VecNP c = s * Helper::blending_matrix_ * p4;

    MinSnapResidual out;
    Vec3 snap = Vec3::Zero();
    for (int i = 0; i < N_POS; ++i) {
        out.coeff[i] = c[i];
        snap += c[i] * Eigen::Map<const Vec3>(pos_cps[i]);
    }
    out.residual = snap;
    return out;
}

struct AngAccelResidual {
    Eigen::Vector3d residual;
    std::array<Eigen::Matrix3d, 3> d_r_d_knot;   // GTSAM right-tangent (3x3), knots [q0,q1,q2]
};

inline AngAccelResidual angaccel_residual_gtsam(
    const double* q0, const double* q1, const double* q2, bool want_jac = true)
{
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;
    auto so3 = [](const double* q) {
        return Sophus::SO3d(Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized());
    };
    const Sophus::SO3d R0 = so3(q0), R1 = so3(q1), R2 = so3(q2);
    const Vec3 phi_a = (R0.inverse() * R1).log();
    const Vec3 phi_b = (R1.inverse() * R2).log();

    AngAccelResidual out;
    out.residual = phi_b - phi_a;
    if (want_jac) {
        Mat3 JlA, JrA, JlB, JrB;
        Sophus::leftJacobianInvSO3(phi_a, JlA);
        Sophus::rightJacobianInvSO3(phi_a, JrA);
        Sophus::leftJacobianInvSO3(phi_b, JlB);
        Sophus::rightJacobianInvSO3(phi_b, JrB);
        out.d_r_d_knot[0] =  JlA;
        out.d_r_d_knot[1] = -JlB - JrA;
        out.d_r_d_knot[2] =  JrB;
    }
    return out;
}

}  // namespace gtsam_factors
}  // namespace rio
