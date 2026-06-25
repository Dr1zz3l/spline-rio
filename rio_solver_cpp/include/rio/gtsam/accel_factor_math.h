#pragma once

// GTSAM-tangent factor math for the accelerometer residual (Phase 1).
// API-independent core (raw pointers); the gtsam wrapper forwards to this.
//
// Residual:  r = z_acc - (R^T (a_world - g) + b_a)            (3D)
// Mirrors AccelAnalyticFactor; orientation knot Jacobian uses the GTSAM
// right-tangent conversion (J_left * R(knot_i)) instead of tangent_to_ambient.

#include <Eigen/Dense>
#include <array>
#include <rio/trajectory.h>
#include <rio/factors/imu_accel.h>                   // GRAVITY_WORLD
#include <rio/factors/analytic/spline_jacobians.h>

namespace rio {
namespace gtsam_factors {

struct AccelResidual {
    Eigen::Vector3d residual;
    std::array<Eigen::Matrix3d, N_ORI> d_r_d_knot;   // GTSAM right-tangent (3x3)
    std::array<Eigen::Matrix3d, N_POS> d_r_d_cp;     // Euclidean (3x3)
    Eigen::Matrix<double, 3, 6> d_r_d_bias;          // [-I | 0]
};

inline AccelResidual accel_residual_gtsam(
    double const* const* ori_knots, double const* const* pos_cps, const double* bias6,
    const Eigen::Vector3d& z_acc,
    double u_ori, double inv_dt_ori, double u_pos, double inv_dt_pos,
    bool want_jac = true)
{
    using Mat3 = Eigen::Matrix3d;
    using Vec3 = Eigen::Vector3d;
    using analytic::JacobianStruct;
    using analytic::rotation_with_jacobian_manual;

    AccelResidual out;
    JacobianStruct<N_ORI> J_R;
    const Sophus::SO3d R = rotation_with_jacobian_manual<N_ORI>(
        ori_knots, u_ori, inv_dt_ori, want_jac ? &J_R : nullptr);

    using Helper = CeresSplineHelper<N_POS>;
    using VecNP = Eigen::Matrix<double, N_POS, 1>;
    VecNP p2;
    Helper::template baseCoeffsWithTime<2>(p2, u_pos);
    const VecNP pos_coeff = (inv_dt_pos * inv_dt_pos) * Helper::blending_matrix_ * p2;

    Vec3 a_world = Vec3::Zero();
    for (int i = 0; i < N_POS; ++i)
        a_world += pos_coeff[i] * Eigen::Map<const Vec3>(pos_cps[i]);

    const Vec3 sf = R.inverse() * (a_world - GRAVITY_WORLD);   // specific force
    const Eigen::Map<const Vec3> b_a(bias6);
    out.residual = z_acc - sf - b_a;

    if (want_jac) {
        const Mat3 Rt = R.inverse().matrix();
        Mat3 d_r_d_R_tang;
        d_r_d_R_tang <<  0.0,    sf[2], -sf[1],
                        -sf[2],   0.0,   sf[0],
                         sf[1], -sf[0],  0.0;
        const Mat3 d_r_dR_full = d_r_d_R_tang * Rt;            // -[sf]^x R^T
        for (int i = 0; i < N_ORI; ++i) {
            const double* q = ori_knots[i];
            const Mat3 R_i =
                Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized().toRotationMatrix();
            out.d_r_d_knot[i].noalias() = (d_r_dR_full * J_R.d_val_d_knot[i]) * R_i;
        }
        for (int i = 0; i < N_POS; ++i)
            out.d_r_d_cp[i].noalias() = -pos_coeff[i] * Rt;
        out.d_r_d_bias.setZero();
        out.d_r_d_bias.leftCols<3>() = -Eigen::Matrix3d::Identity();
    }
    return out;
}

}  // namespace gtsam_factors
}  // namespace rio
