#pragma once

// GTSAM-tangent factor math for the gyro residual (Phase 1).
//
// API-INDEPENDENT core: takes raw quaternion knot pointers ([x,y,z,w]) + bias,
// returns the residual and the Jacobians w.r.t. the GTSAM RIGHT-tangent of each
// variable (3 per Rot3 knot, 6 for the bias).  The thin gtsam::NoiseModelFactorN
// wrapper just forwards to this; the wrapper needs GTSAM, this header does not,
// so the convention/chain can be unit-tested on the existing toolchain.
//
// Convention bridge (verified in the Python spike, Phase 0a):
//   basalt d_val_d_knot[i] is the LEFT-perturbation Jacobian
//     (q_i -> Exp(eps) q_i).  GTSAM Rot3.retract is RIGHT (q_i -> q_i Exp(delta)).
//   Exp(eps) q_i = q_i Exp(delta)  =>  eps = Ad_{q_i} delta = R_i delta.
//   => d(f)/d(delta) = d(f)/d(eps) * R_i = J_left * R_i.matrix().
//
// Residual:  r = z_gyro - omega_body(knots) - b_g          (3D)

#include <Eigen/Dense>
#include <array>
#include <rio/trajectory.h>
#include <rio/factors/analytic/spline_jacobians.h>

namespace rio {
namespace gtsam_factors {

struct GyroResidual {
    Eigen::Vector3d residual;
    std::array<Eigen::Matrix3d, N_ORI> d_r_d_knot;     // GTSAM right-tangent (3x3)
    Eigen::Matrix<double, 3, 6> d_r_d_bias;            // [0 | -I]
};

// ori_knots: array of N_ORI pointers to [x,y,z,w]; bias6: [ba(3), bg(3)].
inline GyroResidual gyro_residual_gtsam(
    double const* const* ori_knots, const double* bias6,
    const Eigen::Vector3d& z_gyro, double u_ori, double inv_dt_ori,
    bool want_jac = true)
{
    using analytic::JacobianStruct;
    using analytic::body_velocity_with_jacobian_manual;

    GyroResidual out;
    JacobianStruct<N_ORI> Jw;
    const Eigen::Vector3d omega = body_velocity_with_jacobian_manual<N_ORI>(
        ori_knots, u_ori, inv_dt_ori, want_jac ? &Jw : nullptr);

    const Eigen::Map<const Eigen::Vector3d> b_g(bias6 + 3);
    out.residual = z_gyro - omega - b_g;

    if (want_jac) {
        for (int i = 0; i < N_ORI; ++i) {
            const double* q = ori_knots[i];
            const Eigen::Matrix3d R_i =
                Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized().toRotationMatrix();
            // d(r)/d(delta_i) = -d(omega)/d(eps_i) * R_i   (left -> right tangent)
            out.d_r_d_knot[i].noalias() = -Jw.d_val_d_knot[i] * R_i;
        }
        out.d_r_d_bias.setZero();
        out.d_r_d_bias.rightCols<3>() = -Eigen::Matrix3d::Identity();
    }
    return out;
}

}  // namespace gtsam_factors
}  // namespace rio
