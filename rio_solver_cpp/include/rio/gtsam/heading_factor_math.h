#pragma once

// GTSAM-tangent factor math for the heading (pseudo-magnetometer) prior (Phase 2).
// Residual (1D, SO(3)-native, matches heading_prior.h):
//   body_x = R(t) * e1;   r = body_x[0]*sin(yaw_ref) - body_x[1]*cos(yaw_ref)
// = sin(yaw_pred - yaw_ref) * cos(pitch).  GTSAM right-tangent knot Jacobians via
// the same left/right chain as the radar factor.

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <sophus/so3.hpp>
#include <rio/trajectory.h>
#include <rio/factors/analytic/spline_jacobians.h>

namespace rio {
namespace gtsam_factors {

struct HeadingResidual {
    double residual;
    std::array<Eigen::Matrix<double, 1, 3>, N_ORI> d_r_d_knot;   // GTSAM right-tangent (1x3)
};

inline HeadingResidual heading_residual_gtsam(
    double const* const* ori_knots, double yaw_ref,
    double u_ori, double inv_dt_ori, bool want_jac = true)
{
    using Mat3 = Eigen::Matrix3d;
    using analytic::JacobianStruct;
    using analytic::rotation_with_jacobian_manual;

    HeadingResidual out;
    JacobianStruct<N_ORI> J_R_s;
    const Sophus::SO3d R = rotation_with_jacobian_manual<N_ORI>(
        ori_knots, u_ori, inv_dt_ori, want_jac ? &J_R_s : nullptr);

    const Eigen::Vector3d body_x = R.matrix().col(0);   // R * e1
    const double sy = std::sin(yaw_ref), cx = std::cos(yaw_ref);
    out.residual = body_x[0] * sy - body_x[1] * cx;

    if (want_jac) {
        // d r / d body_x = [sy, -cx, 0]; d body_x / d(R right-tangent) = -R [e1]_x
        Mat3 skew_e1; skew_e1 << 0, 0, 0,  0, 0, -1,  0, 1, 0;
        const Eigen::RowVector3d dr_dbx(sy, -cx, 0.0);
        const Eigen::RowVector3d J_right_Rt = dr_dbx * (-R.matrix() * skew_e1);
        const Eigen::RowVector3d J_left_Rt = J_right_Rt * R.inverse().matrix();
        for (int i = 0; i < N_ORI; ++i) {
            const double* q = ori_knots[i];
            const Mat3 R_i =
                Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized().toRotationMatrix();
            out.d_r_d_knot[i].noalias() = (J_left_Rt * J_R_s.d_val_d_knot[i]) * R_i;
        }
    }
    return out;
}

}  // namespace gtsam_factors
}  // namespace rio
