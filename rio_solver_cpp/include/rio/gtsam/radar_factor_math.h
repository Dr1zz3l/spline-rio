#pragma once

// GTSAM-tangent factor math for the radar Doppler residual (Phase 1).
// API-independent core (raw pointers); the gtsam wrapper forwards to this.
//
// Residual (scalar): r = v_meas - v_pred, v_pred = -dot(u_body, v_ant),
// v_ant = R^T v_world + omega x t_bs.  Mirrors RadarAnalyticFactor.
// Depends on 4 ori knots + 6 pos CPs (NOT bias).  Orientation knot Jacobian
// uses the GTSAM right-tangent conversion (J_left * R(knot_i)).

#include <Eigen/Dense>
#include <array>
#include <sophus/so3.hpp>
#include <rio/trajectory.h>
#include <rio/factors/analytic/spline_jacobians.h>
#include <rio/factors/analytic/radar_sensor_jac_gen.h>

namespace rio {
namespace gtsam_factors {

struct RadarResidual {
    double residual;
    std::array<Eigen::Matrix<double, 1, 3>, N_ORI> d_r_d_knot;   // GTSAM right-tangent (1x3)
    std::array<Eigen::Matrix<double, 1, 3>, N_POS> d_r_d_cp;     // 1x3
};

// u_body = R_radar_to_body * u_sensor (precomputed, constant per measurement).
inline RadarResidual radar_residual_gtsam(
    double const* const* ori_knots, double const* const* pos_cps,
    const Eigen::Vector3d& u_body, double v_meas, const Eigen::Vector3d& t_body_sensor,
    double u_ori, double inv_dt_ori, double u_pos, double inv_dt_pos,
    bool want_jac = true)
{
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;
    using analytic::JacobianStruct;
    using analytic::body_velocity_with_jacobian_manual;
    using analytic::rotation_with_jacobian_manual;

    RadarResidual out;
    JacobianStruct<N_ORI> J_omega_s, J_R_s;
    const Vec3 omega = body_velocity_with_jacobian_manual<N_ORI>(
        ori_knots, u_ori, inv_dt_ori, want_jac ? &J_omega_s : nullptr);
    const Sophus::SO3d R = rotation_with_jacobian_manual<N_ORI>(
        ori_knots, u_ori, inv_dt_ori, want_jac ? &J_R_s : nullptr);

    using Helper = CeresSplineHelper<N_POS>;
    using VecNP = Eigen::Matrix<double, N_POS, 1>;
    VecNP p1;
    Helper::template baseCoeffsWithTime<1>(p1, u_pos);
    const VecNP v_coeff = inv_dt_pos * Helper::blending_matrix_ * p1;
    Vec3 v_world = Vec3::Zero();
    for (int i = 0; i < N_POS; ++i)
        v_world += v_coeff[i] * Eigen::Map<const Vec3>(pos_cps[i]);

    const auto quat = R.unit_quaternion();
    const sym::Rot3d sym_R(Eigen::Vector4d(quat.x(), quat.y(), quat.z(), quat.w()));

    Eigen::Matrix<double, 1, 3> J_v, J_R_right, J_omega_sf;
    out.residual = sym::RadarSensorJacWithJacobians012(
        v_world, sym_R, omega, u_body, t_body_sensor, v_meas, 1e-10,
        want_jac ? &J_v : nullptr,
        want_jac ? &J_R_right : nullptr,
        want_jac ? &J_omega_sf : nullptr)[0];

    if (want_jac) {
        const Eigen::Matrix<double, 1, 3> J_R_left = J_R_right * R.inverse().matrix();
        for (int i = 0; i < N_ORI; ++i) {
            const Eigen::Matrix<double, 1, 3> J_local =
                J_R_left * J_R_s.d_val_d_knot[i] + J_omega_sf * J_omega_s.d_val_d_knot[i];
            const double* q = ori_knots[i];
            const Mat3 R_i =
                Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized().toRotationMatrix();
            out.d_r_d_knot[i].noalias() = J_local * R_i;
        }
        for (int i = 0; i < N_POS; ++i)
            out.d_r_d_cp[i].noalias() = v_coeff[i] * J_v;
    }
    return out;
}

}  // namespace gtsam_factors
}  // namespace rio
