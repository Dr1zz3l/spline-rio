#pragma once

// GTSAM-tangent math for the radar POSITION-ONLY Doppler factor (radar_pos_split).
// Orientation R and body-rate omega are FROZEN at the warm-start; the residual is
// then linear in the position CPs, so the radar velocity information flows into
// position without dragging the orientation knots. Mirrors RadarPosOnlyAnalyticFactor.
// Used for the complementary (1-w_omega) weight when the omega soft-gate is active.

#include <Eigen/Dense>
#include <array>
#include <rio/trajectory.h>
#include <rio/factors/analytic/spline_jacobians.h>
#include <rio/factors/analytic/radar_sensor_jac_gen.h>

namespace rio {
namespace gtsam_factors {

struct RadarPosOnlyResidual {
    double residual;
    std::array<Eigen::Matrix<double, 1, 3>, N_POS> d_r_d_cp;   // 1x3 per CP
};

// R_ws (frozen warm-start rotation), omega_ws (frozen warm-start body rate).
inline RadarPosOnlyResidual radar_pos_only_residual_gtsam(
    double const* const* pos_cps, const Eigen::Matrix3d& R_ws, const Eigen::Vector3d& omega_ws,
    const Eigen::Vector3d& u_body, double v_meas, const Eigen::Vector3d& t_body_sensor,
    double u_pos, double inv_dt_pos, bool want_jac = true)
{
    using Vec3 = Eigen::Vector3d;
    using Helper = CeresSplineHelper<N_POS>;
    using VecNP = Eigen::Matrix<double, N_POS, 1>;
    VecNP p1;
    Helper::template baseCoeffsWithTime<1>(p1, u_pos);
    const VecNP v_coeff = inv_dt_pos * Helper::blending_matrix_ * p1;
    Vec3 v_world = Vec3::Zero();
    for (int i = 0; i < N_POS; ++i)
        v_world += v_coeff[i] * Eigen::Map<const Vec3>(pos_cps[i]);

    const Eigen::Quaterniond q(R_ws);
    const sym::Rot3d sym_R(Eigen::Vector4d(q.x(), q.y(), q.z(), q.w()));
    Eigen::Matrix<double, 1, 3> J_v;
    Eigen::Matrix<double, 1, 3>* const nullj = nullptr;

    RadarPosOnlyResidual out;
    out.residual = sym::RadarSensorJacWithJacobians012(
        v_world, sym_R, omega_ws, u_body, t_body_sensor, v_meas, 1e-10,
        want_jac ? &J_v : nullj, nullj, nullj)[0];
    if (want_jac)
        for (int i = 0; i < N_POS; ++i) out.d_r_d_cp[i].noalias() = v_coeff[i] * J_v;
    return out;
}

}  // namespace gtsam_factors
}  // namespace rio
