#pragma once

// GTSAM-tangent math for the FLOOR-PLANE absolute-position factor (plane mapping).
// A static radar return classified as FLOOR must lie on the horizontal floor plane
// world_z = floor_z. Orientation R is FROZEN at the warm-start (R_ws): the return's
// world height is z_world = (R_ws * p_body).z + p_traj(t).z, so the residual
//   r = z_off + sum_i posval_i * CP_i.z  -  floor_z,   z_off := (R_ws * p_body).z
// is LINEAR in the position CPs (z-component only). This injects an ABSOLUTE vertical
// anchor (kills the radar z-bias drift, the dominant position-error component) without
// dragging the gyro-driven orientation (same rationale as RadarPosOnlyFactor).
//
// posval_i = (blending_matrix_ * baseCoeffsWithTime<0>(u_pos))_i  (degree-5 value).

#include <Eigen/Dense>
#include <array>
#include <rio/trajectory.h>
#include <rio/factors/analytic/spline_jacobians.h>

namespace rio {
namespace gtsam_factors {

struct FloorResidual {
    double residual;
    std::array<Eigen::Matrix<double, 1, 3>, N_POS> d_r_d_cp;   // 1x3 per CP (z only)
};

// z_off = (R_ws * p_body).z already folded by the caller.
inline FloorResidual floor_residual_gtsam(
    double const* const* pos_cps, double z_off, double floor_z, double u_pos,
    bool want_jac = true)
{
    using Helper = CeresSplineHelper<N_POS>;
    using VecNP = Eigen::Matrix<double, N_POS, 1>;
    VecNP p0;
    Helper::template baseCoeffsWithTime<0>(p0, u_pos);
    const VecNP w = Helper::blending_matrix_ * p0;          // value weights (no 1/dt)

    double z_traj = 0.0;
    for (int i = 0; i < N_POS; ++i) z_traj += w[i] * pos_cps[i][2];

    FloorResidual out;
    out.residual = z_off + z_traj - floor_z;
    if (want_jac)
        for (int i = 0; i < N_POS; ++i) out.d_r_d_cp[i] = (Eigen::Matrix<double, 1, 3>() << 0.0, 0.0, w[i]).finished();
    return out;
}

}  // namespace gtsam_factors
}  // namespace rio
