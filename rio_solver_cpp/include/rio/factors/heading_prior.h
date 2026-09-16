#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <basalt/spline/ceres_spline_helper.h>
#include <rio/trajectory.h>
#include <cmath>

namespace rio {

// ============================================================================
// HeadingPriorFactor
// ============================================================================
// Soft heading constraint from MoCap (pseudo-magnetometer).
//
// Residual (1D): 2D cross product of the predicted and reference forward
// directions projected onto the world horizontal plane:
//
//   r = (R·x̂)[0] · ref_y − (R·x̂)[1] · ref_x
//
// where (R·x̂) is the body x-axis in world frame and (ref_x, ref_y) is the
// reference direction derived from yaw_ref: (cos(yaw_ref), sin(yaw_ref)).
//
// This equals sin(yaw_pred − yaw_ref) · |body_x projected onto xy|.
//
// Properties:
//  - No Euler angle extraction inside AutoDiff — fully SO(3)-native.
//  - No angle wrapping needed; sin handles the full circle.
//  - Naturally uninformative at |pitch| = 90°: body_x points straight
//    up/down, its xy-projection vanishes, residual = 0.  The constraint
//    gracefully fades rather than producing a wrong value (unlike atan2
//    which returns an arbitrary angle at the gimbal-lock singularity).
//
// Note: yaw_ref is still a scalar derived from the MoCap rotation matrix
// via atan2 on the Python side — that extraction is fine because it is a
// constant (no AutoDiff passes through it).
//
// Parameter blocks:
//   [0..N_ORI-1]: orientation knots (4 params each)

struct HeadingPriorFunctor {
    double yaw_ref;   // reference heading (radians), pre-computed constant
    double u_ori, inv_dt_ori;

    HeadingPriorFunctor(double yaw_ref_, double u_ori_, double inv_dt_ori_)
        : yaw_ref(yaw_ref_), u_ori(u_ori_), inv_dt_ori(inv_dt_ori_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using SO3T = Sophus::SO3<T>;

        SO3T R_body_to_world;
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params, u_ori, inv_dt_ori, &R_body_to_world, nullptr, nullptr);

        // Body x-axis in world frame (first column of R)
        Eigen::Matrix<T, 3, 1> body_x = R_body_to_world * Eigen::Matrix<T, 3, 1>(T(1), T(0), T(0));

        // Reference direction in world xy-plane
        const T ref_x = T(std::cos(yaw_ref));
        const T ref_y = T(std::sin(yaw_ref));

        // 2D cross product: sin(heading_error) * cos(pitch)
        residuals[0] = body_x[0] * ref_y - body_x[1] * ref_x;
        return true;
    }
};

}  // namespace rio
