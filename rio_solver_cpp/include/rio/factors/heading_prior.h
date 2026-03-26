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
// Yaw-only constraint from MoCap pseudo-magnetometer.
// Residual (1D): r = yaw_pred - yaw_ref
// where yaw_pred is extracted from R(t) via atan2(R[1,0], R[0,0]).
//
// Parameter blocks:
//   [0..N_ORI-1]: orientation knots (4 params each)

struct HeadingPriorFunctor {
    double yaw_ref;   // reference yaw (radians)
    double u_ori, inv_dt_ori;

    HeadingPriorFunctor(double yaw_ref_, double u_ori_, double inv_dt_ori_)
        : yaw_ref(yaw_ref_), u_ori(u_ori_), inv_dt_ori(inv_dt_ori_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using SO3T = Sophus::SO3<T>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        SO3T R_body_to_world;
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params, u_ori, inv_dt_ori, &R_body_to_world, nullptr, nullptr);

        Mat3T R = R_body_to_world.matrix();
        // Yaw from rotation matrix: atan2(R[1,0], R[0,0])
        T yaw_pred = ceres::atan2(R(1, 0), R(0, 0));

        // Wrap to [-pi, pi]
        T diff = yaw_pred - T(yaw_ref);
        // Keep diff in (-pi, pi]
        while (diff > T(M_PI))  diff -= T(2 * M_PI);
        while (diff < T(-M_PI)) diff += T(2 * M_PI);

        residuals[0] = diff;
        return true;
    }
};

}  // namespace rio
