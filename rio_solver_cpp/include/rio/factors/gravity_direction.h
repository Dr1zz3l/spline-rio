#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <basalt/spline/ceres_spline_helper.h>
#include <rio/trajectory.h>

namespace rio {

// ============================================================================
// GravityDirectionFactor
// ============================================================================
// Mahony-style roll/pitch constraint from accelerometer.
// Residual (3D): r = normalize(a_debiased) · g_hat - R^T [0,0,g]
//
// Only applied when ||a_debiased|| is close to g (within gravity_threshold).
// Equivalent to the Python lambda_gravity factor.
//
// Parameter blocks:
//   [0..N_ORI-1]: orientation knots (4 params each)
//   [N_ORI]     : bias block (6 params)

struct GravityDirectionFunctor {
    Eigen::Vector3d a_meas;  // raw accelerometer measurement
    double u_ori, inv_dt_ori;
    double gravity_mag;

    GravityDirectionFunctor(const Eigen::Vector3d& a_meas_,
                            double u_ori_, double inv_dt_ori_,
                            double gravity_mag_ = 9.81)
        : a_meas(a_meas_), u_ori(u_ori_), inv_dt_ori(inv_dt_ori_),
          gravity_mag(gravity_mag_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using SO3T = Sophus::SO3<T>;

        SO3T R_body_to_world;
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params, u_ori, inv_dt_ori, &R_body_to_world, nullptr, nullptr);

        // Accel bias
        const T* bias_ptr = params[N_ORI];
        Vec3T b_a(bias_ptr[0], bias_ptr[1], bias_ptr[2]);

        Vec3T a_debiased = a_meas.cast<T>() - b_a;

        // Expected gravity in body frame: R^T [0,0,g]
        Vec3T g_world(T(0), T(0), T(gravity_mag));
        Vec3T g_body = R_body_to_world.inverse() * g_world;

        // Residual: g_body - a_debiased (both should point in same direction when level)
        // We normalise a_debiased to get a direction-only constraint.
        T a_norm = a_debiased.norm();
        Vec3T a_unit = a_debiased / (a_norm + T(1e-8));

        Vec3T g_unit = g_body / (g_body.norm() + T(1e-8));

        Vec3T res = g_unit - a_unit;
        residuals[0] = res[0];
        residuals[1] = res[1];
        residuals[2] = res[2];
        return true;
    }
};

}  // namespace rio
