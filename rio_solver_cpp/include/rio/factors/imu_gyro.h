#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <basalt/spline/ceres_spline_helper.h>
#include <rio/trajectory.h>

namespace rio {

// ============================================================================
// GyroFactor
// ============================================================================
// Residual (3D): r = z_gyro - (omega_body - b_g)
// where omega_body is the body angular velocity from the orientation spline.
//
// Parameter blocks:
//   [0..N_ORI-1]: orientation knots (4 params each)
//   [N_ORI]     : bias block (6 params, gyro bias = params[N_ORI][3..5])

struct GyroFunctor {
    Eigen::Vector3d z_gyro;   // raw gyroscope measurement (rad/s)
    double u_ori, inv_dt_ori;

    GyroFunctor(const Eigen::Vector3d& z_gyro_,
                double u_ori_, double inv_dt_ori_)
        : z_gyro(z_gyro_), u_ori(u_ori_), inv_dt_ori(inv_dt_ori_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using SO3T = Sophus::SO3<T>;

        Vec3T omega_body;
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params, u_ori, inv_dt_ori,
            (SO3T*)nullptr,
            &omega_body,
            nullptr);

        // Gyro bias: indices [3,4,5] of the 6-dof bias block
        const T* bias_ptr = params[N_ORI];
        Vec3T b_g(bias_ptr[3], bias_ptr[4], bias_ptr[5]);

        Vec3T res = z_gyro.cast<T>() - omega_body - b_g;
        residuals[0] = res[0];
        residuals[1] = res[1];
        residuals[2] = res[2];
        return true;
    }
};

}  // namespace rio
