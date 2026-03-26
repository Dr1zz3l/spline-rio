#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <basalt/spline/ceres_spline_helper.h>
#include <rio/trajectory.h>

namespace rio {

// Gravity vector in world frame (FLU, z-up): points downward = negative z.
// Matches Python convention: g_world = [0, 0, -9.81].
static const Eigen::Vector3d GRAVITY_WORLD{0.0, 0.0, -9.81};

// ============================================================================
// AccelFactor
// ============================================================================
// Residual (3D): r = z_acc - (R_bw^T (a_world - g) - b_a)
// where a_world is the second time derivative of the position spline.
//
// Parameter blocks:
//   [0..N_ORI-1]      : orientation knots (4 params each)
//   [N_ORI..N_ORI+N_POS-1]: position CPs  (3 params each)
//   [N_ORI+N_POS]     : bias block (6 params)

struct AccelFunctor {
    Eigen::Vector3d z_acc;   // raw accelerometer measurement (m/s²)
    double u_ori, inv_dt_ori;
    double u_pos, inv_dt_pos;

    AccelFunctor(const Eigen::Vector3d& z_acc_,
                 double u_ori_, double inv_dt_ori_,
                 double u_pos_, double inv_dt_pos_)
        : z_acc(z_acc_),
          u_ori(u_ori_), inv_dt_ori(inv_dt_ori_),
          u_pos(u_pos_), inv_dt_pos(inv_dt_pos_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;

        SO3T R_body_to_world;
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params, u_ori, inv_dt_ori, &R_body_to_world, nullptr, nullptr);

        Vec3T a_world;
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 2>(
            params + N_ORI, u_pos, inv_dt_pos, &a_world);

        // Specific force in body frame
        Vec3T specific_force = R_body_to_world.inverse() * (a_world - GRAVITY_WORLD.cast<T>());

        // Accel bias (first 3 of the 6-dof bias block)
        Eigen::Map<const Vec3T> b_a(params[N_ORI + N_POS]);

        Vec3T res = z_acc.cast<T>() - specific_force - b_a;
        residuals[0] = res[0];
        residuals[1] = res[1];
        residuals[2] = res[2];
        return true;
    }
};

}  // namespace rio
