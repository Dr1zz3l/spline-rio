#pragma once

// Analytical Jacobian implementation of GyroFunctor.
//
// Replaces DynamicAutoDiffCostFunction<GyroFunctor, 4> with a hand-derived
// ceres::CostFunction that computes d(r)/d(knot_i) and d(r)/d(bias) directly,
// bypassing Jet arithmetic entirely.
//
// Residual:  r = z_gyro - omega_body(knots) - b_g      (3D)
//
// Jacobians:
//   d(r)/d(knot_i) = -d(omega)/d(knot_i)    [3×4 ambient]
//   d(r)/d(bias)   = [0₃ₓ₃ | -I₃ₓ₃]         [3×6]
//
// d(omega)/d(knot_i) is computed analytically via spline_jacobians.h, which
// ports basalt::So3Spline::velocityBody(t, J) to work with raw quaternion pointers.
//
// Parameter blocks (same layout as GyroFunctor / make_gyro_cost):
//   [0..N_ORI-1]: orientation knots (4 params each, XYZW)
//   [N_ORI]     : bias block (6 params: [b_a₀,b_a₁,b_a₂,b_g₀,b_g₁,b_g₂])

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <rio/trajectory.h>
#include <rio/factors/analytic/spline_jacobians.h>

namespace rio {
namespace analytic {

class GyroAnalyticFactor : public ceres::CostFunction {
public:
    GyroAnalyticFactor(const Eigen::Vector3d& z_gyro,
                       double u_ori, double inv_dt_ori)
        : z_gyro_(z_gyro), u_ori_(u_ori), inv_dt_ori_(inv_dt_ori)
    {
        set_num_residuals(3);
        for (int i = 0; i < N_ORI; ++i) mutable_parameter_block_sizes()->push_back(4);
        mutable_parameter_block_sizes()->push_back(6);  // bias
    }

    bool Evaluate(double const* const* params,
                  double* residuals,
                  double** jacobians) const override
    {
        JacobianStruct<N_ORI> J_omega;
        const Eigen::Vector3d omega = body_velocity_with_jacobian_manual<N_ORI>(
            params, u_ori_, inv_dt_ori_, jacobians ? &J_omega : nullptr);

        // Gyro bias: indices [3,4,5] of the 6-DOF bias block
        const Eigen::Map<const Eigen::Vector3d> b_g(params[N_ORI] + 3);

        Eigen::Map<Eigen::Vector3d>(residuals).noalias() = z_gyro_ - omega - b_g;

        if (!jacobians) return true;

        // Orientation knot Jacobians: d(r)/d(knot_i) = -d(omega)/d(knot_i)
        // Convert 3×3 local (tangent) Jacobian → 3×4 ambient for Ceres.
        for (int i = 0; i < N_ORI; ++i) {
            if (!jacobians[i]) continue;
            // J_ambient (3×4) = J_local (3×3) · PlusJac^T (3×4)
            const Eigen::Matrix<double, 3, 4> plus_jac_T = tangent_to_ambient(params[i]);
            Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> Jq(jacobians[i]);
            Jq.noalias() = -J_omega.d_val_d_knot[i] * plus_jac_T;
        }

        // Bias Jacobian: d(r)/d([b_a, b_g]) = [0₃ₓ₃ | -I₃ₓ₃]
        if (jacobians[N_ORI]) {
            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jb(jacobians[N_ORI]);
            Jb.leftCols<3>().setZero();
            Jb.rightCols<3>() = -Eigen::Matrix3d::Identity();
        }

        return true;
    }

private:
    Eigen::Vector3d z_gyro_;
    double u_ori_, inv_dt_ori_;
};

}  // namespace analytic
}  // namespace rio
