#pragma once

// Analytical Jacobian implementation of AccelFunctor.
//
// Residual:  r = z_acc - (R^T (a_world - g) + b_a)     (3D)
//
// Jacobians via two-level chain rule:
//
//   Level 1 — sensor model at current linearization point:
//     d(r)/d(R_tangent)   = -[R^T(a_world-g)]^×     (3×3, skew of specific force)
//     d(r)/d(a_world)     = -R^T                     (3×3)
//     d(r)/d(b_a)         = -I                       (3×3)
//
//   Level 2 — spline model:
//     d(R_tangent)/d(ori_knot_i)  from rotation_with_jacobian  (3×3 local)
//     d(a_world)/d(pos_CP_i)      = coeff[i] * I               (scalar)
//
//   Full:
//     d(r)/d(ori_knot_i) = -skew(R^T(a-g)) * J_R[i] * PlusJac_i^T   (3×4 ambient)
//     d(r)/d(pos_CP_i)   = -R^T * coeff[i]                           (3×3)
//     d(r)/d(bias)       = [-I | 0]                                   (3×6)
//
// Parameter blocks (same layout as AccelFunctor / make_accel_cost):
//   [0..N_ORI-1]           : orientation knots (4 params each, XYZW)
//   [N_ORI..N_ORI+N_POS-1] : position CPs (3 params each)
//   [N_ORI+N_POS]          : bias block (6 params: [b_a₀..b_a₂, b_g₀..b_g₂])

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <rio/trajectory.h>
#include <rio/factors/imu_accel.h>   // for GRAVITY_WORLD
#include <rio/factors/analytic/spline_jacobians.h>

namespace rio {
namespace analytic {

class AccelAnalyticFactor : public ceres::CostFunction {
public:
    AccelAnalyticFactor(const Eigen::Vector3d& z_acc,
                        double u_ori, double inv_dt_ori,
                        double u_pos, double inv_dt_pos)
        : z_acc_(z_acc),
          u_ori_(u_ori), inv_dt_ori_(inv_dt_ori),
          u_pos_(u_pos), inv_dt_pos_(inv_dt_pos)
    {
        set_num_residuals(3);
        for (int i = 0; i < N_ORI; ++i) mutable_parameter_block_sizes()->push_back(4);
        for (int i = 0; i < N_POS; ++i) mutable_parameter_block_sizes()->push_back(3);
        mutable_parameter_block_sizes()->push_back(6);  // bias
    }

    bool Evaluate(double const* const* params,
                  double* residuals,
                  double** jacobians) const override
    {
        using Mat3 = Eigen::Matrix3d;
        using Vec3 = Eigen::Vector3d;

        // ---- Level-2: spline evaluations (double arithmetic) -----------------
        JacobianStruct<N_ORI> J_R;
        Sophus::SO3d R = rotation_with_jacobian_manual<N_ORI>(
            params, u_ori_, inv_dt_ori_, jacobians ? &J_R : nullptr);

        // Position spline 2nd derivative (world acceleration) and its coefficients
        // a_world = sum_i pos_coeff[i] * pos_CP[i]
        // coeff = inv_dt² * blending_matrix_ * baseCoeffsWithTime<2>(u_pos)
        using Helper = CeresSplineHelper<N_POS>;
        using VecNP  = Eigen::Matrix<double, N_POS, 1>;
        VecNP p2;
        Helper::template baseCoeffsWithTime<2>(p2, u_pos_);
        VecNP pos_coeff = (inv_dt_pos_ * inv_dt_pos_) *
                          Helper::blending_matrix_ * p2;

        Vec3 a_world = Vec3::Zero();
        for (int i = 0; i < N_POS; ++i) {
            Eigen::Map<const Vec3> cp(params[N_ORI + i]);
            a_world += pos_coeff[i] * cp;
        }

        // ---- Level-1: residual -----------------------------------------------
        const Vec3 specific_force = R.inverse() * (a_world - GRAVITY_WORLD);
        Eigen::Map<const Vec3> b_a(params[N_ORI + N_POS]);
        Eigen::Map<Vec3>(residuals).noalias() = z_acc_ - specific_force - b_a;

        if (!jacobians) return true;

        // ---- Jacobians -------------------------------------------------------
        const Mat3 Rt = R.inverse().matrix();   // R^T  (body-to-world inverse)

        // d(r)/d(R_tangent) = -skew(R^T(a-g)) = skew of specific_force (negated)
        // skew(v) = [[  0, -vz,  vy],
        //            [ vz,   0, -vx],
        //            [-vy,  vx,   0]]
        // d(r)/d(delta)|_{delta=0} = -d(R^T(a-g))/d(delta) = -[sf]^×
        // where sf = R^T(a_world - g) = specific_force, and
        // d(R^T v)/d(delta) = +[R^T v]^× (skew of rotated vector, right perturbation)
        const Vec3 sf = specific_force;
        Mat3 d_r_d_R_tang;
        d_r_d_R_tang <<  0.0,    sf[2], -sf[1],
                        -sf[2],   0.0,   sf[0],
                         sf[1], -sf[0],  0.0;

        // Orientation knot Jacobians.
        // basalt's evaluate(t,J) uses LEFT-output convention: R_new = exp(hat(J_k ε)) * R.
        // This introduces an extra Rt = R^T factor in the chain:
        //   d(r)/d(ε_L_k) = d_r_d_R_tang * Rt * J_R.d_val_d_knot[k]
        //                  = -[sf]^× * R^T * J_k
        // velocityBody uses LEFT-knot/vector convention without the extra Rt factor.
        const Mat3 d_r_dR_full = d_r_d_R_tang * Rt;  // -[sf]^× * R^T  (3×3)

        for (int i = 0; i < N_ORI; ++i) {
            if (!jacobians[i]) continue;
            const Mat3 J_local = d_r_dR_full * J_R.d_val_d_knot[i];
            const Eigen::Matrix<double, 3, 4> plus_jac_T = tangent_to_ambient(params[i]);
            Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> Jq(jacobians[i]);
            Jq.noalias() = J_local * plus_jac_T;
        }

        // Position CP Jacobians: d(r)/d(pos_CP_i) = -R^T * pos_coeff[i]  (3×3)
        for (int i = 0; i < N_POS; ++i) {
            if (!jacobians[N_ORI + i]) continue;
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jp(jacobians[N_ORI + i]);
            Jp.noalias() = -pos_coeff[i] * Rt;
        }

        // Bias Jacobian: d(r)/d([b_a, b_g]) = [-I | 0]
        if (jacobians[N_ORI + N_POS]) {
            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jb(jacobians[N_ORI + N_POS]);
            Jb.leftCols<3>()  = -Eigen::Matrix3d::Identity();
            Jb.rightCols<3>().setZero();
        }

        return true;
    }

private:
    Eigen::Vector3d z_acc_;
    double u_ori_, inv_dt_ori_, u_pos_, inv_dt_pos_;
};

}  // namespace analytic
}  // namespace rio
