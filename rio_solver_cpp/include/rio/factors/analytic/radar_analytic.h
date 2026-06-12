#pragma once

// Analytic Jacobian implementations of RadarDopplerFunctor and
// RadarDopplerWithPitchFunctor.
//
// Sensor-model Jacobians (∂r/∂v_world, ∂r/∂R_bw, ∂r/∂omega) are provided by
// sym::RadarSensorJacWithJacobians012 — a SymForce-generated, dependency-free
// Eigen template.  Spline Jacobians come from spline_jacobians.h.
//
// Convention notes:
//   SymForce emits RIGHT-perturbation Jacobians for Rot3 arguments.
//   basalt/spline_jacobians.h uses LEFT-perturbation convention.
//   Conversion:  J_left = J_right * R^T   (applied before chaining to knots)
//
// Parameter block layout (identical to functor variants):
//   [0..N_ORI-1]           : orientation knots (4 params each, XYZW)
//   [N_ORI..N_ORI+N_POS-1] : position CPs      (3 params each)
//   [N_ORI+N_POS]          : bias block         (6 params)
//   (WithPitch only) [N_ORI+N_POS+1] : pitch_delta (1 param)

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <sophus/so3.hpp>

#include <rio/trajectory.h>
#include <rio/factors/analytic/spline_jacobians.h>
#include <rio/factors/analytic/radar_sensor_jac_gen.h>

namespace rio {
namespace analytic {

// ============================================================================
// RadarAnalyticFactor  (fixed extrinsics)
// ============================================================================
class RadarAnalyticFactor : public ceres::CostFunction {
public:
    RadarAnalyticFactor(const Eigen::Vector3d& u_sensor,
                        double v_meas,
                        double u_ori,  double inv_dt_ori,
                        double u_pos,  double inv_dt_pos,
                        const Sophus::SO3d& R_radar_to_body,
                        const Eigen::Vector3d& t_body_sensor)
        : v_meas_(v_meas),
          u_ori_(u_ori), inv_dt_ori_(inv_dt_ori),
          u_pos_(u_pos), inv_dt_pos_(inv_dt_pos),
          t_body_sensor_(t_body_sensor)
    {
        u_body_ = R_radar_to_body.matrix() * u_sensor;
        set_num_residuals(1);
        for (int i = 0; i < N_ORI; ++i) mutable_parameter_block_sizes()->push_back(4);
        for (int i = 0; i < N_POS; ++i) mutable_parameter_block_sizes()->push_back(3);
        mutable_parameter_block_sizes()->push_back(6);  // bias (no dependence)
    }

    bool Evaluate(double const* const* params,
                  double* residuals,
                  double** jacobians) const override
    {
        using Vec3 = Eigen::Vector3d;

        const bool jacs = (jacobians != nullptr);

        // Spline evaluations
        JacobianStruct<N_ORI> J_omega_s, J_R_s;
        const Vec3 omega = body_velocity_with_jacobian_manual<N_ORI>(
            params, u_ori_, inv_dt_ori_, jacs ? &J_omega_s : nullptr);
        const Sophus::SO3d R = rotation_with_jacobian_manual<N_ORI>(
            params, u_ori_, inv_dt_ori_, jacs ? &J_R_s : nullptr);

        // Position first-derivative: v_world = Σ v_coeff[i] * pos_CP[i]
        using Helper = CeresSplineHelper<N_POS>;
        using VecNP  = Eigen::Matrix<double, N_POS, 1>;
        VecNP p1;
        Helper::template baseCoeffsWithTime<1>(p1, u_pos_);
        const VecNP v_coeff = inv_dt_pos_ * Helper::blending_matrix_ * p1;
        Vec3 v_world = Vec3::Zero();
        for (int i = 0; i < N_POS; ++i)
            v_world += v_coeff[i] * Eigen::Map<const Vec3>(params[N_ORI + i]);

        // Sensor-model via SymForce-generated function
        const auto quat = R.unit_quaternion();
        const sym::Rot3d sym_R(Eigen::Vector4d(quat.x(), quat.y(), quat.z(), quat.w()));

        Eigen::Matrix<double, 1, 3> J_v, J_R_right, J_omega_sf;
        residuals[0] = sym::RadarSensorJacWithJacobians012(
            v_world, sym_R, omega, u_body_, t_body_sensor_, v_meas_, 1e-10,
            jacs ? &J_v        : nullptr,
            jacs ? &J_R_right  : nullptr,
            jacs ? &J_omega_sf : nullptr)[0];

        if (!jacs) return true;

        // RIGHT → LEFT for R Jacobian: J_left = J_right * R^T
        const Eigen::Matrix<double, 1, 3> J_R_left = J_R_right * R.inverse().matrix();

        // Orientation knot Jacobians (1×4 ambient each)
        for (int i = 0; i < N_ORI; ++i) {
            if (!jacobians[i]) continue;
            const Eigen::Matrix<double, 1, 3> J_local =
                J_R_left * J_R_s.d_val_d_knot[i]
              + J_omega_sf * J_omega_s.d_val_d_knot[i];
            Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>>(jacobians[i]).noalias()
                = J_local * tangent_to_ambient(params[i]);
        }

        // Position CP Jacobians (1×3 each)
        for (int i = 0; i < N_POS; ++i) {
            if (!jacobians[N_ORI + i]) continue;
            Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>>(jacobians[N_ORI + i]).noalias()
                = v_coeff[i] * J_v;
        }

        // Bias Jacobian — no dependence
        if (jacobians[N_ORI + N_POS])
            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>>(jacobians[N_ORI + N_POS]).setZero();

        return true;
    }

private:
    Eigen::Vector3d u_body_;       // pre-rotated bearing (constant per measurement)
    double v_meas_;
    double u_ori_, inv_dt_ori_, u_pos_, inv_dt_pos_;
    Eigen::Vector3d t_body_sensor_;
};


// ============================================================================
// RadarAnalyticWithPitchFactor  (optimises pitch_delta extrinsic)
// ============================================================================
class RadarAnalyticWithPitchFactor : public ceres::CostFunction {
public:
    RadarAnalyticWithPitchFactor(const Eigen::Vector3d& u_sensor,
                                  double v_meas,
                                  double u_ori,  double inv_dt_ori,
                                  double u_pos,  double inv_dt_pos,
                                  const Sophus::SO3d& R_radar_to_body,
                                  const Eigen::Vector3d& t_body_sensor)
        : u_sensor_(u_sensor),
          v_meas_(v_meas),
          u_ori_(u_ori), inv_dt_ori_(inv_dt_ori),
          u_pos_(u_pos), inv_dt_pos_(inv_dt_pos),
          t_body_sensor_(t_body_sensor),
          R_radar_to_body_(R_radar_to_body)
    {
        set_num_residuals(1);
        for (int i = 0; i < N_ORI; ++i) mutable_parameter_block_sizes()->push_back(4);
        for (int i = 0; i < N_POS; ++i) mutable_parameter_block_sizes()->push_back(3);
        mutable_parameter_block_sizes()->push_back(6);  // bias (no dependence)
        mutable_parameter_block_sizes()->push_back(1);  // pitch_delta
    }

    bool Evaluate(double const* const* params,
                  double* residuals,
                  double** jacobians) const override
    {
        using Vec3 = Eigen::Vector3d;

        const bool jacs = (jacobians != nullptr);

        // Spline evaluations
        JacobianStruct<N_ORI> J_omega_s, J_R_s;
        const Vec3 omega = body_velocity_with_jacobian_manual<N_ORI>(
            params, u_ori_, inv_dt_ori_, jacs ? &J_omega_s : nullptr);
        const Sophus::SO3d R = rotation_with_jacobian_manual<N_ORI>(
            params, u_ori_, inv_dt_ori_, jacs ? &J_R_s : nullptr);

        // Position velocity
        using Helper = CeresSplineHelper<N_POS>;
        using VecNP  = Eigen::Matrix<double, N_POS, 1>;
        VecNP p1;
        Helper::template baseCoeffsWithTime<1>(p1, u_pos_);
        const VecNP v_coeff = inv_dt_pos_ * Helper::blending_matrix_ * p1;
        Vec3 v_world = Vec3::Zero();
        for (int i = 0; i < N_POS; ++i)
            v_world += v_coeff[i] * Eigen::Map<const Vec3>(params[N_ORI + i]);

        // Pitch-perturbed bearing: u_body = R_nominal * Ry(pd) * u_sensor
        const double pd = params[N_ORI + N_POS + 1][0];
        Eigen::Matrix3d Ry;
        Ry <<  std::cos(pd), 0.0, std::sin(pd),
               0.0,          1.0, 0.0,
              -std::sin(pd), 0.0, std::cos(pd);
        const Vec3 u_body = R_radar_to_body_.matrix() * Ry * u_sensor_;

        // Sensor-model via SymForce-generated function
        const auto quat = R.unit_quaternion();
        const sym::Rot3d sym_R(Eigen::Vector4d(quat.x(), quat.y(), quat.z(), quat.w()));

        Eigen::Matrix<double, 1, 3> J_v, J_R_right, J_omega_sf;
        residuals[0] = sym::RadarSensorJacWithJacobians012(
            v_world, sym_R, omega, u_body, t_body_sensor_, v_meas_, 1e-10,
            jacs ? &J_v        : nullptr,
            jacs ? &J_R_right  : nullptr,
            jacs ? &J_omega_sf : nullptr)[0];

        if (!jacs) return true;

        // RIGHT → LEFT for R Jacobian
        const Eigen::Matrix<double, 1, 3> J_R_left = J_R_right * R.inverse().matrix();

        // Orientation knot Jacobians (1×4 ambient each)
        for (int i = 0; i < N_ORI; ++i) {
            if (!jacobians[i]) continue;
            const Eigen::Matrix<double, 1, 3> J_local =
                J_R_left * J_R_s.d_val_d_knot[i]
              + J_omega_sf * J_omega_s.d_val_d_knot[i];
            Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>>(jacobians[i]).noalias()
                = J_local * tangent_to_ambient(params[i]);
        }

        // Position CP Jacobians (1×3 each)
        for (int i = 0; i < N_POS; ++i) {
            if (!jacobians[N_ORI + i]) continue;
            Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>>(jacobians[N_ORI + i]).noalias()
                = v_coeff[i] * J_v;
        }

        // Bias Jacobian — no dependence
        if (jacobians[N_ORI + N_POS])
            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>>(jacobians[N_ORI + N_POS]).setZero();

        // Pitch Jacobian: ∂r/∂pd = v_ant · (∂u_body/∂pd)
        // r = v_meas + u_body · v_ant, so ∂r/∂pd = v_ant · (R_rb * dRy/dpd * u_sensor)
        if (jacobians[N_ORI + N_POS + 1]) {
            Eigen::Matrix3d dRy;
            dRy << -std::sin(pd), 0.0, std::cos(pd),
                    0.0,          0.0, 0.0,
                   -std::cos(pd), 0.0, -std::sin(pd);
            const Vec3 d_u_dpd = R_radar_to_body_.matrix() * dRy * u_sensor_;
            const Vec3 v_ant = R.inverse() * v_world + omega.cross(t_body_sensor_);
            jacobians[N_ORI + N_POS + 1][0] = v_ant.dot(d_u_dpd);
        }

        return true;
    }

private:
    Eigen::Vector3d u_sensor_;
    double v_meas_;
    double u_ori_, inv_dt_ori_, u_pos_, inv_dt_pos_;
    Eigen::Vector3d t_body_sensor_;
    Sophus::SO3d R_radar_to_body_;
};


// ============================================================================
// RadarPosOnlyAnalyticFactor  (asymmetric ω-gate split: orientation frozen)
// ============================================================================
// Same Doppler model as RadarAnalyticFactor but with R(t), ω(t) FROZEN at
// their warm-start values (evaluated at problem-build time).  The residual is
// then LINEAR in the position CPs: the radar's velocity information flows
// into position without dragging the orientation knots — the complementary
// half of the ω soft gate (radar_pos_split): the full factor carries weight
// w = 1/(1+(|ω|/ω₀)²), this factor carries (1−w)·radar_pos_split.
// Accuracy caveat: the velocity projection is only as good as the warm-start
// orientation at that time (~5–7° mid-flip ⇒ ~0.1·|v| systematic), which is
// well below the backflips radar noise core (2.47 m/s).
// Parameter blocks: [0..N_POS-1] position CPs (3 each).
class RadarPosOnlyAnalyticFactor : public ceres::CostFunction {
public:
    RadarPosOnlyAnalyticFactor(const Eigen::Vector3d& u_sensor,
                               double v_meas,
                               double u_pos, double inv_dt_pos,
                               const Sophus::SO3d& R_ws,
                               const Eigen::Vector3d& omega_ws,
                               const Sophus::SO3d& R_radar_to_body,
                               const Eigen::Vector3d& t_body_sensor)
        : v_meas_(v_meas), u_pos_(u_pos), inv_dt_pos_(inv_dt_pos),
          R_ws_(R_ws), omega_ws_(omega_ws), t_body_sensor_(t_body_sensor)
    {
        u_body_ = R_radar_to_body.matrix() * u_sensor;
        set_num_residuals(1);
        for (int i = 0; i < N_POS; ++i) mutable_parameter_block_sizes()->push_back(3);
    }

    bool Evaluate(double const* const* params,
                  double* residuals,
                  double** jacobians) const override
    {
        using Vec3 = Eigen::Vector3d;
        using Helper = CeresSplineHelper<N_POS>;
        using VecNP  = Eigen::Matrix<double, N_POS, 1>;

        VecNP p1;
        Helper::template baseCoeffsWithTime<1>(p1, u_pos_);
        const VecNP v_coeff = inv_dt_pos_ * Helper::blending_matrix_ * p1;
        Vec3 v_world = Vec3::Zero();
        for (int i = 0; i < N_POS; ++i)
            v_world += v_coeff[i] * Eigen::Map<const Vec3>(params[i]);

        const auto quat = R_ws_.unit_quaternion();
        const sym::Rot3d sym_R(Eigen::Vector4d(quat.x(), quat.y(), quat.z(), quat.w()));
        Eigen::Matrix<double, 1, 3> J_v;
        Eigen::Matrix<double, 1, 3>* const nullj = nullptr;
        residuals[0] = sym::RadarSensorJacWithJacobians012(
            v_world, sym_R, omega_ws_, u_body_, t_body_sensor_, v_meas_, 1e-10,
            jacobians ? &J_v : nullj, nullj, nullj)[0];

        if (!jacobians) return true;
        for (int i = 0; i < N_POS; ++i) {
            if (!jacobians[i]) continue;
            Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>>(jacobians[i]).noalias()
                = v_coeff[i] * J_v;
        }
        return true;
    }

private:
    Eigen::Vector3d u_body_;       // pre-rotated bearing (constant per measurement)
    double v_meas_, u_pos_, inv_dt_pos_;
    Sophus::SO3d R_ws_;            // warm-start rotation at measurement time (frozen)
    Eigen::Vector3d omega_ws_;     // warm-start body rate (frozen; lever-arm term)
    Eigen::Vector3d t_body_sensor_;
};

}  // namespace analytic
}  // namespace rio
