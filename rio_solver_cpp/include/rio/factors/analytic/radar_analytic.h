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
                        const Eigen::Vector3d& t_body_sensor,
                        bool with_zbias = false)
        : v_meas_(v_meas),
          u_ori_(u_ori), inv_dt_ori_(inv_dt_ori),
          u_pos_(u_pos), inv_dt_pos_(inv_dt_pos),
          t_body_sensor_(t_body_sensor),
          with_zbias_(with_zbias)
    {
        u_body_ = R_radar_to_body.matrix() * u_sensor;
        set_num_residuals(1);
        for (int i = 0; i < N_ORI; ++i) mutable_parameter_block_sizes()->push_back(4);
        for (int i = 0; i < N_POS; ++i) mutable_parameter_block_sizes()->push_back(3);
        mutable_parameter_block_sizes()->push_back(6);  // bias (no dependence)
        // Optional trailing radar z-bias state (1 param): r -= b_z * u_z(body)
        if (with_zbias_) mutable_parameter_block_sizes()->push_back(1);
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

        // Radar z-bias state: measurement model v_meas = v_true + b_z u_z,
        // so the residual r = v_meas - v_pred gains -b_z * u_z(body).
        if (with_zbias_)
            residuals[0] -= params[N_ORI + N_POS + 1][0] * u_body_.z();

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

        // z-bias Jacobian: d r / d b_z = -u_z(body), constant per point
        if (with_zbias_ && jacobians[N_ORI + N_POS + 1])
            jacobians[N_ORI + N_POS + 1][0] = -u_body_.z();

        return true;
    }

private:
    Eigen::Vector3d u_body_;       // pre-rotated bearing (constant per measurement)
    double v_meas_;
    double u_ori_, inv_dt_ori_, u_pos_, inv_dt_pos_;
    Eigen::Vector3d t_body_sensor_;
    bool with_zbias_{false};
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

// ============================================================================
// RadarFrameWhitenedFactor  (per-frame correlated noise, stochastic model)
// ============================================================================
// Stacks all n returns of one radar frame into a single n-dim residual block
// and whitens a per-frame correlated noise model.  S_w = diag(sqrt(w_i)) are the
// per-point weights (intensity x alias x Huber IRLS), applied BEFORE whitening.
//
//   radar_frame_hetero = 0 (default):  Σ = I + σ_c²·11ᵀ
//     r' = W (S_w r),  W = I − (γ/n)·11ᵀ,  γ = 1 − 1/sqrt(1 + n σ_c²)
//     EXACT only when the w_i are equal.  With unequal weights the implied
//     shared covariance is σ_c²/sqrt(w_i w_j) rather than σ_c², i.e. the
//     frame-shared term is inflated for down-weighted returns.
//   radar_frame_hetero = 1:  Σ' = I + σ_c²·v vᵀ,  v_i = sqrt(w_i)
//     W = I − (γ/S)·v vᵀ,  S = vᵀv = Σ w_i,  γ = 1 − 1/sqrt(1 + σ_c² S)
//     Exact for unequal weights, and identical to the above at w_i ≡ 1.
//
// WHAT σ_c IS (2026-08-08, measured; supersedes an earlier note here claiming
// hetero=1 makes it "absolute").  The rows are scaled by sqrt(w_i) above, and
// the identity coefficient below is fixed at 1, so this factor asserts
// Var(sqrt(w_i) r_i) = 1.  Writing the true covariance of the scaled stack,
//     Cov(r') = σ_0²·I + σ_c²·v vᵀ = σ_0²·[ I + (σ_c/σ_0)²·v vᵀ ],
// and noting the leading σ_0² is absorbed by the frame ScaledLoss
// (s_frame = w_omega·radar_weight), radar_frame_sigma_c is exactly the RATIO
//     σ_c / σ_0,   σ_0² = Var(sqrt(w_i) r_i)
// in BOTH branches.  hetero changes the rank-one term's DIRECTION (1 -> v),
// not its units; neither branch is absolute, and neither is a ratio to s0.
//
// Consequence: the right value depends on WHAT THE WEIGHTS NORMALISE TO, so it
// must be re-derived whenever the weight channel changes.  Measured on the
// deployed stream (characterize_shape_contradiction.py --block sigmacw, median
// over 6 prefilter seeds): deployed weights (w_int × alias) need 0.81; the
// --bearing-weight channel would need 1.24.  Deployed default is 0.4.
//
// There was also a comment here claiming the weights are "clipped [0.25,4]".
// They are not clipped anywhere; measured per-frame spreads reach 9-11x.
// All returns of a frame share one timestamp, hence identical parameter-block
// support: [N_ORI ori knots | N_POS pos CPs | bias] — the same layout as
// RadarAnalyticFactor, so window/marginalization bookkeeping is unchanged.
// ----------------------------------------------------------------------------
// FrameWhitenCore: the frame factors' whitening operator, generalized to two
// nuisance directions.  W = Sigma^{-1/2} applied to the row-scaled stack with
//     Sigma = I + sigma_c^2 v v^T + sigma_z^2 z z^T,
// v = the shared-error direction (1 or sqrt(w)), z = row-scaled elevation
// pattern (sqrt(w)_i * u_z,i): the measured frame-varying elevation
// coefficient (sigma_z ~ 0.26-0.64 m/s across bags, 2026-08-11) is a
// GEOMETRY-CORRELATED structured error; whitening it out stops any weighting
// change from re-exposing the radar's elevation bias.  Math: with U = [v z],
// D = diag(sigma_c^2, sigma_z^2), B = D^{1/2} U^T U D^{1/2} = Q L Q^T,
//     W = I + U T U^T,  T = D^{1/2} Q f(L) Q^T D^{1/2},
//     f(l) = (1/sqrt(1+l) - 1)/l  (limit -1/2 at l = 0),
// which reduces EXACTLY to the single-direction gamma/S form when sigma_z = 0
// (bit-identical code path kept for that case).
struct FrameWhitenCore {
    std::vector<double> d1, d2;
    double T11{0.0}, T12{0.0}, T22{0.0};
    double g_over_s{0.0};
    bool two_dir{false};

    void init(const std::vector<double>& sqrt_v, double sigma_c,
              const std::vector<double>& uz, double sigma_z)
    {
        const int n = static_cast<int>(sqrt_v.size());
        d1 = sqrt_v;
        double sc = sigma_c, sz = sigma_z;
        if (sc <= 0.0 && sz > 0.0 && static_cast<int>(uz.size()) == n) {
            // degenerate: only the elevation direction active
            for (int i = 0; i < n; ++i) d1[i] = sqrt_v[i] * uz[i];
            sc = sz; sz = 0.0;
        }
        if (sz > 0.0 && static_cast<int>(uz.size()) == n) {
            two_dir = true;
            d2.resize(n);
            for (int i = 0; i < n; ++i) d2[i] = sqrt_v[i] * uz[i];
            double g11 = 0.0, g12 = 0.0, g22 = 0.0;
            for (int i = 0; i < n; ++i) {
                g11 += d1[i] * d1[i];
                g12 += d1[i] * d2[i];
                g22 += d2[i] * d2[i];
            }
            const double b11 = sc * sc * g11, b12 = sc * sz * g12,
                         b22 = sz * sz * g22;
            const double tr = b11 + b22, det = b11 * b22 - b12 * b12;
            const double disc = std::sqrt(std::max(tr * tr * 0.25 - det, 0.0));
            const double l1 = tr * 0.5 + disc, l2 = tr * 0.5 - disc;
            auto f = [](double l) {
                return (l > 1e-12) ? (1.0 / std::sqrt(1.0 + l) - 1.0) / l
                                   : -0.5 + 0.375 * l;
            };
            double v1x, v1y;
            if (std::abs(b12) > 1e-14 * (std::abs(b11) + std::abs(b22) + 1e-300)) {
                v1x = b12; v1y = l1 - b11;
            } else {
                v1x = (b11 >= b22) ? 1.0 : 0.0;
                v1y = (b11 >= b22) ? 0.0 : 1.0;
            }
            double nrm = std::sqrt(v1x * v1x + v1y * v1y);
            if (nrm < 1e-300) { v1x = 1.0; v1y = 0.0; nrm = 1.0; }
            v1x /= nrm; v1y /= nrm;
            const double v2x = -v1y, v2y = v1x;
            const double f1 = f(l1), f2 = f(l2);
            const double F11 = f1 * v1x * v1x + f2 * v2x * v2x;
            const double F12 = f1 * v1x * v1y + f2 * v2x * v2y;
            const double F22 = f1 * v1y * v1y + f2 * v2y * v2y;
            T11 = sc * sc * F11; T12 = sc * sz * F12; T22 = sz * sz * F22;
        } else {
            double s = 0.0;
            for (double v : d1) s += v * v;
            if (s <= 0.0) s = static_cast<double>(n);
            const double gamma = 1.0 - 1.0 / std::sqrt(1.0 + s * sc * sc);
            g_over_s = gamma / s;
        }
    }

    inline void apply(double* x, int n, int stride) const
    {
        if (!two_dir) {
            double rs = 0.0;
            for (int i = 0; i < n; ++i) rs += d1[i] * x[i * stride];
            for (int i = 0; i < n; ++i) x[i * stride] -= g_over_s * d1[i] * rs;
        } else {
            double c1 = 0.0, c2 = 0.0;
            for (int i = 0; i < n; ++i) {
                c1 += d1[i] * x[i * stride];
                c2 += d2[i] * x[i * stride];
            }
            const double a1 = T11 * c1 + T12 * c2;
            const double a2 = T12 * c1 + T22 * c2;
            for (int i = 0; i < n; ++i)
                x[i * stride] += d1[i] * a1 + d2[i] * a2;
        }
    }
};


class RadarFrameWhitenedFactor : public ceres::CostFunction {
public:
    RadarFrameWhitenedFactor(std::vector<std::unique_ptr<RadarAnalyticFactor>> pts,
                             std::vector<double> sqrt_w,
                             double sigma_c, bool hetero = false,
                             std::vector<double> uz = {},
                             double sigma_z = 0.0,
                             bool with_zbias = false)
        : pts_(std::move(pts)), sqrt_w_(std::move(sqrt_w)), with_zbias_(with_zbias)
    {
        n_ = static_cast<int>(pts_.size());
        // Heteroscedastic frame whitening.  The per-point weights make each
        // return's own variance sigma_0^2, but the frame-SHARED error c_f is a
        // single scalar common to every return, so after the row scaling
        // r'_i = v_i r_i (v_i = sqrt(w_i)) its contribution is
        //     Cov(r') = sigma_0^2 I + sigma_c^2 v v^T
        // -- a rank-one term along v, NOT along 1.  Whitening as if it were
        // along 1 (which is what this factor did until 2026-08-06) implies a
        // shared covariance sigma_c^2 / sqrt(w_i w_j), i.e. it INFLATES the
        // modelled frame-shared error for exactly the returns a per-point law
        // down-weights.  Sherman-Morrison for the actual v:
        //     Sigma^-1/2 = I - (gamma / S) v v^T,  S = v^T v = sum_j w_j,
        //     gamma = 1 - 1/sqrt(1 + sigma_c^2 S)
        // With all w_j = 1 this reduces exactly to the previous form (S = n,
        // uniform column mean), so the deployed unweighted behaviour is
        // unchanged bit-for-bit.
        if (!hetero) sqrt_v_.assign(n_, 1.0);   // legacy: shared term along 1
        else         sqrt_v_ = sqrt_w_;
        core_.init(sqrt_v_, sigma_c, uz, sigma_z);
        set_num_residuals(n_);
        for (int i = 0; i < N_ORI; ++i) mutable_parameter_block_sizes()->push_back(4);
        for (int i = 0; i < N_POS; ++i) mutable_parameter_block_sizes()->push_back(3);
        mutable_parameter_block_sizes()->push_back(6);  // bias (no dependence)
        if (with_zbias_) mutable_parameter_block_sizes()->push_back(1);  // b_z
    }

    bool Evaluate(double const* const* params,
                  double* residuals,
                  double** jacobians) const override
    {
        constexpr int NBMAX = N_ORI + N_POS + 2;
        const int NB = N_ORI + N_POS + 1 + (with_zbias_ ? 1 : 0);
        int bsz[NBMAX];
        for (int b = 0; b < N_ORI; ++b)          bsz[b] = 4;
        for (int b = 0; b < N_POS; ++b)          bsz[N_ORI + b] = 3;
        bsz[N_ORI + N_POS] = 6;
        if (with_zbias_) bsz[N_ORI + N_POS + 1] = 1;

        // Per-point evaluation, rows written directly into the frame Jacobian
        double* jrow[NBMAX];
        for (int i = 0; i < n_; ++i) {
            if (jacobians) {
                for (int b = 0; b < NB; ++b)
                    jrow[b] = jacobians[b] ? jacobians[b] + i * bsz[b] : nullptr;
            }
            if (!pts_[i]->Evaluate(params, &residuals[i],
                                   jacobians ? jrow : nullptr))
                return false;
            const double s = sqrt_w_[i];
            if (s != 1.0) {
                residuals[i] *= s;
                if (jacobians)
                    for (int b = 0; b < NB; ++b)
                        if (jrow[b])
                            for (int c = 0; c < bsz[b]; ++c) jrow[b][c] *= s;
            }
        }

        // Whitening: W = I + U T U^T (FrameWhitenCore; single-direction path
        // is bit-identical to the old gamma/S subtraction)
        core_.apply(residuals, n_, 1);
        if (jacobians) {
            for (int b = 0; b < NB; ++b) {
                if (!jacobians[b]) continue;
                for (int c = 0; c < bsz[b]; ++c)
                    core_.apply(jacobians[b] + c, n_, bsz[b]);
            }
        }
        return true;
    }

private:
    std::vector<std::unique_ptr<RadarAnalyticFactor>> pts_;
    std::vector<double> sqrt_w_;
    std::vector<double> sqrt_v_;    // the shared term's direction: 1 or sqrt(w)
    int n_{0};
    bool with_zbias_{false};
    FrameWhitenCore core_;
};

// ============================================================================
// RadarFrameWhitenedWithPitchFactor  (frame whitening, pitch_delta variant)
// ============================================================================
// Identical whitening to RadarFrameWhitenedFactor but wrapping the WithPitch
// per-point factor; extra trailing pitch_delta block (1 param).
class RadarFrameWhitenedWithPitchFactor : public ceres::CostFunction {
public:
    RadarFrameWhitenedWithPitchFactor(
        std::vector<std::unique_ptr<RadarAnalyticWithPitchFactor>> pts,
        std::vector<double> sqrt_w,
        double sigma_c, bool hetero = false,
        std::vector<double> uz = {},
        double sigma_z = 0.0)
        : pts_(std::move(pts)), sqrt_w_(std::move(sqrt_w))
    {
        n_ = static_cast<int>(pts_.size());
        // Heteroscedastic frame whitening.  The per-point weights make each
        // return's own variance sigma_0^2, but the frame-SHARED error c_f is a
        // single scalar common to every return, so after the row scaling
        // r'_i = v_i r_i (v_i = sqrt(w_i)) its contribution is
        //     Cov(r') = sigma_0^2 I + sigma_c^2 v v^T
        // -- a rank-one term along v, NOT along 1.  Whitening as if it were
        // along 1 (which is what this factor did until 2026-08-06) implies a
        // shared covariance sigma_c^2 / sqrt(w_i w_j), i.e. it INFLATES the
        // modelled frame-shared error for exactly the returns a per-point law
        // down-weights.  Sherman-Morrison for the actual v:
        //     Sigma^-1/2 = I - (gamma / S) v v^T,  S = v^T v = sum_j w_j,
        //     gamma = 1 - 1/sqrt(1 + sigma_c^2 S)
        // With all w_j = 1 this reduces exactly to the previous form (S = n,
        // uniform column mean), so the deployed unweighted behaviour is
        // unchanged bit-for-bit.
        if (!hetero) sqrt_v_.assign(n_, 1.0);   // legacy: shared term along 1
        else         sqrt_v_ = sqrt_w_;
        core_.init(sqrt_v_, sigma_c, uz, sigma_z);
        set_num_residuals(n_);
        for (int i = 0; i < N_ORI; ++i) mutable_parameter_block_sizes()->push_back(4);
        for (int i = 0; i < N_POS; ++i) mutable_parameter_block_sizes()->push_back(3);
        mutable_parameter_block_sizes()->push_back(6);  // bias (no dependence)
        mutable_parameter_block_sizes()->push_back(1);  // pitch_delta
    }

    bool Evaluate(double const* const* params,
                  double* residuals,
                  double** jacobians) const override
    {
        constexpr int NB = N_ORI + N_POS + 2;
        int bsz[NB];
        for (int b = 0; b < N_ORI; ++b)          bsz[b] = 4;
        for (int b = 0; b < N_POS; ++b)          bsz[N_ORI + b] = 3;
        bsz[N_ORI + N_POS]     = 6;
        bsz[N_ORI + N_POS + 1] = 1;

        double* jrow[NB];
        for (int i = 0; i < n_; ++i) {
            if (jacobians) {
                for (int b = 0; b < NB; ++b)
                    jrow[b] = jacobians[b] ? jacobians[b] + i * bsz[b] : nullptr;
            }
            if (!pts_[i]->Evaluate(params, &residuals[i],
                                   jacobians ? jrow : nullptr))
                return false;
            const double s = sqrt_w_[i];
            if (s != 1.0) {
                residuals[i] *= s;
                if (jacobians)
                    for (int b = 0; b < NB; ++b)
                        if (jrow[b])
                            for (int c = 0; c < bsz[b]; ++c) jrow[b][c] *= s;
            }
        }

        core_.apply(residuals, n_, 1);
        if (jacobians) {
            for (int b = 0; b < NB; ++b) {
                if (!jacobians[b]) continue;
                for (int c = 0; c < bsz[b]; ++c)
                    core_.apply(jacobians[b] + c, n_, bsz[b]);
            }
        }
        return true;
    }

private:
    std::vector<std::unique_ptr<RadarAnalyticWithPitchFactor>> pts_;
    std::vector<double> sqrt_w_;
    std::vector<double> sqrt_v_;    // the shared term's direction: 1 or sqrt(w)
    int n_{0};
    FrameWhitenCore core_;
};

}  // namespace analytic
}  // namespace rio
