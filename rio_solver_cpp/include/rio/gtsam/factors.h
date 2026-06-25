#pragma once

// Thin gtsam::NoiseModelFactor wrappers forwarding to the verified, GTSAM-tangent
// factor math (gyro/accel/radar/reg *_factor_math.h).  Dynamic NoiseModelFactor
// (n-ary) is used instead of variadic NoiseModelFactorN to avoid 10-11 template
// args for the radar/accel factors.  GTSAM whitens; we return the raw residual +
// raw tangent Jacobians (H[i] = d residual / d tangent(key_i)).
//
// Key order conventions (positional, in this->keys()):
//   Gyro:     [ori0..ori3, bias]
//   Accel:    [ori0..ori3, pos0..pos5, bias]
//   Radar:    [ori0..ori3, pos0..pos5]            (no bias dependence)
//   MinSnap:  [pos0..pos5]
//   AngAccel: [ori0, ori1, ori2]

#include <boost/optional.hpp>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Vector.h>

#include <rio/trajectory.h>
#include <rio/gtsam/gyro_factor_math.h>
#include <rio/gtsam/accel_factor_math.h>
#include <rio/gtsam/radar_factor_math.h>
#include <rio/gtsam/reg_factor_math.h>
#include <rio/gtsam/heading_factor_math.h>
#include <rio/gtsam/radar_pos_only_factor_math.h>

namespace rio {
namespace gtsam_factors {

inline void rot3_to_xyzw(const gtsam::Rot3& R, double* out) {
    const gtsam::Quaternion q = R.toQuaternion();
    out[0] = q.x(); out[1] = q.y(); out[2] = q.z(); out[3] = q.w();
}

// ---------------------------------------------------------------- Gyro (3D)
class GyroFactor : public gtsam::NoiseModelFactor {
public:
    GyroFactor(const gtsam::SharedNoiseModel& m, const gtsam::KeyVector& keys,
               const Eigen::Vector3d& z_gyro, double u_ori, double inv_dt_ori)
        : gtsam::NoiseModelFactor(m, keys), z_(z_gyro), u_(u_ori), inv_dt_(inv_dt_ori) {}

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const override {
        double q[N_ORI][4]; const double* qp[N_ORI];
        for (int i = 0; i < N_ORI; ++i) { rot3_to_xyzw(x.at<gtsam::Rot3>(keys()[i]), q[i]); qp[i] = q[i]; }
        const gtsam::Vector6 bias = x.at<gtsam::Vector6>(keys()[N_ORI]);
        auto r = gyro_residual_gtsam(qp, bias.data(), z_, u_, inv_dt_, bool(H));
        if (H) {
            auto& Hs = *H; Hs.resize(N_ORI + 1);
            for (int i = 0; i < N_ORI; ++i) Hs[i] = r.d_r_d_knot[i];
            Hs[N_ORI] = r.d_r_d_bias;
        }
        return r.residual;
    }
private:
    Eigen::Vector3d z_; double u_, inv_dt_;
};

// ---------------------------------------------------------------- Accel (3D)
class AccelFactor : public gtsam::NoiseModelFactor {
public:
    AccelFactor(const gtsam::SharedNoiseModel& m, const gtsam::KeyVector& keys,
                const Eigen::Vector3d& z_acc,
                double u_ori, double inv_dt_ori, double u_pos, double inv_dt_pos)
        : gtsam::NoiseModelFactor(m, keys), z_(z_acc),
          u_o_(u_ori), idt_o_(inv_dt_ori), u_p_(u_pos), idt_p_(inv_dt_pos) {}

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const override {
        double q[N_ORI][4]; const double* qp[N_ORI];
        for (int i = 0; i < N_ORI; ++i) { rot3_to_xyzw(x.at<gtsam::Rot3>(keys()[i]), q[i]); qp[i] = q[i]; }
        double cp[N_POS][3]; const double* cpp[N_POS];
        for (int i = 0; i < N_POS; ++i) {
            const gtsam::Vector3 c = x.at<gtsam::Vector3>(keys()[N_ORI + i]);
            cp[i][0] = c[0]; cp[i][1] = c[1]; cp[i][2] = c[2]; cpp[i] = cp[i];
        }
        const gtsam::Vector6 bias = x.at<gtsam::Vector6>(keys()[N_ORI + N_POS]);
        auto r = accel_residual_gtsam(qp, cpp, bias.data(), z_, u_o_, idt_o_, u_p_, idt_p_, bool(H));
        if (H) {
            auto& Hs = *H; Hs.resize(N_ORI + N_POS + 1);
            for (int i = 0; i < N_ORI; ++i) Hs[i] = r.d_r_d_knot[i];
            for (int i = 0; i < N_POS; ++i) Hs[N_ORI + i] = r.d_r_d_cp[i];
            Hs[N_ORI + N_POS] = r.d_r_d_bias;
        }
        return r.residual;
    }
private:
    Eigen::Vector3d z_; double u_o_, idt_o_, u_p_, idt_p_;
};

// ---------------------------------------------------------------- Radar (1D)
class RadarFactor : public gtsam::NoiseModelFactor {
public:
    RadarFactor(const gtsam::SharedNoiseModel& m, const gtsam::KeyVector& keys,
                const Eigen::Vector3d& u_body, double v_meas, const Eigen::Vector3d& t_bs,
                double u_ori, double inv_dt_ori, double u_pos, double inv_dt_pos)
        : gtsam::NoiseModelFactor(m, keys), u_body_(u_body), v_meas_(v_meas), t_bs_(t_bs),
          u_o_(u_ori), idt_o_(inv_dt_ori), u_p_(u_pos), idt_p_(inv_dt_pos) {}

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const override {
        double q[N_ORI][4]; const double* qp[N_ORI];
        for (int i = 0; i < N_ORI; ++i) { rot3_to_xyzw(x.at<gtsam::Rot3>(keys()[i]), q[i]); qp[i] = q[i]; }
        double cp[N_POS][3]; const double* cpp[N_POS];
        for (int i = 0; i < N_POS; ++i) {
            const gtsam::Vector3 c = x.at<gtsam::Vector3>(keys()[N_ORI + i]);
            cp[i][0] = c[0]; cp[i][1] = c[1]; cp[i][2] = c[2]; cpp[i] = cp[i];
        }
        auto r = radar_residual_gtsam(qp, cpp, u_body_, v_meas_, t_bs_, u_o_, idt_o_, u_p_, idt_p_, bool(H));
        if (H) {
            auto& Hs = *H; Hs.resize(N_ORI + N_POS);
            for (int i = 0; i < N_ORI; ++i) Hs[i] = r.d_r_d_knot[i];
            for (int i = 0; i < N_POS; ++i) Hs[N_ORI + i] = r.d_r_d_cp[i];
        }
        return (gtsam::Vector(1) << r.residual).finished();
    }
private:
    Eigen::Vector3d u_body_; double v_meas_; Eigen::Vector3d t_bs_;
    double u_o_, idt_o_, u_p_, idt_p_;
};

// ---------------------------------------------------------------- MinSnap (3D)
class MinSnapFactor : public gtsam::NoiseModelFactor {
public:
    MinSnapFactor(const gtsam::SharedNoiseModel& m, const gtsam::KeyVector& keys,
                  double u_pos, double inv_dt_pos)
        : gtsam::NoiseModelFactor(m, keys), u_p_(u_pos), idt_p_(inv_dt_pos) {}

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const override {
        double cp[N_POS][3]; const double* cpp[N_POS];
        for (int i = 0; i < N_POS; ++i) {
            const gtsam::Vector3 c = x.at<gtsam::Vector3>(keys()[i]);
            cp[i][0] = c[0]; cp[i][1] = c[1]; cp[i][2] = c[2]; cpp[i] = cp[i];
        }
        auto r = minsnap_residual_gtsam(cpp, u_p_, idt_p_);
        if (H) {
            auto& Hs = *H; Hs.resize(N_POS);
            for (int i = 0; i < N_POS; ++i) Hs[i] = r.coeff[i] * Eigen::Matrix3d::Identity();
        }
        return r.residual;
    }
private:
    double u_p_, idt_p_;
};

// ---------------------------------------------------------------- AngAccel (3D)
class AngAccelFactor : public gtsam::NoiseModelFactor {
public:
    AngAccelFactor(const gtsam::SharedNoiseModel& m, const gtsam::KeyVector& keys)
        : gtsam::NoiseModelFactor(m, keys) {}

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const override {
        double q[3][4];
        for (int i = 0; i < 3; ++i) rot3_to_xyzw(x.at<gtsam::Rot3>(keys()[i]), q[i]);
        auto r = angaccel_residual_gtsam(q[0], q[1], q[2], bool(H));
        if (H) { auto& Hs = *H; Hs.resize(3); for (int i = 0; i < 3; ++i) Hs[i] = r.d_r_d_knot[i]; }
        return r.residual;
    }
};

// ------------------------------------------------ Radar position-only (1D)
class RadarPosOnlyFactor : public gtsam::NoiseModelFactor {
public:
    RadarPosOnlyFactor(const gtsam::SharedNoiseModel& m, const gtsam::KeyVector& keys,
                       const Eigen::Matrix3d& R_ws, const Eigen::Vector3d& omega_ws,
                       const Eigen::Vector3d& u_body, double v_meas, const Eigen::Vector3d& t_bs,
                       double u_pos, double inv_dt_pos)
        : gtsam::NoiseModelFactor(m, keys), R_ws_(R_ws), omega_ws_(omega_ws),
          u_body_(u_body), v_meas_(v_meas), t_bs_(t_bs), u_p_(u_pos), idt_p_(inv_dt_pos) {}

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const override {
        double cp[N_POS][3]; const double* cpp[N_POS];
        for (int i = 0; i < N_POS; ++i) {
            const gtsam::Vector3 c = x.at<gtsam::Vector3>(keys()[i]);
            cp[i][0] = c[0]; cp[i][1] = c[1]; cp[i][2] = c[2]; cpp[i] = cp[i];
        }
        auto r = radar_pos_only_residual_gtsam(cpp, R_ws_, omega_ws_, u_body_, v_meas_, t_bs_,
                                               u_p_, idt_p_, bool(H));
        if (H) { auto& Hs = *H; Hs.resize(N_POS); for (int i = 0; i < N_POS; ++i) Hs[i] = r.d_r_d_cp[i]; }
        return (gtsam::Vector(1) << r.residual).finished();
    }
private:
    Eigen::Matrix3d R_ws_; Eigen::Vector3d omega_ws_, u_body_; double v_meas_;
    Eigen::Vector3d t_bs_; double u_p_, idt_p_;
};

// ---------------------------------------------------------------- Heading (1D)
class HeadingFactor : public gtsam::NoiseModelFactor {
public:
    HeadingFactor(const gtsam::SharedNoiseModel& m, const gtsam::KeyVector& keys,
                  double yaw_ref, double u_ori, double inv_dt_ori)
        : gtsam::NoiseModelFactor(m, keys), yaw_(yaw_ref), u_(u_ori), inv_dt_(inv_dt_ori) {}

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const override {
        double q[N_ORI][4]; const double* qp[N_ORI];
        for (int i = 0; i < N_ORI; ++i) { rot3_to_xyzw(x.at<gtsam::Rot3>(keys()[i]), q[i]); qp[i] = q[i]; }
        auto r = heading_residual_gtsam(qp, yaw_, u_, inv_dt_, bool(H));
        if (H) { auto& Hs = *H; Hs.resize(N_ORI); for (int i = 0; i < N_ORI; ++i) Hs[i] = r.d_r_d_knot[i]; }
        return (gtsam::Vector(1) << r.residual).finished();
    }
private:
    double yaw_, u_, inv_dt_;
};

}  // namespace gtsam_factors
}  // namespace rio
