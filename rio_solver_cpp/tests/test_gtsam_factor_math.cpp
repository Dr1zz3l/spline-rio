// Phase 1 unit test: GTSAM-tangent factor math vs (a) numerical differentiation
// using the GTSAM right-retract (q -> q*Exp(eps)) and (b) the existing Ceres
// analytic factors (residual parity).  Needs NO GTSAM (validates the math/chain).

#include <cstdio>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <sophus/so3.hpp>

#include <rio/gtsam/gyro_factor_math.h>
#include <rio/gtsam/accel_factor_math.h>
#include <rio/gtsam/radar_factor_math.h>
#include <rio/gtsam/reg_factor_math.h>
#include <rio/factors/analytic/gyro_analytic.h>
#include <rio/factors/analytic/accel_analytic.h>
#include <rio/factors/analytic/radar_analytic.h>

using Eigen::Vector3d;
using Eigen::Matrix3d;
using Sophus::SO3d;
using rio::N_ORI;
using rio::N_POS;

static int g_fail = 0;
static void check(const char* name, double err, double tol) {
    bool ok = (err < tol);
    std::printf("  %-40s max_err=%.3e  tol=%.1e  %s\n", name, err, tol, ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
}

// quaternion ([x,y,z,w]) right-retract: q * Exp(eps)
static void retract(const double* q_xyzw, const Vector3d& eps, double* out_xyzw) {
    SO3d R(Eigen::Quaterniond(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]));
    SO3d Rp = R * SO3d::exp(eps);
    Eigen::Quaterniond qp = Rp.unit_quaternion();
    out_xyzw[0] = qp.x(); out_xyzw[1] = qp.y(); out_xyzw[2] = qp.z(); out_xyzw[3] = qp.w();
}

int main() {
    std::printf("test_gtsam_factor_math: gyro residual + GTSAM-tangent Jacobians\n");
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);

    const double inv_dt = 1.0 / 0.008;
    const double u = 0.37;

    double max_knot_err = 0.0, max_bias_err = 0.0, max_res_parity = 0.0;

    for (int trial = 0; trial < 20; ++trial) {
        // random knot quaternions (moderate rotations) + bias + measurement
        double q[N_ORI][4];
        const double* qptr[N_ORI];
        for (int i = 0; i < N_ORI; ++i) {
            Vector3d w(nd(rng) * 0.3, nd(rng) * 0.3, nd(rng) * 0.3);
            Eigen::Quaterniond qq = SO3d::exp(w).unit_quaternion();
            q[i][0] = qq.x(); q[i][1] = qq.y(); q[i][2] = qq.z(); q[i][3] = qq.w();
            qptr[i] = q[i];
        }
        double bias[6];
        for (int k = 0; k < 6; ++k) bias[k] = nd(rng) * 0.05;
        Vector3d z_gyro(nd(rng), nd(rng), nd(rng));

        auto a = rio::gtsam_factors::gyro_residual_gtsam(qptr, bias, z_gyro, u, inv_dt);

        // (a) numerical Jacobian wrt each knot (right-retract)
        const double eps = 1e-6;
        for (int i = 0; i < N_ORI; ++i) {
            Matrix3d Jnum;
            for (int axis = 0; axis < 3; ++axis) {
                Vector3d d = Vector3d::Zero(); d[axis] = eps;
                double qpert[4]; retract(q[i], d, qpert);
                const double* qp[N_ORI];
                for (int k = 0; k < N_ORI; ++k) qp[k] = (k == i) ? qpert : qptr[k];
                auto ap = rio::gtsam_factors::gyro_residual_gtsam(qp, bias, z_gyro, u, inv_dt, false);
                Jnum.col(axis) = (ap.residual - a.residual) / eps;
            }
            max_knot_err = std::max(max_knot_err, (Jnum - a.d_r_d_knot[i]).cwiseAbs().maxCoeff());
        }
        // bias numerical Jacobian
        Eigen::Matrix<double, 3, 6> Jb_num;
        for (int k = 0; k < 6; ++k) {
            double bp[6]; for (int j = 0; j < 6; ++j) bp[j] = bias[j];
            bp[k] += eps;
            auto ap = rio::gtsam_factors::gyro_residual_gtsam(qptr, bp, z_gyro, u, inv_dt, false);
            Jb_num.col(k) = (ap.residual - a.residual) / eps;
        }
        max_bias_err = std::max(max_bias_err, (Jb_num - a.d_r_d_bias).cwiseAbs().maxCoeff());

        // (b) residual parity vs the Ceres GyroAnalyticFactor
        rio::analytic::GyroAnalyticFactor ceres_f(z_gyro, u, inv_dt);
        std::vector<const double*> params;
        for (int i = 0; i < N_ORI; ++i) params.push_back(qptr[i]);
        params.push_back(bias);
        Vector3d r_ceres;
        ceres_f.Evaluate(params.data(), r_ceres.data(), nullptr);
        max_res_parity = std::max(max_res_parity, (r_ceres - a.residual).cwiseAbs().maxCoeff());
    }

    check("gyro d_r/d_knot vs numerical", max_knot_err, 1e-5);
    check("gyro d_r/d_bias vs numerical", max_bias_err, 1e-6);  // [0|-I] exact; FD roundoff
    check("gyro residual vs Ceres factor", max_res_parity, 1e-12);

    // ---------- ACCEL ----------
    const double inv_dt_pos = 1.0 / 0.005;
    const double u_pos = 0.62;
    double a_knot_err = 0.0, a_cp_err = 0.0, a_bias_err = 0.0, a_res_parity = 0.0;
    for (int trial = 0; trial < 20; ++trial) {
        double q[N_ORI][4]; const double* qptr[N_ORI];
        for (int i = 0; i < N_ORI; ++i) {
            Vector3d w(nd(rng) * 0.3, nd(rng) * 0.3, nd(rng) * 0.3);
            Eigen::Quaterniond qq = SO3d::exp(w).unit_quaternion();
            q[i][0] = qq.x(); q[i][1] = qq.y(); q[i][2] = qq.z(); q[i][3] = qq.w(); qptr[i] = q[i];
        }
        // CPs scaled so a_world (= inv_dt_pos^2 * coeff * cp) is physical (~O(10)).
        double cp[N_POS][3]; const double* cptr[N_POS];
        for (int i = 0; i < N_POS; ++i) {
            for (int k = 0; k < 3; ++k) cp[i][k] = nd(rng) * 1e-3;
            cptr[i] = cp[i];
        }
        double bias[6]; for (int k = 0; k < 6; ++k) bias[k] = nd(rng) * 0.05;
        Vector3d z_acc(nd(rng), nd(rng), nd(rng));

        auto a = rio::gtsam_factors::accel_residual_gtsam(
            qptr, cptr, bias, z_acc, u, inv_dt, u_pos, inv_dt_pos);
        const double eps = 1e-6;
        auto eval = [&](const double* const* qp, const double* const* cpq, const double* bp) {
            return rio::gtsam_factors::accel_residual_gtsam(
                qp, cpq, bp, z_acc, u, inv_dt, u_pos, inv_dt_pos, false).residual;
        };
        // knot Jacobians (right-retract, CENTRAL difference)
        for (int i = 0; i < N_ORI; ++i) {
            Matrix3d Jnum;
            for (int axis = 0; axis < 3; ++axis) {
                Vector3d d = Vector3d::Zero(); d[axis] = eps;
                double qpp[4], qpm[4]; retract(q[i], d, qpp); retract(q[i], -d, qpm);
                const double *qp_p[N_ORI], *qp_m[N_ORI];
                for (int k = 0; k < N_ORI; ++k) { qp_p[k] = (k == i) ? qpp : qptr[k]; qp_m[k] = (k == i) ? qpm : qptr[k]; }
                Jnum.col(axis) = (eval(qp_p, cptr, bias) - eval(qp_m, cptr, bias)) / (2 * eps);
            }
            a_knot_err = std::max(a_knot_err, (Jnum - a.d_r_d_knot[i]).cwiseAbs().maxCoeff());
        }
        // CP Jacobians (Euclidean, central)
        for (int i = 0; i < N_POS; ++i) {
            Matrix3d Jnum;
            for (int axis = 0; axis < 3; ++axis) {
                double cpp[3], cpm[3];
                for (int k = 0; k < 3; ++k) { cpp[k] = cp[i][k]; cpm[k] = cp[i][k]; }
                cpp[axis] += eps; cpm[axis] -= eps;
                const double *cq_p[N_POS], *cq_m[N_POS];
                for (int k = 0; k < N_POS; ++k) { cq_p[k] = (k == i) ? cpp : cptr[k]; cq_m[k] = (k == i) ? cpm : cptr[k]; }
                Jnum.col(axis) = (eval(qptr, cq_p, bias) - eval(qptr, cq_m, bias)) / (2 * eps);
            }
            a_cp_err = std::max(a_cp_err, (Jnum - a.d_r_d_cp[i]).cwiseAbs().maxCoeff());
        }
        // bias (central)
        Eigen::Matrix<double, 3, 6> Jb;
        for (int k = 0; k < 6; ++k) {
            double bp[6], bm[6];
            for (int j = 0; j < 6; ++j) { bp[j] = bias[j]; bm[j] = bias[j]; }
            bp[k] += eps; bm[k] -= eps;
            Jb.col(k) = (eval(qptr, cptr, bp) - eval(qptr, cptr, bm)) / (2 * eps);
        }
        a_bias_err = std::max(a_bias_err, (Jb - a.d_r_d_bias).cwiseAbs().maxCoeff());
        // residual parity vs Ceres
        rio::analytic::AccelAnalyticFactor cf(z_acc, u, inv_dt, u_pos, inv_dt_pos);
        std::vector<const double*> params;
        for (int i = 0; i < N_ORI; ++i) params.push_back(qptr[i]);
        for (int i = 0; i < N_POS; ++i) params.push_back(cptr[i]);
        params.push_back(bias);
        Vector3d rc; cf.Evaluate(params.data(), rc.data(), nullptr);
        a_res_parity = std::max(a_res_parity, (rc - a.residual).cwiseAbs().maxCoeff());
    }
    check("accel d_r/d_knot vs numerical", a_knot_err, 1e-4);
    check("accel d_r/d_cp vs numerical", a_cp_err, 1e-4);
    check("accel d_r/d_bias vs numerical", a_bias_err, 1e-6);
    check("accel residual vs Ceres factor", a_res_parity, 1e-12);

    // ---------- RADAR ----------
    const SO3d R_bs = SO3d::exp(Vector3d(3.14159, 0.48, 0.0));   // ~ extrinsic
    const Vector3d t_bs(0.08, 0.02, -0.01);
    double r_knot_err = 0.0, r_cp_err = 0.0, r_res_parity = 0.0;
    for (int trial = 0; trial < 20; ++trial) {
        double q[N_ORI][4]; const double* qptr[N_ORI];
        for (int i = 0; i < N_ORI; ++i) {
            Vector3d w(nd(rng) * 0.3, nd(rng) * 0.3, nd(rng) * 0.3);
            Eigen::Quaterniond qq = SO3d::exp(w).unit_quaternion();
            q[i][0] = qq.x(); q[i][1] = qq.y(); q[i][2] = qq.z(); q[i][3] = qq.w(); qptr[i] = q[i];
        }
        double cp[N_POS][3]; const double* cptr[N_POS];   // scaled so v_world ~ O(1) m/s
        for (int i = 0; i < N_POS; ++i) {
            for (int k = 0; k < 3; ++k) cp[i][k] = nd(rng) * 1e-2;
            cptr[i] = cp[i];
        }
        Vector3d u_sensor(nd(rng), nd(rng), nd(rng)); u_sensor.normalize();
        const Vector3d u_body = R_bs.matrix() * u_sensor;
        double v_meas = nd(rng);

        auto a = rio::gtsam_factors::radar_residual_gtsam(
            qptr, cptr, u_body, v_meas, t_bs, u, inv_dt, u_pos, inv_dt_pos);
        const double eps = 1e-6;
        auto reval = [&](const double* const* qp, const double* const* cpq) {
            return rio::gtsam_factors::radar_residual_gtsam(
                qp, cpq, u_body, v_meas, t_bs, u, inv_dt, u_pos, inv_dt_pos, false).residual;
        };
        for (int i = 0; i < N_ORI; ++i) {
            Eigen::Matrix<double, 1, 3> Jnum;
            for (int axis = 0; axis < 3; ++axis) {
                Vector3d d = Vector3d::Zero(); d[axis] = eps;
                double qpp[4], qpm[4]; retract(q[i], d, qpp); retract(q[i], -d, qpm);
                const double *qp[N_ORI], *qm[N_ORI];
                for (int k = 0; k < N_ORI; ++k) { qp[k] = (k == i) ? qpp : qptr[k]; qm[k] = (k == i) ? qpm : qptr[k]; }
                Jnum(0, axis) = (reval(qp, cptr) - reval(qm, cptr)) / (2 * eps);
            }
            r_knot_err = std::max(r_knot_err, (Jnum - a.d_r_d_knot[i]).cwiseAbs().maxCoeff());
        }
        for (int i = 0; i < N_POS; ++i) {
            Eigen::Matrix<double, 1, 3> Jnum;
            for (int axis = 0; axis < 3; ++axis) {
                double cpp[3], cpm[3];
                for (int k = 0; k < 3; ++k) { cpp[k] = cp[i][k]; cpm[k] = cp[i][k]; }
                cpp[axis] += eps; cpm[axis] -= eps;
                const double *cp_p[N_POS], *cp_m[N_POS];
                for (int k = 0; k < N_POS; ++k) { cp_p[k] = (k == i) ? cpp : cptr[k]; cp_m[k] = (k == i) ? cpm : cptr[k]; }
                Jnum(0, axis) = (reval(qptr, cp_p) - reval(qptr, cp_m)) / (2 * eps);
            }
            r_cp_err = std::max(r_cp_err, (Jnum - a.d_r_d_cp[i]).cwiseAbs().maxCoeff());
        }
        // residual parity vs Ceres RadarAnalyticFactor
        rio::analytic::RadarAnalyticFactor cf(u_sensor, v_meas, u, inv_dt, u_pos, inv_dt_pos, R_bs, t_bs);
        std::vector<const double*> params;
        for (int i = 0; i < N_ORI; ++i) params.push_back(qptr[i]);
        for (int i = 0; i < N_POS; ++i) params.push_back(cptr[i]);
        double bias[6] = {0, 0, 0, 0, 0, 0}; params.push_back(bias);
        double rc; cf.Evaluate(params.data(), &rc, nullptr);
        r_res_parity = std::max(r_res_parity, std::abs(rc - a.residual));
    }
    check("radar d_r/d_knot vs numerical", r_knot_err, 1e-4);
    check("radar d_r/d_cp vs numerical", r_cp_err, 1e-4);
    check("radar residual vs Ceres factor", r_res_parity, 1e-12);

    // ---------- REGULARIZERS ----------
    double ms_err = 0.0, aa_err = 0.0;
    for (int trial = 0; trial < 20; ++trial) {
        // min-snap: residual linear in CPs; check d_r/d_cp[i] = coeff[i] I via numerical
        double cp[N_POS][3]; const double* cptr[N_POS];
        for (int i = 0; i < N_POS; ++i) {
            for (int k = 0; k < 3; ++k) cp[i][k] = nd(rng) * 1e-2;
            cptr[i] = cp[i];
        }
        auto ms = rio::gtsam_factors::minsnap_residual_gtsam(cptr, u_pos, inv_dt_pos);
        const double eps = 1e-6;
        for (int i = 0; i < N_POS; ++i) {
            for (int axis = 0; axis < 3; ++axis) {
                double cpp[3], cpm[3];
                for (int k = 0; k < 3; ++k) { cpp[k] = cp[i][k]; cpm[k] = cp[i][k]; }
                cpp[axis] += eps; cpm[axis] -= eps;
                const double *cp_p[N_POS], *cp_m[N_POS];
                for (int k = 0; k < N_POS; ++k) { cp_p[k] = (k == i) ? cpp : cptr[k]; cp_m[k] = (k == i) ? cpm : cptr[k]; }
                Vector3d col = (rio::gtsam_factors::minsnap_residual_gtsam(cp_p, u_pos, inv_dt_pos).residual
                              - rio::gtsam_factors::minsnap_residual_gtsam(cp_m, u_pos, inv_dt_pos).residual) / (2 * eps);
                Vector3d expect = Vector3d::Zero(); expect[axis] = ms.coeff[i];  // coeff[i]*I col
                ms_err = std::max(ms_err, (col - expect).cwiseAbs().maxCoeff() / std::max(1.0, std::abs(ms.coeff[i])));
            }
        }
        // ang-accel: 3 knots, central-diff right-retract
        double q[3][4];
        for (int i = 0; i < 3; ++i) {
            Vector3d w(nd(rng) * 0.3, nd(rng) * 0.3, nd(rng) * 0.3);
            Eigen::Quaterniond qq = SO3d::exp(w).unit_quaternion();
            q[i][0] = qq.x(); q[i][1] = qq.y(); q[i][2] = qq.z(); q[i][3] = qq.w();
        }
        auto aa = rio::gtsam_factors::angaccel_residual_gtsam(q[0], q[1], q[2]);
        auto aeval = [&](const double* a0, const double* a1, const double* a2) {
            return rio::gtsam_factors::angaccel_residual_gtsam(a0, a1, a2, false).residual;
        };
        for (int i = 0; i < 3; ++i) {
            Matrix3d Jnum;
            for (int axis = 0; axis < 3; ++axis) {
                Vector3d d = Vector3d::Zero(); d[axis] = eps;
                double qpp[4], qpm[4]; retract(q[i], d, qpp); retract(q[i], -d, qpm);
                const double* a_p[3] = {q[0], q[1], q[2]};
                const double* a_m[3] = {q[0], q[1], q[2]};
                a_p[i] = qpp; a_m[i] = qpm;
                Jnum.col(axis) = (aeval(a_p[0], a_p[1], a_p[2]) - aeval(a_m[0], a_m[1], a_m[2])) / (2 * eps);
            }
            aa_err = std::max(aa_err, (Jnum - aa.d_r_d_knot[i]).cwiseAbs().maxCoeff());
        }
    }
    check("minsnap d_r/d_cp vs numerical (rel)", ms_err, 1e-5);
    check("angaccel d_r/d_knot vs numerical", aa_err, 1e-4);

    std::printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
