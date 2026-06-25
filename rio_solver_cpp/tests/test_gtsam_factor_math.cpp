// Phase 1 unit test: GTSAM-tangent factor math vs (a) numerical differentiation
// using the GTSAM right-retract (q -> q*Exp(eps)) and (b) the existing Ceres
// analytic factors (residual parity).  Needs NO GTSAM (validates the math/chain).

#include <cstdio>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <sophus/so3.hpp>

#include <rio/gtsam/gyro_factor_math.h>
#include <rio/factors/analytic/gyro_analytic.h>

using Eigen::Vector3d;
using Eigen::Matrix3d;
using Sophus::SO3d;
using rio::N_ORI;

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

    std::printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
