// Phase 1 wrapper test: gtsam::NoiseModelFactor wrappers vs gtsam's own
// numerical Jacobian (linearizeNumerically, which differentiates through GTSAM's
// retract).  Validates the full wrapper + convention end-to-end inside GTSAM.

#include <cstdio>
#include <random>
#include <Eigen/Dense>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Rot3.h>

#include <rio/gtsam/factors.h>

using gtsam::Rot3;
using gtsam::Values;
using gtsam::Symbol;
using gtsam::Matrix;
using Eigen::Vector3d;

static int g_fail = 0;

static Rot3 rand_rot(std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, 0.3);
    return Rot3::Expmap(Vector3d(nd(rng), nd(rng), nd(rng)));
}

static void check(const char* name, const gtsam::NoiseModelFactor& f, const Values& v) {
    auto gf = f.linearize(v);
    auto jf = boost::dynamic_pointer_cast<gtsam::JacobianFactor>(gf);
    Matrix A_a = jf->jacobian().first;
    gtsam::JacobianFactor jn = gtsam::linearizeNumerically(f, v, 1e-6);
    Matrix A_n = jn.jacobian().first;
    double err = (A_a - A_n).cwiseAbs().maxCoeff();
    double scale = std::max(1.0, A_a.cwiseAbs().maxCoeff());   // min-snap J ~ inv_dt_pos^4 ~ 1e7
    double rel = err / scale;
    bool ok = (rel < 1e-5);
    std::printf("  %-32s J_err=%.3e  rel=%.3e  %s\n", name, err, rel, ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
}

int main() {
    std::printf("test_gtsam_factors: wrappers vs gtsam linearizeNumerically\n");
    std::mt19937 rng(11);
    std::normal_distribution<double> nd(0.0, 1.0);
    const int NO = rio::N_ORI, NP = rio::N_POS;
    const double u_o = 0.37, idt_o = 1.0 / 0.008, u_p = 0.62, idt_p = 1.0 / 0.005;

    // Variables
    Values v;
    gtsam::KeyVector ori, pos;
    for (int i = 0; i < NO + 2; ++i) { Symbol k('r', i); v.insert(k, rand_rot(rng)); ori.push_back(k); }
    for (int i = 0; i < NP + 2; ++i) {
        Symbol k('p', i);
        v.insert(k, gtsam::Vector3(nd(rng) * 1e-2, nd(rng) * 1e-2, nd(rng) * 1e-2));
        pos.push_back(k);
    }
    Symbol bk('b', 0);
    gtsam::Vector6 bias; for (int i = 0; i < 6; ++i) bias[i] = nd(rng) * 0.05;
    v.insert(bk, bias);

    auto N3 = gtsam::noiseModel::Isotropic::Sigma(3, 0.5);
    auto N1 = gtsam::noiseModel::Isotropic::Sigma(1, 1.0);

    // Gyro: [ori0..3, bias]
    { gtsam::KeyVector ks(ori.begin(), ori.begin() + NO); ks.push_back(bk);
      rio::gtsam_factors::GyroFactor f(N3, ks, Vector3d(nd(rng), nd(rng), nd(rng)), u_o, idt_o);
      check("gyro", f, v); }

    // Accel: [ori0..3, pos0..5, bias]
    { gtsam::KeyVector ks(ori.begin(), ori.begin() + NO);
      for (int i = 0; i < NP; ++i) ks.push_back(pos[i]); ks.push_back(bk);
      rio::gtsam_factors::AccelFactor f(N3, ks, Vector3d(nd(rng), nd(rng), nd(rng)), u_o, idt_o, u_p, idt_p);
      check("accel", f, v); }

    // Radar: [ori0..3, pos0..5]
    { gtsam::KeyVector ks(ori.begin(), ori.begin() + NO);
      for (int i = 0; i < NP; ++i) ks.push_back(pos[i]);
      Rot3 R_bs = Rot3::Expmap(Vector3d(3.14159, 0.48, 0.0));
      Vector3d u_sensor(nd(rng), nd(rng), nd(rng)); u_sensor.normalize();
      Vector3d u_body = R_bs.matrix() * u_sensor;
      rio::gtsam_factors::RadarFactor f(N1, ks, u_body, nd(rng), Vector3d(0.08, 0.02, -0.01), u_o, idt_o, u_p, idt_p);
      check("radar", f, v); }

    // MinSnap: [pos0..5]
    { gtsam::KeyVector ks(pos.begin(), pos.begin() + NP);
      rio::gtsam_factors::MinSnapFactor f(N3, ks, u_p, idt_p);
      check("minsnap", f, v); }

    // AngAccel: [ori0..2]
    { gtsam::KeyVector ks(ori.begin(), ori.begin() + 3);
      rio::gtsam_factors::AngAccelFactor f(N3, ks);
      check("angaccel", f, v); }

    std::printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
