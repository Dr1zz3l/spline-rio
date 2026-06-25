#include <rio/isam_sliding_window_solver.h>
#include <rio/gtsam/factors.h>

#include <chrono>
#include <cmath>
#include <memory>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/ISAM2Params.h>

namespace rio {

using gtsam::Symbol;
using gtsam::Rot3;
using gtsam::Vector3;
using gtsam::Vector6;
using gtsam::Key;
using KTMap = gtsam::FixedLagSmoother::KeyTimestampMap;

namespace nm = gtsam::noiseModel;

static Key RK(int i) { return Symbol('r', i); }
static Key PK(int i) { return Symbol('p', i); }
static Key BK(int i) { return Symbol('b', i); }

static Rot3 quat_to_rot3(const std::array<double, 4>& q) {  // q = xyzw
    return Rot3(Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized());
}

IsamSolver::IsamSolver(const IsamConfig& cfg, const ExtrinsicConfig& ext)
    : cfg_(cfg), ext_(ext) {
    // R_radar_to_body, ZYX (Rz*Ry*Rx); matches solver.cpp ExtrinsicConfig (avoids
    // linking the Ceres rio_solver lib just for this one function).
    const double er = ext.roll_deg * M_PI / 180.0;
    const double ep = ext.pitch_deg * M_PI / 180.0;
    const double ey = ext.yaw_deg * M_PI / 180.0;
    Eigen::Matrix3d Rz, Ry, Rx;
    Rz << std::cos(ey), -std::sin(ey), 0, std::sin(ey), std::cos(ey), 0, 0, 0, 1;
    Ry << std::cos(ep), 0, std::sin(ep), 0, 1, 0, -std::sin(ep), 0, std::cos(ep);
    Rx << 1, 0, 0, 0, std::cos(er), -std::sin(er), 0, std::sin(er), std::cos(er);
    R_bs_ = Rz * Ry * Rx;
    t_bs_ = Eigen::Vector3d(ext.tx, ext.ty, ext.tz);
    inv_dt_ori_ = 1.0 / cfg.dt_ori;
    inv_dt_pos_ = 1.0 / cfg.dt_pos;

    auto sig = [](int d, double s) { return nm::Isotropic::Sigma(d, s); };
    n_acc_   = sig(3, 1.0 / std::sqrt(cfg.lambda_accel));
    n_gyr_   = sig(3, 1.0 / std::sqrt(cfg.lambda_gyro));
    n_snap_  = sig(3, 1.0 / std::sqrt(cfg.lambda_snap_pos));
    n_aacc_  = sig(3, 1.0 / std::sqrt(cfg.lambda_ori_accel));
    n_heading_ = (cfg.lambda_heading > 0) ? sig(1, 1.0 / std::sqrt(cfg.lambda_heading))
                                          : gtsam::SharedNoiseModel();
    n_bias_prior_ = sig(6, 1.0 / std::sqrt(cfg.lambda_bias_prior));
    n_bias_rw_    = sig(6, cfg.bias_rw_sigma);
    n_boundary_r_ = sig(3, cfg.boundary_sigma);
    n_boundary_p_ = sig(3, cfg.boundary_sigma);
    // radar: Huber(delta) around unit (sigma=1) noise (raw m/s)
    n_radar_ = nm::Robust::Create(
        nm::mEstimator::Huber::Create(cfg.huber_delta),
        nm::Isotropic::Sigma(1, 1.0));

    gtsam::ISAM2Params p;
    p.relinearizeThreshold = cfg.relinearize_threshold;
    p.relinearizeSkip = cfg.relinearize_skip;
    p.factorization = cfg.use_qr ? gtsam::ISAM2Params::QR : gtsam::ISAM2Params::CHOLESKY;
    smoother_ = std::make_unique<gtsam::IncrementalFixedLagSmoother>(cfg.lag, p);
}

void IsamSolver::initialize(
    const std::vector<std::array<double, 3>>& pos_cps,
    const std::vector<std::array<double, 4>>& ori_quats,
    const std::array<double, 6>& biases, double t_ref) {
    init_pos_ = pos_cps;
    init_ori_ = ori_quats;
    init_bias_ = biases;
    last_bias_ = biases;
    t_ref_ = t_ref;
    n_pos_ = static_cast<int>(pos_cps.size());
    n_ori_ = static_cast<int>(ori_quats.size());
}

bool IsamSolver::ori_active(double t_abs, int& k, double& u) const {
    const double t_rel = t_abs - t_ref_;
    k = static_cast<int>(t_rel / cfg_.dt_ori);
    k = std::max(N_ORI - 1, std::min(k, n_ori_ - 1));
    u = (t_rel - k * cfg_.dt_ori) / cfg_.dt_ori;
    u = std::min(std::max(u, 0.0), 1.0 - 1e-10);
    return in_domain(t_abs);
}

bool IsamSolver::pos_active(double t_abs, int& k, double& u) const {
    const double t_rel = t_abs - t_ref_;
    k = static_cast<int>(t_rel / cfg_.dt_pos);
    k = std::max(N_POS - 1, std::min(k, n_pos_ - 1));
    u = (t_rel - k * cfg_.dt_pos) / cfg_.dt_pos;
    u = std::min(std::max(u, 0.0), 1.0 - 1e-10);
    return in_domain(t_abs);
}

bool IsamSolver::in_domain(double t_abs) const {
    const double t_rel = t_abs - t_ref_;
    const bool ok_o = (N_ORI - 1) * cfg_.dt_ori <= t_rel && t_rel <= (n_ori_ - 1) * cfg_.dt_ori;
    const bool ok_p = (N_POS - 1) * cfg_.dt_pos <= t_rel && t_rel <= (n_pos_ - 1) * cfg_.dt_pos;
    return ok_o && ok_p;
}

double IsamSolver::update(
    const std::vector<RadarFrame>& radar,
    const std::vector<ImuSample>& imu,
    const std::vector<std::pair<double, double>>& heading,
    double t_now) {

    gtsam::NonlinearFactorGraph g;
    gtsam::Values v;
    KTMap ts;

    // --- warm-start alignment: rigidly align entering knots to the current
    // solved boundary (else new knots enter at stale P1-P3 dead-reckoned values
    // -> a seam the limited per-update iterations cannot fix). Mirrors the Ceres
    // SW warm_start_align (methodology.tex).
    Eigen::Matrix3d align_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d align_t = Eigen::Vector3d::Zero();
    if (!first_) {
        const gtsam::Values cur = smoother_->calculateEstimate();
        for (auto it = added_ori_.rbegin(); it != added_ori_.rend(); ++it)
            if (cur.exists(RK(*it))) {
                align_R = cur.at<Rot3>(RK(*it)).matrix()
                          * quat_to_rot3(init_ori_[*it]).matrix().transpose();
                break;
            }
        for (auto it = added_pos_.rbegin(); it != added_pos_.rend(); ++it)
            if (cur.exists(PK(*it))) {
                const Vector3 pe = cur.at<Vector3>(PK(*it));
                align_t = pe - Vector3(init_pos_[*it][0], init_pos_[*it][1], init_pos_[*it][2]);
                break;
            }
    }
    const Rot3 align_R3((Eigen::Quaterniond(align_R)).normalized());

    auto ensure = [&](int ko, int kp) {
        for (int j = ko - 3; j <= ko; ++j)
            if (j >= 0 && j < n_ori_) {
                if (!added_ori_.count(j)) {
                    v.insert(RK(j), align_R3 * quat_to_rot3(init_ori_[j]));
                    added_ori_.insert(j);
                }
                ts[RK(j)] = t_now;
            }
        for (int i = kp - 5; i <= kp; ++i)
            if (i >= 0 && i < n_pos_) {
                if (!added_pos_.count(i)) {
                    v.insert(PK(i), Vector3(init_pos_[i][0] + align_t[0],
                                            init_pos_[i][1] + align_t[1],
                                            init_pos_[i][2] + align_t[2]));
                    added_pos_.insert(i);
                }
                ts[PK(i)] = t_now;
            }
    };

    // --- bias (random walk) ---
    Key bkey;
    if (first_) {
        Vector6 b0; for (int i = 0; i < 6; ++i) b0[i] = init_bias_[i];
        v.insert(BK(0), b0);
        g.add(gtsam::PriorFactor<Vector6>(BK(0), b0, n_bias_prior_));
        bkey = BK(0);
    } else {
        ++bias_k_;
        Vector6 b0; for (int i = 0; i < 6; ++i) b0[i] = last_bias_[i];
        v.insert(BK(bias_k_), b0);
        g.add(gtsam::BetweenFactor<Vector6>(BK(bias_k_ - 1), BK(bias_k_), Vector6::Zero(), n_bias_rw_));
        ts[BK(bias_k_ - 1)] = t_now;
        bkey = BK(bias_k_);
    }
    ts[bkey] = t_now;

    // --- ensure knots for all measurements first ---
    std::vector<int> imu_ko(imu.size()), imu_kp(imu.size());
    std::vector<double> imu_uo(imu.size()), imu_up(imu.size());
    for (size_t i = 0; i < imu.size(); ++i) {
        if (!in_domain(imu[i].timestamp)) { imu_ko[i] = -1; continue; }
        ori_active(imu[i].timestamp, imu_ko[i], imu_uo[i]);
        pos_active(imu[i].timestamp, imu_kp[i], imu_up[i]);
        ensure(imu_ko[i], imu_kp[i]);
    }
    for (const auto& f : radar) {
        if (f.points.empty() || !in_domain(f.timestamp)) continue;
        int ko, kp; double uo, up;
        ori_active(f.timestamp, ko, uo); pos_active(f.timestamp, kp, up);
        ensure(ko, kp);
    }
    const bool use_heading = n_heading_ && !heading.empty();
    if (use_heading)
        for (const auto& h : heading) {
            if (!in_domain(h.first)) continue;
            int ko, kp; double uo, up;
            ori_active(h.first, ko, uo); pos_active(h.first, kp, up);
            ensure(ko, kp);
        }

    // --- gauge anchor on the very first knots ---
    if (first_) {
        int c = 0;
        for (int j : added_ori_) { if (c++ >= 4) break;
            g.add(gtsam::PriorFactor<Rot3>(RK(j), quat_to_rot3(init_ori_[j]), n_boundary_r_)); }
        c = 0;
        for (int i : added_pos_) { if (c++ >= 6) break;
            g.add(gtsam::PriorFactor<Vector3>(
                PK(i), Vector3(init_pos_[i][0], init_pos_[i][1], init_pos_[i][2]), n_boundary_p_)); }
    }

    // --- sensor factors ---
    for (size_t i = 0; i < imu.size(); ++i) {
        if (imu_ko[i] < 0) continue;
        gtsam::KeyVector ko, kg;
        for (int j = 0; j < N_ORI; ++j) { ko.push_back(RK(imu_ko[i] - 3 + j)); kg.push_back(RK(imu_ko[i] - 3 + j)); }
        for (int j = 0; j < N_POS; ++j) ko.push_back(PK(imu_kp[i] - 5 + j));
        ko.push_back(bkey); kg.push_back(bkey);
        g.add(gtsam_factors::AccelFactor(n_acc_, ko,
            Eigen::Vector3d(imu[i].ax, imu[i].ay, imu[i].az), imu_uo[i], inv_dt_ori_, imu_up[i], inv_dt_pos_));
        g.add(gtsam_factors::GyroFactor(n_gyr_, kg,
            Eigen::Vector3d(imu[i].gx, imu[i].gy, imu[i].gz), imu_uo[i], inv_dt_ori_));
    }
    for (const auto& f : radar) {
        if (f.points.empty() || !in_domain(f.timestamp)) continue;
        int ko, kp; double uo, up;
        ori_active(f.timestamp, ko, uo); pos_active(f.timestamp, kp, up);
        gtsam::KeyVector keys;
        for (int j = 0; j < N_ORI; ++j) keys.push_back(RK(ko - 3 + j));
        for (int j = 0; j < N_POS; ++j) keys.push_back(PK(kp - 5 + j));
        for (const auto& pt : f.points) {
            Eigen::Vector3d xyz(pt.x, pt.y, pt.z);
            const double rng = xyz.norm();
            if (rng < cfg_.min_range) continue;
            const Eigen::Vector3d u_body = R_bs_ * (xyz / rng);
            g.add(gtsam_factors::RadarFactor(n_radar_, keys, u_body, pt.v, t_bs_, uo, inv_dt_ori_, up, inv_dt_pos_));
        }
    }
    if (use_heading)
        for (const auto& h : heading) {
            if (!in_domain(h.first)) continue;
            int ko; double uo; ori_active(h.first, ko, uo);
            gtsam::KeyVector keys;
            for (int j = 0; j < N_ORI; ++j) keys.push_back(RK(ko - 3 + j));
            g.add(gtsam_factors::HeadingFactor(n_heading_, keys, h.second, uo, inv_dt_ori_));
        }

    // --- regularizers (only when all member knots present) ---
    if (!added_pos_.empty()) {
        const int lo = *added_pos_.begin(), hi = *added_pos_.rbegin();
        for (int seg = lo; seg <= hi - (N_POS - 1); ++seg) {
            if (added_snap_.count(seg)) continue;
            bool all = true; for (int l = 0; l < N_POS; ++l) all &= bool(added_pos_.count(seg + l));
            if (!all) continue;
            gtsam::KeyVector ks; for (int l = 0; l < N_POS; ++l) ks.push_back(PK(seg + l));
            g.add(gtsam_factors::MinSnapFactor(n_snap_, ks, 0.5, inv_dt_pos_));
            added_snap_.insert(seg);
        }
    }
    if (!added_ori_.empty()) {
        const int lo = *added_ori_.begin(), hi = *added_ori_.rbegin();
        for (int i = lo; i <= hi - 2; ++i) {
            if (added_aacc_.count(i)) continue;
            if (!(added_ori_.count(i) && added_ori_.count(i + 1) && added_ori_.count(i + 2))) continue;
            gtsam::KeyVector ks{RK(i), RK(i + 1), RK(i + 2)};
            g.add(gtsam_factors::AngAccelFactor(n_aacc_, ks));
            added_aacc_.insert(i);
        }
    }

    // --- solve ---
    auto t0 = std::chrono::steady_clock::now();
    smoother_->update(g, v, ts);
    const gtsam::Values est = smoother_->calculateEstimate();
    auto t1 = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(t1 - t0).count();

    // --- record live-edge estimate + update bias ---
    for (int j : added_ori_) {
        if (est.exists(RK(j))) {
            const Eigen::Quaterniond q = est.at<Rot3>(RK(j)).toQuaternion();
            live_ori_[j] = {q.x(), q.y(), q.z(), q.w()};
        }
    }
    for (int i : added_pos_) {
        if (est.exists(PK(i))) {
            const Vector3 c = est.at<Vector3>(PK(i));
            live_pos_[i] = {c[0], c[1], c[2]};
        }
    }
    if (est.exists(bkey)) {
        const Vector6 b = est.at<Vector6>(bkey);
        for (int i = 0; i < 6; ++i) last_bias_[i] = b[i];
    }
    num_active_ = static_cast<int>(smoother_->timestamps().size());
    first_ = false;
    return dt;
}

std::vector<std::pair<int, std::array<double, 4>>> IsamSolver::ori_knots() const {
    std::vector<std::pair<int, std::array<double, 4>>> out;
    for (const auto& kv : live_ori_) out.emplace_back(kv.first, kv.second);
    return out;
}
std::vector<std::pair<int, std::array<double, 3>>> IsamSolver::pos_cps() const {
    std::vector<std::pair<int, std::array<double, 3>>> out;
    for (const auto& kv : live_pos_) out.emplace_back(kv.first, kv.second);
    return out;
}

}  // namespace rio
