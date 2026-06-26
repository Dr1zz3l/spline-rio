#include <rio/isam_sliding_window_solver.h>
#include <rio/gtsam/factors.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

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
static Key FK(int i) { return Symbol('f', i); }   // floor-offset landmark (Phase 1)

static Rot3 quat_to_rot3(const std::array<double, 4>& q) {  // q = xyzw
    return Rot3(Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized());
}

static gtsam::SharedNoiseModel sf_iso3(double lam) {
    return nm::Isotropic::Sigma(3, 1.0 / std::sqrt(std::max(lam, 1e-30)));
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

    sigma_g_ = 1.0 / std::sqrt(cfg.lambda_gyro);   // NIS-adaptive starting points
    sigma_a_ = 1.0 / std::sqrt(cfg.lambda_accel);
    sigma_r_ = 1.0;

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
    if (!first_ && cfg_.warm_start_align) {
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
                    const Vector3 anchor(init_pos_[i][0] + align_t[0],
                                         init_pos_[i][1] + align_t[1],
                                         init_pos_[i][2] + align_t[2]);
                    v.insert(PK(i), anchor);
                    added_pos_.insert(i);
                    // position tether to the RAW P1-P3 init (radar-velocity-
                    // integrated, drift-resistant -- NOT the aligned value, which
                    // would follow the drift). Backflips: radar sparsity in flips.
                    if (cfg_.lambda_pos_init_prior > 0.0)
                        g.add(gtsam::PriorFactor<Vector3>(
                            PK(i), Vector3(init_pos_[i][0], init_pos_[i][1], init_pos_[i][2]),
                            sf_iso3(cfg_.lambda_pos_init_prior)));
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

    // --- floor-offset landmark f0 (Phase 1, free-offset floor) ---
    // Bootstrap once from the first stride's LOWEST return cluster (drift-free: the
    // lowest dense layer is the floor regardless of absolute z), then keep f0 alive
    // (persistent, never marginalized) by refreshing its timestamp each stride.
    if (cfg_.floor_free) {
        if (!floor_init_) {
            // Buffer candidate world-z over the first ~1.5 s (a single 0.3 s stride is
            // too short to reliably see the floor on slow flight) before committing f0.
            for (const auto& f : radar) {
                if (f.points.empty() || !in_domain(f.timestamp)) continue;
                int ko, kp; double uo, up;
                ori_active(f.timestamp, ko, uo); pos_active(f.timestamp, kp, up);
                double qk[N_ORI][4]; const double* qp[N_ORI];
                for (int j = 0; j < N_ORI; ++j) {
                    const int idx = std::max(0, std::min(ko - 3 + j, n_ori_ - 1));
                    const Eigen::Quaterniond qq = (align_R3 * quat_to_rot3(init_ori_[idx])).toQuaternion();
                    qk[j][0] = qq.x(); qk[j][1] = qq.y(); qk[j][2] = qq.z(); qk[j][3] = qq.w(); qp[j] = qk[j];
                }
                const Eigen::Matrix3d Rw =
                    analytic::rotation_with_jacobian_manual<N_ORI>(qp, uo, inv_dt_ori_, nullptr).matrix();
                using Helper = CeresSplineHelper<N_POS>;
                Eigen::Matrix<double, N_POS, 1> p0; Helper::template baseCoeffsWithTime<0>(p0, up);
                const Eigen::Matrix<double, N_POS, 1> wv = Helper::blending_matrix_ * p0;
                double pz = 0.0;
                for (int i = 0; i < N_POS; ++i) {
                    const int idx = std::max(0, std::min(kp - 5 + i, n_pos_ - 1));
                    pz += wv[i] * (init_pos_[idx][2] + align_t[2]);
                }
                for (const auto& pt : f.points) {
                    Eigen::Vector3d xyz(pt.x, pt.y, pt.z);
                    if (xyz.norm() < cfg_.min_range) continue;
                    floor_cand_.push_back((Rw * (R_bs_ * xyz + t_bs_)).z() + pz);
                }
            }
            ++floor_boot_strides_;
            if (floor_cand_.size() >= 200 || floor_boot_strides_ >= 6) {
                double f0 = cfg_.floor_z;
                if (!floor_cand_.empty()) {
                    std::sort(floor_cand_.begin(), floor_cand_.end());
                    const size_t n = std::max<size_t>(1, floor_cand_.size() / 4);  // lowest 25%
                    f0 = floor_cand_[n / 2];                                        // its median
                }
                floor_off_est_ = f0;
                v.insert(FK(0), (gtsam::Vector1() << f0).finished());
                g.add(gtsam::PriorFactor<gtsam::Vector1>(
                    FK(0), (gtsam::Vector1() << f0).finished(),
                    gtsam::noiseModel::Isotropic::Sigma(1, 1.0)));   // weak (1 m) gauge anchor
                floor_init_ = true;
                std::vector<double>().swap(floor_cand_);
            }
        }
        if (floor_init_) ts[FK(0)] = t_now;   // persistent: never ages out of the lag
    }

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

    // |omega| proxy = gyro-measurement magnitude (close to the spline rate the
    // Ceres SW gates on, but available without a build-time spline eval).
    auto gmag = [](const ImuSample& s) { return std::sqrt(s.gx*s.gx + s.gy*s.gy + s.gz*s.gz); };
    auto gmag_near = [&](double t) {
        double best = 1e18, m = 0.0;
        for (const auto& s : imu) { double d = std::abs(s.timestamp - t); if (d < best) { best = d; m = gmag(s); } }
        return m;
    };

    // --- IMU sensor factors (omega-adaptive gyro + accel soft-gate) ---
    for (size_t i = 0; i < imu.size(); ++i) {
        if (imu_ko[i] < 0) continue;
        const double wn = gmag(imu[i]);
        double w_g = 1.0;
        if (cfg_.lambda_gyro_omega_sigma > 0.0)
            w_g = 1.0 + std::pow(wn / cfg_.lambda_gyro_omega_sigma, cfg_.lambda_gyro_omega_pow);
        double w_acc = 1.0;
        if (cfg_.accel_soft_sigma > 0.0) { double q = wn / cfg_.accel_soft_sigma; w_acc = 1.0 / (1.0 + q * q); }
        auto ng = (w_g == 1.0) ? n_gyr_ : sf_iso3(cfg_.lambda_gyro * w_g);
        auto na = (w_acc == 1.0) ? n_acc_ : sf_iso3(cfg_.lambda_accel * w_acc);

        gtsam::KeyVector ko, kg;
        for (int j = 0; j < N_ORI; ++j) { ko.push_back(RK(imu_ko[i] - 3 + j)); kg.push_back(RK(imu_ko[i] - 3 + j)); }
        for (int j = 0; j < N_POS; ++j) ko.push_back(PK(imu_kp[i] - 5 + j));
        ko.push_back(bkey); kg.push_back(bkey);
        g.add(gtsam_factors::AccelFactor(na, ko,
            Eigen::Vector3d(imu[i].ax, imu[i].ay, imu[i].az), imu_uo[i], inv_dt_ori_, imu_up[i], inv_dt_pos_));
        g.add(gtsam_factors::GyroFactor(ng, kg,
            Eigen::Vector3d(imu[i].gx, imu[i].gy, imu[i].gz), imu_uo[i], inv_dt_ori_));
    }

    // --- radar (omega soft-gate + per-point intensity + z-bias) ---
    for (const auto& f : radar) {
        if (f.points.empty() || !in_domain(f.timestamp)) continue;
        int ko, kp; double uo, up;
        ori_active(f.timestamp, ko, uo); pos_active(f.timestamp, kp, up);
        gtsam::KeyVector keys;
        for (int j = 0; j < N_ORI; ++j) keys.push_back(RK(ko - 3 + j));
        for (int j = 0; j < N_POS; ++j) keys.push_back(PK(kp - 5 + j));

        double w_omega = 1.0;
        if (cfg_.omega_soft_sigma > 0.0) { double q = gmag_near(f.timestamp) / cfg_.omega_soft_sigma; w_omega = 1.0 / (1.0 + q * q); }
        // radar_pos_split: warm-start R, omega (frozen) for the position-only factor.
        // The floor factor also needs the frozen warm-start R (and the predicted
        // world-z baseline) to classify + anchor floor returns.
        Eigen::Matrix3d R_ws = Eigen::Matrix3d::Identity(); Eigen::Vector3d w_ws = Eigen::Vector3d::Zero();
        const bool do_split = (cfg_.radar_pos_split > 0.0 && w_omega < 1.0);
        const bool do_floor = (cfg_.lambda_floor > 0.0) && (!cfg_.floor_free || floor_init_);
        double pz_pred = 0.0;   // warm-start predicted trajectory z at this frame
        if (do_split || do_floor) {
            double qk[N_ORI][4]; const double* qp[N_ORI];
            for (int j = 0; j < N_ORI; ++j) {
                const int idx = std::max(0, std::min(ko - 3 + j, n_ori_ - 1));
                const Eigen::Quaterniond qq = (align_R3 * quat_to_rot3(init_ori_[idx])).toQuaternion();
                qk[j][0] = qq.x(); qk[j][1] = qq.y(); qk[j][2] = qq.z(); qk[j][3] = qq.w(); qp[j] = qk[j];
            }
            R_ws = analytic::rotation_with_jacobian_manual<N_ORI>(qp, uo, inv_dt_ori_, nullptr).matrix();
            if (do_split)
                w_ws = analytic::body_velocity_with_jacobian_manual<N_ORI>(qp, uo, inv_dt_ori_, nullptr);
            if (do_floor) {
                using Helper = CeresSplineHelper<N_POS>;
                Eigen::Matrix<double, N_POS, 1> p0;
                Helper::template baseCoeffsWithTime<0>(p0, up);
                const Eigen::Matrix<double, N_POS, 1> wv = Helper::blending_matrix_ * p0;
                for (int i = 0; i < N_POS; ++i) {
                    const int idx = std::max(0, std::min(kp - 5 + i, n_pos_ - 1));
                    pz_pred += wv[i] * (init_pos_[idx][2] + align_t[2]);
                }
            }
        }
        gtsam::KeyVector pkeys(keys.begin() + N_ORI, keys.end());   // 6 pos CPs
        double inv_med_I = 0.0;
        if (cfg_.radar_intensity_weight > 0.0) {
            std::vector<double> ints;
            for (const auto& pt : f.points) if (pt.intensity > 0.0) ints.push_back(pt.intensity);
            if (!ints.empty()) { std::sort(ints.begin(), ints.end()); double med = ints[ints.size() / 2]; if (med > 0) inv_med_I = 1.0 / med; }
        }
        for (const auto& pt : f.points) {
            Eigen::Vector3d xyz(pt.x, pt.y, pt.z);
            const double rng = xyz.norm();
            if (rng < cfg_.min_range) continue;
            const Eigen::Vector3d u_sensor = xyz / rng;
            const Eigen::Vector3d u_body = R_bs_ * u_sensor;
            const double v_c = pt.v - cfg_.radar_zbias_fixed * u_sensor.z();
            double w_int = 1.0;
            if (inv_med_I > 0.0 && pt.intensity > 0.0)
                w_int = std::min(4.0, std::max(0.25, std::pow(pt.intensity * inv_med_I, cfg_.radar_intensity_weight)));
            const double w = w_omega * w_int;
            auto nr = (w == 1.0) ? n_radar_
                      : gtsam::noiseModel::Robust::Create(
                            gtsam::noiseModel::mEstimator::Huber::Create(cfg_.huber_delta),
                            gtsam::noiseModel::Isotropic::Sigma(1, 1.0 / std::sqrt(std::max(w, 1e-6))));
            g.add(gtsam_factors::RadarFactor(nr, keys, u_body, v_c, t_bs_, uo, inv_dt_ori_, up, inv_dt_pos_));
            // complementary position-only factor (radar velocity -> position, frozen ori)
            if (do_split) {
                const double ws = (1.0 - w_omega) * cfg_.radar_pos_split * w_int;
                if (ws > 1e-6) {
                    auto ns = gtsam::noiseModel::Robust::Create(
                        gtsam::noiseModel::mEstimator::Huber::Create(cfg_.huber_delta),
                        gtsam::noiseModel::Isotropic::Sigma(1, 1.0 / std::sqrt(ws)));
                    g.add(gtsam_factors::RadarPosOnlyFactor(ns, pkeys, R_ws, w_ws, u_body, v_c, t_bs_, up, inv_dt_pos_));
                }
            }
            // floor-plane absolute-z anchor: classify by predicted world-z, then add
            // a point-to-plane factor (frozen ori) pinning the vertical position.
            if (do_floor) {
                const Eigen::Vector3d p_body = R_bs_ * xyz + t_bs_;
                const double z_off = (R_ws * p_body).z();
                const double z_pred_world = z_off + pz_pred;
                // free-offset: gate on the current f0 estimate (self-calibrating);
                // fixed: gate on the hardcoded floor_z.
                const double ref = cfg_.floor_free ? floor_off_est_ : cfg_.floor_z;
                if (std::abs(z_pred_world - ref) <= cfg_.floor_band) {
                    auto nf = gtsam::noiseModel::Robust::Create(
                        gtsam::noiseModel::mEstimator::Huber::Create(cfg_.floor_huber),
                        gtsam::noiseModel::Isotropic::Sigma(1, 1.0 / std::sqrt(cfg_.lambda_floor)));
                    if (cfg_.floor_free) {
                        gtsam::KeyVector fk(pkeys); fk.push_back(FK(0));
                        g.add(gtsam_factors::FloorPlaneFreeFactor(nf, fk, z_off, up));
                    } else {
                        g.add(gtsam_factors::FloorPlaneFactor(nf, pkeys, z_off, cfg_.floor_z, up));
                    }
                    ++n_floor_;
                }
            }
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

    // NOTE: "selective FEJ" (exclude active ori knots from the marginal freeze via
    // a vendored-GTSAM setNoFixKeys patch) was tried here and REJECTED -- worse
    // everywhere (ISAM2_MIGRATION.md "Selective FEJ"). The freeze is load-bearing
    // for marginal validity, so the plumbing was reverted.

    // --- solve ---
    auto t0 = std::chrono::steady_clock::now();
    auto res = smoother_->update(g, v, ts);
    // Extra empty updates: ISAM2 does one gated GN step per update; the Ceres SW
    // re-solves each window to convergence. Extra passes re-linearize + re-solve
    // the marked variables (recovers roll/pitch from the weak-accel P1-P3 init).
    // With extra_iters_rtol>0, stop early once the error reduction plateaus.
    double prev_err = res.getError();
    for (int e = 0; e < cfg_.extra_iters; ++e) {
        auto r2 = smoother_->update();
        if (cfg_.extra_iters_rtol > 0.0) {
            const double err = r2.getError();
            if ((prev_err - err) < cfg_.extra_iters_rtol * std::max(prev_err, 1e-9)) break;
            prev_err = err;
        }
    }
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
    if (cfg_.floor_free && est.exists(FK(0)))
        floor_off_est_ = est.at<gtsam::Vector1>(FK(0))[0];   // next stride's gate ref
    num_active_ = static_cast<int>(smoother_->timestamps().size());
    bkey_last_ = bkey;
    if (cfg_.adapt_noise_stride > 0 && (++stride_count_ % cfg_.adapt_noise_stride == 0))
        adapt_noise(est, radar, imu);
    first_ = false;
    return dt;
}

// NIS-adaptive noise: set each sensor's sigma = std of its residuals at the
// current solution (data-driven whitening, ROADMAP 4c). Updates the BASE noise
// models used by the non-omega-weighted (racing) factors.
void IsamSolver::adapt_noise(const gtsam::Values& est,
                             const std::vector<RadarFrame>& radar,
                             const std::vector<ImuSample>& imu) {
    auto get_ori = [&](int ko, double q[N_ORI][4], const double* qp[N_ORI]) {
        for (int j = 0; j < N_ORI; ++j) {
            Key k = RK(ko - 3 + j); if (!est.exists(k)) return false;
            const Eigen::Quaterniond qq = est.at<Rot3>(k).toQuaternion();
            q[j][0] = qq.x(); q[j][1] = qq.y(); q[j][2] = qq.z(); q[j][3] = qq.w(); qp[j] = q[j];
        } return true;
    };
    auto get_pos = [&](int kp, double c[N_POS][3], const double* cp[N_POS]) {
        for (int j = 0; j < N_POS; ++j) {
            Key k = PK(kp - 5 + j); if (!est.exists(k)) return false;
            const Vector3 v = est.at<Vector3>(k); c[j][0] = v[0]; c[j][1] = v[1]; c[j][2] = v[2]; cp[j] = c[j];
        } return true;
    };
    Eigen::Matrix<double, 6, 1> bias = est.exists(bkey_last_) ? est.at<Vector6>(bkey_last_) : Vector6::Zero();

    std::vector<double> rg, ra, rr;
    for (size_t i = 0; i < imu.size(); i += 10) {
        if (!in_domain(imu[i].timestamp)) continue;
        int ko, kp; double uo, up; ori_active(imu[i].timestamp, ko, uo); pos_active(imu[i].timestamp, kp, up);
        double q[N_ORI][4]; const double* qp[N_ORI]; double c[N_POS][3]; const double* cp[N_POS];
        if (!get_ori(ko, q, qp)) continue;
        auto rgy = gtsam_factors::gyro_residual_gtsam(qp, bias.data(),
            Eigen::Vector3d(imu[i].gx, imu[i].gy, imu[i].gz), uo, inv_dt_ori_, false).residual;
        for (int k = 0; k < 3; ++k) rg.push_back(rgy[k]);
        if (get_pos(kp, c, cp)) {
            auto ray = gtsam_factors::accel_residual_gtsam(qp, cp, bias.data(),
                Eigen::Vector3d(imu[i].ax, imu[i].ay, imu[i].az), uo, inv_dt_ori_, up, inv_dt_pos_, false).residual;
            for (int k = 0; k < 3; ++k) ra.push_back(ray[k]);
        }
    }
    for (const auto& f : radar) {
        if (f.points.empty() || !in_domain(f.timestamp)) continue;
        int ko, kp; double uo, up; ori_active(f.timestamp, ko, uo); pos_active(f.timestamp, kp, up);
        double q[N_ORI][4]; const double* qp[N_ORI]; double c[N_POS][3]; const double* cp[N_POS];
        if (!get_ori(ko, q, qp) || !get_pos(kp, c, cp)) continue;
        for (const auto& pt : f.points) {
            Eigen::Vector3d xyz(pt.x, pt.y, pt.z); double rng = xyz.norm(); if (rng < cfg_.min_range) continue;
            rr.push_back(gtsam_factors::radar_residual_gtsam(qp, cp, R_bs_ * (xyz / rng), pt.v, t_bs_,
                uo, inv_dt_ori_, up, inv_dt_pos_, false).residual);
        }
    }
    auto stdev = [](const std::vector<double>& v) {
        if (v.size() < 8) return -1.0;
        double m = 0; for (double x : v) m += x; m /= v.size();
        double s = 0; for (double x : v) s += (x - m) * (x - m); return std::sqrt(s / v.size());
    };
    const double a = cfg_.adapt_noise_alpha;
    double sg = stdev(rg), sa = stdev(ra), sr = stdev(rr);
    if (sg > 0) sigma_g_ = (1 - a) * sigma_g_ + a * std::max(sg, 1e-3);
    if (sa > 0) sigma_a_ = (1 - a) * sigma_a_ + a * std::max(sa, 1e-3);
    if (sr > 0) sigma_r_ = (1 - a) * sigma_r_ + a * std::max(sr, 1e-3);
    n_gyr_ = nm::Isotropic::Sigma(3, sigma_g_);
    n_acc_ = nm::Isotropic::Sigma(3, sigma_a_);
    n_radar_ = nm::Robust::Create(nm::mEstimator::Huber::Create(cfg_.huber_delta),
                                  nm::Isotropic::Sigma(1, sigma_r_));
}

int IsamSolver::num_fixed() const {
    return smoother_ ? static_cast<int>(smoother_->getISAM2().getFixedVariables().size()) : 0;
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
