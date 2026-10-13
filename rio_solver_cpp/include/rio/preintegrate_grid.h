#pragma once
// ============================================================================
// preintegrate_grid (2026-10-13): C++ port of the Python increment builder
// (analysis/lib/imu_preintegration.py::preintegrate +
//  analysis/validate_live_solver.py::_build_preint_factors_cpp), used by the
// live ROS spline node (the Python loop costs tens of ms per window) and
// selectable in the harness.  Same grid, same sample selection, same
// midpoint integration and Forster bias Jacobians, same order of operations.
// ============================================================================
#include <rio/solver.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace rio {

namespace detail {
inline Eigen::Matrix3d pg_skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d K;
    K << 0.0, -v[2], v[1],
         v[2], 0.0, -v[0],
        -v[1], v[0], 0.0;
    return K;
}
inline Eigen::Matrix3d pg_exp(const Eigen::Vector3d& phi) {
    const double th = phi.norm();
    if (th < 1e-10) return Eigen::Matrix3d::Identity() + pg_skew(phi);
    const Eigen::Matrix3d K = pg_skew(phi / th);
    return Eigen::Matrix3d::Identity() + std::sin(th) * K + (1.0 - std::cos(th)) * (K * K);
}
inline Eigen::Matrix3d pg_jr(const Eigen::Vector3d& phi) {
    const double th = phi.norm();
    const Eigen::Matrix3d K = pg_skew(phi);
    if (th < 1e-8) return Eigen::Matrix3d::Identity() - 0.5 * K;
    const double th2 = th * th;
    const double c1 = (1.0 - std::cos(th)) / th2;
    const double c2 = (th - std::sin(th)) / (th * th2);
    return Eigen::Matrix3d::Identity() - c1 * K + c2 * (K * K);
}
}  // namespace detail

// imu: N x 7 rows (t, ax, ay, az, gx, gy, gz), time-sorted.
// Grid: t_ref + k*dt for k = round(t_start-t_ref)/dt .. round(t_end-t_ref)/dt
// (round half to even, as Python's round()).  One factor per non-empty cell.
inline std::vector<PreintFactor> preintegrate_grid(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>>& imu,
    const Eigen::Vector3d& b_a0, const Eigen::Vector3d& b_g0,
    double t_start, double t_end, double dt, double t_ref, bool snap_to_samples)
{
    using detail::pg_exp; using detail::pg_jr; using detail::pg_skew;
    std::vector<PreintFactor> out;
    const long n = imu.rows();
    const long k0 = static_cast<long>(std::nearbyint((t_start - t_ref) / dt));
    const long k1 = static_cast<long>(std::nearbyint((t_end - t_ref) / dt));
    for (long k = k0; k < k1; ++k) {
        const double ti = t_ref + static_cast<double>(k) * dt;
        const double tj = t_ref + static_cast<double>(k + 1) * dt;
        // cell mask (t >= ti) & (t < tj + 1e-9)
        long m_first = -1, m_last = -1, m_count = 0;
        // integration set: ti <= t <= tj
        long s_first = -1, s_last = -1;
        for (long r = 0; r < n; ++r) {
            const double t = imu(r, 0);
            if (t >= ti && t < tj + 1e-9) { if (m_first < 0) m_first = r; m_last = r; ++m_count; }
            if (t >= ti && t <= tj) { if (s_first < 0) s_first = r; s_last = r; }
        }
        if (m_count == 0) continue;
        PreintFactor f;
        f.t_i = ti; f.t_j = tj; f.dt = tj - ti;
        f.b_a0 = b_a0; f.b_g0 = b_g0;
        Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
        Eigen::Vector3d dv = Eigen::Vector3d::Zero(), dp = Eigen::Vector3d::Zero();
        Eigen::Matrix3d dRdbg = Eigen::Matrix3d::Zero(), dvdba = Eigen::Matrix3d::Zero(),
                        dvdbg = Eigen::Matrix3d::Zero(), dpdba = Eigen::Matrix3d::Zero(),
                        dpdbg = Eigen::Matrix3d::Zero();
        if (s_first >= 0 && s_last - s_first + 1 >= 2) {
            for (long r = s_first; r < s_last; ++r) {
                const double h = imu(r + 1, 0) - imu(r, 0);
                if (h <= 0.0) continue;
                const Eigen::Vector3d a_m = 0.5 * (imu.block<1, 3>(r, 1).transpose() + imu.block<1, 3>(r + 1, 1).transpose());
                const Eigen::Vector3d w_m = 0.5 * (imu.block<1, 3>(r, 4).transpose() + imu.block<1, 3>(r + 1, 4).transpose());
                const Eigen::Vector3d a_c = a_m - b_a0;
                const Eigen::Vector3d w_c = w_m - b_g0;
                const Eigen::Vector3d phi = w_c * h;
                const Eigen::Matrix3d Rstep = pg_exp(phi);
                const Eigen::Matrix3d Jr = pg_jr(phi);
                dp += dv * h + 0.5 * (dR * a_c) * h * h;
                dv += (dR * a_c) * h;
                dpdba += dvdba * h - 0.5 * dR * h * h;
                dpdbg += dvdbg * h - 0.5 * (dR * pg_skew(a_c) * dRdbg) * h * h;
                dvdba -= dR * h;
                dvdbg -= dR * pg_skew(a_c) * dRdbg * h;
                dRdbg = Rstep.transpose() * dRdbg - Jr * h;
                dR = dR * Rstep;
            }
        }
        f.delta_R = dR; f.delta_v = dv; f.delta_p = dp;
        f.d_R_d_bg = dRdbg; f.d_v_d_ba = dvdba; f.d_v_d_bg = dvdbg;
        f.d_p_d_ba = dpdba; f.d_p_d_bg = dpdbg;
        if (snap_to_samples) {
            if (m_count < 2) continue;
            f.t_i = imu(m_first, 0); f.t_j = imu(m_last, 0); f.dt = f.t_j - f.t_i;
        }
        out.push_back(f);
    }
    return out;
}

}  // namespace rio
