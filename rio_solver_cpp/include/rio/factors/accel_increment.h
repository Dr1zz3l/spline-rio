#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <basalt/spline/ceres_spline_helper.h>
#include <rio/trajectory.h>
#include <rio/solver.h>              // PreintFactor
#include <rio/factors/imu_accel.h>   // GRAVITY_WORLD

namespace rio {

// ============================================================================
// AccelIncrementFunctor  (EXPERIMENT 2026-09-12; DEPLOYED 2026-09-30 at 10 Hz,
// worklog/2026-09-29_accel-lag.md; solver_cpp.yaml accel_increment_hz)
// ============================================================================
// The accelerometer as a VELOCITY-INCREMENT factor over [t_i, t_j] instead of
// one row per 1 kHz sample on the spline's second derivative: the 1 kHz
// samples are integrated first (Forster TRO-2017 eq. 38-44, r_v only, the
// gyro-integrated rotation increment inside delta_v; bias corrected to first
// order), then weighted once with the GP backend's measured increment sigma.
// The per-sample gyro rows stay; only the accelerometer rows are replaced.
//
//   r_v = R_i^T (v_j - v_i - g dt) - (dv + dv/dba db_a + dv/dbg db_g)
//
// Parameter blocks:
//   params[0..3]                         N_ORI = 4 ori knots at t_i
//   params[4 .. 4+N_POS+k_pos_stride-1]  pos CPs covering t_i (first N_POS)
//                                        and t_j (offset k_pos_stride)
//   params[4+N_POS+k_pos_stride]         bias (ba, bg)

struct AccelIncrementFunctor {
    Eigen::Vector3d delta_v, b_a0, b_g0;
    Eigen::Matrix3d d_v_d_ba, d_v_d_bg;
    double dt;
    double u_ori_i, inv_dt_ori;
    double u_pos_i, u_pos_j, inv_dt_pos;
    int k_pos_stride;
    double scale;   // 1 / sigma_eff

    AccelIncrementFunctor(const PreintFactor& pf,
                          double u_ori_i_, double inv_dt_ori_,
                          double u_pos_i_, double u_pos_j_, double inv_dt_pos_,
                          int k_pos_stride_, double scale_)
        : delta_v(pf.delta_v), b_a0(pf.b_a0), b_g0(pf.b_g0),
          d_v_d_ba(pf.d_v_d_ba), d_v_d_bg(pf.d_v_d_bg), dt(pf.dt),
          u_ori_i(u_ori_i_), inv_dt_ori(inv_dt_ori_),
          u_pos_i(u_pos_i_), u_pos_j(u_pos_j_), inv_dt_pos(inv_dt_pos_),
          k_pos_stride(k_pos_stride_), scale(scale_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using SO3T  = Sophus::SO3<T>;
        SO3T R_i;
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params, u_ori_i, inv_dt_ori, &R_i, nullptr, nullptr);
        Vec3T v_i, v_j;
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 1>(
            params + 4, u_pos_i, inv_dt_pos, &v_i);
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 1>(
            params + 4 + k_pos_stride, u_pos_j, inv_dt_pos, &v_j);
        const int bi = 4 + N_POS + k_pos_stride;
        Vec3T b_a(params[bi][0], params[bi][1], params[bi][2]);
        Vec3T b_g(params[bi][3], params[bi][4], params[bi][5]);
        const Vec3T dv_c = delta_v.cast<T>()
            + d_v_d_ba.cast<T>() * (b_a - b_a0.cast<T>())
            + d_v_d_bg.cast<T>() * (b_g - b_g0.cast<T>());
        const Vec3T r = R_i.inverse()
            * (v_j - v_i - GRAVITY_WORLD.cast<T>() * T(dt)) - dv_c;
        residuals[0] = r[0] * T(scale);
        residuals[1] = r[1] * T(scale);
        residuals[2] = r[2] * T(scale);
        return true;
    }
};

}  // namespace rio
