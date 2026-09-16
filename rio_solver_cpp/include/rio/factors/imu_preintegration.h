#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <basalt/spline/ceres_spline_helper.h>
#include <rio/trajectory.h>
#include <rio/solver.h>      // for PreintFactor
#include <rio/factors/imu_accel.h>   // for GRAVITY_WORLD

namespace rio {

// ============================================================================
// IMUPreintegrationFunctor
// ============================================================================
// Connects two spline evaluation points t_i and t_j via a preintegrated IMU
// factor (Forster TRO-2017, eq. 38-44).
//
// 9 residuals: [r_R(3), r_v(3), r_p(3)]
//
// Parameter block layout (passed to AddResidualBlock via make_auto_cost):
//   params[0..3]                        : N_ORI=4 ori knots at t_i
//   params[1..4]                        : N_ORI=4 ori knots at t_j  (overlaps by 3)
//   --> unique ori blocks: params[0..4] = 5 blocks of 4 doubles each
//
//   params[5..10]                       : N_POS=6 pos CPs at t_i
//   params[5+k_pos_stride..10+k_pos_stride] : N_POS=6 pos CPs at t_j
//   --> unique pos blocks: params[5..5+N_POS+k_pos_stride-1]
//                          = N_POS+k_pos_stride blocks of 3 doubles each
//
//   params[5+N_POS+k_pos_stride]        : bias block (6 doubles: ba(3), bg(3))
//
// Total parameter blocks: 5 + N_POS + k_pos_stride + 1
//   Default (k_pos_stride=2, dt_ori=0.01, dt_pos=0.005): 5+6+2+1 = 14
//
// Adjacency requirement: the caller must verify oi0_j == oi0_i + 1 before
// constructing this functor (ensured by uniform grid aligned to knot times).
//
// Evaluation uses params + offset pointer arithmetic — no local pointer arrays
// needed, and no custom const-cast required.

struct IMUPreintegrationFunctor {
    // Preintegrated measurements (at linearization biases b_a0, b_g0)
    Eigen::Matrix3d delta_R;
    Eigen::Vector3d delta_v, delta_p;
    Eigen::Vector3d b_a0, b_g0;
    Eigen::Matrix3d d_R_d_bg;
    Eigen::Matrix3d d_v_d_ba, d_v_d_bg;
    Eigen::Matrix3d d_p_d_ba, d_p_d_bg;
    double dt;

    // Spline evaluation parameters
    double u_ori_i, u_ori_j, inv_dt_ori;
    double u_pos_i, u_pos_j, inv_dt_pos;
    int k_pos_stride;   // = pi0_j - pi0_i (runtime value, typically 2)

    // Independent scales for r_v and r_p residuals.
    // scale = sqrt(lambda_preint_X / lambda_preint), default 0 (disabled).
    // Keeping them zero uses the factor as rotation-only, which is the safe
    // default because the P1-P3 velocity initialisation differs from the
    // IMU-integrated velocity by ~0.1 m/s (radar noise), causing r_v to
    // dominate the first gradient step and corrupt orientation if enabled.
    double scale_v{0.0};
    double scale_p{0.0};

    IMUPreintegrationFunctor(
        const PreintFactor& pf,
        double u_ori_i_, double u_ori_j_, double inv_dt_ori_,
        double u_pos_i_, double u_pos_j_, double inv_dt_pos_,
        int k_pos_stride_,
        double scale_v_ = 0.0, double scale_p_ = 0.0)
        : delta_R(pf.delta_R),
          delta_v(pf.delta_v),
          delta_p(pf.delta_p),
          b_a0(pf.b_a0),
          b_g0(pf.b_g0),
          d_R_d_bg(pf.d_R_d_bg),
          d_v_d_ba(pf.d_v_d_ba),
          d_v_d_bg(pf.d_v_d_bg),
          d_p_d_ba(pf.d_p_d_ba),
          d_p_d_bg(pf.d_p_d_bg),
          dt(pf.dt),
          u_ori_i(u_ori_i_), u_ori_j(u_ori_j_), inv_dt_ori(inv_dt_ori_),
          u_pos_i(u_pos_i_), u_pos_j(u_pos_j_), inv_dt_pos(inv_dt_pos_),
          k_pos_stride(k_pos_stride_),
          scale_v(scale_v_), scale_p(scale_p_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using SO3T  = Sophus::SO3<T>;

        // ---- Orientation at t_i and t_j ------------------------------------
        // params[0..3] for t_i (u=u_ori_i); params[1..4] for t_j (u=u_ori_j)
        SO3T R_i, R_j;
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params, u_ori_i, inv_dt_ori, &R_i, nullptr, nullptr);
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params + 1, u_ori_j, inv_dt_ori, &R_j, nullptr, nullptr);

        // ---- Position and velocity at t_i and t_j --------------------------
        // pos CPs start at params+5; t_j CPs start at params+5+k_pos_stride
        Vec3T p_i, v_i, p_j, v_j;
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 0>(
            params + 5, u_pos_i, inv_dt_pos, &p_i);
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 1>(
            params + 5, u_pos_i, inv_dt_pos, &v_i);
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 0>(
            params + 5 + k_pos_stride, u_pos_j, inv_dt_pos, &p_j);
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 1>(
            params + 5 + k_pos_stride, u_pos_j, inv_dt_pos, &v_j);

        // ---- Bias ----------------------------------------------------------
        const int bias_idx = 5 + N_POS + k_pos_stride;
        Vec3T b_a(params[bias_idx][0], params[bias_idx][1], params[bias_idx][2]);
        Vec3T b_g(params[bias_idx][3], params[bias_idx][4], params[bias_idx][5]);
        Vec3T db_a = b_a - b_a0.cast<T>();
        Vec3T db_g = b_g - b_g0.cast<T>();

        // ---- Bias-corrected preintegrated measurements ---------------------
        SO3T delta_R_c = SO3T(delta_R.cast<T>()) * SO3T::exp(d_R_d_bg.cast<T>() * db_g);
        Vec3T delta_v_c = delta_v.cast<T>()
                          + d_v_d_ba.cast<T>() * db_a
                          + d_v_d_bg.cast<T>() * db_g;
        Vec3T delta_p_c = delta_p.cast<T>()
                          + d_p_d_ba.cast<T>() * db_a
                          + d_p_d_bg.cast<T>() * db_g;

        // ---- Residuals (Forster TRO-2017, eq. 38-44) ----------------------
        Vec3T g_world = GRAVITY_WORLD.cast<T>();
        T dt_T(dt);

        // r_R = Log(ΔR̃_corr^T · R_i^T · R_j)
        Vec3T r_R = (delta_R_c.inverse() * R_i.inverse() * R_j).log();

        // r_v = R_i^T · (v_j - v_i - g·dt) - Δṽ_corr
        Vec3T r_v = R_i.inverse() * (v_j - v_i - g_world * dt_T) - delta_v_c;

        // r_p = R_i^T · (p_j - p_i - v_i·dt - ½g·dt²) - Δp̃_corr
        Vec3T r_p = R_i.inverse() *
                    (p_j - p_i - v_i * dt_T - T(0.5) * g_world * dt_T * dt_T)
                    - delta_p_c;

        residuals[0] = r_R[0]; residuals[1] = r_R[1]; residuals[2] = r_R[2];
        residuals[3] = r_v[0] * T(scale_v);
        residuals[4] = r_v[1] * T(scale_v);
        residuals[5] = r_v[2] * T(scale_v);
        residuals[6] = r_p[0] * T(scale_p);
        residuals[7] = r_p[1] * T(scale_p);
        residuals[8] = r_p[2] * T(scale_p);

        return true;
    }
};

}  // namespace rio
