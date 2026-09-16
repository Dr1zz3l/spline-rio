#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <ceres/ceres.h>
#include <basalt/spline/ceres_spline_helper.h>
#include <rio/trajectory.h>

namespace rio {

// ============================================================================
// RadarDopplerFactor
// ============================================================================
// Residual: r = v_meas - v_pred
// where v_pred = -dot(u_body, v_ant)
//   u_body = R_radar_to_body.T @ u_sensor  (unit bearing in body frame)
//   v_ant  = R_body_to_world.T @ v_world + omega_body × t_body_sensor
//          (world velocity rotated to body frame + lever arm correction)
//
// Lever arm: the antenna is not at the CoM. True Doppler reflects v_CoM + ω×r.
// omega_body is extracted from evaluate_lie (body-frame angular rate).
// t_body_sensor: translation from body CoM to antenna, expressed in body frame.
//
// The negation (−) is the physically correct TI IWR6843 convention:
// positive Doppler = receding target.
//
// Parameter blocks (via Ceres DynamicAutoDiffCostFunction):
//   [0..N_ORI-1]  : orientation quaternion knots   (4 params each)
//   [N_ORI]       : position CP block i0+0          (3 params)
//   ...
//   [N_ORI+N_POS-1]: position CP block i0+N_POS-1  (3 params)
//   [N_ORI+N_POS]  : bias block                     (6 params)
//
// The extrinsic rotation is passed as a constant (not a parameter block) for
// now; when optimize_pitch_only is true only pitch changes and we handle it
// via a separate 1-param block in the full solver.

struct RadarDopplerFunctor {
    // measurement
    Eigen::Vector3d u_sensor;  // unit bearing in sensor (radar) frame
    double v_meas;             // measured Doppler velocity (m/s)

    // spline parameters
    double u_ori;     // normalised time in ori spline [0,1)
    double inv_dt_ori;

    double u_pos;     // normalised time in pos spline [0,1)
    double inv_dt_pos;

    // extrinsic (constant)
    Sophus::SO3d R_radar_to_body;
    Eigen::Vector3d t_body_sensor;  // antenna position in body frame (m)

    RadarDopplerFunctor(const Eigen::Vector3d& u_sensor_,
                        double v_meas_,
                        double u_ori_, double inv_dt_ori_,
                        double u_pos_, double inv_dt_pos_,
                        const Sophus::SO3d& R_radar_to_body_,
                        const Eigen::Vector3d& t_body_sensor_)
        : u_sensor(u_sensor_), v_meas(v_meas_),
          u_ori(u_ori_), inv_dt_ori(inv_dt_ori_),
          u_pos(u_pos_), inv_dt_pos(inv_dt_pos_),
          R_radar_to_body(R_radar_to_body_),
          t_body_sensor(t_body_sensor_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;

        // 1. Evaluate orientation and body-frame angular rate at measurement time
        SO3T R_body_to_world;
        Vec3T omega_body;

        // Orientation knots: params[0..N_ORI-1] each of size 4
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params, u_ori, inv_dt_ori,
            &R_body_to_world,  // transform_out
            &omega_body,       // vel_out: body-frame angular velocity (rad/s)
            nullptr);

        // 2. Evaluate position velocity: params[N_ORI..N_ORI+N_POS-1] each size 3
        Vec3T body_vel_world;
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 1>(
            params + N_ORI, u_pos, inv_dt_pos,
            &body_vel_world);

        // 3. Transform bearing to body frame using constant extrinsic
        Vec3T u_body = R_radar_to_body.matrix().template cast<T>() * u_sensor.cast<T>();

        // 4. Velocity at antenna = R_bw^T @ v_world + omega_body × t_body_sensor
        //    (lever arm: antenna is offset from CoM by t_body_sensor)
        Vec3T v_CoM_body = R_body_to_world.inverse() * body_vel_world;
        Vec3T v_lever = omega_body.cross(t_body_sensor.cast<T>());
        Vec3T v_ant = v_CoM_body + v_lever;

        // 5. Predicted Doppler: -dot(u_body, v_ant)
        T v_pred = -u_body.dot(v_ant);

        residuals[0] = v_meas - v_pred;
        return true;
    }
};

// ============================================================================
// RadarDopplerWithPitchFunctor
// ============================================================================
// Same as RadarDopplerFunctor but accepts one extra parameter block:
//   params[N_ORI + N_POS + 1] = pitch_delta (1 scalar, radians)
//
// Composition: R_total = R_nominal * Ry(pitch_delta)
// where R_nominal is the fixed roll=180°, yaw=0° part of the extrinsic.
// Only pitch is observable from Doppler; roll/yaw are not added as parameters.
//
// Used when lock_extrinsics=false (default). Falls back to RadarDopplerFunctor
// when lock_extrinsics=true.

struct RadarDopplerWithPitchFunctor {
    Eigen::Vector3d u_sensor;
    double v_meas;
    double u_ori, inv_dt_ori;
    double u_pos, inv_dt_pos;
    Sophus::SO3d R_radar_to_body;
    Eigen::Vector3d t_body_sensor;

    RadarDopplerWithPitchFunctor(const Eigen::Vector3d& u_sensor_,
                                  double v_meas_,
                                  double u_ori_, double inv_dt_ori_,
                                  double u_pos_, double inv_dt_pos_,
                                  const Sophus::SO3d& R_radar_to_body_,
                                  const Eigen::Vector3d& t_body_sensor_)
        : u_sensor(u_sensor_), v_meas(v_meas_),
          u_ori(u_ori_), inv_dt_ori(inv_dt_ori_),
          u_pos(u_pos_), inv_dt_pos(inv_dt_pos_),
          R_radar_to_body(R_radar_to_body_),
          t_body_sensor(t_body_sensor_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        // 1. Orientation and angular rate
        SO3T R_body_to_world;
        Vec3T omega_body;
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params, u_ori, inv_dt_ori, &R_body_to_world, &omega_body, nullptr);

        // 2. Position velocity
        Vec3T body_vel_world;
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 1>(
            params + N_ORI, u_pos, inv_dt_pos, &body_vel_world);

        // 3. Apply pitch perturbation: R_total = R_nominal * Ry(pitch_delta)
        T pd = params[N_ORI + N_POS + 1][0];
        Mat3T Ry;
        Ry << cos(pd), T(0), sin(pd),
              T(0),    T(1), T(0),
             -sin(pd), T(0), cos(pd);
        Mat3T R_total = R_radar_to_body.matrix().template cast<T>() * Ry;
        Vec3T u_body = R_total * u_sensor.cast<T>();

        // 4. Velocity at antenna with lever arm
        Vec3T v_CoM_body = R_body_to_world.inverse() * body_vel_world;
        Vec3T v_lever = omega_body.cross(t_body_sensor.cast<T>());
        Vec3T v_ant = v_CoM_body + v_lever;

        // 5. Predicted Doppler
        T v_pred = -u_body.dot(v_ant);
        residuals[0] = v_meas - v_pred;
        return true;
    }
};

}  // namespace rio
