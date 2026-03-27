#pragma once

#include <Eigen/Dense>
#include <basalt/spline/ceres_spline_helper.h>
#include <rio/trajectory.h>

namespace rio {

// ============================================================================
// MinSnapFactor
// ============================================================================
// Position minimum-snap regularization: r = p^(4)(t)   (4th derivative)
// We use derivative=4, which for a quintic (order-6) spline is the snap.
//
// Parameter blocks: N_POS consecutive position CPs (3 params each)

struct MinSnapFunctor {
    double u_pos, inv_dt_pos;

    MinSnapFunctor(double u_pos_, double inv_dt_pos_)
        : u_pos(u_pos_), inv_dt_pos(inv_dt_pos_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;

        Vec3T snap;
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 4>(
            params, u_pos, inv_dt_pos, &snap);

        residuals[0] = snap[0];
        residuals[1] = snap[1];
        residuals[2] = snap[2];
        return true;
    }
};

// ============================================================================
// OrientationRegFactor
// ============================================================================
// Penalise the magnitude of the log-difference between consecutive orientation
// knots: r = log(q_i^{-1} * q_{i+1})  (3D body-rate increment)
//
// Parameter blocks: [q_i (4 params), q_{i+1} (4 params)]

struct OrientationRegFunctor {
    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using SO3T = Sophus::SO3<T>;

        Eigen::Map<SO3T const> q0(params[0]);
        Eigen::Map<SO3T const> q1(params[1]);

        auto log_delta = (q0.inverse() * q1).log();
        residuals[0] = log_delta[0];
        residuals[1] = log_delta[1];
        residuals[2] = log_delta[2];
        return true;
    }
};

// ============================================================================
// AngularAccelRegFactor
// ============================================================================
// Penalise angular acceleration: second finite difference of rotation in so(3).
//   omega_prev = log(q_{i-1}^{-1} * q_i)
//   omega_next = log(q_i^{-1}     * q_{i+1})
//   r = omega_next - omega_prev   (3D, units: rad per knot interval)
//
// Zero for constant angular rate (steady-state bank, uniform flip rotation).
// Only fires when omega changes abruptly — correct prior for aggressive flight.
// Analogue of min-snap for position. Replaces OrientationRegFunctor.
//
// Parameter blocks: [q_{i-1} (4), q_i (4), q_{i+1} (4)]

struct AngularAccelRegFunctor {
    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using SO3T = Sophus::SO3<T>;

        Eigen::Map<SO3T const> q0(params[0]);
        Eigen::Map<SO3T const> q1(params[1]);
        Eigen::Map<SO3T const> q2(params[2]);

        auto omega_prev = (q0.inverse() * q1).log();
        auto omega_next = (q1.inverse() * q2).log();

        auto alpha = omega_next - omega_prev;
        residuals[0] = alpha[0];
        residuals[1] = alpha[1];
        residuals[2] = alpha[2];
        return true;
    }
};

// ============================================================================
// BoundaryPosFactor
// ============================================================================
// Pin position spline to a reference value at the boundary.
// Parameter blocks: N_POS consecutive position CPs (3 params each)

struct BoundaryPosFunctor {
    Eigen::Vector3d p_ref;
    double u_pos, inv_dt_pos;

    BoundaryPosFunctor(const Eigen::Vector3d& p_ref_,
                       double u_pos_, double inv_dt_pos_)
        : p_ref(p_ref_), u_pos(u_pos_), inv_dt_pos(inv_dt_pos_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;

        Vec3T pos;
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 0>(
            params, u_pos, inv_dt_pos, &pos);

        Vec3T res = pos - p_ref.cast<T>();
        residuals[0] = res[0];
        residuals[1] = res[1];
        residuals[2] = res[2];
        return true;
    }
};

// ============================================================================
// BoundaryVelFactor
// ============================================================================
// Pin velocity at boundary.

struct BoundaryVelFunctor {
    Eigen::Vector3d v_ref;
    double u_pos, inv_dt_pos;

    BoundaryVelFunctor(const Eigen::Vector3d& v_ref_,
                       double u_pos_, double inv_dt_pos_)
        : v_ref(v_ref_), u_pos(u_pos_), inv_dt_pos(inv_dt_pos_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;

        Vec3T vel;
        CeresSplineHelper<N_POS>::template evaluate<T, 3, 1>(
            params, u_pos, inv_dt_pos, &vel);

        Vec3T res = vel - v_ref.cast<T>();
        residuals[0] = res[0];
        residuals[1] = res[1];
        residuals[2] = res[2];
        return true;
    }
};

// ============================================================================
// BoundaryOriFactor
// ============================================================================
// Pin orientation at boundary (full SO3 residual).

struct BoundaryOriFunctor {
    Sophus::SO3d R_ref;
    double u_ori, inv_dt_ori;

    BoundaryOriFunctor(const Sophus::SO3d& R_ref_,
                       double u_ori_, double inv_dt_ori_)
        : R_ref(R_ref_), u_ori(u_ori_), inv_dt_ori(inv_dt_ori_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        using SO3T = Sophus::SO3<T>;

        SO3T R_pred;
        CeresSplineHelper<N_ORI>::template evaluate_lie<T, Sophus::SO3>(
            params, u_ori, inv_dt_ori, &R_pred, nullptr, nullptr);

        auto log_err = (R_pred * R_ref.inverse().template cast<T>()).log();
        residuals[0] = log_err[0];
        residuals[1] = log_err[1];
        residuals[2] = log_err[2];
        return true;
    }
};

// ============================================================================
// PitchDeltaPriorFactor
// ============================================================================
// Anchors pitch_delta near zero (= nominal extrinsic pitch).
// residual = pitch_delta   →   cost = lambda_extrinsic_prior * pitch_delta²
// Parameter blocks: [pitch_delta (1 param)]

struct PitchDeltaPriorFunctor {
    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        residuals[0] = params[0][0];
        return true;
    }
};

}  // namespace rio
