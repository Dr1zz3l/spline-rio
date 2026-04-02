#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <ceres/ceres.h>

#include <array>
#include <vector>

namespace rio {

// ============================================================================
// MarginalizationPrior
// ============================================================================
// Stores the Schur complement prior from a previous window solve.
//
// After marginalizing out CPs/knots in the "stride zone", the remaining
// "boundary" CPs/knots + bias carry forward information as a dense Gaussian:
//
//   E = 0.5 * local_x^T * S * local_x   (S = Schur complement)
//
// where local_x = [pos_local (Euclidean), ori_local (SO3 log), bias_local].
// Residual: r = sqrt_info^T * local_x  so that ||r||^2 = local_x^T * S * local_x.
// sqrt_info = L  (lower Cholesky factor, S = L * L^T).
//
// Boundary convention (for window stride s, dt_pos, dt_ori):
//   n_bound_pos = N_POS - 1  (= 5 for quintic spline)
//   n_bound_ori = N_ORI - 1  (= 3 for cubic spline)
// ============================================================================

struct MarginalizationPrior {
    bool valid{false};

    // Cholesky factor of Schur complement (lower triangular L, S = L*L^T)
    // Dimensions: d_b × d_b  where d_b = 3*n_bound_pos + 3*n_bound_ori + 6
    Eigen::MatrixXd sqrt_info;
    int d_b{0};

    // Linearization point (global coordinates, updated after each solve)
    std::vector<std::array<double, 3>> bound_pos;   // n_bound_pos × 3
    std::vector<std::array<double, 4>> bound_ori;   // n_bound_ori × 4 (xyzw Sophus)
    std::array<double, 6> biases{};

    // Global indices into the persistent Trajectory for boundary params
    int pos_start{0};    // first boundary pos CP global index
    int ori_start{0};    // first boundary ori knot global index

    // Diagnostics from the last compute_prior() call (set regardless of valid)
    double      cond_number{0.0};
    double      min_eigenvalue{0.0};
    double      max_eigenvalue{0.0};
    int         numerical_rank{0};
    std::string drop_reason;   // "" if valid; reason string if valid=false

    // Covariance of boundary state = S^{-1}  (d_b × d_b)
    // trace_cov = tr(S^{-1}): sum of marginal variances across all boundary DOF
    // adaptive_scale = sqrt(lambda_boundary_pos / max_eigenvalue_of_S)
    //   — the scale that would normalise max S eigenvalue to lambda_boundary_pos
    Eigen::MatrixXd covariance;           // d_b × d_b (empty if not valid)
    double          trace_cov{0.0};
    double          adaptive_scale{0.0};
    double          last_adaptive_scale{0.0};  // as applied in add_prior_to_problem
};

// ============================================================================
// MargPriorFunctor
// ============================================================================
// DynamicAutoDiff functor for the marginalization prior.
// Parameter blocks (in order): [bound_pos CPs (3D each), bound_ori knots (4D each), bias (6D)]
// Residual: d_b dimensional vector = sqrt_info^T * local_x
// ============================================================================
struct MargPriorFunctor {
    explicit MargPriorFunctor(const MarginalizationPrior& p) : prior(p) {}

    template<typename T>
    bool operator()(T const* const* params, T* residuals) const {
        const int n_pos = static_cast<int>(prior.bound_pos.size());
        const int n_ori = static_cast<int>(prior.bound_ori.size());
        const int d = prior.d_b;

        Eigen::Matrix<T, Eigen::Dynamic, 1> local_x(d);
        int offset = 0;

        // Boundary pos CPs (Euclidean: local = x - x0)
        for (int i = 0; i < n_pos; ++i) {
            const T* p = params[i];
            local_x(offset++) = p[0] - T(prior.bound_pos[i][0]);
            local_x(offset++) = p[1] - T(prior.bound_pos[i][1]);
            local_x(offset++) = p[2] - T(prior.bound_pos[i][2]);
        }

        // Boundary ori knots (SO3: local = log(Q0^{-1} * Q))
        // Sophus convention: [x, y, z, w]
        for (int i = 0; i < n_ori; ++i) {
            const T* q = params[n_pos + i];
            Sophus::SO3<T> Q(Eigen::Quaternion<T>(q[3], q[0], q[1], q[2]));
            const auto& q0 = prior.bound_ori[i];
            T q0w = static_cast<T>(q0[3]);
            T q0x = static_cast<T>(q0[0]);
            T q0y = static_cast<T>(q0[1]);
            T q0z = static_cast<T>(q0[2]);
            Sophus::SO3<T> Q0(Eigen::Quaternion<T>(q0w, q0x, q0y, q0z));
            Eigen::Matrix<T, 3, 1> log_diff = (Q0.inverse() * Q).log();
            local_x(offset++) = log_diff(0);
            local_x(offset++) = log_diff(1);
            local_x(offset++) = log_diff(2);
        }

        // Bias (Euclidean)
        const T* b = params[n_pos + n_ori];
        for (int j = 0; j < 6; ++j)
            local_x(offset++) = b[j] - T(prior.biases[j]);

        // r = sqrt_info^T * local_x
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> r(residuals, d);
        r = prior.sqrt_info.template cast<T>().transpose() * local_x;
        return true;
    }

    const MarginalizationPrior& prior;
};

}  // namespace rio
