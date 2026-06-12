#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace rio {

// ============================================================================
// BiasPriorFactor
// ============================================================================
// Soft prior on biases: r_i = sqrt_w_i * (b_i - b_init_i)
// Parameter block: bias (6 params: [b_ax,b_ay,b_az, b_gx,b_gy,b_gz])
//
// Per-component sqrt weights allow lambda_bias_prior_accel (components 0-2)
// and lambda_bias_prior_gyro (components 3-5) to differ.  The previous
// implementation applied sqrt(lambda_a * lambda_g) (geometric mean) uniformly
// via ScaledLoss — correct only when the two lambdas are equal.

struct BiasPriorFunctor {
    Eigen::Matrix<double, 6, 1> b_init;
    Eigen::Matrix<double, 6, 1> sqrt_w;

    explicit BiasPriorFunctor(const Eigen::Matrix<double, 6, 1>& b_init_)
        : b_init(b_init_), sqrt_w(Eigen::Matrix<double, 6, 1>::Ones()) {}

    BiasPriorFunctor(const Eigen::Matrix<double, 6, 1>& b_init_,
                     double lambda_accel, double lambda_gyro)
        : b_init(b_init_) {
        const double wa = std::sqrt(std::max(0.0, lambda_accel));
        const double wg = std::sqrt(std::max(0.0, lambda_gyro));
        sqrt_w << wa, wa, wa, wg, wg, wg;
    }

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        for (int i = 0; i < 6; ++i)
            residuals[i] = T(sqrt_w[i]) * (params[0][i] - T(b_init[i]));
        return true;
    }
};

}  // namespace rio
