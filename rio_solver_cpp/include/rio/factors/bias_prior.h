#pragma once

#include <Eigen/Dense>

namespace rio {

// ============================================================================
// BiasPriorFactor
// ============================================================================
// Soft prior on biases: r = b - b_init
// Parameter block: bias (6 params: [b_ax,b_ay,b_az, b_gx,b_gy,b_gz])

struct BiasPriorFunctor {
    Eigen::Matrix<double, 6, 1> b_init;

    explicit BiasPriorFunctor(const Eigen::Matrix<double, 6, 1>& b_init_)
        : b_init(b_init_) {}

    template <class T>
    bool operator()(T const* const* params, T* residuals) const {
        for (int i = 0; i < 6; ++i)
            residuals[i] = params[0][i] - T(b_init[i]);
        return true;
    }
};

}  // namespace rio
