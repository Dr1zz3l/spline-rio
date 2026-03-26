#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <vector>
#include <stdexcept>
#include <cmath>

#include <basalt/spline/ceres_spline_helper.h>

using basalt::CeresSplineHelper;

namespace rio {

// ============================================================================
// Trajectory
// ============================================================================
// Combines a quintic (order-6) Euclidean B-spline for position with a cubic
// (order-4) cumulative SO(3) B-spline for orientation.
//
// Basalt convention: order N = degree + 1.  We use N=6 (quintic) for pos and
// N=4 (cubic) for ori to match the Python solver.
//
// Control points / knots are stored as flat vectors of raw doubles so that
// Ceres can hold them as parameter blocks directly (no copies).

static constexpr int N_POS = 6;  // order=6 → degree=5 (quintic position)
static constexpr int N_ORI = 4;  // order=4 → degree=3 (cubic orientation)

struct Trajectory {
    // Time reference (absolute seconds).  All other times are offsets from here.
    double t_ref{0.0};

    // Position knot spacing (seconds).
    double dt_pos{0.005};

    // Orientation knot spacing (seconds).
    double dt_ori{0.008};

    // Position control points: [N_pos_cps][3]  (x, y, z per point)
    std::vector<std::array<double, 3>> pos_cps;

    // Orientation quaternion knots: [N_ori_knots][4]  (Sophus storage: x,y,z,w)
    // Each knot is a Sophus::SO3d stored as 4 doubles via Eigen::Map<Sophus::SO3d>.
    std::vector<std::array<double, 4>> ori_knots;

    // Biases: [b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
    std::array<double, 6> biases{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Extrinsic: radar-to-body rotation (3 Euler angles [roll, pitch, yaw] deg)
    // During optimization only pitch (index 1) changes if optimize_pitch_only.
    std::array<double, 3> extrinsic_euler_deg{180.0, 25.5, 0.0};

    // ---- Initializers --------------------------------------------------------

    void init(double t_ref_, double dt_pos_, double dt_ori_,
              int n_pos_cps, int n_ori_knots) {
        t_ref = t_ref_;
        dt_pos = dt_pos_;
        dt_ori = dt_ori_;
        pos_cps.assign(n_pos_cps, {0.0, 0.0, 0.0});
        // Default: identity quaternion (Sophus: x=0,y=0,z=0,w=1)
        ori_knots.assign(n_ori_knots, {0.0, 0.0, 0.0, 1.0});
    }

    // ---- Time helpers --------------------------------------------------------

    // Number of pos control points fully within the trajectory.
    int n_pos_cps() const { return static_cast<int>(pos_cps.size()); }
    int n_ori_knots() const { return static_cast<int>(ori_knots.size()); }

    // Valid time range for position spline [t_ref + offset_s, t_ref + ...].
    // For order-N uniform B-spline with n_cp control points, the valid range
    // is [knot[N-1], knot[n_cp]] where knots are spaced dt_pos apart.
    double t_pos_start() const { return t_ref + (N_POS - 1) * dt_pos; }
    double t_pos_end()   const { return t_ref + (n_pos_cps() - 1) * dt_pos; }

    double t_ori_start() const { return t_ref + (N_ORI - 1) * dt_ori; }
    double t_ori_end()   const { return t_ref + (n_ori_knots() - 1) * dt_ori; }

    // Return normalised time u ∈ [0,1) and first-knot index i0 for position.
    // i0 .. i0+N_POS-1 are the N_POS control points involved.
    bool pos_index(double t_abs, double& u, int& i0) const {
        double t_rel = t_abs - t_ref;
        // Knot index of the segment start
        int k = static_cast<int>(t_rel / dt_pos);
        // First active CP is k - (N_POS - 1) in standard B-spline indexing,
        // but basalt's evaluate() takes N consecutive knots starting at i0.
        // For order-N B-spline, segment k uses CPs [k, k+1, ..., k+N-1].
        // Valid segment range: k = N_POS-1 .. n_pos_cps-1
        k = std::max(N_POS - 1, std::min(k, n_pos_cps() - 1));
        i0 = k - (N_POS - 1);
        u = (t_rel - k * dt_pos) / dt_pos;
        // Clamp u to [0,1)
        u = std::max(0.0, std::min(u, 1.0 - 1e-10));
        return (t_abs >= t_pos_start() && t_abs <= t_pos_end());
    }

    bool ori_index(double t_abs, double& u, int& i0) const {
        double t_rel = t_abs - t_ref;
        int k = static_cast<int>(t_rel / dt_ori);
        k = std::max(N_ORI - 1, std::min(k, n_ori_knots() - 1));
        i0 = k - (N_ORI - 1);
        u = (t_rel - k * dt_ori) / dt_ori;
        u = std::max(0.0, std::min(u, 1.0 - 1e-10));
        return (t_abs >= t_ori_start() && t_abs <= t_ori_end());
    }

    // ---- Raw pointer accessors for Ceres ------------------------------------

    double* pos_cp_data(int i) { return pos_cps[i].data(); }
    const double* pos_cp_data(int i) const { return pos_cps[i].data(); }

    double* ori_knot_data(int i) { return ori_knots[i].data(); }
    const double* ori_knot_data(int i) const { return ori_knots[i].data(); }

    double* bias_data() { return biases.data(); }
    const double* bias_data() const { return biases.data(); }

    // ---- Accel / gyro bias helpers ------------------------------------------
    Eigen::Vector3d ba() const {
        return Eigen::Vector3d(biases[0], biases[1], biases[2]);
    }
    Eigen::Vector3d bg() const {
        return Eigen::Vector3d(biases[3], biases[4], biases[5]);
    }
};

}  // namespace rio
