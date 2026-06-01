#pragma once

#include <rio/solver.h>
#include <rio/trajectory.h>
#include <rio/marginalization.h>

#include <ceres/ceres.h>
#include <vector>
#include <utility>

namespace rio {

// ============================================================================
// SlidingWindowSolver
// ============================================================================
// Stateful fixed-lag smoother with Schur complement marginalization.
//
// Owns a persistent global Trajectory (all CPs/knots from t_ref to end).
// Each solve_window() call:
//   1. Builds a windowed Ceres problem (global index range)
//   2. Attaches the marginalization prior from the previous window
//   3. Solves with Ceres LM
//   4. Computes the new marginalization prior via Schur complement
//
// Marginalization: after each window, the "stride zone" CPs/knots are
// eliminated and their information is compressed into a dense Gaussian
// prior on the boundary CPs/knots + bias.  This prior is carried forward
// to the next window.
//
// Usage (from pybind / Python):
//   solver = SlidingWindowSolver(cfg, ext)
//   solver.initialize(all_pos_cps, all_ori_quats, all_biases, t_ref)
//   for each [t_start, t_end] window:
//       result = solver.solve_window(radar, imu, heading, t_start, t_end)
//   pos = solver.get_pos_cps()   // full global trajectory
//   ori = solver.get_ori_knots()
// ============================================================================
class SlidingWindowSolver {
public:
    SlidingWindowSolver(SolverConfig cfg, ExtrinsicConfig ext);

    // Initialize with the full P1-P3 trajectory (covers the entire dataset).
    // Call this once before any solve_window() calls.
    void initialize(
        const std::vector<std::array<double, 3>>& pos_cps,
        const std::vector<std::array<double, 4>>& ori_knots,
        const std::array<double, 6>& biases,
        double t_ref);

    // Solve one window [t_start, t_end].
    // stride: how far the window advances per call (seconds).  Used to
    //         determine which CPs to marginalize.
    // Returns the SolverResult for the committed portion.
    SolverResult solve_window(
        const std::vector<RadarFrame>& radar_frames,
        const std::vector<ImuSample>& imu_samples,
        const std::vector<PreintFactor>& preint_factors,
        const std::vector<std::pair<double, double>>& heading_samples,
        double t_start, double t_end, double stride);

    // ---- Accessors for the full global trajectory ----------------------------
    const std::vector<std::array<double, 3>>& pos_cps()   const { return traj_.pos_cps; }
    const std::vector<std::array<double, 4>>& ori_knots() const { return traj_.ori_knots; }
    const std::array<double, 6>&              biases()    const { return traj_.biases; }

private:
    SolverConfig    cfg_;
    ExtrinsicConfig ext_;
    Trajectory      traj_;
    bool            initialized_{false};

    // Marginalization prior (carried across window advances)
    MarginalizationPrior prior_;

    // Initial biases from stationary detection (captured in initialize()).
    // Used as the absolute anchor for the per-window bias prior.
    // The marg prior is correctly re-centered every window (curvature-only).
    // The bias prior must NOT be re-centered — it is an absolute measurement
    // anchor tied to the stationary calibration estimate.
    std::array<double, 6> init_biases_{};

    // Initial position CPs from P1-P3 radar-velocity integration (captured in
    // initialize()).  Used as the anchor for lambda_pos_init_prior when > 0:
    // each window CP is softly pinned to its init value, preventing radar-sparse
    // windows from drifting position while orientation is being refined.
    // Same discipline as init_biases_: anchors to the init, never the warm-start.
    std::vector<std::array<double, 3>> init_pos_cps_;

    // Add the marginalization prior to a Ceres problem.
    // Connects to traj_ indices [prior_.pos_start, +n_bound_pos) etc.
    void add_prior_to_problem(ceres::Problem& problem);

    // Compute new marginalization prior after solving a window.
    // pi0_raw / oi0_raw: first "active" (non-extended-leading) index.
    // k_stride_pos / k_stride_ori: number of CP/knot slots per stride.
    void compute_prior(ceres::Problem& problem,
                       int pi0_raw, int oi0_raw,
                       int k_stride_pos, int k_stride_ori);
};

}  // namespace rio
