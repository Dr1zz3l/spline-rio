#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <rio/solver.h>
#include <rio/sliding_window_solver.h>

namespace py = pybind11;
using namespace rio;

// ============================================================================
// pybind11 module: rio_solver
// ============================================================================
// Python usage:
//
//   import sys; sys.path.insert(0, 'path/to/analysis')
//   import rio_solver
//
//   result = rio_solver.solve(
//       radar_frames,      # list of dicts: {'timestamp': float, 'points': Nx4 array}
//       imu_np,            # Nx7 numpy array [t, ax,ay,az, gx,gy,gz]
//       config,            # dict matching SolverConfig fields
//       extrinsic,         # dict: {roll_deg, pitch_deg, yaw_deg, tx, ty, tz}
//       init_pos_cps,      # Nx3 numpy array
//       init_ori_quats,    # Nx4 numpy array (x,y,z,w Sophus convention)
//       init_biases,       # length-6 array [bax,bay,baz, bgx,bgy,bgz]
//       t_ref,             # float
//       heading_samples,   # list of (timestamp, yaw_rad) pairs (optional)
//   )
//
//   result.pos_cps         # Nx3 numpy array
//   result.ori_knots       # Nx4 numpy array
//   result.biases          # length-6 array
//   result.cost_history    # list of floats
//   result.solve_time_s    # float
//   result.solver_summary  # string

// ---- Convert numpy Nx3 → vector<array<double,3>> ---------------------------
static std::vector<std::array<double, 3>> np_to_pos_cps(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
    auto r = arr.unchecked<2>();
    int n = static_cast<int>(r.shape(0));
    std::vector<std::array<double, 3>> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = {r(i, 0), r(i, 1), r(i, 2)};
    return out;
}

// ---- Convert numpy Nx4 → vector<array<double,4>> ---------------------------
static std::vector<std::array<double, 4>> np_to_ori_knots(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
    auto r = arr.unchecked<2>();
    int n = static_cast<int>(r.shape(0));
    std::vector<std::array<double, 4>> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = {r(i, 0), r(i, 1), r(i, 2), r(i, 3)};
    return out;
}

// ---- Convert vector<array<double,3>> → numpy Nx3 ---------------------------
static py::array_t<double> pos_cps_to_np(
        const std::vector<std::array<double, 3>>& v) {
    py::array_t<double> out({(int)v.size(), 3});
    auto r = out.mutable_unchecked<2>();
    for (int i = 0; i < (int)v.size(); ++i) {
        r(i, 0) = v[i][0];
        r(i, 1) = v[i][1];
        r(i, 2) = v[i][2];
    }
    return out;
}

// ---- Convert vector<array<double,4>> → numpy Nx4 ---------------------------
static py::array_t<double> ori_knots_to_np(
        const std::vector<std::array<double, 4>>& v) {
    py::array_t<double> out({(int)v.size(), 4});
    auto r = out.mutable_unchecked<2>();
    for (int i = 0; i < (int)v.size(); ++i) {
        r(i, 0) = v[i][0];
        r(i, 1) = v[i][1];
        r(i, 2) = v[i][2];
        r(i, 3) = v[i][3];
    }
    return out;
}

PYBIND11_MODULE(rio_solver, m) {
    m.doc() = "Rio C++ Ceres solver — Python bridge";

    // ---- BlockMapEntry ------------------------------------------------------
    py::class_<BlockMapEntry>(m, "BlockMapEntry")
        .def_readonly("type_id",     &BlockMapEntry::type_id)
        .def_readonly("index",       &BlockMapEntry::index)
        .def_readonly("col_offset",  &BlockMapEntry::col_offset)
        .def_readonly("tangent_size",&BlockMapEntry::tangent_size);

    // ---- SystemDump ---------------------------------------------------------
    // Linearized system snapshot (J, r, grad) at either the warm-start or the
    // converged point.  Populated when SolverConfig.dump_system = True.
    // Reconstruct H = JᵀJ in Python via:
    //   import scipy.sparse as sp
    //   J = sp.csr_matrix((dump.jac_values, dump.jac_cols, dump.jac_row_ptr),
    //                     shape=(dump.jac_num_rows, dump.jac_num_cols))
    //   H = J.T @ J     # (n_cols × n_cols) normal-equation matrix
    //   g = J.T @ dump.residuals   # gradient (use -g for RHS of Hδx = -g)
    py::class_<SolverResult::SystemDump>(m, "SystemDump")
        .def_readonly("valid",        &SolverResult::SystemDump::valid)
        .def_readonly("jac_num_rows", &SolverResult::SystemDump::jac_num_rows)
        .def_readonly("jac_num_cols", &SolverResult::SystemDump::jac_num_cols)
        .def_property_readonly("jac_values",
            [](const SolverResult::SystemDump& d) {
                return py::array_t<double>(d.jac_values.size(), d.jac_values.data());
            })
        .def_property_readonly("jac_cols",
            [](const SolverResult::SystemDump& d) {
                return py::array_t<int>(d.jac_cols.size(), d.jac_cols.data());
            })
        .def_property_readonly("jac_row_ptr",
            [](const SolverResult::SystemDump& d) {
                return py::array_t<int>(d.jac_row_ptr.size(), d.jac_row_ptr.data());
            })
        .def_property_readonly("residuals",
            [](const SolverResult::SystemDump& d) {
                return py::array_t<double>(d.residuals.size(), d.residuals.data());
            })
        .def_property_readonly("gradient",
            [](const SolverResult::SystemDump& d) {
                return py::array_t<double>(d.gradient.size(), d.gradient.data());
            })
        .def_readonly("block_map", &SolverResult::SystemDump::block_map);

    // ---- SolverConfig -------------------------------------------------------
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("dt_pos", &SolverConfig::dt_pos)
        .def_readwrite("dt_ori", &SolverConfig::dt_ori)
        .def_readwrite("huber_delta", &SolverConfig::huber_delta)
        .def_readwrite("min_range", &SolverConfig::min_range)
        .def_readwrite("lambda_accel", &SolverConfig::lambda_accel)
        .def_readwrite("lambda_gyro", &SolverConfig::lambda_gyro)
        .def_readwrite("huber_delta_accel", &SolverConfig::huber_delta_accel)
        .def_readwrite("lambda_snap_pos", &SolverConfig::lambda_snap_pos)
        .def_readwrite("lambda_ori_reg", &SolverConfig::lambda_ori_reg)
        .def_readwrite("lambda_ori_accel", &SolverConfig::lambda_ori_accel)
        .def_readwrite("lambda_gravity", &SolverConfig::lambda_gravity)
        .def_readwrite("gravity_accel_threshold", &SolverConfig::gravity_accel_threshold)
        .def_readwrite("lambda_heading", &SolverConfig::lambda_heading)
        .def_readwrite("lambda_bias_prior_accel", &SolverConfig::lambda_bias_prior_accel)
        .def_readwrite("lambda_bias_prior_gyro", &SolverConfig::lambda_bias_prior_gyro)
        .def_readwrite("lambda_boundary_pos", &SolverConfig::lambda_boundary_pos)
        .def_readwrite("lambda_boundary_vel", &SolverConfig::lambda_boundary_vel)
        .def_readwrite("lambda_boundary_ori", &SolverConfig::lambda_boundary_ori)
        .def_readwrite("lambda_boundary_ori_yaw", &SolverConfig::lambda_boundary_ori_yaw)
        .def_readwrite("lock_extrinsics", &SolverConfig::lock_extrinsics)
        .def_readwrite("lock_gyro_bias", &SolverConfig::lock_gyro_bias)
        .def_readwrite("lambda_pos_init_prior", &SolverConfig::lambda_pos_init_prior)
        .def_readwrite("omega_gate_threshold", &SolverConfig::omega_gate_threshold)
        .def_readwrite("optimize_pitch_only", &SolverConfig::optimize_pitch_only)
        .def_readwrite("lambda_extrinsic_prior", &SolverConfig::lambda_extrinsic_prior)
        .def_readwrite("max_iterations", &SolverConfig::max_iterations)
        .def_readwrite("num_threads", &SolverConfig::num_threads)
        .def_readwrite("n_fix_leading_pos", &SolverConfig::n_fix_leading_pos)
        .def_readwrite("n_fix_leading_ori", &SolverConfig::n_fix_leading_ori)
        .def_readwrite("marg_prior_scale", &SolverConfig::marg_prior_scale)
        .def_readwrite("use_adaptive_marg_scale", &SolverConfig::use_adaptive_marg_scale)
        .def_readwrite("marg_prior_cauchy_delta", &SolverConfig::marg_prior_cauchy_delta)
        .def_readwrite("marg_prior_eig_clip",    &SolverConfig::marg_prior_eig_clip)
        .def_readwrite("use_preintegration", &SolverConfig::use_preintegration)
        .def_readwrite("lambda_preint",   &SolverConfig::lambda_preint)
        .def_readwrite("lambda_preint_v", &SolverConfig::lambda_preint_v)
        .def_readwrite("lambda_preint_p", &SolverConfig::lambda_preint_p)
        .def_readwrite("preint_hz",       &SolverConfig::preint_hz)
        .def_readwrite("dump_system",     &SolverConfig::dump_system)
        .def_readwrite("use_banded_schur", &SolverConfig::use_banded_schur);

    // ---- ExtrinsicConfig ----------------------------------------------------
    py::class_<ExtrinsicConfig>(m, "ExtrinsicConfig")
        .def(py::init<>())
        .def_readwrite("roll_deg",  &ExtrinsicConfig::roll_deg)
        .def_readwrite("pitch_deg", &ExtrinsicConfig::pitch_deg)
        .def_readwrite("yaw_deg",   &ExtrinsicConfig::yaw_deg)
        .def_readwrite("tx", &ExtrinsicConfig::tx)
        .def_readwrite("ty", &ExtrinsicConfig::ty)
        .def_readwrite("tz", &ExtrinsicConfig::tz);

    // ---- RadarPoint, RadarFrame, ImuSample ----------------------------------
    py::class_<RadarPoint>(m, "RadarPoint")
        .def(py::init<>())
        .def_readwrite("x", &RadarPoint::x)
        .def_readwrite("y", &RadarPoint::y)
        .def_readwrite("z", &RadarPoint::z)
        .def_readwrite("v", &RadarPoint::v);

    py::class_<RadarFrame>(m, "RadarFrame")
        .def(py::init<>())
        .def_readwrite("timestamp", &RadarFrame::timestamp)
        .def_readwrite("points", &RadarFrame::points);

    py::class_<ImuSample>(m, "ImuSample")
        .def(py::init<>())
        .def_readwrite("timestamp", &ImuSample::timestamp)
        .def_readwrite("ax", &ImuSample::ax)
        .def_readwrite("ay", &ImuSample::ay)
        .def_readwrite("az", &ImuSample::az)
        .def_readwrite("gx", &ImuSample::gx)
        .def_readwrite("gy", &ImuSample::gy)
        .def_readwrite("gz", &ImuSample::gz);

    py::class_<PreintFactor>(m, "PreintFactor")
        .def(py::init<>())
        .def_readwrite("t_i", &PreintFactor::t_i)
        .def_readwrite("t_j", &PreintFactor::t_j)
        .def_readwrite("dt",  &PreintFactor::dt)
        .def_readwrite("delta_R",   &PreintFactor::delta_R)
        .def_readwrite("delta_v",   &PreintFactor::delta_v)
        .def_readwrite("delta_p",   &PreintFactor::delta_p)
        .def_readwrite("b_a0",      &PreintFactor::b_a0)
        .def_readwrite("b_g0",      &PreintFactor::b_g0)
        .def_readwrite("d_R_d_bg",  &PreintFactor::d_R_d_bg)
        .def_readwrite("d_v_d_ba",  &PreintFactor::d_v_d_ba)
        .def_readwrite("d_v_d_bg",  &PreintFactor::d_v_d_bg)
        .def_readwrite("d_p_d_ba",  &PreintFactor::d_p_d_ba)
        .def_readwrite("d_p_d_bg",  &PreintFactor::d_p_d_bg);

    // ---- SolverResult -------------------------------------------------------
    py::class_<SolverResult>(m, "SolverResult")
        .def_property_readonly("pos_cps",
            [](const SolverResult& r) { return pos_cps_to_np(r.pos_cps); })
        .def_property_readonly("ori_knots",
            [](const SolverResult& r) { return ori_knots_to_np(r.ori_knots); })
        .def_property_readonly("biases",
            [](const SolverResult& r) -> py::array_t<double> {
                py::array_t<double> out(6);
                auto buf = out.mutable_unchecked<1>();
                for (int i = 0; i < 6; ++i) buf(i) = r.biases[i];
                return out;
            })
        .def_property_readonly("extrinsic_euler_deg",
            [](const SolverResult& r) -> py::array_t<double> {
                py::array_t<double> out(3);
                auto buf = out.mutable_unchecked<1>();
                for (int i = 0; i < 3; ++i) buf(i) = r.extrinsic_euler_deg[i];
                return out;
            })
        .def_readonly("cost_history", &SolverResult::cost_history)
        .def_readonly("solve_time_s", &SolverResult::solve_time_s)
        .def_readonly("num_iterations", &SolverResult::num_iterations)
        .def_readonly("solver_summary", &SolverResult::solver_summary)
        .def_readonly("time_residual_eval_s", &SolverResult::time_residual_eval_s)
        .def_readonly("time_jacobian_eval_s", &SolverResult::time_jacobian_eval_s)
        .def_readonly("time_linear_solver_s", &SolverResult::time_linear_solver_s)
        .def_readonly("marg_prior_valid",     &SolverResult::marg_prior_valid)
        .def_readonly("marg_prior_dim",       &SolverResult::marg_prior_dim)
        .def_readonly("marg_cond_number",     &SolverResult::marg_cond_number)
        .def_readonly("marg_min_eigenvalue",  &SolverResult::marg_min_eigenvalue)
        .def_readonly("marg_max_eigenvalue",  &SolverResult::marg_max_eigenvalue)
        .def_readonly("marg_numerical_rank",  &SolverResult::marg_numerical_rank)
        .def_readonly("marg_drop_reason",      &SolverResult::marg_drop_reason)
        .def_readonly("marg_trace_cov",             &SolverResult::marg_trace_cov)
        .def_readonly("marg_adaptive_scale",        &SolverResult::marg_adaptive_scale)
        .def_readonly("marg_applied_scale",         &SolverResult::marg_applied_scale)
        .def_readonly("marg_prior_residual_norm",   &SolverResult::marg_prior_residual_norm)
        .def_readonly("boundary_cov_valid",    &SolverResult::boundary_cov_valid)
        .def_readonly("boundary_cov_trace",    &SolverResult::boundary_cov_trace)
        .def_readonly("window_cov_trace",      &SolverResult::window_cov_trace)
        .def_property_readonly("boundary_covariance",
            [](const SolverResult& r) -> py::object {
                if (!r.boundary_cov_valid) return py::none();
                return py::cast(r.boundary_covariance);
            })
        .def_property_readonly("window_covariance",
            [](const SolverResult& r) -> py::object {
                if (!r.boundary_cov_valid) return py::none();
                return py::cast(r.window_covariance);
            })
        // ---- dump_system snapshots ------------------------------------------
        .def_readonly("dump_pre",  &SolverResult::dump_pre)
        .def_readonly("dump_post", &SolverResult::dump_post);

    // ---- Main solve() interface (numpy-friendly) ----------------------------
    m.def("solve",
        [](const std::vector<RadarFrame>& radar_frames,
           const std::vector<ImuSample>& imu_samples,
           const std::vector<PreintFactor>& preint_factors,
           const SolverConfig& cfg,
           const ExtrinsicConfig& extrinsic,
           py::array_t<double, py::array::c_style | py::array::forcecast> init_pos_cps_np,
           py::array_t<double, py::array::c_style | py::array::forcecast> init_ori_quats_np,
           py::array_t<double, py::array::c_style | py::array::forcecast> init_biases_np,
           double t_ref,
           const std::vector<std::pair<double, double>>& heading_samples) {

            auto init_pos_cps   = np_to_pos_cps(init_pos_cps_np);
            auto init_ori_knots = np_to_ori_knots(init_ori_quats_np);

            std::array<double, 6> init_biases{};
            {
                auto r = init_biases_np.unchecked<1>();
                for (int i = 0; i < 6; ++i) init_biases[i] = r(i);
            }

            return solve(radar_frames, imu_samples, preint_factors, cfg, extrinsic,
                         init_pos_cps, init_ori_knots, init_biases, t_ref,
                         heading_samples);
        },
        py::arg("radar_frames"),
        py::arg("imu_samples"),
        py::arg("preint_factors") = std::vector<PreintFactor>{},
        py::arg("cfg"),
        py::arg("extrinsic"),
        py::arg("init_pos_cps"),
        py::arg("init_ori_quats"),
        py::arg("init_biases"),
        py::arg("t_ref"),
        py::arg("heading_samples") = std::vector<std::pair<double, double>>{},
        "Run the C++ Ceres RIO solver. Returns SolverResult."
    );

    // ---- Convenience: convert Python radar data (list of dicts) to RadarFrame
    m.def("make_radar_frame",
        [](double timestamp,
           py::array_t<double, py::array::c_style | py::array::forcecast> pts) {
            RadarFrame frame;
            frame.timestamp = timestamp;
            auto r = pts.unchecked<2>();
            int n = static_cast<int>(r.shape(0));
            frame.points.resize(n);
            for (int i = 0; i < n; ++i) {
                frame.points[i].x = r(i, 0);
                frame.points[i].y = r(i, 1);
                frame.points[i].z = r(i, 2);
                frame.points[i].v = r(i, 3);
            }
            return frame;
        },
        py::arg("timestamp"), py::arg("points_nx4"),
        "Create a RadarFrame from a Nx4 numpy array [x,y,z,v].");

    // ---- SlidingWindowSolver ------------------------------------------------
    py::class_<SlidingWindowSolver>(m, "SlidingWindowSolver")
        .def(py::init<SolverConfig, ExtrinsicConfig>(),
             py::arg("cfg"), py::arg("ext"))
        .def("initialize",
            [](SlidingWindowSolver& self,
               py::array_t<double, py::array::c_style | py::array::forcecast> pos_cps_np,
               py::array_t<double, py::array::c_style | py::array::forcecast> ori_knots_np,
               py::array_t<double, py::array::c_style | py::array::forcecast> biases_np,
               double t_ref) {
                auto pos_cps   = np_to_pos_cps(pos_cps_np);
                auto ori_knots = np_to_ori_knots(ori_knots_np);
                std::array<double, 6> biases{};
                {
                    auto r = biases_np.unchecked<1>();
                    for (int i = 0; i < 6; ++i) biases[i] = r(i);
                }
                self.initialize(pos_cps, ori_knots, biases, t_ref);
            },
            py::arg("pos_cps"), py::arg("ori_knots"), py::arg("biases"), py::arg("t_ref"),
            "Initialize with full P1-P3 trajectory. Call once before solve_window().")
        .def("solve_window",
            [](SlidingWindowSolver& self,
               const std::vector<RadarFrame>& radar_frames,
               const std::vector<ImuSample>& imu_samples,
               const std::vector<PreintFactor>& preint_factors,
               const std::vector<std::pair<double, double>>& heading_samples,
               double t_start, double t_end, double stride) {
                return self.solve_window(radar_frames, imu_samples, preint_factors,
                                         heading_samples, t_start, t_end, stride);
            },
            py::arg("radar_frames"), py::arg("imu_samples"),
            py::arg("preint_factors") = std::vector<PreintFactor>{},
            py::arg("heading_samples"),
            py::arg("t_start"), py::arg("t_end"), py::arg("stride"),
            "Solve one window [t_start, t_end]. Returns SolverResult.")
        .def_property_readonly("pos_cps",
            [](const SlidingWindowSolver& s) { return pos_cps_to_np(s.pos_cps()); })
        .def_property_readonly("ori_knots",
            [](const SlidingWindowSolver& s) { return ori_knots_to_np(s.ori_knots()); })
        .def_property_readonly("biases",
            [](const SlidingWindowSolver& s) -> py::array_t<double> {
                py::array_t<double> out(6);
                auto buf = out.mutable_unchecked<1>();
                for (int i = 0; i < 6; ++i) buf(i) = s.biases()[i];
                return out;
            });

    m.def("make_imu_samples",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> imu_np) {
            auto r = imu_np.unchecked<2>();
            int n = static_cast<int>(r.shape(0));
            std::vector<ImuSample> out(n);
            for (int i = 0; i < n; ++i) {
                out[i].timestamp = r(i, 0);
                out[i].ax = r(i, 1); out[i].ay = r(i, 2); out[i].az = r(i, 3);
                out[i].gx = r(i, 4); out[i].gy = r(i, 5); out[i].gz = r(i, 6);
            }
            return out;
        },
        py::arg("imu_nx7"),
        "Create ImuSample list from a Nx7 numpy array [t, ax,ay,az, gx,gy,gz].");
}
