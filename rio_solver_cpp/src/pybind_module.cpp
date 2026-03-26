#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <rio/solver.h>

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
        .def_readwrite("optimize_pitch_only", &SolverConfig::optimize_pitch_only)
        .def_readwrite("lambda_extrinsic_prior", &SolverConfig::lambda_extrinsic_prior)
        .def_readwrite("max_iterations", &SolverConfig::max_iterations)
        .def_readwrite("n_fix_leading_pos", &SolverConfig::n_fix_leading_pos)
        .def_readwrite("n_fix_leading_ori", &SolverConfig::n_fix_leading_ori);

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
        .def_readonly("solver_summary", &SolverResult::solver_summary)
        .def_readonly("time_residual_eval_s", &SolverResult::time_residual_eval_s)
        .def_readonly("time_jacobian_eval_s", &SolverResult::time_jacobian_eval_s)
        .def_readonly("time_linear_solver_s", &SolverResult::time_linear_solver_s);

    // ---- Main solve() interface (numpy-friendly) ----------------------------
    m.def("solve",
        [](const std::vector<RadarFrame>& radar_frames,
           const std::vector<ImuSample>& imu_samples,
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

            return solve(radar_frames, imu_samples, cfg, extrinsic,
                         init_pos_cps, init_ori_knots, init_biases, t_ref,
                         heading_samples);
        },
        py::arg("radar_frames"),
        py::arg("imu_samples"),
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
