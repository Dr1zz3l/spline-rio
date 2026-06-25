// pybind module for the GTSAM IncrementalFixedLagSmoother backend (Phase 2).
// Separate module (rio_isam) so the GTSAM-linked code stays isolated from the
// Ceres rio_solver module.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <rio/isam_sliding_window_solver.h>

namespace py = pybind11;
using namespace rio;

static std::vector<std::array<double, 3>> mat_to_cps(const Eigen::MatrixXd& m) {
    std::vector<std::array<double, 3>> out(m.rows());
    for (int i = 0; i < m.rows(); ++i) out[i] = {m(i, 0), m(i, 1), m(i, 2)};
    return out;
}
static std::vector<std::array<double, 4>> mat_to_quats(const Eigen::MatrixXd& m) {
    std::vector<std::array<double, 4>> out(m.rows());
    for (int i = 0; i < m.rows(); ++i) out[i] = {m(i, 0), m(i, 1), m(i, 2), m(i, 3)};
    return out;
}

PYBIND11_MODULE(rio_isam, m) {
    m.doc() = "GTSAM IncrementalFixedLagSmoother RIO backend (Phase 2)";

    py::class_<IsamConfig>(m, "IsamConfig")
        .def(py::init<>())
        .def_readwrite("dt_pos", &IsamConfig::dt_pos)
        .def_readwrite("dt_ori", &IsamConfig::dt_ori)
        .def_readwrite("lambda_accel", &IsamConfig::lambda_accel)
        .def_readwrite("lambda_gyro", &IsamConfig::lambda_gyro)
        .def_readwrite("huber_delta", &IsamConfig::huber_delta)
        .def_readwrite("lambda_snap_pos", &IsamConfig::lambda_snap_pos)
        .def_readwrite("lambda_ori_accel", &IsamConfig::lambda_ori_accel)
        .def_readwrite("lambda_heading", &IsamConfig::lambda_heading)
        .def_readwrite("lambda_bias_prior", &IsamConfig::lambda_bias_prior)
        .def_readwrite("bias_rw_sigma", &IsamConfig::bias_rw_sigma)
        .def_readwrite("boundary_sigma", &IsamConfig::boundary_sigma)
        .def_readwrite("min_range", &IsamConfig::min_range)
        .def_readwrite("lag", &IsamConfig::lag)
        .def_readwrite("relinearize_threshold", &IsamConfig::relinearize_threshold)
        .def_readwrite("relinearize_skip", &IsamConfig::relinearize_skip)
        .def_readwrite("extra_iters", &IsamConfig::extra_iters)
        .def_readwrite("use_qr", &IsamConfig::use_qr)
        .def_readwrite("fej", &IsamConfig::fej)
        .def_readwrite("warm_start_align", &IsamConfig::warm_start_align)
        .def_readwrite("adapt_noise_stride", &IsamConfig::adapt_noise_stride)
        .def_readwrite("adapt_noise_alpha", &IsamConfig::adapt_noise_alpha)
        .def_readwrite("lambda_gyro_omega_sigma", &IsamConfig::lambda_gyro_omega_sigma)
        .def_readwrite("lambda_gyro_omega_pow", &IsamConfig::lambda_gyro_omega_pow)
        .def_readwrite("omega_soft_sigma", &IsamConfig::omega_soft_sigma)
        .def_readwrite("accel_soft_sigma", &IsamConfig::accel_soft_sigma)
        .def_readwrite("radar_zbias_fixed", &IsamConfig::radar_zbias_fixed)
        .def_readwrite("radar_intensity_weight", &IsamConfig::radar_intensity_weight)
        .def_readwrite("lambda_pos_init_prior", &IsamConfig::lambda_pos_init_prior);

    py::class_<ExtrinsicConfig>(m, "ExtrinsicConfig")
        .def(py::init<>())
        .def_readwrite("roll_deg", &ExtrinsicConfig::roll_deg)
        .def_readwrite("pitch_deg", &ExtrinsicConfig::pitch_deg)
        .def_readwrite("yaw_deg", &ExtrinsicConfig::yaw_deg)
        .def_readwrite("tx", &ExtrinsicConfig::tx)
        .def_readwrite("ty", &ExtrinsicConfig::ty)
        .def_readwrite("tz", &ExtrinsicConfig::tz);

    py::class_<IsamSolver>(m, "IsamSolver")
        .def(py::init<const IsamConfig&, const ExtrinsicConfig&>())
        .def("initialize",
             [](IsamSolver& s, const Eigen::MatrixXd& pos_cps,
                const Eigen::MatrixXd& ori_quats, const Eigen::VectorXd& biases, double t_ref) {
                 std::array<double, 6> b{};
                 for (int i = 0; i < 6 && i < biases.size(); ++i) b[i] = biases[i];
                 s.initialize(mat_to_cps(pos_cps), mat_to_quats(ori_quats), b, t_ref);
             },
             py::arg("pos_cps"), py::arg("ori_quats"), py::arg("biases"), py::arg("t_ref"))
        .def("update",
             [](IsamSolver& s, const py::list& radar, const Eigen::MatrixXd& imu,
                const std::vector<std::pair<double, double>>& heading, double t_now) {
                 // radar: list of (timestamp, Nx5 [x,y,z,v,intensity])
                 std::vector<RadarFrame> frames;
                 for (const auto& item : radar) {
                     auto tup = item.cast<std::pair<double, Eigen::MatrixXd>>();
                     RadarFrame f; f.timestamp = tup.first;
                     const Eigen::MatrixXd& P = tup.second;
                     for (int i = 0; i < P.rows(); ++i) {
                         RadarPoint pt; pt.x = P(i, 0); pt.y = P(i, 1); pt.z = P(i, 2);
                         pt.v = P(i, 3); pt.intensity = (P.cols() > 4) ? P(i, 4) : 0.0;
                         f.points.push_back(pt);
                     }
                     frames.push_back(std::move(f));
                 }
                 std::vector<ImuSample> samples(imu.rows());
                 for (int i = 0; i < imu.rows(); ++i)
                     samples[i] = {imu(i, 0), imu(i, 1), imu(i, 2), imu(i, 3), imu(i, 4), imu(i, 5), imu(i, 6)};
                 return s.update(frames, samples, heading, t_now);
             },
             py::arg("radar"), py::arg("imu"), py::arg("heading"), py::arg("t_now"))
        .def("ori_knots", &IsamSolver::ori_knots)
        .def("pos_cps", &IsamSolver::pos_cps)
        .def("biases", &IsamSolver::biases)
        .def("num_active", &IsamSolver::num_active)
        .def("num_fixed", &IsamSolver::num_fixed);
}
