"""
Capture the EXACT batch problem that the C++ Ceres solver receives for a bag,
plus the C++ solved trajectory (the parity reference), into an .npz.

We do not reimplement the 2900-line driver's load/preunwrap/RANSAC/P1-P3 init.
Instead we monkeypatch validate_live_solver._solve_cpp to snapshot its inputs,
then let the real batch solve run (run_once -> main()).  The gtsam spike then
builds its factor graph from this snapshot and compares against the C++ result.

Usage (from analysis/):
    ../.venv/bin/python3 isam_spike/capture_problem.py slow_racing_best_velocity
Writes: isam_spike/_cache/<bag>_batch.npz
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, os.path.join(ANALYSIS, 'lib'))

import validate_live_solver as vls          # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

_CAPTURE = {}


def _capture_wrapper(orig):
    def wrapped(initial_state, solver_radar_frames, imu_data,
                extrinsics_cfg, solver_cfg, heading_priors=None):
        ori_spline = initial_state.ori_spline
        pos_bspline = initial_state.pos_bspline

        # --- init knots (basalt absolute-knot convention, xyzw) ---
        R_abs = ori_spline._base_rotations
        _CAPTURE['init_ori_quats'] = Rotation.from_matrix(R_abs).as_quat()
        _CAPTURE['init_pos_cps'] = pos_bspline.control_points.copy()
        _CAPTURE['init_biases'] = np.concatenate(
            [initial_state.acc_bias, initial_state.gyr_bias])
        _CAPTURE['t_ref'] = float(pos_bspline.t_ref)
        _CAPTURE['pos_degree'] = int(pos_bspline.degree)
        _CAPTURE['ori_dt'] = float(ori_spline.dt)
        _CAPTURE['pos_dt'] = float(pos_bspline.dt)

        # --- radar frames (post preunwrap + RANSAC) ---
        rf_ts, rf_pos, rf_vel, rf_int, rf_split = [], [], [], [], [0]
        for frame in solver_radar_frames:
            n = frame.num_points()
            if n == 0:
                continue
            rf_ts.append(float(frame.timestamp))
            rf_pos.append(np.asarray(frame.positions[:n], float))
            rf_vel.append(np.asarray(frame.velocities[:n], float))
            inten = (np.asarray(frame.intensities[:n], float)
                     if frame.intensities is not None else np.zeros(n))
            rf_int.append(inten)
            rf_split.append(rf_split[-1] + n)
        _CAPTURE['radar_ts'] = np.asarray(rf_ts)
        _CAPTURE['radar_pos'] = np.vstack(rf_pos) if rf_pos else np.zeros((0, 3))
        _CAPTURE['radar_vel'] = np.concatenate(rf_vel) if rf_vel else np.zeros(0)
        _CAPTURE['radar_int'] = np.concatenate(rf_int) if rf_int else np.zeros(0)
        _CAPTURE['radar_split'] = np.asarray(rf_split)   # CSR-style frame offsets

        # --- IMU (M,7): t, ax,ay,az, gx,gy,gz ---
        imu_np = np.zeros((len(imu_data), 7))
        for i, s in enumerate(imu_data):
            imu_np[i, 0] = s.timestamp
            imu_np[i, 1:4] = s.linear_acceleration
            imu_np[i, 4:7] = s.angular_velocity
        _CAPTURE['imu'] = imu_np

        # --- heading priors (K,2): t, yaw_rad ---
        if heading_priors:
            head = np.array([(float(t), float(np.arctan2(R[1, 0], R[0, 0])))
                             for t, R in heading_priors])
        else:
            head = np.zeros((0, 2))
        _CAPTURE['heading'] = head

        # --- extrinsics + full solver cfg ---
        _CAPTURE['ext_euler_deg'] = np.asarray(
            extrinsics_cfg.get('rotation_euler_deg', [180.0, 25.5, 0.0]), float)
        _CAPTURE['ext_trans_m'] = np.asarray(
            extrinsics_cfg.get('translation_body_m', [0.08, 0.02, -0.01]), float)
        _CAPTURE['solver_cfg'] = dict(solver_cfg)

        return orig(initial_state, solver_radar_frames, imu_data,
                    extrinsics_cfg, solver_cfg, heading_priors)
    return wrapped


def main():
    bag = sys.argv[1] if len(sys.argv) > 1 else 'slow_racing_best_velocity'
    vls._solve_cpp = _capture_wrapper(vls._solve_cpp)

    argv = [bag, '--mocap-yaw', '--cpp', '--no-plot']
    result = vls.run_once(argv)   # runs real batch solve; returns raw C++ SolverResult

    # C++ solved trajectory = parity reference
    _CAPTURE['cpp_pos_cps'] = np.asarray(result.pos_cps)
    _CAPTURE['cpp_ori_knots'] = np.asarray(result.ori_knots)   # xyzw
    _CAPTURE['cpp_biases'] = np.asarray(result.biases)
    _CAPTURE['cpp_ext_euler_deg'] = np.asarray(result.extrinsic_euler_deg)

    out_dir = os.path.join(HERE, '_cache')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{bag}_batch.npz')
    np.savez(out, **{k: (np.asarray(v, dtype=object) if isinstance(v, dict) else v)
                     for k, v in _CAPTURE.items()})

    print(f"\n[capture] wrote {out}")
    print(f"[capture]  pos_cps {_CAPTURE['init_pos_cps'].shape}"
          f"  ori_knots {_CAPTURE['init_ori_quats'].shape}"
          f"  biases {_CAPTURE['init_biases'].shape}")
    print(f"[capture]  radar frames {len(_CAPTURE['radar_ts'])}"
          f"  points {_CAPTURE['radar_pos'].shape[0]}"
          f"  imu {_CAPTURE['imu'].shape}  heading {_CAPTURE['heading'].shape}")
    print(f"[capture]  dt_pos={_CAPTURE['pos_dt']:.4f} dt_ori={_CAPTURE['ori_dt']:.4f}"
          f"  t_ref={_CAPTURE['t_ref']:.3f}  ext_euler={_CAPTURE['ext_euler_deg']}")


if __name__ == '__main__':
    main()
