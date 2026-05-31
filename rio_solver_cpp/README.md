# rio_solver_cpp — C++ Ceres RIO Solver

Phase 1 of the Python→C++ migration for the radar-inertial odometry solver.

## Status

- [x] **Phase 1**: Splines + all residual factors (build + tests passing)
- [x] **Phase 2**: Full problem + pybind11 integration with `validate_live_solver.py --cpp`
  - Current (with lever arm + extrinsic opt): slow 0.174 m/1.08°, fast 0.758 m/2.58°, backflips 1.817 m/8.31°
  - Phase 2 initial (pre-lever-arm): slow 0.358 m/2.53° (Python baseline: 0.374 m/3.32°)
- [x] **Phase 3**: Extrinsic pitch optimization, angular-acceleration regularization, lever-arm correction, full-rate IMU, per-bag config overrides
- [x] **Phase 4**: Sliding window + Schur complement marginalization (`--sliding-window`)
  - `SlidingWindowSolver` in `src/sliding_window_solver.cpp`
  - `MargPriorFunctor` in `include/rio/marginalization.h`
  - Per-window diagnostics: S condition number, eigenvalue range, tr(S⁻¹), tr(H_bb⁻¹)
  - Live-edge (settled) results: slow 0.393 m/2.21° (0.218 m/1.57°), fast 0.877 m/4.16° (0.804 m/3.10°)
- [ ] **Phase 5**: ROS node (real-time subscriber + odometry publisher)

## Build

```bash
# Dependencies (Ubuntu Noble)
sudo apt-get install libceres-dev libsuitesparse-dev libfmt-dev

# Build
cd rio_solver_cpp
mkdir -p build_release && cd build_release
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Or use the script:
./scripts/build.sh
./scripts/build.sh --install   # copies .so to analysis/
```

## Test

```bash
cd build_release
./test_spline        # 11 tests: spline eval + trajectory indexing
ctest --output-on-failure
```

## Python usage (after build --install)

```python
import sys; sys.path.insert(0, '../analysis')
import rio_solver

cfg = rio_solver.SolverConfig()
cfg.dt_pos = 0.005
cfg.lambda_gravity = 0.001

ext = rio_solver.ExtrinsicConfig()
ext.roll_deg = 180.0; ext.pitch_deg = 25.5; ext.yaw_deg = 0.0

# (Pass init_pos_cps, init_ori_quats, init_biases from Python P1-P3 init)
result = rio_solver.solve(radar_frames, imu_samples, cfg, ext,
                          init_pos_cps, init_ori_quats, init_biases, t_ref)

print(result.solver_summary)
print(f"Solve time: {result.solve_time_s:.2f}s")
```

## Architecture

- `include/rio/trajectory.h` — Trajectory state (pos CPs + ori quaternion knots + biases)
- `include/rio/factors/` — Ceres cost functors (radar Doppler, accel, gyro, gravity, heading, regularization)
- `include/rio/solver.h` — Public API
- `src/solver.cpp` — Problem construction + Ceres solve
- `src/pybind_module.cpp` — Python↔C++ bridge
- `tests/test_spline.cpp` — Phase 1 validation tests

## Dependencies

- Eigen 3.4+, Ceres 2.2+, Sophus (bundled in basalt-headers), basalt-headers, pybind11, fmt
- basalt-headers path: `../lie-spline-experiments/thirdparty/basalt/thirdparty/basalt-headers`
