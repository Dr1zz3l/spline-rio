# rio_solver_cpp — C++ Ceres RIO Solver

Phase 1 of the Python→C++ migration for the radar-inertial odometry solver.

## Status

- [x] **Phase 1**: Splines + all residual factors (build + tests passing)
- [x] **Phase 2**: Full problem + pybind11 integration with `validate_live_solver.py --cpp`
  - pos 0.358m / ori 2.53° on slow_racing_best_velocity (Python baseline: 0.374m / 3.32°)
  - Bugs fixed: gravity sign, gyro bias sign, boundary priors (vel+ori added, anchor corrected)
- [ ] **Phase 3**: Performance + preintegrated IMU factor
- [ ] **Phase 4**: Sliding window + ROS node

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
