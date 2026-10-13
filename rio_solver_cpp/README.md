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

There is also a **second, parallel backend**: a GTSAM `IncrementalFixedLagSmoother`
(iSAM2) on branch `isam2-backend`, exposed as `--isam` instead of `--sliding-window`.
It beats this Ceres SW on racing (accuracy + speed) and matches it on backflips
position/velocity. See **[iSAM2 / GTSAM backend](#isam2--gtsam-backend---isam)** below
and `documentation/ISAM2_MIGRATION.md` for the full record.

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

## iSAM2 / GTSAM backend (`--isam`)

The `rio_isam` pybind module (`src/isam_sliding_window_solver.cpp`,
`include/rio/gtsam/`) is the incremental GTSAM backend. It is built by the **same**
`cmake --build .` above, but **only if the vendored GTSAM is present**: the CMake
auto-detects `gtsam_vendored_install/libgtsam_unstable.so` and skips the `rio_isam`
target otherwise (the Ceres build is unaffected).

**Why vendored:** apt `libgtsam-dev` ships no `gtsam_unstable` (no
`IncrementalFixedLagSmoother`) and is baseline-x86-64, which collides with our
`-march=native` Eigen alignment (heap corruption). Build GTSAM 4.2.1 from source
with matching flags (the install dir is gitignored, so a fresh checkout must
rebuild it):

```bash
git clone --branch 4.2.1 --depth 1 https://github.com/borglab/gtsam.git
cd gtsam && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_BUILD_UNSTABLE=ON \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_WITH_TBB=ON \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=ON \
  -DGTSAM_BUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=<repo>/rio_solver_cpp/gtsam_vendored_install
make -j$(nproc) install
# then rebuild rio_solver_cpp: cmake .. && cmake --build . --target rio_isam -j$(nproc)
```

CMake links `libgtsam.so` directly (not `find_package`, because Ubuntu's
`GTSAMConfig.cmake` is broken) and sets RPATH to the vendored lib dir for the bundled
`libmetis-gtsam.so`.
Run it from `analysis/` (parallel to `--sliding-window`):

```bash
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity \
  --mocap-yaw --cpp --isam --set lambda_heading=10 \
  --set window_duration=1.5 --set extra_iters=3
```

Opt-in accuracy/diagnostic flags (all default off, one config across bags): floor
anchor `--set lambda_floor=15 --set floor_free=1 --set floor_cluster=1`, online
z-bias `--set radar_zbias_estimate=1`, adaptive lag `--set lag_flip=8`, NEES/
consistency `--set nees=1`. See `documentation/ISAM2_MIGRATION.md` (FINAL SUMMARY +
per-bag config) and `documentation/ALGO_IMPROVEMENTS.md` (D5/D6).

## Python usage (after build --install)

```python
import sys; sys.path.insert(0, '../analysis')
import rio_solver

cfg = rio_solver.SolverConfig()
cfg.dt_pos = 0.005
cfg.lambda_gravity = 0.0   # gravity-direction (tilt) factor, DISABLED. Inert on the
                           # racing/backflips headline flights (near-hover gate never trips;
                           # verified bit-identical); the full-stride gravity=0 ICINS re-run
                           # reproduces the published baselines to within run-to-run noise, so
                           # it is omitted from the paper and off by default.

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
- `include/rio/factors/` — Ceres cost functors (radar Doppler, accel, gyro, gravity [off by default `lambda_gravity=0`; inert on the headline flights, omitted from the paper], heading, regularization)
- `include/rio/solver.h` — Public API
- `src/solver.cpp` — Problem construction + Ceres solve
- `src/pybind_module.cpp` — Python↔C++ bridge
- `tests/test_spline.cpp` — Phase 1 validation tests

## Dependencies

- Eigen 3.4+, Ceres 2.2+, Sophus (bundled in basalt-headers), basalt-headers, pybind11, fmt
- basalt-headers path: `../lie-spline-experiments/thirdparty/basalt/thirdparty/basalt-headers`
