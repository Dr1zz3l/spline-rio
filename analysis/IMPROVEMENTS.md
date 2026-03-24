# Live Solver Improvement Roadmap

## Status

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Decouple `--mocap-yaw` into `--mocap-init` / `--mocap-heading` | ✅ Done | `--mocap-yaw` still works as shorthand |
| 2 | Yaw-free boundary orientation priors (`lambda_boundary_ori_yaw`) | ✅ Done | Default 0.0 = free yaw gauge; SE3 alignment recovers global frame |
| 3 | Proper B-spline LS init (`build_position_spline_from_radar_integration`) | ✅ Done | Replaced interpolation with `min ||B @ c - p||²` |
| A1 | IMU preintegration (`--preintegrate` flag) | ⏳ Next | See below |
| A2 | Multi-bag evaluation script | ⏳ Next | `eval_bags.py` |
| A3 | GNC outlier rejection | 🔜 Planned | Yang et al. RA-L 2020 |

---

## Python Phase Remaining (~1 week)

### A1. IMU Preintegration (`--preintegrate` flag) — 2-3 days

**Problem**: ~5000 per-sample IMU residuals at 200 Hz dominate solver cost and future real-time latency.

**Fix**: Preintegrate IMU between consecutive radar frames (~10 Hz), yielding ~200 preintegrated 9D factors (ΔR, Δv, Δp) instead of ~5000 3D residuals. Each factor has first-order bias correction Jacobians so bias updates don't require re-integration.

**Implementation**:
- `codegen/derive_jacobians_symforce.py`: add `preintegrated_imu_residual()` — SymForce-derived Jacobians for guaranteed correctness
- `lib/imu_preintegration.py`: `PreintegratedIMUMeasurement` + `preintegrate(imu_samples, ba, bg)` — must use `imu_data_full` (original rate), NOT the downsampled `imu_data`
- `validate_nonlinear_solver.py`: new `preintegrated_imu_factors` path in `compute_jacobian_analytical` / `compute_residuals_only` / `solve_trajectory_nonlinear`
- `validate_live_solver.py`: `--preintegrate` flag, pass `imu_data_full` to preintegration

**IMU downsampling note**: Both `validate_live_solver.py:628-630` and `validate_nonlinear_solver.py:2003-2005` downsample IMU to ~200 Hz for the per-sample solver. The preintegration must use the full-rate `imu_data_full` to avoid integration error.

**Validation**: RMSE within ~5% of per-sample baseline; solver wall-clock time significantly reduced.

**Literature**: Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" (TRO 2017)

---

### A2. Multi-Bag Evaluation Script — 0.5 day

`analysis/eval_bags.py`: runs `validate_live_solver.py` on a set of bags, parses RMSE from stdout, saves JSON result. Used to establish baselines and compare against preintegration.

---

### A3. GNC Outlier Rejection — 1 day (optional)

Replace fixed Huber loss with Graduated Non-Convexity: anneal shape parameter in outer loop over LM iterations. Better rejection of structured Doppler outliers (aliased returns that survive unwrapping). ~50-line change to `solve_trajectory_nonlinear`.

**Literature**: Yang et al., "Graduated Non-Convexity for Robust Spatial Perception" (RA-L 2020)

---

## C++ Migration (After Python Phase)

### B1. Ceres Factor Library (1 week)

Port factor types as `ceres::CostFunction`:
- `RadarDopplerFactor` (1D residual)
- `PreintegratedIMUFactor` (9D residual: ΔR, Δv, Δp)
- `BiasPriorFactor` (6D)
- `SplineRegularization` (min-snap pos, increment-penalty ori)

### B2. C++ B-spline + SO(3) Spline (1 week)

Port `UniformBSpline` and `CumulativeSO3BSpline` (Eigen only). Omega knots are ℝ³ → standard Euclidean parameterization.

### B3. Sliding Window (1-2 weeks)

- Maintain `ceres::Problem` for current window (1-2 s)
- On window advance: marginalize old CPs/knots via Schur complement prior
- Cumulative SO(3) anchor: freeze `R_base` at window boundary
- Biases stay active across all windows

**Literature**:
- OKVIS: Leutenegger et al., IJRR 2015
- VINS-Mono: Qin et al., TRO 2018
- CLIC (continuous-time): Lv et al., IROS 2023

### B4. ROS Node (3-5 days)

Subscribers for `/mmWaveDataHdl/RScanVelocity` + `/agiros_pilot/imu`. Publish odometry at ~10 Hz.

---

## Architecture Note: Why Not Python Sliding Window

The cumulative SO(3) B-spline has a left-triangular Jacobian structure (each R(t) depends on all prior Ω knots via `R_base`). The `ori_base_jacobian_window=20` parameter already approximates the sliding window anchor approach. No novel algorithm to validate — skip Python prototype, implement directly in Ceres where marginalization infrastructure exists natively.
