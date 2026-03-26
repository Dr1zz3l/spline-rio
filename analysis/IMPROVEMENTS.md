# Live Solver Improvement Roadmap

## Status

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Decouple `--mocap-yaw` into `--mocap-init` / `--mocap-heading` | ✅ Done | `--mocap-yaw` still works as shorthand |
| 2 | Yaw-free boundary orientation priors (`lambda_boundary_ori_yaw`) | ✅ Done | Default 0.0 = free yaw gauge |
| 3 | Position B-spline init from radar dead-reckoning | ✅ Done | Linear interp at CP times; LS fit was 14× underdetermined and reverted |
| A1 | IMU preintegration (`--preintegrate` flag) | ✅ Done | Forster TRO-2017; replaces accel rows, keeps gyro; see FINDINGS_PREINTEGRATION.md |
| A2 | Multi-bag evaluation script | ✅ Done | `eval_bags.py`; results in `eval_results/` |
| A3 | GNC outlier rejection | ✅ Done | `--gnc` flag; Geman-McClure loss with phase annealing. Does NOT improve results with corrected extrinsics — avoid. |
| A4 | Fix extrinsic roll/yaw drift | ✅ Done | `optimize_pitch_only: true`; was inflating ori RMSE (9.5° → 5.8° slow, 5.7° → 4.2° fast) |

---

## C++ Migration (after Python phase)

### B1. Ceres Factor Library

Port factor types as `ceres::CostFunction`:
- `RadarDopplerFactor` (1D residual)
- `PreintegratedIMUFactor` (9D residual: ΔR, Δv, Δp)
- `BiasPriorFactor` (6D)
- `SplineRegularization` (min-snap pos, increment-penalty ori)

### B2. C++ B-spline + SO(3) Spline

Port `UniformBSpline` and `CumulativeSO3BSpline` (Eigen only).
Omega knots are ℝ³ → standard Euclidean parameterisation.

### B3. Sliding Window

- Maintain `ceres::Problem` for current window (1–2 s)
- On window advance: marginalise old CPs/knots via Schur complement prior
- Cumulative SO(3) anchor: freeze `R_base` at window boundary
- Biases stay active across all windows

**Literature**: OKVIS (Leutenegger et al., IJRR 2015), VINS-Mono (Qin et al.,
TRO 2018), CLIC (Lv et al., IROS 2023)

### B4. ROS Node

Subscribers for `/mmWaveDataHdl/RScanVelocity` + `/agiros_pilot/imu`.
Publish odometry at ~10 Hz.

---

## Architecture note: why not Python sliding window

The cumulative SO(3) B-spline has a left-triangular Jacobian structure (each
R(t) depends on all prior Ω knots via `R_base`). The `ori_base_jacobian_window=20`
parameter already approximates the sliding-window anchor approach. Skip Python
prototype; implement directly in Ceres where marginalisation infrastructure
exists natively.
