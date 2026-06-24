# Radar-Inertial Odometry: The Forward Measurement Model

This document defines the **Forward Model** for a quadrotor equipped with a
Pixhawk IMU and a TI IWR6843AOPEVM mmWave radar.

The **Forward Model** answers: *"Given a known drone state (position, velocity,
orientation), what values will the sensors produce?"*

The estimation pipeline (the "Reverse Model") attempts to invert these equations
to find the state that best explains observed measurements.  See `Backward Model.md`.

> **Orientation parameterization note:**  This document describes the
> **current** cumulative SO(3) B-spline representation, which does **not**
> require a MoCap-nominal trajectory.  Earlier versions of this document used a
> tangent-space perturbation `δ(t)` around a MoCap-SLERP reference — that
> formulation is superseded.

---

## 1. Coordinate Systems & Notation

All frames are right-handed.

1. **World Frame (W):** The fixed inertial frame.
   - Z-axis: aligned with gravity direction (pointing **away** from Earth,
     i.e., gravity vector is `[0, 0, −9.81]` m/s²).
   - Origin: drone start position.

2. **Body Frame (B):** The moving frame attached to the drone.
   - **FLU**: X=Forward, Y=Left, Z=Up.
   - Origin: center of the Pixhawk IMU.
   - **Not** the radar board's own IMU — the Pixhawk is the only IMU in use.

3. **Sensor/Radar Frame (S):** The frame attached to the radar.
   - Origin: phase center of the antenna array.
   - Related to B via the extrinsic calibration below.

### 1.1 Extrinsic Calibration

The radar is mounted **upside-down** (180° roll) with a downward tilt (30° CAD
nominal; the 3D-printed, hand-assembled fixture is a few degrees short).

| Parameter | Value | Notes |
|-----------|-------|-------|
| Roll  | 180° | upside-down mounting |
| Pitch | **27.5° (frozen)** | **Physically MEASURED at 27–28°** (inclinometer on radar face, ref surface 0°) — the SOLE founded anchor. The self-cal alone is **init-DEPENDENT** (lands at 27.0/27.2° ONLY because seeded at 25.5°; 2026-06-24 sweep: 25.5°→27°, 30°→36°, 33°→42° both racing bags), so freeze, don't estimate, and do NOT cite it as corroboration. A few degrees under the 30° CAD nominal. `extrinsics.yaml` keeps 25.5° only as the batch self-cal *seed* (stale early output, don't change); deployed runs lock 27.5° via `--set-ext`. (2026-06-24: old "init-independent recovery" claim retracted; paper/report state only the measured value.) |
| Yaw   | 0° | unobservable from Doppler; locked |
| Translation `t_bs` | `[0.08, +0.02, −0.01]` m | 8 cm fwd, 2 cm left, 1 cm down in body frame |

`R_bs`: rotation from Sensor frame to Body frame (i.e., `R_bs = Rx(180°)·Ry(27.5°)`).
`t_bs`: translation from body origin to radar antenna center, expressed in body frame.

**Extrinsic observability**: only the pitch angle is observable from Doppler
(rotating the radar about its boresight does not meaningfully change radial velocity
predictions).  `optimize_pitch_only: true` in the solver; roll and yaw are locked.

---

## 2. State Representation

### 2.1 Position B-Spline

The world-frame position trajectory `p_w(t) ∈ ℝ³` is a **quintic (degree-5)
uniform B-spline** with knot spacing `Δt_p = 5 ms`.

Velocity `v_w(t) = ṗ_w(t)` and acceleration `a_w(t) = p̈_w(t)` are obtained as
analytical derivatives of the spline (see `Backward Model.md §2`).

### 2.2 Orientation: Cumulative SO(3) B-Spline

The orientation is parameterized as a **cumulative SO(3) B-spline** following
Sommer et al. 2020 and the basalt VIO formulation:

$$R(t) = R_{\text{base}}[k-3] \cdot \prod_{j=k-3}^{k} \text{Exp}\!\bigl(\tilde{B}_j(t)\,\Omega_j\bigr)$$

where:
- `k` = the active knot span index for time `t`
- `Ω_j ∈ so(3)` = incremental rotation control knots (optimization variables)
- `B̃_j(t)` = cumulative cubic basis functions (suffix sums of standard B-spline basis)
- `R_base[i]` = precomputed left anchor: `Exp(Ω_0)·…·Exp(Ω_i)`; refreshed
  after each parameter update
- Knot spacing: `Δt_o = 8 ms`

**Key properties:**
- Exact on-manifold representation — no re-linearization around a MoCap nominal needed.
- `R_base` is recomputed from scratch at each optimizer update; it is not an optimization variable.
- `Ω_j = 0` → identity increments → initial `R(t)` from gyro-integrated orientation (P1 init).

### 2.3 Angular Velocity

The body-frame angular velocity at time `t` (within knot span `k`) is:

$$\omega_b(t) = \sum_{j=k-3}^{k} \frac{d\tilde{B}_j}{dt}(t) \cdot R_{\text{suffix},j}^\top \, \Omega_j$$

where `R_suffix,j = Exp(B̃_{j+1}·Ω_{j+1})·…·Exp(B̃_k·Ω_k)` is the partial
product of exponential factors **after** index `j` within the span.

This is obtained analytically from `evaluate_lie()` in the C++ solver
(`basalt::CeresSplineHelper<N>::evaluate_lie()`) or from
`analysis/lib/cumulative_so3_bspline.py`.

### 2.4 Sensor Biases

Constant accelerometer bias `b_a ∈ ℝ³` and gyroscope bias `b_g ∈ ℝ³`.
Real MEMS thermal drift is present — biases differ between flights.

---

## 3. Accelerometer Forward Model

The accelerometer measures **Specific Force** — not coordinate acceleration.

$$z_a(t) = R_{bw}(t)\,\bigl(a_w(t) - g_w\bigr) + b_a + n_a(t)$$

where:
- `R_bw(t) = R_wb(t)ᵀ = R(t)ᵀ`: rotation from world to body at time `t`
- `a_w(t)`: true coordinate acceleration from the position B-spline 2nd derivative
- `g_w = [0, 0, −9.81]ᵀ` m/s²: gravity in world frame
- `b_a`: accelerometer bias
- `n_a`: additive white Gaussian noise

During stationary hover: `a_w ≈ 0`, `R_bw g_w ≈ [0,0,−9.81]ᵀ` in world → the
sensor reads approximately `[0, 0, +9.81]ᵀ` in body (FLU body, z=up). ✓

---

## 4. Gyroscope Forward Model

$$z_g(t) = \omega_b(t) + b_g + n_g(t)$$

where:
- `ω_b(t)`: body-frame angular velocity from the cumulative SO(3) B-spline (§2.3)
- `b_g`: gyroscope bias (~0.18–0.28 rad/s z-axis thermal drift confirmed across bags)
- `n_g`: additive white Gaussian noise

The Pixhawk IMU measures at ~993 Hz (raw rate used directly in the C++ solver).
`z_g` is in the **body frame** (confirmed: body-frame RMSE ≪ world-frame RMSE
in `diagnostics/diagnose_gyro.py`).

---

## 5. Radar Doppler Forward Model

### 5.1 Antenna Velocity (with Lever Arm)

The velocity of the radar antenna in the body frame includes the lever-arm
contribution from body rotation:

$$v_{ant,b}(t) = v_b(t) + \omega_b(t) \times t_{bs}$$

where:
- `v_b(t) = R_bw · v_w(t)`: linear velocity rotated into body frame
- `ω_b(t)`: angular velocity from the SO(3) spline (§2.3)
- `t_bs = [0.08, 0.02, −0.01]ᵀ` m: offset from body origin to antenna center

The lever-arm term `ω × t_bs` is most significant during the backflips bag
(angular rates ~10 rad/s × 8 cm offset ≈ 0.8 m/s correction per axis).

### 5.2 Bearing Direction

A radar point detected at position `p_s` in the sensor frame has bearing:

$$\hat{u}_s = \frac{p_s}{\|p_s\|}$$

Rotated into the body frame via the extrinsic rotation:

$$\hat{u}_b = R_{bs} \, \hat{u}_s$$

### 5.3 Doppler Measurement (Confirmed Sign Convention)

The TI IWR6843 reports Doppler **positive for a receding target** (range rate
`ṙ > 0`). The forward model is:

$$v_D = -\hat{u}_b^\top v_{ant,b}(t) + \epsilon$$

Expanding:

$$\boxed{v_D = -\hat{u}_b^\top \!\left( R_{bw}(t)\,v_w(t) + \omega_b(t) \times t_{bs} \right) + \epsilon}$$

**Sign convention confirmed** (`diagnostics/diagnose_doppler_sign.py`,
`slow_racing_best_velocity`):

| Convention | corr(v_meas, v_pred) | RMSE | Huber-suppressed |
|---|---|---|---|
| `−dot(u, v)` (correct) | **+0.85** | **0.83 m/s** | **4.3 %** |
| `+dot(u, v)` (wrong)   | −0.85     | 2.92 m/s     | 76.4 %   |

The negation in `v_pred = −u·v` is physically correct. **Do not remove it.**

### 5.4 Known Systematic Bias

Limited elevation diversity (2 TX antennas) causes a systematic z-velocity
underestimate of −0.5 to −0.65 m/s.  Huber loss threshold `δ = 1.0 m/s` is
chosen to be ≥ this bias magnitude (the 0.049 m/s Doppler bin size in the
best-velocity firmware configuration sets a lower bound but does not drive the
choice).

### 5.5 Doppler Aliasing

The radar firmware aliases Doppler into `[−v_max, +v_max]`.  In the best-velocity
configuration `v_max ≈ 3.136 m/s`.  The solver unwraps aliases in-loop:
`k = round(−r / (2·v_max))`.  An IMU-aided pre-unwrapping pass also runs before
the solver (see RESEARCH_NOTES.md §2).

---

## 6. Summary: What We Measure vs. What We Estimate

| Quantity | Type | Derived from |
|----------|------|--------------|
| `p_w(t)` | **State** | Position B-spline control points |
| `v_w(t)` | **State** | Position B-spline 1st derivative |
| `a_w(t)` | **State** | Position B-spline 2nd derivative |
| `R(t)`   | **State** | Cumulative SO(3) B-spline (Ω_j knots) |
| `ω_b(t)` | **State** | Analytical derivative of SO(3) spline (§2.3) |
| `b_a`    | **State** | Optimization variable |
| `b_g`    | **State** | Optimization variable |
| `z_a`    | **Measurement** | IMU accelerometer ~993 Hz |
| `z_g`    | **Measurement** | IMU gyroscope ~993 Hz |
| `v_D`    | **Measurement** | Radar Doppler ~11 Hz per point |

---

## 7. ROS Topics

| Topic | Description |
|-------|-------------|
| `/angrybird2/imu` | Raw IMU (accelerometer + gyroscope) from Pixhawk |
| `/mmWaveDataHdl/RScanVelocity` | Radar point cloud (x, y, z, velocity, intensity, …) |
| `/angrybird2/agiros_pilot/state` | MoCap-derived state — used only for RMSE evaluation |

---

## 8. References

- Sommer et al., "Why and How to Avoid the Flipped Quaternion Multiplication" (2020)
- Hug et al., "Continuous-Time Radar-Inertial and Lidar-Inertial Odometry" (2022)
- Usenko et al., "Visual-Inertial Mapping with Non-Linear Factor Recovery" (basalt, 2020)
- Furgale et al., "Unified Temporal and Spatial Calibration for Multi-Sensor Systems" (2013)
