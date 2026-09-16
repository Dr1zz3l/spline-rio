# Radar-Inertial Odometry: The Estimation Model

While the Forward Model describes how a known state generates measurements, the
Estimation Model (Backward Model) describes how we find the state that best
explains noisy observations.

We formulate this as a **Maximum A Posteriori (MAP) estimation** problem solved
as Non-Linear Least Squares (NLLS) via the C++ Ceres solver.

> **Parameterization note:** This document describes the **current** cumulative
> SO(3) B-spline + Ceres LM architecture.  The earlier Python solver used a
> tangent-space perturbation `δ(t)` around a MoCap-SLERP reference with
> SymForce-generated analytical Jacobians and an explicit re-linearization loop —
> that formulation is superseded.  See `RESEARCH_NOTES.md §7` for context.

---

## 1. State Parameterization

### 1.1 Position B-Spline

The world-frame position `p_w(t) ∈ ℝ³` is a **quintic (degree-5) uniform
B-spline** with knot spacing `Δt_p = 5 ms`:

$$\mathcal{X}_{pos} = \{ P_0, P_1, \ldots, P_{N_p} \} \in \mathbb{R}^{N_p \times 3}$$

Velocity and acceleration are analytical derivatives of the spline (§2).

### 1.2 Orientation: Cumulative SO(3) B-Spline

The body-to-world rotation `R(t) ∈ SO(3)` is a **cumulative SO(3) B-spline**
(cubic, degree 3, following Sommer et al. 2020 / basalt):

$$R(t) = R_{\text{base}}[k-3] \cdot \prod_{j=k-3}^{k} \text{Exp}\!\bigl(\tilde{B}_j(t)\,\Omega_j\bigr)$$

where:
- `k` = active knot span for time `t`
- `Ω_j ∈ so(3)` = incremental rotation control knots (**the optimization variables**)
- `B̃_j(t)` = cumulative cubic basis functions
- `R_base[i]` = left-anchor precomputed as `Exp(Ω_0)·…·Exp(Ω_i)`; recomputed
  after each parameter update (not an optimization variable)
- Knot spacing: `Δt_o = 8 ms` (racing bags); `Δt_o = 0.8 ms` (backflips batch)

**Key advantage over perturbation formulations:** Exact on-manifold
representation at any rotation magnitude — no re-linearization around a MoCap
nominal needed, no gauge-freedom issue from MoCap SLERP re-initialization.

The angular velocity from this spline is (see `Forward Model.md §2.3`):

$$\omega_b(t) = \sum_{j=k-3}^{k} \frac{d\tilde{B}_j}{dt}(t) \cdot R_{\text{suffix},j}^\top \, \Omega_j$$

### 1.3 Sensor Biases

Constant accelerometer bias `b_a ∈ ℝ³` and gyroscope bias `b_g ∈ ℝ³`.

### 1.4 Extrinsic Pitch (optional)

A 1-DOF scalar `pitch_delta ∈ ℝ` is optimized when `lock_extrinsics = false`
(default for racing bags).  Composition: `R_total = R_nominal · Ry(pitch_delta)`,
with a soft prior (`lambda_extrinsic_prior = 10.0`).  Only pitch is observable
from Doppler; roll and yaw are locked.

### 1.5 Full State Vector

$$\mathcal{X} = \{ P_0, \ldots, P_{N_p},\; \Omega_0, \ldots, \Omega_{N_o},\; b_a,\; b_g,\; [\text{pitch\_delta}] \}$$

For a 26 s bag at the default knot spacings: ~5 200 pos CPs + ~3 250 ori knots +
6 bias DOF + 1 extrinsic = ~25 957 parameters.

---

## 2. B-Spline Evaluation

### 2.1 Position Spline (Uniform B-Spline, Degree 5)

For time `t` in knot span `[t_k, t_{k+1})`:

$$p_w(t) = \sum_{i=k-5}^{k} N_{i,5}(t)\, P_i$$

Basis functions are defined by the Cox-de Boor recursion:

**Base case** (`p=0`): `N_{i,0}(t) = 1` if `t_i ≤ t < t_{i+1}`, else 0.

**Recursive step**:
$$N_{i,p}(t) = \frac{t - t_i}{t_{i+p} - t_i} N_{i,p-1}(t) + \frac{t_{i+p+1} - t}{t_{i+p+1} - t_{i+1}} N_{i+1,p-1}(t)$$

**Velocity** and **acceleration** are degree-(p−1) and degree-(p−2) B-splines
respectively, computed by differencing adjacent control points:

$$v_w(t) = \dot{p}_w(t), \quad a_w(t) = \ddot{p}_w(t)$$

**Snap (4th derivative)** enters the minimum-snap regularization (§5.1).

### 2.2 Orientation Spline (Cumulative SO(3))

The cumulative basis functions `B̃_j(t)` are prefix sums of standard cubic
B-spline basis values, evaluated within the active span.  See
`analysis/lib/cumulative_so3_bspline.py` and
`rio_solver_cpp/include/rio/trajectory.h` for implementation details.

---

## 3. Optimization Problem

We seek the state minimizing the weighted sum of factor costs:

$$\mathcal{X}^* = \arg\min_{\mathcal{X}} \Bigl( E_\text{radar} + \lambda_a E_\text{accel} + \lambda_g E_\text{gyro} + E_\text{gravity} + E_\text{heading} + E_\text{reg} + E_\text{boundary} + E_\text{bias} \Bigr)$$

Solved via **Levenberg-Marquardt** (Ceres `SPARSE_NORMAL_CHOLESKY` with
`SUITE_SPARSE` backend; Ceres autodiff Jet arithmetic for Jacobians).
See CLAUDE.md Key Hyperparameters for current weight values.

---

## 4. Radar Doppler Factor

$$r_\text{rad} = v_{D,\text{meas}} - v_{D,\text{pred}}$$

$$v_{D,\text{pred}} = -\hat{u}_b^\top \!\left( R_{bw}(t)\,v_w(t) + \omega_b(t) \times t_{bs} \right)$$

where `R_bw = R(t)ᵀ`, `hat_u_b = R_bs · hat_u_s` (bearing in body frame),
`t_bs = [0.08, 0.02, −0.01]ᵀ` m (lever arm).

**Loss:** Huber loss, `δ = 1.0 m/s`.

$$E_\text{radar} = \sum_{k} \rho_\text{Huber}(r_{\text{rad},k};\; 1.0)$$

The `−` sign is the TI IWR6843 convention (positive Doppler = receding target).
**Critical: do not remove the negation.**

---

## 5. IMU Factors

### 5.1 Accelerometer

$$\mathbf{r}_a = z_a - R_{bw}(t)\!\left(a_w(t) - g_w\right) - b_a, \quad g_w = [0,\,0,\,-9.81]^\top \text{ m/s}^2$$

Loss: L2, weight `λ_a = 0.01`.
$$E_\text{accel} = \lambda_a \sum_m \|\mathbf{r}_{a,m}\|^2$$

### 5.2 Gyroscope

$$\mathbf{r}_g = z_g - \omega_b(t) - b_g$$

Loss: L2, weight `λ_g = 4.0` (C++ solver).  Dense constraints at ~993 Hz;
the dominant orientation anchor.

$$E_\text{gyro} = \lambda_g \sum_m \|\mathbf{r}_{g,m}\|^2$$

### 5.3 Gravity Direction (Optional)

During near-hover phases, a Mahony-style roll/pitch constraint:

$$\mathbf{r}_\text{tilt} = \frac{z_a - b_a}{\|z_a - b_a\|} - R_{bw}^\top \hat{g}$$

Active only when `|‖z_a − b_a‖ − g| < 3.0 m/s²` (flight dynamics threshold).
Weight `λ_\text{gravity} = 0.001`.

---

## 6. Heading Prior

Yaw is unobservable from Doppler alone (all yaw-equivalent trajectories predict
identical radial velocities under pure rotation about gravity).  A heading
reference is required:
- **During development / evaluation**: MoCap yaw via `--mocap-yaw`
- **In deployment**: magnetometer or visual compass

The residual `r_ψ = ψ_est − ψ_ref` is added at ~100 Hz, weight `λ_ψ = 0.6`.

---

## 7. Regularization

### 7.1 Minimum Snap (Position)

$$E_\text{snap} = \lambda_s \int_{t_0}^{t_f} \!\|p_w^{(4)}(t)\|^2\, dt, \quad \lambda_s = 2\times10^{-5}$$

Penalizes the 4th derivative of the position spline.  Computed in closed form
from the control points.

### 7.2 Minimum Angular Acceleration (Orientation)

Rather than penalizing angular velocity increments (minimum-ω), which opposes
every banked turn and the entire backflip maneuver, we penalize the **second
finite difference** on SO(3):

$$\mathbf{r}_\alpha = \text{Log}(R_{i-1}^\top R_i) - \text{Log}(R_i^\top R_{i+1})$$

This is zero for constant angular rate.  Only fires at maneuver onset/offset.

$$E_\text{ori\_accel} = \lambda_\alpha \sum_i \|\mathbf{r}_{\alpha,i}\|^2, \quad \lambda_\alpha = 0.1$$

Lambda sweep: 0.1 is the best compromise (1.0 blows up backflips; 0.0 degrades
fast-racing position).  Scaled `λ ∝ dt_ori³` when knot spacing changes to
maintain consistent continuous `∫‖α‖²dt` penalty.

---

## 8. Boundary Priors

**In the current live solver (P1–P3 MoCap-free init)**, boundary priors anchor
the trajectory start using **sensor-only estimates** (not MoCap ground truth):

| Prior | Target | Weight |
|-------|--------|--------|
| Position at t=0 | P2 dead-reckoned position | λ = 1000 |
| Velocity at t=0 | P2 WLS velocity estimate | λ = 1000 |
| Orientation at t=0 | P1 gyro-integrated R(0) | λ = 1000 |

MoCap is loaded after the solve for RMSE evaluation only.

**In the older batch solver** (`validate_nonlinear_solver.py`), boundary priors
used MoCap ground truth — accurate but not deployable without an external pose source.

---

## 9. Bias Prior

Soft prior preventing biases from absorbing dynamics:

$$E_\text{bias} = \lambda_{ba}\|b_a\|^2 + \lambda_{bg}\|b_g\|^2$$

C++ solver: `λ_ba = λ_bg = 10 000` (full-rate IMU provides enough constraints;
prevents trash-can behaviour where biases absorb systematic residuals).

---

## 10. Solver: C++ Ceres LM

### 10.1 Problem Construction

`rio_solver_cpp/src/solver.cpp` builds a `ceres::Problem` by iterating over:
1. Radar frames → `RadarDopplerFunctor` (or `RadarDopplerWithPitchFunctor`)
2. IMU samples → `AccelFunctor` + `GyroFunctor`
3. Gravity samples → `GravityFunctor`
4. Heading priors → `HeadingPriorFunctor`
5. Regularization → `SnapRegFunctor` + `AngularAccelRegFunctor`
6. Boundary → `BoundaryPosFunctor` + `BoundaryVelFunctor` + `BoundaryOriFunctor`
7. Bias priors → `BiasPriorFunctor`
8. Marginalization prior (SW only) → `MargPriorFunctor`

All functors use Ceres **automatic differentiation** (Jet arithmetic).  The
Python legacy solver (`codegen/generated_jacobians.py`) used SymForce-generated
analytical Jacobians; those remain available but are not used by the C++ solver.

### 10.2 Orientation Representation in C++

The C++ solver stores orientation knots as unit quaternions `[x,y,z,w]`
(`_base_rotations[i]` from Python's `CumulativeSO3BSpline`).  The basalt
`CeresSplineHelper<N>::evaluate_lie()` function evaluates the cumulative product
and its derivative within Ceres' autodiff framework.

### 10.3 Algorithm

```
Ceres LM options:
  max_num_iterations: 400
  linear_solver_type: SPARSE_NORMAL_CHOLESKY
  sparse_linear_algebra_library_type: SUITE_SPARSE
  minimizer_progress_to_stdout: false
```

No explicit re-linearization loop is needed because the cumulative SO(3)
B-spline is an exact on-manifold representation — Ceres LM updates `Ω_j` in
the local tangent space and the `R_base` anchors are recomputed from the updated
knots at each iteration.

---

## 11. Sensor-Only Initialization (P1–P3)

The live solver bootstraps without any MoCap data:

**P1 — Orientation (gyro integration):**
Initial roll/pitch from the stationary accelerometer (gravity direction).
Yaw set to zero (gauge freedom; corrected by heading prior during optimization).
Full orientation trajectory from forward-integrating debiased gyroscope:
`R(t + Δt) = R(t) · Exp((z_g − b_g)Δt)`.

**P2 — Position (radar dead-reckoning):**
Per radar frame, WLS estimates 3D ego-velocity from Doppler measurements.
Doppler alias unwrapping uses IMU-integrated world-frame velocity.
World-frame velocity integrated forward (Euler) to produce initial position trajectory.

**P3 — Boundary priors:**
Position, velocity, and orientation priors at `t = 0` from P1–P2 sensor estimates.
`λ_bnd = 1000` for position and velocity.

---

## 12. Sliding Window with Schur Complement Marginalization

The fixed-lag smoother (Phase 4b) advances a 3 s window in 0.3 s strides.

### 12.1 Window Structure

At each step, the window contains:
- **Stride zone CPs** (to be marginalized): the control points covering the
  oldest 0.3 s
- **Boundary CPs** (to be retained): the control points at the window boundary

### 12.2 Schur Complement Prior

After each Ceres LM solve, the stride-zone variables `a` are marginalized:

$$S = H_{bb} - H_{ab}^\top H_{aa}^{-1} H_{ab}$$

where `a` indexes stride-zone CPs/knots and `b` indexes boundary CPs/knots +
bias (dimension 30).  `S` is factored via Cholesky as `S = LLᵀ` and stored as
a `MargPriorFunctor` for the next window.

### 12.3 Re-Centered Prior

Before each window solve, the prior's linearization point `x₀` is updated to
the current warm-start.  The prior residual is:

$$\mathbf{r}_\text{marg} = L^\top(\mathbf{x} - \mathbf{x}_0)$$

in local SO(3) coordinates.  This makes the prior contribute only **curvature**
(the Hessian shape), not a gradient pull toward a stale historical estimate.
Without re-centering the prior fights the new data and the estimator diverges.

### 12.4 Prior Scaling

The raw Schur complement has eigenvalues spanning `[10⁴, 5.5×10¹⁴]` (condition
number ≈ 5.5×10¹⁰) due to the gyroscope constraint density at 1 kHz.  A scalar
`marg_prior_scale` is applied:

| Mission type | Scale | Rationale |
|---|---|---|
| Gentle / slow | 10⁻⁷ | Windows are self-sufficient; near-zero prior is best |
| Aggressive / fast | 2×10⁻⁴ | Inter-window position continuity needed |

There is a harmful intermediate regime (10⁻⁶ to 10⁻⁵) that partially constrains
without providing useful continuity — results are worse than either extreme.
No universal (clip, scale) pair beats per-mission-type tuning.
Configuration: `marg_prior_scale` in `config/bags.yaml` per-bag `solver_overrides`.

---

## 13. Current Results

See **CLAUDE.md** for up-to-date results tables (batch and sliding window,
settled and live-edge metrics).

---

## 14. References

**Continuous-Time B-Spline Estimation**
- Sommer et al., "Why and How to Avoid the Flipped Quaternion Multiplication" (2020)
- Furgale et al., "Unified Temporal and Spatial Calibration" (2013)
- Hug et al., "Continuous-Time Radar-Inertial and Lidar-Inertial Odometry" (2022)
- Usenko et al., "Visual-Inertial Mapping with Non-Linear Factor Recovery" (basalt, 2020)

**Sliding Window Marginalization**
- Leutenegger et al., "Keyframe-Based Visual-Inertial Odometry" (OKVIS, 2015)
- Qin et al., "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator" (2018)

**Radar Odometry**
- Kramer et al., "Asynchronous Multi-Sensor Fusion for Navigation" (2020)
- Doer & Trommer, "An EKF-Based Approach to Radar Inertial Odometry" (2021)

**Lie Groups for Robotics**
- Sola et al., "A Micro Lie Theory for State Estimation in Robotics" (2018)
- Forster et al., "On-Manifold Preintegration for Real-Time VIO" (2017)
