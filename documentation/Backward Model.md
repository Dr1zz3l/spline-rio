# Radar-Inertial Odometry: The Reverse Estimation Model

While the Forward Model describes how a state generates measurements, the Reverse Model (Estimation) describes how we find the state that best explains the noisy measurements we observed.

We formulate this as a Maximum A Posteriori (MAP) estimation problem, implemented as Non-Linear Least Squares (NLLS) optimization.

## 1. State Parameterization

We estimate continuous trajectories for both **position** and **orientation** using B-splines, plus constant sensor biases.

### 1.1 Position B-Spline (Translation in World Frame)

The translational trajectory $\mathbf{p}_w(t)$ is represented by a degree-$p$ B-spline with control points:

$$\mathcal{X}_{pos} = \{ \mathbf{c}_0^p, \mathbf{c}_1^p, \dots, \mathbf{c}_{N_p}^p \} \in \mathbb{R}^{N_p \times 3}$$

### 1.2 Orientation B-Spline (Lie Algebra so(3) Parameterization)

Instead of directly parameterizing rotations (which are constrained to the SO(3) manifold), we use a **tangent-space perturbation** around a nominal trajectory:

$$\mathbf{R}_{w \gets b}(t) = \mathbf{R}_{nom}(t) \cdot \exp(\boldsymbol{\delta}(t))$$

Where:
- $\mathbf{R}_{nom}(t)$: Nominal rotation trajectory from **MoCap SLERP interpolation** (updated by re-linearization)
- $\boldsymbol{\delta}(t) \in \mathbb{R}^3$: Correction vector in the Lie algebra **so(3)** (tangent space at identity)
- $\exp(\cdot): so(3) \to SO(3)$: Exponential map (Rodrigues' formula)

**Why Lie algebra?** SO(3) is a curved manifold — we cannot add rotations or perform unconstrained optimization. The Lie algebra so(3) is a flat vector space where standard optimization applies.

The correction vector $\boldsymbol{\delta}(t)$ is represented by a B-spline:

$$\mathcal{X}_{ori} = \{ \mathbf{c}_0^o, \mathbf{c}_1^o, \dots, \mathbf{c}_{N_o}^o \} \in \mathbb{R}^{N_o \times 3}$$

Control points are **elements of the Lie algebra** so(3), making them suitable for unconstrained optimization.

### 1.3 Angular Velocity from Orientation State

The body-frame angular velocity is derived from the orientation parameterization using the **exact** formula:

$$\boldsymbol{\omega}_b(t) = \exp(-[\boldsymbol{\delta}]_\times) \, \boldsymbol{\omega}_{nom}(t) + \mathbf{J}_r(\boldsymbol{\delta}) \, \dot{\boldsymbol{\delta}}(t)$$

Where:
- $\boldsymbol{\omega}_{nom}(t)$: Nominal angular velocity (from MoCap, linearly interpolated)
- $\mathbf{J}_r(\boldsymbol{\delta})$: The **right Jacobian of SO(3)**, defined as:

$$\mathbf{J}_r(\boldsymbol{\phi}) = \mathbf{I} - \frac{1 - \cos||\boldsymbol{\phi}||}{||\boldsymbol{\phi}||^2} [\boldsymbol{\phi}]_\times + \frac{||\boldsymbol{\phi}|| - \sin||\boldsymbol{\phi}||}{||\boldsymbol{\phi}||^3} [\boldsymbol{\phi}]_\times^2$$

This formula and all its Jacobians are computed **exactly via SymForce codegen** (no manual derivation, no small-angle approximation). For $\boldsymbol{\delta} \to 0$: $\boldsymbol{\omega}_b \approx \boldsymbol{\omega}_{nom} + \dot{\boldsymbol{\delta}}$ (recovers the linear model).

### 1.4 Sensor Biases

$$\mathbf{b}_a \in \mathbb{R}^3, \quad \mathbf{b}_g \in \mathbb{R}^3$$

Constant accelerometer and gyroscope biases. Currently estimated (`LOCK_BIASES=False`). The gyroscope z-bias is a real MEMS thermal bias (~0.18–0.28 rad/s, confirmed by `diagnostics/diagnose_gyro.py`). The accelerometer bias is constrained by a strong prior (`LAMBDA_BIAS_PRIOR_ACCEL=10.0`) to prevent gravity leakage.

### 1.5 Full State Vector

$$\mathcal{X} = \{ \mathcal{X}_{pos}, \mathcal{X}_{ori}, \mathbf{b}_a, \mathbf{b}_g \} \in \mathbb{R}^{3N_p + 3N_o + 6}$$

## 2. B-Spline Evaluation

### 2.1 Knot Vector and Degree

To define the basis functions for degree $p$, we use a Uniform Knot Vector (constant time spacing $\Delta t$).

Current configuration:
* **Position spline**: Degree $p = 5$ (Quintic) for continuous snap, $\Delta t = 0.05$s
* **Orientation spline**: Degree $p = 3$ (Cubic) for smoothness, $\Delta t = 0.05$s
* **Local influence**: At any time $t$, the spline value depends on $p+1$ local control points

For a 5-second trajectory this yields ~106 position CPs and ~106 orientation CPs, totaling ~642 optimization variables (including 6 bias parameters).

### 2.2 Cox-de Boor Evaluation

For a time $t$ falling in knot span $[t_k, t_{k+1})$, the spline value is:

$$\mathbf{s}(t) = \sum_{i=k-p}^{k} N_{i,p}(t) \mathbf{c}_i$$

Where basis functions $N_{i,p}(t)$ are defined recursively:

**Base case** ($p=0$):
$$N_{i,0}(t) = \begin{cases} 1 & \text{if } t_i \le t < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

**Recursive step**:
$$N_{i,p}(t) = \frac{t - t_i}{t_{i+p} - t_i} N_{i, p-1}(t) + \frac{t_{i+p+1} - t}{t_{i+p+1} - t_{i+1}} N_{i+1, p-1}(t)$$

### 2.3 Analytical Derivatives

The $k$-th derivative of a degree-$p$ B-spline is a degree-$(p-k)$ B-spline.

**Velocity** (1st derivative):
$$\mathbf{v}_w(t) = \frac{d}{dt}\mathbf{p}(t) = \sum_{i=k-p+1}^{k} N_{i, p-1}(t) \cdot \mathbf{c}'_i$$

Where velocity control points are:
$$\mathbf{c}'_i = \frac{p}{t_{i+p+1} - t_{i+1}} (\mathbf{c}_{i+1} - \mathbf{c}_i)$$

**Acceleration** (2nd derivative):
$$\mathbf{a}_w(t) = \frac{d^2}{dt^2}\mathbf{p}(t) = \sum_{i=k-p+2}^{k} N_{i, p-2}(t) \cdot \mathbf{c}''_i$$

Where acceleration control points are:
$$\mathbf{c}''_i = \frac{p-1}{t_{i+p} - t_{i+1}} (\mathbf{c}'_{i+1} - \mathbf{c}'_i)$$

**Orientation perturbation and its derivative**: The orientation delta and its time derivative are evaluated the same way from the orientation B-spline:
$$\boldsymbol{\delta}(t) = \sum_j N_j^{ori}(t) \mathbf{c}_j^o, \quad \dot{\boldsymbol{\delta}}(t) = \sum_j {N'}_j^{ori}(t) \mathbf{c}_j^o$$

These are then combined via the exact angular velocity formula (§1.3) to produce $\boldsymbol{\omega}_b(t)$.

## 3. The Optimization Problem

We seek to find the optimal state $\mathcal{X}^*$ that minimizes the weighted sum of residuals:

$$\mathcal{X}^* = \arg\min_{\mathcal{X}} \left( E_{radar} + \lambda_{acc} E_{accel} + \lambda_{gyr} E_{gyro} + E_{reg} + E_{boundary} \right)$$

Where:
- $E_{radar}$: Doppler velocity residuals (Huber loss)
- $E_{accel}$: Accelerometer residuals (optional Huber loss)
- $E_{gyro}$: Gyroscope residuals (L2 loss)
- $E_{reg}$: Regularization (minimum snap smoothness)
- $E_{boundary}$: Boundary priors at trajectory start
- $\lambda_{acc}, \lambda_{gyr}$: Weights balancing sensor modalities

This is solved using **Levenberg-Marquardt** (damped Gauss-Newton) with **conditional re-linearization** (triggered when max orientation delta exceeds a threshold).

## 4. The Radar Residual ($r_{rad}$)

### 4.1 The Error Term

The residual is the difference between measured and predicted Doppler velocity:

$$r_{rad, k} = v_{D, meas} - h_{rad}(\mathcal{X}, t, \hat{\mathbf{u}}_s)$$

Expanding $h_{rad}$ using the Lie algebra parameterization (with TI sign convention: positive = receding):

$$h_{rad} = -\hat{\mathbf{u}}_b^T \left[ \mathbf{R}_{w \gets b}(t)^T \mathbf{v}_w(t) + \boldsymbol{\omega}_b(t) \times \mathbf{T}_{b \gets s} \right]$$

Where:
- $\hat{\mathbf{u}}_b = \mathbf{R}_{b\gets s} \hat{\mathbf{u}}_s$: Ray direction in body frame
- $\mathbf{v}_w(t)$: Velocity from position B-spline derivative
- $\mathbf{R}_{w \gets b}(t) = \mathbf{R}_{nom}(t) \exp(\boldsymbol{\delta}(t))$: Rotation via Lie algebra
- $\boldsymbol{\omega}_b(t) = \exp(-[\boldsymbol{\delta}]_\times) \boldsymbol{\omega}_{nom} + \mathbf{J}_r(\boldsymbol{\delta}) \dot{\boldsymbol{\delta}}$: Angular velocity (exact, §1.3)

**Note:** Since $\boldsymbol{\omega}_b$ depends on both $\boldsymbol{\delta}(t)$ and $\dot{\boldsymbol{\delta}}(t)$, the radar residual depends on orientation control points through **both** the B-spline value and its derivative.

### 4.2 Loss Function (Robust Estimation)

Radar data contains outliers. We apply a **Huber loss** $\rho(\cdot)$ to down-weight large errors:

$$E_{radar} = \sum_{k \in \mathcal{P}} \rho_{Huber} \left( r_{rad, k}; \delta_{hub} \right)$$

Where:
$$\rho_{Huber}(x; \delta) = \begin{cases} \frac{1}{2}x^2 & |x| \le \delta \\ \delta |x| - \frac{1}{2}\delta^2 & |x| > \delta \end{cases}$$

Current setting: $\delta_{hub} = 1.0$ m/s (increased from 0.5 to account for 0.63 m/s Doppler quantization).

## 5. The Accelerometer Residual ($r_{acc}$)

### 5.1 The Error Term

$$\mathbf{r}_{acc} = \mathbf{z}_{acc} - \mathbf{R}_{w \gets b}(t)^T \left( \mathbf{a}_w(t) - \mathbf{g}_w \right) - \mathbf{b}_a$$

Where:
- $\mathbf{a}_w(t)$: Acceleration from position B-spline 2nd derivative
- $\mathbf{R}_{w \gets b}(t) = \mathbf{R}_{nom}(t) \exp(\boldsymbol{\delta}(t))$: From orientation state
- $\mathbf{g}_w = [0, 0, -9.81]^T$ m/s²: Gravity in world frame
- $\mathbf{b}_a$: Accelerometer bias (optimization variable)

### 5.2 Loss Function

Operates with optional Huber loss on the 3D residual norm, weighted by $\lambda_{acc}$:

$$E_{accel} = \lambda_{acc} \sum_{m} \rho_{Huber}(||\mathbf{r}_{acc,m}||; \delta_{acc})$$

Current settings: $\lambda_{acc} = 0.01$, $\delta_{acc} = 2.0$ m/s².

**Note on accel–orientation coupling:** The accelerometer residual depends on orientation through $R_{wb}^\top$. Increasing $\lambda_{acc}$ causes the optimizer to adjust orientation to reduce accel residuals, which can degrade orientation accuracy. At $\lambda_{acc} \geq 0.05$, the accel cost dominates and pulls orientation away from the gyro-determined solution. The current value of 0.01 keeps the accel contribution balanced.

## 6. The Gyroscope Residual ($r_{gyr}$)

### 6.1 The Error Term

$$\mathbf{r}_{gyr} = \mathbf{z}_{gyr} - \boldsymbol{\omega}_b(t) - \mathbf{b}_g$$

Expanding the angular velocity model:

$$\mathbf{r}_{gyr} = \mathbf{z}_{gyr} - \left[ \exp(-[\boldsymbol{\delta}]_\times) \boldsymbol{\omega}_{nom}(t) + \mathbf{J}_r(\boldsymbol{\delta}) \dot{\boldsymbol{\delta}}(t) \right] - \mathbf{b}_g$$

Where:
- $\boldsymbol{\omega}_{nom}(t)$: Nominal angular velocity from MoCap (linearly interpolated)
- $\boldsymbol{\delta}(t), \dot{\boldsymbol{\delta}}(t)$: Orientation B-spline value and derivative
- $\mathbf{J}_r(\boldsymbol{\delta})$: SO(3) right Jacobian (exact, SymForce-generated)
- $\mathbf{b}_g$: Gyroscope bias

**Key insight:** The gyro residual depends on orientation control points through **both** the value (via $\boldsymbol{\delta}$) and the derivative (via $\dot{\boldsymbol{\delta}}$). This is different from the old linear model where only the derivative appeared.

### 6.2 Loss Function

Gyroscope noise is modeled as Gaussian (L2 loss):

$$E_{gyro} = \lambda_{gyr} \sum_{m} || \mathbf{r}_{gyr, m} ||^2$$

Current setting: $\lambda_{gyr} = 0.50$.

## 7. Boundary Priors

To anchor the trajectory at the start (where MoCap ground truth is available), we add soft priors on multiple state quantities within a window of $W = 0.3$s from the trajectory start.

$$E_{boundary} = E_{bnd,pos} + E_{bnd,vel} + E_{bnd,ori} + E_{bnd,acc} + E_{bnd,gyr}$$

Each term takes the form $\lambda \cdot || x_{est}(t) - x_{target}(t) ||^2$ at sample points within the boundary window:

| Prior | Target | Weight |
| :--- | :--- | :--- |
| Position | $\mathbf{p}_{MoCap}(t)$ | $\lambda_{bnd,pos} = 1000$ |
| Velocity | $\mathbf{v}_{MoCap}(t)$ | $\lambda_{bnd,vel} = 1000$ |
| Orientation | $\boldsymbol{\delta}(t) = 0$ (trust nominal) | $\lambda_{bnd,ori} = 100$ |
| Acceleration | $\mathbf{a}_{MoCap}(t)$ | $\lambda_{bnd,acc} = 0.001$ |
| Ang. velocity | $\dot{\boldsymbol{\delta}}(t) = 0$ (trust nominal $\omega$) | $\lambda_{bnd,gyr} = 10$ |

These are applied at the **start edge only** (no end priors), using multiple sample points within the boundary window.

## 8. Regularization Terms

### 8.1 Minimum Snap Regularization

For both position and orientation splines, we penalize the highest useful derivative (snap = 4th derivative for quintic, jerk = 3rd derivative for cubic):

$$E_{reg} = \lambda_{snap,pos} \int_{t_0}^{t_f} || \mathbf{p}^{(4)}(t) ||^2 dt + \lambda_{snap,ori} \int_{t_0}^{t_f} || \boldsymbol{\delta}^{(3)}(t) ||^2 dt$$

These integrals are computed in closed form using the B-spline control points and added to the normal equations as a fixed quadratic penalty matrix $\mathbf{R}_{snap}$.

Current settings: $\lambda_{snap,pos} = 0$, $\lambda_{snap,ori} = 0$ (disabled — sensor data provides sufficient constraints).

## 9. Analytical Jacobians (SymForce Codegen)

All Jacobians are computed **analytically via SymForce code generation**. The codegen pipeline (`codegen/derive_jacobians_symforce.py`) produces three functions in `codegen/generated_jacobians.py` (pure NumPy, no SymForce runtime dependency):

1. **`radar_residual_with_jacobians`**: $r_{rad}$ and $\frac{\partial r}{\partial \mathbf{v}_w}, \frac{\partial r}{\partial \boldsymbol{\delta}}, \frac{\partial r}{\partial \boldsymbol{\omega}}$
2. **`accel_residual_with_jacobians`**: $\mathbf{r}_{acc}$ and $\frac{\partial \mathbf{r}}{\partial \mathbf{a}_w}, \frac{\partial \mathbf{r}}{\partial \boldsymbol{\delta}}, \frac{\partial \mathbf{r}}{\partial \mathbf{b}_a}$
3. **`gyro_residual_with_jacobians`**: $\mathbf{r}_{gyr}$ and $\frac{\partial \mathbf{r}}{\partial \boldsymbol{\delta}}, \frac{\partial \mathbf{r}}{\partial \dot{\boldsymbol{\delta}}}, \frac{\partial \mathbf{r}}{\partial \mathbf{b}_g}$

### 9.1 Chain Rule: from SymForce Jacobians to Control Point Jacobians

The SymForce functions compute Jacobians w.r.t. **evaluated** quantities ($\mathbf{v}_w$, $\boldsymbol{\delta}$, etc.). To get Jacobians w.r.t. **control points** $\mathbf{c}_i$, we apply the chain rule via basis functions:

**Position control points** (velocity enters via 1st derivative):
$$\frac{\partial r}{\partial \mathbf{c}_i^p} = \frac{\partial r}{\partial \mathbf{v}_w} \cdot M_i'(t)$$

where $M_i'(t)$ is the derivative basis function coefficient for control point $i$.

**Orientation control points** (enter via both value and derivative):
$$\frac{\partial r}{\partial \mathbf{c}_j^o} = \frac{\partial r}{\partial \boldsymbol{\delta}} \cdot M_j(t) + \frac{\partial r}{\partial \dot{\boldsymbol{\delta}}} \cdot M_j'(t)$$

**Radar orientation chain rule** (full, including omega dependency):

Since the radar residual depends on $\boldsymbol{\omega}_b$ which itself depends on $\boldsymbol{\delta}$ and $\dot{\boldsymbol{\delta}}$:

$$\frac{\partial r_{rad}}{\partial \mathbf{c}_j^o} = \underbrace{\left(\frac{\partial r}{\partial \boldsymbol{\delta}} + \frac{\partial r}{\partial \boldsymbol{\omega}} \frac{\partial \boldsymbol{\omega}}{\partial \boldsymbol{\delta}}\right)}_{\text{effective value Jacobian}} M_j(t) + \underbrace{\left(\frac{\partial r}{\partial \boldsymbol{\omega}} \frac{\partial \boldsymbol{\omega}}{\partial \dot{\boldsymbol{\delta}}}\right)}_{\text{effective derivative Jacobian}} M_j'(t)$$

The inner Jacobians $\frac{\partial \boldsymbol{\omega}}{\partial \boldsymbol{\delta}}$ and $\frac{\partial \boldsymbol{\omega}}{\partial \dot{\boldsymbol{\delta}}}$ are extracted from the SymForce gyro function via the `compute_omega_and_jacobians()` wrapper.

### 9.2 Sparsity

Due to B-spline local support, each residual only depends on $p+1$ control points → the Jacobian is **>98% sparse** (stored in CSR format). Typical size: ~9000 residuals × ~1500 variables.

## 10. Solver: Levenberg-Marquardt with Re-linearization

### 10.1 Algorithm

The solver uses a single LM loop with **conditional SO(3) re-linearization** and an **accelerometer warm-up** phase:

```
lambda = 1e-3

for iteration in range(max_iterations):
    J, r = build_jacobian(state, lambda_accel)
    H = J^T J + lambda * I + R_snap
    delta_x = solve(H, -J^T r)
    
    state_new = state + delta_x
    if cost(state_new) < cost(state):
        state = state_new
        lambda *= 0.1
        if max(|delta_ori|) >= relinearize_threshold:
            state.relinearize()     # absorb delta into nominal
    else:
        lambda *= 10            # reject step, increase damping
    
    if ||delta_x|| < 1e-4:
        break                   # converged
```

### 10.2 Re-linearization

After accepted LM steps **where the maximum orientation delta exceeds `RELINEARIZE_THRESHOLD_DEG`** (currently 10°), `relinearize()` absorbs the current delta perturbation into the nominal trajectory:

1. **Dense sampling**: Evaluate $\mathbf{R}(t) = \mathbf{R}_{nom}(t) \exp(\boldsymbol{\delta}(t))$ and $\boldsymbol{\omega}_b(t)$ at ~200 time points
2. **Rebuild SLERP**: Construct new `scipy.spatial.transform.Slerp` from the dense rotation samples
3. **Rebuild omega interpolation**: Construct new `scipy.interpolate.interp1d` from the dense angular velocity samples
4. **Reset delta**: Set all orientation control points to zero

This keeps $\boldsymbol{\delta}$ near zero, which improves numerical conditioning. Because the angular velocity model uses the **exact** SO(3) right Jacobian $\mathbf{J}_r(\boldsymbol{\delta})$ (via SymForce), re-linearization is safe at any delta magnitude — there is no small-angle approximation to violate.

### 10.3 Current Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `MAX_ITERATIONS` | 20 | LM iterations |
| `LAMBDA_ACCEL` | 1.0 | Accelerometer weight |
| `LAMBDA_GYRO` | 100.0 | Gyroscope weight (high: gyro is the primary orientation sensor) |
| `LAMBDA_SNAP_POS` | 0.0001 | Position smoothness |
| `LAMBDA_SNAP_ORI` | 0.0001 | Orientation smoothness |
| `HUBER_DELTA` | 1.0 m/s | Radar Huber threshold (≥ 0.63 m/s quantization bin) |
| `HUBER_DELTA_ACCEL` | 2.0 m/s² | Accelerometer Huber threshold |
| `LOCK_BIASES` | False | Biases estimated (gyro z-bias confirmed real) |
| `LAMBDA_BIAS_PRIOR_ACCEL` | 1000.0 | Strong prior on accel bias (prevents gravity leakage) |
| `LAMBDA_BIAS_PRIOR_GYRO` | 10000.0 | Very strong prior on gyro bias |
| `BSPLINE_DEGREE` | 5 (pos) / 5 (ori) | B-spline degrees (both quintic) |
| `DT_POS` / `DT_ORI` | 0.05s / 0.05s | Knot spacings |
| `IMU_MOCAP_OFFSET` | +20 ms | IMU/radar timestamps shifted forward to align with MoCap |
| `RELINEARIZE_THRESHOLD_DEG` | 10.0° | Re-linearize only when max delta exceeds this |
| `ACCEL_WARMUP_ITERS` | 0 | Disabled (all sensors active from iteration 0) |
| `ROTATION_EULER_DEG` | [180, 30, 0] | Radar extrinsic rotation (upside-down + 30° tilt) |

## 11. Implementation Pipeline

### 11.1 Current Approach: Batch Optimization (Offline)

**Purpose**: Validation, calibration, ground truth generation

**Method**:
1. Load MoCap + IMU + Radar from rosbag
2. Initialize position B-spline from MoCap positions (least-squares fit)
3. Initialize orientation nominal from MoCap rotations (SLERP), delta = 0
4. Run LM optimization with re-linearization
5. Evaluate against MoCap ground truth

**Initialization phases**:
1. **Phase 1 (Forward model)**: Validate prediction equations using known MoCap state
2. **Phase 2 (Linear solver)**: Solve for position assuming known orientation
3. **Phase 3 (Full nonlinear)**: Joint position + orientation + bias estimation ← **current**

### 11.2 SymForce Codegen Pipeline

The SymForce codegen runs from the project venv:

```bash
cd radar-iwr6843-driver
source .venv/bin/activate
python analysis/codegen/derive_jacobians_symforce.py
```

This generates `analysis/codegen/generated_jacobians.py` — a pure NumPy file with no SymForce dependency, containing:
- `Rot3` class (quaternion representation with `from_rotation_matrix` converter)
- Three residual-with-Jacobians functions (radar, accel, gyro)
- Built-in validation against finite differences (run at generation time)

## 12. Current Results (2026-03-16)

On the **slow_racing_best_velocity** bag (10 seconds, no yaw flip, correct Doppler sign):

| Metric | Value |
| :--- | :--- |
| Position RMSE | 0.4 m |
| Velocity RMSE | 0.2 m/s |
| Angular velocity RMSE | 0.1 rad/s |
| Acceleration RMSE | 2.9 m/s² |
| Orientation RMSE | 1.4° |

**Historical baseline** (before Doppler sign fix, with yaw flip):

| Metric | Before fix |
| :--- | :--- |
| Position RMSE | 2.0 m |
| Velocity RMSE | 1.3 m/s |
| Orientation RMSE | 2.1° |

The ~5× improvement in position/velocity RMSE is attributed to the Doppler sign fix (see FINDINGS.md §11). With the correct sign, radar residuals at MoCap ground truth have RMSE 0.83 m/s with only 4.3% Huber-suppressed (vs 1.1 m/s / 16% before).

## 13. Future: Sliding Window for Real-Time Estimation

Once the batch (global) optimizer produces reliable results, the next step is a **sliding window** formulation for real-time operation on live sensor data.

### 13.1 Concept

Instead of optimizing over the entire trajectory at once, we maintain a fixed-duration window (e.g., 1–2 seconds) that slides forward in time as new measurements arrive:

1. New radar/IMU measurements arrive
2. Extend the B-spline with a new control point at the leading edge
3. Marginalize (drop) the oldest control point at the trailing edge
4. Run a few LM iterations (1–3), warm-started from the previous solution
5. Publish the state at the trailing edge (now "frozen" and outside the window)

### 13.2 Why Sliding Window?

| Property | Batch (Current) | Sliding Window |
| :--- | :--- | :--- |
| State size | ~640 variables (5s) | ~30–60 variables (1–2s) |
| Compute | Minutes | Target: 10–100 ms/window |
| Latency | Offline only | Real-time capable |
| Accuracy | Best (global context) | Slightly worse (limited horizon) |
| Drift | None (fixed window) | Accumulates over time |

### 13.3 Open Questions

- **Marginalization strategy**: How to properly marginalize the trailing control point — Schur complement prior, or simply drop it? The Schur complement preserves information but adds complexity.
- **Initialization without MoCap**: The current batch solver uses MoCap for $\mathbf{R}_{nom}$ and boundary priors. A real-time system needs bootstrapping from IMU-only integration.
- **Bias observability**: With a short window, biases may not be observable. May need to carry bias estimates across windows.
- **Re-linearization frequency**: In the batch solver we re-linearize every accepted step. In a sliding window, the warm-start means delta is already small — re-linearization may only be needed occasionally.
- **Target platform**: Jetson Orin or similar embedded GPU.

### 10.4 Next Steps

📋 **Analytical Jacobian**: Implement equations from Section 8 for 20-50× speedup  
📋 **Sliding window simulator**: Loop through rosbag with 1-2s windows  
📋 **Real-time testing**: Validate performance on longer flight sequences  
📋 **Jetson deployment**: Port to embedded platform  

## 11. References

**B-spline Continuous-Time Estimation**:
- Furgale et al., "Unified Temporal and Spatial Calibration" (2013)
- Anderson & Barfoot, "Full STEAM Ahead" (2015)
- Müller et al., "Continuous-Time Visual-Inertial Odometry" (2022)

**Radar Odometry**:
- Kramer et al., "Asynchronous Multi-Sensor Fusion" (2020)
- Doer et al., "Radar Inertial Odometry" (2021)

**Lie Algebra and SO(3) Optimization**:
- Sola et al., "Micro Lie Theory for State Estimation in Robotics" (2018)  
  *(Comprehensive tutorial on Lie groups, Lie algebras, and their use in robotics)*
- Barfoot & Furgale, "Associating Uncertainty With Three-Dimensional Poses" (2014)
- Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" (2017)