# Radar-Inertial Odometry: The Reverse Estimation Model

While the Forward Model describes how a state generates measurements, the Reverse Model (Estimation) describes how we find the state that best explains the noisy measurements we observed.

We formulate this as a Maximum A Posteriori (MAP) estimation problem, implemented as Non-Linear Least Squares (NLLS) optimization.

## 1. State Parameterization

We estimate continuous trajectories for both **position** and **orientation** using B-splines, plus constant sensor biases.

### 1.1 Position B-Spline (Translation in World Frame)

The translational trajectory $\mathbf{p}_w(t)$ is represented by a degree-$p$ B-spline with control points:

$$\mathcal{X}_{pos} = \{ \mathbf{c}_0^p, \mathbf{c}_1^p, \dots, \mathbf{c}_{N_p}^p \} \in \mathbb{R}^{N_p \times 3}$$

### 1.2 Orientation B-Spline (Lie Algebra so(3) Parameterization)

Instead of directly parameterizing rotations (which are constrained to the SO(3) manifold), we use a **Lie algebra parameterization**:

$$\mathbf{R}_{w \gets b}(t) = \mathbf{R}_{nom}(t) \cdot \exp(\boldsymbol{\delta\omega}(t))$$

Where:
- $\mathbf{R}_{nom}(t)$: Nominal rotation trajectory (e.g., from gyroscope integration)
- $\boldsymbol{\delta\omega}(t) \in \mathbb{R}^3$: Small correction in the Lie algebra **so(3)** (tangent space at identity)
- $\exp(\cdot): so(3) \to SO(3)$: Exponential map (Rodrigues' formula)

**Why Lie algebra?** SO(3) is a curved manifold - we cannot add rotations or perform unconstrained optimization. The Lie algebra so(3) is a flat vector space where standard optimization applies. Each 3-vector $\boldsymbol{\omega} \in \mathbb{R}^3$ corresponds to a skew-symmetric matrix $[\boldsymbol{\omega}]_\times \in so(3)$:

$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}$$

The correction vector $\boldsymbol{\delta\omega}(t)$ is represented by a B-spline:

$$\mathcal{X}_{ori} = \{ \mathbf{c}_0^o, \mathbf{c}_1^o, \dots, \mathbf{c}_{N_o}^o \} \in \mathbb{R}^{N_o \times 3}$$

Control points are **elements of the Lie algebra** so(3), making them suitable for unconstrained optimization.

Typically $N_o < N_p$ (fewer orientation control points, e.g., 3× sparser) since orientation changes more slowly.

### 1.3 Sensor Biases

$$\mathbf{b}_a \in \mathbb{R}^3, \quad \mathbf{b}_g \in \mathbb{R}^3$$

Constant accelerometer and gyroscope biases.

### 1.4 Full State Vector

$$\mathcal{X} = \{ \mathcal{X}_{pos}, \mathcal{X}_{ori}, \mathbf{b}_a, \mathbf{b}_g \} \in \mathbb{R}^{3N_p + 3N_o + 6}$$

## 2. B-Spline Evaluation

### 2.1 Knot Vector and Degree

To define the basis functions for degree $p$, we use a Uniform Knot Vector (constant time spacing $\Delta t$).

* **Position spline**: Degree $p = 5$ (Quintic) for continuous snap, $\Delta t \approx 0.15$s
* **Orientation spline**: Degree $p = 3$ (Cubic) for smoothness, $\Delta t \approx 0.45$s (3× sparser)
* **Local influence**: At any time $t$, the spline value depends on $p+1$ local control points

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

**Angular velocity** from orientation spline:
$$\boldsymbol{\omega}_b(t) \approx \frac{d}{dt}\boldsymbol{\delta\omega}(t) = \sum_{i} N_{i, p-1}^{ori}(t) \cdot \mathbf{c}_i^o$$

(First-order approximation valid for small perturbations in the Lie algebra. For larger perturbations, the exact relationship involves the Jacobian of the exponential map.)

## 3. The Optimization Problem

We seek to find the optimal state $\mathcal{X}^*$ that minimizes the weighted sum of residuals:

$$\mathcal{X}^* = \arg\min_{\mathcal{X}} \left( E_{radar} + \lambda_{acc} E_{accel} + \lambda_{gyr} E_{gyro} + E_{reg} \right)$$

Where:
- $E_{radar}$: Doppler velocity residuals
- $E_{accel}$: Accelerometer residuals
- $E_{gyro}$: Gyroscope residuals
- $E_{reg}$: Regularization (smoothness constraints)
- $\lambda_{acc}, \lambda_{gyr}$: Weights balancing sensor modalities

This is solved using **Levenberg-Marquardt** (damped Gauss-Newton) optimization.

## 4. The Radar Residual ($r_{rad}$)

### 4.1 The Error Term

The residual is the difference between measured and predicted Doppler velocity:

$$r_{rad, k} = v_{D, meas} - h_{rad}(\mathcal{X}, t, \hat{\mathbf{u}}_s)$$

Expanding $h_{rad}$ using the Lie algebra parameterization:

$$h_{rad} = \underbrace{(\mathbf{R}_{b \gets s} \hat{\mathbf{u}}_s)^T}_{=\hat{\mathbf{u}}_b^T} \left[ \underbrace{\mathbf{R}_{w \gets b}(t)^T \mathbf{v}_w(t)}_{\mathbf{v}_b} + \underbrace{\boldsymbol{\omega}_b(t) \times \mathbf{T}_{b \gets s}}_{\text{Lever arm vel}} \right]$$

Where:
- $\mathbf{v}_w(t) = \sum_i N_i^{p-1}(t) \mathbf{c}_i^{p'}$: Velocity from position spline
- $\mathbf{R}_{w \gets b}(t) = \mathbf{R}_{nom}(t) \exp\left(\sum_j N_j(t) \mathbf{c}_j^o\right)$: Rotation via Lie algebra exponential map
- $\boldsymbol{\omega}_b(t) = \sum_j N_j^{p-1}(t) \mathbf{c}_j^o$: Angular velocity from orientation spline derivative

### 4.2 Loss Function (Robust Estimation)

Radar data contains outliers. We apply a **Huber loss** $\rho(\cdot)$ to down-weight large errors:

$$E_{radar} = \sum_{k \in \mathcal{P}} \rho_{Huber} \left( \frac{r_{rad, k}}{\sigma_{rad}} \right)$$

Where:
$$\rho_{Huber}(x) = \begin{cases} \frac{1}{2}x^2 & |x| \le \delta \\ \delta |x| - \frac{1}{2}\delta^2 & |x| > \delta \end{cases}$$

Typical: $\delta = 0.5$ m/s, $\sigma_{rad} = 0.1$ m/s

## 5. The Accelerometer Residual ($r_{acc}$)

### 5.1 The Error Term

The residual is the vector difference between measured specific force and predicted specific force:

$$\mathbf{r}_{acc} = \mathbf{z}_{acc} - h_{acc}(\mathcal{X}, t)$$

Expanding $h_{acc}$:

$$h_{acc} = \mathbf{R}_{w \gets b}(t)^T \left( \mathbf{a}_w(t) - \mathbf{g}_w \right) + \mathbf{b}_a$$

Where:
- $\mathbf{a}_w(t) = \sum_i N_i^{p-2}(t) \mathbf{c}_i^{p''}$: Acceleration from position spline (2nd derivative)
- $\mathbf{R}_{w \gets b}(t) = \mathbf{R}_{nom}(t) \exp\left(\sum_j N_j(t) \mathbf{c}_j^o\right)$: From orientation spline
- $\mathbf{g}_w = [0, 0, -9.81]^T$ m/s²: Gravity in world frame
- $\mathbf{b}_a$: Accelerometer bias (optimization variable)

### 5.2 Loss Function

Accelerometer noise is modeled as Gaussian:

$$E_{accel} = \sum_{m \in \mathcal{M}} || \mathbf{r}_{acc, m} ||^2_{\Sigma_{acc}^{-1}}$$

## 6. The Gyroscope Residual ($r_{gyr}$)

### 6.1 The Error Term

The residual is the difference between measured and predicted angular velocity:

$$\mathbf{r}_{gyr} = \mathbf{z}_{gyr} - h_{gyr}(\mathcal{X}, t)$$

Expanding $h_{gyr}$:

$$h_{gyr} = \boldsymbol{\omega}_b(t) + \mathbf{b}_g$$

Where:
- $\boldsymbol{\omega}_b(t) = \sum_j N_j^{p-1}(t) \mathbf{c}_j^o$: Angular velocity from orientation spline derivative
- $\mathbf{b}_g$: Gyroscope bias (optimization variable)

### 6.2 Loss Function

Gyroscope noise is also Gaussian:

$$E_{gyro} = \sum_{m \in \mathcal{M}} || \mathbf{r}_{gyr, m} ||^2_{\Sigma_{gyr}^{-1}}$$

## 7. Regularization Terms

### 7.1 Acceleration Regularization

Penalizes large accelerations to prefer smooth motion:

$$E_{reg,acc} = \lambda_{acc} \int_{t_0}^{t_f} || \mathbf{a}_w(t) ||^2 dt$$

For B-splines, this can be computed in closed form using the control points.

### 7.2 Snap Regularization

For position spline (degree 5), penalizes the 4th derivative (snap):

$$E_{reg,snap,pos} = \lambda_{snap,pos} \int_{t_0}^{t_f} || \mathbf{p}^{(4)}(t) ||^2 dt$$

For orientation spline (degree 3), penalizes angular snap:

$$E_{reg,snap,ori} = \lambda_{snap,ori} \int_{t_0}^{t_f} || \boldsymbol{\delta\omega}^{(3)}(t) ||^2 dt$$

These ensure smooth trajectories between sparse measurements.

## 8. Analytical Jacobians (for Real-Time Performance)

The computational bottleneck is Jacobian computation. **Analytical Jacobians** provide 20-50× speedup over numerical finite differences.

### 8.1 Jacobian Structure

For Levenberg-Marquardt, we need:

$$\mathbf{J} = \frac{\partial \mathbf{r}}{\partial \mathcal{X}} \in \mathbb{R}^{M \times (3N_p + 3N_o + 6)}$$

Where $M$ is the total number of residuals (radar + accel + gyro).

**Key property**: Due to B-spline local support, each residual only depends on $p+1$ control points → **sparse Jacobian** (>90% zeros).

### 8.2 Radar Jacobian w.r.t. Position Control Points

Given radar residual:
$$r_{rad} = v_{D,meas} - \hat{\mathbf{u}}_b^T \left( \mathbf{R}_{w \gets b}^T \mathbf{v}_w + \boldsymbol{\omega}_b \times \mathbf{T}_{b \gets s} \right)$$

Take derivative w.r.t. position control point $\mathbf{c}_i^p$:

$$\frac{\partial r_{rad}}{\partial \mathbf{c}_i^p} = -\hat{\mathbf{u}}_b^T \mathbf{R}_{w \gets b}^T \frac{\partial \mathbf{v}_w}{\partial \mathbf{c}_i^p}$$

Since $\mathbf{v}_w(t) = \sum_j N_j^{p-1}(t) \mathbf{c}_j^{p'}$ where $\mathbf{c}_j^{p'} = \frac{p}{\Delta t}(\mathbf{c}_{j+1}^p - \mathbf{c}_j^p)$:

$$\frac{\partial \mathbf{v}_w}{\partial \mathbf{c}_i^p} = N_{i-1}^{p-1}(t) \cdot \frac{p}{\Delta t} \mathbf{I} - N_i^{p-1}(t) \cdot \frac{p}{\Delta t} \mathbf{I}$$

**Final form**:
$$\frac{\partial r_{rad}}{\partial \mathbf{c}_i^p} = -\frac{p}{\Delta t} \left( N_{i-1}^{p-1}(t) - N_i^{p-1}(t) \right) (\mathbf{R}_{w \gets b}^T \hat{\mathbf{u}}_b)^T$$

This is a **row vector** $\in \mathbb{R}^{1 \times 3}$.

### 8.3 Radar Jacobian w.r.t. Orientation Control Points

The rotation enters as $\mathbf{R}_{w \gets b} = \mathbf{R}_{nom} \exp(\boldsymbol{\delta\omega})$ where $\boldsymbol{\delta\omega} = \sum_j N_j(t) \mathbf{c}_j^o$.

Using the chain rule via the Lie algebra so(3):

$$\frac{\partial r_{rad}}{\partial \mathbf{c}_i^o} = \frac{\partial r_{rad}}{\partial \boldsymbol{\delta\omega}} \frac{\partial \boldsymbol{\delta\omega}}{\partial \mathbf{c}_i^o}$$

Where:
$$\frac{\partial \boldsymbol{\delta\omega}}{\partial \mathbf{c}_i^o} = N_i(t) \mathbf{I}$$

For small perturbations in the Lie algebra, the derivative w.r.t. rotation is:

$$\frac{\partial r_{rad}}{\partial \boldsymbol{\delta\omega}} = -\hat{\mathbf{u}}_b^T [\mathbf{v}_b]_\times$$

Where $[\mathbf{v}_b]_\times$ is the skew-symmetric matrix of $\mathbf{v}_b = \mathbf{R}_{w \gets b}^T \mathbf{v}_w$.

**Final form**:
$$\frac{\partial r_{rad}}{\partial \mathbf{c}_i^o} = -N_i(t) \hat{\mathbf{u}}_b^T [\mathbf{v}_b]_\times$$

### 8.4 Accelerometer Jacobian w.r.t. Position Control Points

Given:
$$\mathbf{r}_{acc} = \mathbf{z}_{acc} - \mathbf{R}_{w \gets b}^T(\mathbf{a}_w - \mathbf{g}_w) - \mathbf{b}_a$$

Take derivative w.r.t. $\mathbf{c}_i^p$:

$$\frac{\partial \mathbf{r}_{acc}}{\partial \mathbf{c}_i^p} = -\mathbf{R}_{w \gets b}^T \frac{\partial \mathbf{a}_w}{\partial \mathbf{c}_i^p}$$

Since $\mathbf{a}_w(t) = \sum_j N_j^{p-2}(t) \mathbf{c}_j^{p''}$ where $\mathbf{c}_j^{p''} = \frac{p(p-1)}{\Delta t^2}(\mathbf{c}_{j+2}^p - 2\mathbf{c}_{j+1}^p + \mathbf{c}_j^p)$:

$$\frac{\partial \mathbf{a}_w}{\partial \mathbf{c}_i^p} = \frac{p(p-1)}{\Delta t^2} \left( N_{i-2}^{p-2}(t) - 2N_{i-1}^{p-2}(t) + N_i^{p-2}(t) \right) \mathbf{I}$$

**Final form** (matrix $\in \mathbb{R}^{3 \times 3}$):
$$\frac{\partial \mathbf{r}_{acc}}{\partial \mathbf{c}_i^p} = -\frac{p(p-1)}{\Delta t^2} \left( N_{i-2}^{p-2}(t) - 2N_{i-1}^{p-2}(t) + N_i^{p-2}(t) \right) \mathbf{R}_{w \gets b}^T$$

### 8.5 Accelerometer Jacobian w.r.t. Orientation Control Points

$$\frac{\partial \mathbf{r}_{acc}}{\partial \mathbf{c}_i^o} = \frac{\partial \mathbf{r}_{acc}}{\partial \boldsymbol{\delta\omega}} \frac{\partial \boldsymbol{\delta\omega}}{\partial \mathbf{c}_i^o}$$

The derivative of $\mathbf{R}^T \mathbf{x}$ w.r.t. rotation perturbation is:

$$\frac{\partial}{\partial \boldsymbol{\delta\omega}} \left( \mathbf{R}_{w \gets b}^T \mathbf{x} \right) = [\mathbf{R}_{w \gets b}^T \mathbf{x}]_\times = -[\mathbf{x}]_\times \mathbf{R}_{w \gets b}^T$$

**Final form** (matrix $\in \mathbb{R}^{3 \times 3}$):
$$\frac{\partial \mathbf{r}_{acc}}{\partial \mathbf{c}_i^o} = N_i(t) [\mathbf{a}_w - \mathbf{g}_w]_\times \mathbf{R}_{w \gets b}^T$$

### 8.6 Accelerometer Jacobian w.r.t. Bias

$$\frac{\partial \mathbf{r}_{acc}}{\partial \mathbf{b}_a} = -\mathbf{I}$$

### 8.7 Gyroscope Jacobian w.r.t. Orientation Control Points

Given:
$$\mathbf{r}_{gyr} = \mathbf{z}_{gyr} - \boldsymbol{\omega}_b - \mathbf{b}_g$$

Where $\boldsymbol{\omega}_b(t) = \sum_j N_j^{p-1}(t) \mathbf{c}_j^{o'}$ (derivative of orientation spline):

$$\frac{\partial \mathbf{r}_{gyr}}{\partial \mathbf{c}_i^o} = -\frac{p_{ori}}{\Delta t_{ori}} \left( N_{i-1}^{p-1}(t) - N_i^{p-1}(t) \right) \mathbf{I}$$

### 8.8 Gyroscope Jacobian w.r.t. Bias

$$\frac{\partial \mathbf{r}_{gyr}}{\partial \mathbf{b}_g} = -\mathbf{I}$$

### 8.9 Implementation Notes

1. **Basis function caching**: Compute $N_i^p(t)$ once per measurement, reuse for all derivatives
2. **Sparse matrix storage**: Use CSR format, only store non-zero Jacobian blocks
3. **Vectorization**: Process all measurements at same timestamp together
4. **Expected speedup**: 20-50× faster than numerical finite differences (currently ~20 min/iteration → ~30 sec/iteration)

## 9. Implementation Strategies

### 9.1 Current Approach: Batch Optimization (Offline)

**Purpose**: Validation, calibration, ground truth generation

**Method**:
- Fixed time window (e.g., 15 seconds)
- Optimize all variables simultaneously
- Multiple iterations until convergence (10-15 iterations)
- High accuracy but computationally expensive

**State size**: 
- Position: 105 control points × 3 = 315 variables
- Orientation: 35 control points × 3 = 105 variables  
- Biases: 6 variables
- **Total: 426 variables**

**Initialization**:
1. Phase 1 (Forward model): Validate prediction equations
2. Phase 2 (Linear solver): Solve for position assuming known orientation
3. Phase 3 (Full nonlinear): Joint position + orientation estimation

**Runtime**: 
- Numerical Jacobian: ~20 min/iteration × 10 iterations = **3-4 hours** for 15s
- Analytical Jacobian (future): ~30 sec/iteration × 10 iterations = **5 minutes** for 15s

### 9.2 Future Approach: Sliding Window (Real-Time)

**Purpose**: Online state estimation for autonomous flight

**Method**:
- Rolling window of 1-2 seconds
- As new measurements arrive:
  1. Drop oldest control point
  2. Add new control point at current time
  3. Quick optimization (1-3 iterations) using warm-start
  4. Publish trajectory for oldest control point (now "frozen")

**State size** (1s window, dt=0.15s):
- Position: ~7 control points × 3 = 21 variables
- Orientation: ~3 control points × 3 = 9 variables
- Biases: 6 variables
- **Total: ~36 variables** (12× smaller than batch)

**Target performance**:
- Process 1 second of data in < 1 second
- With analytical Jacobian: ~100-500 ms/window
- Real-time capable on Jetson Orin

**Sliding window algorithm**:
```python
window = deque(maxlen=window_size)
state = initialize_from_imu()

while new_measurement_available():
    # Add new data to window
    window.append(new_measurement)
    
    # Quick optimization (warm-started from previous)
    state = optimize(state, window, max_iterations=3)
    
    # Output oldest state (now outside optimization window)
    if len(window) == window_size:
        publish_pose(state.get_oldest())
```

**Advantages**:
- Constant memory (fixed window size)
- Constant compute (fixed number of variables)
- Graceful degradation (can skip iterations if running slow)

**Disadvantages vs Batch**:
- Slightly less accurate (shorter optimization horizon)
- Cannot go back and fix errors (no global refinement)
- Drift accumulation over long flights

### 9.3 Hybrid Approach (Recommended)

**Online**: Sliding window for real-time odometry during flight

**Offline**: Batch optimization for post-processing
- Use full flight data (takeoff → landing)
- Generate high-accuracy ground truth
- Calibrate sensor extrinsics and biases
- Validate sliding window performance

This mirrors successful systems like VINS-Mono, Kimera, ORB-SLAM.

## 10. Current Implementation Status

### 10.1 Completed (Phase 1-3)

✅ **Forward model validation**: Doppler prediction RMSE = 0.48 m/s  
✅ **Linear solver (Phase 2)**: Position RMSE = 2.11 m, Velocity = 0.67 m/s  
✅ **Nonlinear solver (Phase 3)**: Joint position + orientation estimation  
✅ **Lie algebra parameterization**: Proper SO(3) optimization via so(3)  
✅ **Numerical Jacobian**: Finite differences (slow but working)  
✅ **Levenberg-Marquardt**: Damped optimization with convergence detection  

### 10.2 Bug Fixes Applied

🐛 **Timestamp conversion**: B-splines expect relative time, not absolute *(CRITICAL)*  
🐛 **Orientation indexing**: Changed from knot-span to time-based interpolation  
🐛 **Angular velocity**: Analytical derivative instead of numerical differentiation  
🐛 **Weight balancing**: Reduced LAMBDA_ACCEL from 0.1 to 0.01  

### 10.3 In Progress

🔄 **Current validation run**: Testing timestamp fix (expected ~2.5 hours runtime)  
🔄 **Performance analysis**: Evaluating position/velocity/orientation accuracy  

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