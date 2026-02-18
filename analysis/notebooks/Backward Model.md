# Radar-Inertial Odometry: The Reverse Estimation Model

While the Forward Model describes how a state generates measurements, the Reverse Model (Estimation) describes how we find the state that best explains the noisy measurements we observed.

We formulate this as a Maximum A Posteriori (MAP) estimation problem, implemented as Non-Linear Least Squares (NLLS) optimization.

## 1. State Parameterization (General B-Spline)

Instead of estimating discrete poses, we estimate a continuous trajectory function $\mathbf{p}_w(t)$. To support high-order derivatives (like Jerk and Snap), we use a B-Spline of variable degree $p$.

### 1.1 The Control Points & Knot Vector

The state is defined by a set of Control Points in the World Frame:

$$\mathcal{X} = \{ \mathbf{c}_0, \mathbf{c}_1, \dots, \mathbf{c}_N \}$$

To define the basis functions for degree $p$, we use a Uniform Knot Vector (assuming constant time spacing $\Delta t$).

* Degree $p$:
  * $p=3$ (Cubic): Continuous Acceleration.
  * $p=4$ (Quartic): Continuous Jerk.
  * $p=5$ (Quintic): Continuous Snap.
* Influence: At any time $t$, the spline value depends on $p+1$ local control points.

### 1.2 Evaluation (The De Boor Algorithm)

Unlike the fixed cubic matrix, we evaluate the spline using the recursive Cox-De Boor formula. For a time $t$ falling in the knot span $[t_k, t_{k+1})$, the position $\mathbf{p}(t)$ is evaluated by:

$$\mathbf{p}(t) = \sum_{i=k-p}^{k} N_{i,p}(t) \mathbf{c}_i$$

Where the basis functions $N_{i,p}(t)$ are defined recursively:

1. Base Case ($p=0$):

$$N_{i,0}(t) = \begin{cases} 1 & \text{if } t_i \le t < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

2. Recursive Step:

$$N_{i,p}(t) = \frac{t - t_i}{t_{i+p} - t_i} N_{i, p-1}(t) + \frac{t_{i+p+1} - t}{t_{i+p+1} - t_{i+1}} N_{i+1, p-1}(t)$$

### 1.3 Analytical Derivatives (Velocity, Accel, Snap)

To compute the derivatives needed for the sensor models, we use the property that the derivative of a degree $p$ B-Spline is a degree $p-1$ B-Spline.

Velocity ($\mathbf{v}_w$):

$$\mathbf{v}_w(t) = \frac{d}{dt}\mathbf{p}(t) = \sum_{i=k-p+1}^{k} N_{i, p-1}(t) \cdot \mathbf{c}'_i$$

The "Velocity Control Points" $\mathbf{c}'_i$ are computed from the original points:

$$\mathbf{c}'_i = \frac{p}{t_{i+p+1} - t_{i+1}} (\mathbf{c}_{i+1} - \mathbf{c}_i)$$

Acceleration ($\mathbf{a}_w$):
We apply the reduction again. The "Acceleration Control Points" $\mathbf{c}''_i$ are:

$$\mathbf{c}''_i = \frac{p-1}{t_{i+p} - t_{i+1}} (\mathbf{c}'_{i+1} - \mathbf{c}'_i)$$

Then evaluate using basis functions $N_{i, p-2}(t)$.

## 1. The Optimization Problem

We seek to find the optimal set of Control Points $\mathcal{X}^*$ that minimizes the total error:

$$\mathcal{X}^* = \arg\min_{\mathcal{X}} \left( E_{radar} + \lambda_{acc} E_{accel} + \lambda_{reg} E_{reg} \right)$$

**Assumed Knowns (The Split-Spline Approach)**

For this formulation, we assume the Orientation $\mathbf{R}_{w \gets b}(t)$ and Angular Velocity $\boldsymbol{\omega}_b(t)$ are inputs provided by the pre-integrated gyroscope. They are not optimization variables here.

## 3. The Radar Residual ($r_{rad}$)

### 3.1 The Error Term

The residual is the difference between the Measured Doppler and the Predicted Doppler.

$$r_{rad, k} = v_{D, meas} - h_{rad}(\mathcal{X}, t, \hat{\mathbf{u}}_s)$$

Expanding $h_{rad}$ using our Spline variables and arrow notation:

$$h_{rad} = \underbrace{(\mathbf{R}_{b \gets s} \hat{\mathbf{u}}_s)}_{\text{Ray in Body}} \cdot \left[ \underbrace{\mathbf{R}_{w \gets b}(t)^T \mathbf{v}_w(t)}_{\mathbf{v}_b} + \underbrace{\boldsymbol{\omega}_b(t) \times \mathbf{T}_{b \gets s}}_{\text{Lever Arm Vel}} \right]$$

* $\mathbf{v}_w(t)$: Computed using the De Boor derivative recursion (Degree $p \to p-1$).

### 3.2 Loss Function (Robust Estimation)

Radar data contains outliers. We apply a Huber Loss function $\rho(\cdot)$ to down-weight large errors.

$$E_{radar} = \sum_{k \in \mathcal{P}} \rho_{Huber} \left( \frac{r_{rad, k}^2}{\sigma_{rad}^2} \right)$$

## 4. The Inertial Residual ($r_{acc}$)

### 4.1 The Error Term

The residual is the vector difference between the Measured Specific Force and the Predicted Specific Force.

$$\mathbf{r}_{acc} = \mathbf{z}_{acc} - h_{acc}(\mathcal{X}, t)$$

Expanding $h_{acc}$:

$$h_{acc} = \mathbf{R}_{w \gets b}(t)^T \left( \mathbf{a}_w(t) - \mathbf{g}_w \right) + \mathbf{b}_a$$

* $\mathbf{a}_w(t)$: Spline 2nd derivative computed via De Boor (Degree $p \to p-2$).
* $\mathbf{b}_a$: Accelerometer bias (estimated as a static variable).

### 4.2 Loss Function

Accelerometer noise is modeled as Gaussian:

$$E_{accel} = \sum_{m \in \mathcal{M}} || \mathbf{r}_{acc, m} ||^2_{\Sigma_{acc}^{-1}}$$

## 5. Summary of the Linear System (Jacobian Update)

When minimizing this function via Gauss-Newton, we solve $H \Delta \mathcal{X} = -b$.

Sparsity: Because we use B-splines, each measurement only affects $p+1$ control points. This results in a banded Hessian matrix, allowing for $O(N)$ optimization time.

Regularization: $\lambda_{reg} E_{reg}$ often penalizes the integral of the squared Snap ($\int ||\mathbf{p}^{(4)}(t)||^2 dt$) to ensure a smooth, "minimum-snap" trajectory between sparse radar frames.