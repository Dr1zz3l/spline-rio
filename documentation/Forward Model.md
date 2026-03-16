# Radar-Inertial Odometry: The Forward Measurement Model

This document defines the mathematical **Forward Model** for a drone equipped with an IMU (Accelerometer/Gyroscope) and a Doppler Radar (TI IWR6843AOP).

The **Forward Model** answers the question: *"Given a known drone state (position, velocity, orientation), what values will the sensors produce?"*

The estimation pipeline (the "Reverse" model) attempts to invert these equations to find the state that best explains the observed measurements.

## 1. Coordinate Systems & Notation

We define three Cartesian coordinate frames. All frames are Right-Handed.

1. **World Frame ($\mathcal{W}$):** The fixed inertial frame.
   * **Z-axis:** Aligned with gravity (pointing opposite to gravity vector).
   * **Origin:** Arbitrary (usually the drone's start position).

2. **Body Frame ($\mathcal{B}$):** The moving frame attached to the drone.
   * **Origin:** The center of the IMU (specifically the Accelerometer).
   * **Orientation:** Aligned with the flight controller's axes — **FLU** (X-Forward, Y-Left, Z-Up).
   * **Note:** Some bags (`circle_fwd`, `loopings`, `backflips`, `slow_racing_best_velocity`) were previously believed to require a 180° yaw-flip. This was a workaround for a sign error in the Doppler forward model — see FINDINGS.md §11 for the full investigation. With the sign fix applied, the need for per-bag flipping is under re-evaluation.

3. **Sensor/Radar Frame ($\mathcal{S}$):** The moving frame attached to the Radar.
   * **Origin:** The phase center of the radar antenna array.
   * **Orientation:** Defined by the radar hardware (often different from Body frame).

### Extrinsic Calibration
The relationship between Body ($\mathcal{B}$) and Sensor ($\mathcal{S}$) is fixed and rigid.
* $\mathbf{T}_{b\gets s}$: Translation vector from Body Origin to Sensor Origin, expressed in the **Body Frame**.
* $\mathbf{R}_{s\gets b}$: Rotation matrix rotating a vector from the **Body Frame** to the **Sensor Frame**.
* $\mathbf{R}_{b\gets s}$: Rotation matrix rotating a vector from the **Sensor Frame** to the **Body Frame** ($\mathbf{R}_{b\gets s} = \mathbf{R}_{s\gets b}^T$).

## 2. The Drone State Vector

At any continuous time $t$, the physical state of the drone is defined by:

| Symbol | Definition | Frame |
| :--- | :--- | :--- |
| $\mathbf{p}_w(t)$ | Position of the Body Center | World ($\mathcal{W}$) |
| $\mathbf{v}_w(t)$ | Linear Velocity of the Body Center ($\dot{\mathbf{p}}_w$) | World ($\mathcal{W}$) |
| $\mathbf{a}_w(t)$ | Linear Acceleration of the Body Center ($\ddot{\mathbf{p}}_w$) | World ($\mathcal{W}$) |
| $\mathbf{R}_{w\gets b}(t)$ | Orientation (Rotation from Body to World) | $\mathcal{B} \to \mathcal{W}$ |
| $\boldsymbol{\omega}_b(t)$ | Angular Velocity of the Body | Body ($\mathcal{B}$) |

### 2.1 Orientation Parameterization

The orientation is parameterized using a tangent-space perturbation around a nominal trajectory:

$$\mathbf{R}_{w\gets b}(t) = \mathbf{R}_{nom}(t) \cdot \exp(\boldsymbol{\delta}(t))$$

Where $\mathbf{R}_{nom}(t)$ is the nominal (reference) rotation from MoCap and $\boldsymbol{\delta}(t) \in \mathbb{R}^3$ is a correction in the Lie algebra so(3). See the Backward Model for details.

### 2.2 Angular Velocity Model

The angular velocity in the body frame is derived from the orientation parameterization:

$$\boldsymbol{\omega}_b(t) = \exp(-[\boldsymbol{\delta}]_\times) \, \boldsymbol{\omega}_{nom}(t) + \mathbf{J}_r(\boldsymbol{\delta}) \, \dot{\boldsymbol{\delta}}(t)$$

Where:
* $\boldsymbol{\omega}_{nom}(t)$: Nominal angular velocity (from MoCap, interpolated)
* $\mathbf{J}_r(\boldsymbol{\delta})$: The **right Jacobian of SO(3)**, computed exactly via SymForce
* $\dot{\boldsymbol{\delta}}(t)$: Time derivative of the perturbation spline

This is the **exact** formula (not a small-angle approximation). For $\boldsymbol{\delta} \to 0$, it simplifies to $\boldsymbol{\omega}_b \approx \boldsymbol{\omega}_{nom} + \dot{\boldsymbol{\delta}}$.

## 3. The Inertial Forward Model (Accelerometer)

The accelerometer is the reference for the Body Frame. It measures **Specific Force**, not coordinate acceleration.

### Physics
The accelerometer measures the difference between the body's kinematic acceleration and the gravitational field vector. When hovering stationary, the drone must exert an upward force to counteract gravity; the accelerometer measures this upward force (approx $9.81 \, m/s^2$).

### The Equation
$$
\mathbf{z}_{acc}(t) = \mathbf{R}_{b\gets w}(t) \left( \mathbf{a}_w(t) - \mathbf{g}_w \right) + \mathbf{b}_a + \mathbf{n}_a(t)
$$

### Parameter Breakdown
* $\mathbf{z}_{acc}(t)$: The measured 3D acceleration vector from the IMU [m/s²].
* $\mathbf{R}_{b\gets w}(t) = \mathbf{R}_{w\gets b}(t)^T$: Rotates the World Frame force into the Body Frame.
* $\mathbf{a}_w(t)$: The true 2nd derivative of the position trajectory.
* $\mathbf{g}_w$: The gravity vector in World Frame.
  * Convention: $\mathbf{g}_w = [0, 0, -9.81]^T$.
* $\mathbf{b}_a$: Accelerometer Bias (modeled as constant over the trajectory window; estimated when `LOCK_BIASES=False`).
* $\mathbf{n}_a(t)$: Additive White Gaussian Noise (AWGN).

## 4. The Gyroscope Forward Model

The gyroscope measures angular velocity in the body frame.

### The Equation
$$
\mathbf{z}_{gyr}(t) = \boldsymbol{\omega}_b(t) + \mathbf{b}_g + \mathbf{n}_g(t)
$$

Expanding using the orientation parameterization:

$$
\mathbf{z}_{gyr}(t) = \exp(-[\boldsymbol{\delta}]_\times) \, \boldsymbol{\omega}_{nom}(t) + \mathbf{J}_r(\boldsymbol{\delta}) \, \dot{\boldsymbol{\delta}}(t) + \mathbf{b}_g + \mathbf{n}_g(t)
$$

### Parameter Breakdown
* $\mathbf{z}_{gyr}(t)$: The measured 3D angular velocity from the IMU [rad/s].
* $\boldsymbol{\omega}_{nom}(t)$: Nominal angular velocity from the reference trajectory.
* $\boldsymbol{\delta}(t)$: Orientation perturbation (B-spline, optimization variable).
* $\mathbf{J}_r(\boldsymbol{\delta})$: SO(3) right Jacobian (exact, via SymForce codegen).
* $\mathbf{b}_g$: Gyroscope Bias (modeled as constant; estimated when `LOCK_BIASES=False`). Real MEMS gyro z-bias of ~0.18–0.28 rad/s has been confirmed across bags (thermal drift between flights).
* $\mathbf{n}_g(t)$: Additive White Gaussian Noise.

## 5. The Radar Forward Model (Doppler)

The radar measures the **Relative Radial Velocity** along the line-of-sight.

### 5.1. Kinematics: Antenna Velocity
First, we calculate the velocity of the radar antenna itself in the **Body Frame**. This includes the drone's linear velocity plus the "Lever Arm" effect caused by the drone's rotation.

$$
\mathbf{v}_{ant, b} = \mathbf{v}_b(t) + \boldsymbol{\omega}_b(t) \times \mathbf{T}_{b\gets s}
$$

* $\mathbf{v}_b(t) = \mathbf{R}_{w\gets b}^T \mathbf{v}_w$: Body linear velocity.
* $\boldsymbol{\omega}_b(t)$: Body angular velocity (from §2.2).
* $\mathbf{T}_{b\gets s}$: The fixed offset of the radar from the body origin.
* $\times$: Cross product.

### 5.2. Geometry: The Ray Direction
The radar detects a point at $\mathbf{p}_s$ (in Sensor Frame). The unit direction vector of this ray in the **Sensor Frame** is:
$$
\hat{\mathbf{u}}_s = \frac{\mathbf{p}_s}{||\mathbf{p}_s||}
$$

To compare this with our Body Frame velocity $\mathbf{v}_{ant, b}$, we must rotate this ray into the **Body Frame** using $\mathbf{R}_{b\gets s}$:
$$
\hat{\mathbf{u}}_b = \mathbf{R}_{b\gets s} \hat{\mathbf{u}}_s
$$

### 5.3. The Projection (Doppler Constraint)
The TI IWR6843 reports Doppler as **positive when the target is receding** (range rate $\dot{r} > 0$). This is the **opposite** of the naive dot-product sign. The forward model is:

$$
v_{D} = -\hat{\mathbf{u}}_{b} \cdot \mathbf{v}_{ant, b} + \epsilon
$$

Substituting the full terms:

$$
v_{D} = -(\mathbf{R}_{b\gets s} \hat{\mathbf{u}}_s) \cdot \left( \mathbf{R}_{w\gets b}^T \mathbf{v}_w(t) + \boldsymbol{\omega}_b(t) \times \mathbf{T}_{b\gets s} \right) + \epsilon
$$

### Sign Convention (Confirmed)
The negation was confirmed experimentally by `diagnostics/diagnose_doppler_sign.py` on the `slow_racing_best_velocity` bag:
* `corr(v_meas, -dot(u,v)) = +0.85` (correct sign, flip=OFF)
* `corr(v_meas, +dot(u,v)) = -0.85` (wrong sign)
* RMSE with correct sign: 0.83 m/s (vs 2.92 m/s with wrong sign)
* Only 4.3% of points Huber-suppressed with correct sign (vs 76.4% with wrong sign)

This is implemented in `predict_doppler_velocity()` (returns `-dot`) and in the SymForce residual source (`derive_jacobians_symforce.py`: `v_pred = -u_body.dot(v_ant)`).

## 6. Summary Table: What We Estimate vs. What We Measure

| Quantity | Type | Source | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| $\mathbf{p}_w(t)$ | **State** | Position B-spline control points | **Unknown** (To be solved) |
| $\mathbf{v}_w(t)$ | **State** | Position B-spline 1st derivative | **Unknown** (derived from position CPs) |
| $\mathbf{a}_w(t)$ | **State** | Position B-spline 2nd derivative | **Unknown** (derived from position CPs) |
| $\boldsymbol{\delta}(t)$ | **State** | Orientation B-spline control points | **Unknown** (To be solved) |
| $\boldsymbol{\omega}_b(t)$ | **State** | Orientation B-spline value + derivative | **Unknown** (derived from ori CPs via §2.2) |
| $\mathbf{b}_a$ | **State** | Estimator variable | **Unknown** (To be solved, or locked to zero) |
| $\mathbf{b}_g$ | **State** | Estimator variable | **Unknown** (To be solved, or locked to zero) |
| $\mathbf{R}_{nom}(t)$ | **Prior** | MoCap SLERP | Nominal orientation (re-linearization point) |
| $\boldsymbol{\omega}_{nom}(t)$ | **Prior** | MoCap angular velocity | Nominal angular velocity |
| $\mathbf{z}_{acc}$ | **Measurement** | IMU Topic | Constrains $\mathbf{a}_w$ and $\boldsymbol{\delta}$ |
| $\mathbf{z}_{gyr}$ | **Measurement** | IMU Topic | Constrains $\boldsymbol{\delta}$ and $\dot{\boldsymbol{\delta}}$ |
| $v_{D,k}$ | **Measurement** | Radar Topic | Constrains $\mathbf{v}_w$, $\boldsymbol{\delta}$, and $\dot{\boldsymbol{\delta}}$ |

## 7. Rosbag Topics
| Topic | Description |
| :--- | :--- |
| `/angrybird2/agiros_pilot/state` | Kalman-smoothed MoCap data. Pose is accurate; used for nominal orientation $\mathbf{R}_{nom}$. Angular velocity is body-frame Kalman-filtered (confirmed, NOT world-frame). |
| `/angrybird2/imu` | Raw IMU data from the **drone's own Pixhawk IMU** (accelerometer + gyroscope). This is NOT the radar board IMU — the sensor is at the drone center, so `R_bs` does NOT apply to IMU data. |
| `/mmWaveDataHdl/RScanVelocity` | Raw Radar Data. The radar is mounted **upside-down** (180° roll) and tilted 30° downward from horizontal. `ROTATION_EULER_DEG = [180, 30, 0]`. Translation: `[0.08, 0.02, -0.01]` m in body frame (8 cm forward, 2 cm left, 1 cm down). |