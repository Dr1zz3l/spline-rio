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
   * **Orientation:** Aligned with the flight controller's axes (usually X-Forward, Y-Left, Z-Up).

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

## 3. The Inertial Forward Model (Accelerometer)

The accelerometer is the reference for the Body Frame. It measures **Specific Force**, not coordinate acceleration.

### Physics
The accelerometer measures the difference between the body's kinematic acceleration and the gravitational field vector. When hovering stationary, the drone must exert an upward force to counteract gravity; the accelerometer measures this upward force (approx $9.81 \, m/s^2$).

### The Equation
$$
\mathbf{z}_{acc}(t) = \mathbf{R}_{b\gets w}(t) \left( \mathbf{a}_w(t) - \mathbf{g}_w \right) + \mathbf{b}_a(t) + \mathbf{n}_a(t)
$$

### Parameter Breakdown
* $\mathbf{z}_{acc}(t)$: The measured 3D acceleration vector from the IMU [m/s²].
* $\mathbf{R}_{b\gets w}(t)$: Rotates the World Frame force into the Body Frame.
* $\mathbf{a}_w(t)$: The true 2nd derivative of the Spline trajectory.
* $\mathbf{g}_w$: The gravity vector in World Frame.
  * Convention: $\mathbf{g}_w \approx [0, 0, -9.81]^T$.
  * *Note:* The term becomes $(\mathbf{a}_w - (-9.81)) = \mathbf{a}_w + 9.81$.
* $\mathbf{b}_a(t)$: Accelerometer Bias. A slowly varying offset (modeled as a Random Walk).
* $\mathbf{n}_a(t)$: Additive White Gaussian Noise (AWGN).

## 4. The Radar Forward Model (Doppler) Corrected

The radar measures the **Relative Radial Velocity** along the line-of-sight.

### 4.1. Kinematics: Antenna Velocity
First, we calculate the velocity of the radar antenna itself in the **Body Frame**. This includes the drone's linear velocity plus the "Lever Arm" effect caused by the drone's rotation.

$$
\mathbf{v}_{ant, b} = \mathbf{v}_b(t) + \boldsymbol{\omega}_b(t) \times \mathbf{T}_{b\gets s}
$$

* $\mathbf{v}_b(t)$: Body linear velocity ($\mathbf{R}_{w\gets b}^T \mathbf{v}_w$).
* $\boldsymbol{\omega}_b(t)$: Body angular velocity.
* $\mathbf{T}_{b\gets s}$: The fixed offset of the radar from the center of mass.
* $\times$: Cross product.

### 4.2. Geometry: The Ray Direction
The radar detects a point at $\mathbf{p}_s$ (in Sensor Frame). The unit direction vector of this ray in the **Sensor Frame** is:
$$
\hat{\mathbf{u}}_s = \frac{\mathbf{p}_s}{||\mathbf{p}_s||}
$$

To compare this with our Body Frame velocity $\mathbf{v}_{ant, b}$, we must rotate this ray into the **Body Frame** using $\mathbf{R}_{b\gets s}$:
$$
\hat{\mathbf{u}}_b = \mathbf{R}_{b\gets s} \hat{\mathbf{u}}_s
$$

### 4.3. The Projection (Doppler Constraint)
The Doppler measurement $v_D$ is the dot product of the antenna velocity and the ray direction (both now in Body Frame):

$$
v_{D} = \hat{\mathbf{u}}_{b} \cdot \mathbf{v}_{ant, b} + \epsilon
$$

Substituting the full terms:

$$
v_{D} = (\mathbf{R}_{b\gets s} \hat{\mathbf{u}}_s) \cdot \left( \mathbf{v}_b(t) + \boldsymbol{\omega}_b(t) \times \mathbf{T}_{b\gets s} \right) + \epsilon
$$


### Important Note on Sign Convention
The formula above produces a **positive** value when the sensor moves **towards** the target (closing speed).
* **Check your Radar:** Many radars (including TI default) report closing speed as **negative** (range rate $\dot{r} < 0$).
* **Implementation:** If your radar reports negative for closing, you must use $v_D = - (\dots)$ or flip the sign of your measurements during pre-processing.

## 5. Summary Table: What We Estimate vs. What We Measure

| Quantity | Type | Source | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| $\mathbf{p}_w(t)$ | **State** | Spline Control Points | **Unknown** (To be solved) |
| $\mathbf{v}_w(t)$ | **State** | Spline Derivative | **Unknown** (To be solved) |
| $\mathbf{a}_w(t)$ | **State** | Spline 2nd Derivative | **Unknown** (To be solved) |
| $\mathbf{b}_a(t)$ | **State** | Estimator Variable | **Unknown** (To be solved) |
| $\mathbf{z}_{acc}$ | **Measure** | IMU Topic | Constraint (Target for $\mathbf{a}_w$) |
| $\boldsymbol{\omega}_b$ | **Measure** | IMU Topic | Input (Drives Rotation $R_{w\gets b}$) |
| $v_{D,k}$ | **Measure** | Radar Topic | Constraint (Target for $\mathbf{v}_w$) |

## 6. Rosbag Topics: 
| Title | Description |
| `/angrybird2/agiros_pilot/state` | kalman-smoothed mocap data. Pose is accurate, derivative may be slightly jagged (if it is, fit spline to pose and derive analytically) |
|`/angrybird2/imu` | Raw IMU data from Pixhawk | 
| `/mmWaveDataHdl/RScanVelocity` | Raw Radar Data. Analysis shows a 0.018879s delay relative to integrated IMU curve. The radar is mounted tilting downwards 30deg from horizontat, about 7cm forward in x-axis. |