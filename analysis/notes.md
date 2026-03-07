## 1. Coordinate System Transformations — Are They 100% Correct?

**Mostly yes, with one concern.**

The chain is:
- **World frame**: z-up, gravity = [0, 0, -9.81]
- **Body frame (agiros)**: FLU. Quaternion `[qx,qy,qz,qw]` → `R_world_from_body` — validated by IMU hover accel reading `[−0.88, 0.07, +10.04]` (z ≈ +g) ✓
- **Sensor→Body**: `R_body_from_sensor = R_y(+30°)`, translation `[0.07, 0, 0]` — physically correct (30° downward tilt, radar forward of CoG) ✓ [Note by user: the 7cm forward offset of the radar is eyeballed. it is not guaranteed. there also may be a slight z offset. im not sure if our current quantization is detailed enough to allow calibration of this offset]
  - **Translation calibration analysis**: Cannot calibrate with current data (0.63 m/s resolution). The lever arm term $|\omega \times T| \approx 0.105$ m/s at typical $\omega \approx 1.5$ rad/s — that's 1/6 of a Doppler bin. A 2cm translation error changes the lever arm by ~0.03 m/s (5% of one bin), completely invisible. Even with 0.06 m/s resolution the lever arm is only ~1.7 bins, and calibrating a 3D offset from a scalar Doppler measurement under orientation uncertainty is extremely ill-conditioned. The dominant velocity term $R^T v_w$ is 30–50× larger. **Translation uncertainty is not affecting results.**
- **Sensor coordinates**: TI driver maps mmWave native → ROS FLU: `x←mmWave_y, y←-mmWave_x, z←mmWave_z` ✓
- **Flipped bags**: `circle_fwd`, `loopings` get 180° yaw correction — documented and implemented ✓

**One concern**: The quaternion is passed to `scipy.spatial.transform.Rotation.from_quat()` and also to our custom `quat_to_rotation_matrix()`. Both expect `[qx, qy, qz, qw]` and the rosbag loader stores it that way. ✓ Consistent.

**The SymForce Rot3 shim** also uses `[x, y, z, w]` and `from_rotation_matrix()` uses Shepperd's method. The conversion path is: MoCap quat → `Rotation.from_quat` → `.as_matrix()` → SLERP → `.as_matrix()` → `Rot3.from_rotation_matrix()` → SymForce codegen. This is correct but involves a potentially lossy rotation-matrix-to-quaternion step. I see no actual bug here.

**Verdict: Transforms look correct.**

---

## 2. Doppler Sign — Is It Correct?

**Yes, but there's an important subtlety to verify.**

The forward model computes:
$$v_D = \hat{\mathbf{u}}_b \cdot (\mathbf{R}^T_{wb} \cdot \mathbf{v}_w + \boldsymbol{\omega}_b \times \mathbf{T}_{b \leftarrow s})$$

Positive $v_D$ = velocity component toward the point (closing). The TI IWR6843 SDK 3.x gives velocity directly as a float — positive means approaching the target.

Your FINDINGS.md Section 11 shows `corr(v_body_x, mean_Doppler)` is **positive** for correctly-oriented bags (original: +0.826, circle: +0.773). Since the radar looks forward (+x body +30° down), forward motion should produce positive Doppler (closing on targets ahead). ✓ Sign is correct.

**BUT**: The correlations are surprisingly weak for the fast bags (circle_fast: 0.13, loopings: 0.26). This hints at **Doppler aliasing** — the max unambiguous velocity is only ±4.99 m/s, and some of these bags have speeds exceeding that.

**Verdict: Sign is correct. Weak correlations on fast bags are likely aliasing, not a sign error.**

---

## 3. Doppler Quantization — Do We Need to Model It?

**This is a significant issue you should address.**

From 6843AOP_3d.cfg:
- **Velocity resolution: 0.63 m/s per bin** (only ~16 discrete velocity levels across ±4.99 m/s)
- **Max unambiguous velocity: ±4.99 m/s**

Your FINDINGS.md Section 8 confirms: *"16 chirps per frame → Doppler resolution = V_MAX/(N_chirps/2) = 4.99/8 ≈ 0.624 m/s per bin. Observed as vertical lines in pred-vs-meas scatter plots. Unique measured velocity values: 15–16 per flight phase."*

**Impact on the solver**: Your Huber delta is 0.5 m/s, but the quantization step is 0.63 m/s. This means a measurement that's "correct" could still be 0.315 m/s off from the true Doppler just from quantization alone — that's 63% of your Huber threshold! The solver is treating quantization noise as if it were meaningful information.

**Options**:
1. **Increase Huber delta to ~1.0 m/s** (simple fix — acknowledge that errors up to half a bin are pure noise)
2. **Switch to 6843AOP_best_velocity.cfg** for future data collection — that gives **0.06 m/s resolution** (10× better), though at the cost of range resolution
3. **Model quantization in the forward model** — replace the smooth residual with a "flat bottom" loss where errors within ±half-bin are zero-cost. This is doable but complex.

**My recommendation**: Option 1 is the quick fix. Option 2 is the real fix for future data. Option 3 is theoretically correct but probably overkill — Huber loss with a properly tuned threshold achieves a similar effect.

**Verdict: Quantization is a significant issue (0.63 m/s resolution vs 0.5 m/s Huber threshold). Increase Huber delta to at least 0.8-1.0 m/s.**

---

## 4. Initialization — How Good Is It?

**The initialization is excellent for position but has a structural problem for orientation.**

**Position init** ([validate_nonlinear_solver.py lines 1462-1467](analysis/validate_nonlinear_solver.py#L1462-L1467)):
- Cubic interpolation of MoCap positions at B-spline control point times
- This is the best possible init since it starts from ground truth

**Orientation init** ([lines 1473-1500](analysis/validate_nonlinear_solver.py#L1473-L1500)):
- `R(t) = R_nominal(t) · exp(delta(t))`
- Nominal rotations = SLERP of MoCap quaternions (ground truth!)
- Delta CPs initialized to zero → initial trajectory IS the MoCap trajectory
- This is also excellent

**So why is the solver struggling?** The init is actually very close to optimal. The problem is that the solver has to IMPROVE upon MoCap-initialized trajectories using only radar + IMU data. With:
- 0.63 m/s Doppler quantization
- Accel cost dominating (96% of total, per FINDINGS.md Section 12)
- Only 5s of data

The solver is being pulled AWAY from a good init by noisy accelerometer data. The orientation drifts "from 0° → 34° over the first 11 iterations" as your findings note.

**Key issue**: `USE_PHASE2_INIT = False`. When False, position comes from MoCap cubic interpolation — essentially cheating with ground truth. When True, position comes from the Phase 2 linear solver (radar+accel only), which is a more realistic starting point.

**Verdict: Init is fine. The problem is not initialization — it's that the accel cost dominates and corrupts what starts as a good estimate.**

---

## 5. Is SymForce Usage Correct?

**Yes, with one subtle thing to verify.**

The three SymForce-generated residuals are:
1. **Radar**: `res = v_meas - u_body·(R^T·v_world + ω×T)` — matches the predict function in radar_velocity_utils.py ✓
2. **Accel**: `res = z_acc - R^T·(a_world - g) - b_a` — standard IMU model ✓
3. **Gyro**: `res = z_gyro - (R_delta^T·ω_nom + J_r(delta)·δ̇) - b_g` — correct SO(3) model ✓

The clever trick of using `gyro_residual(z_gyro=0, b_g=0)` to extract `ω` and its Jacobians in `compute_omega_and_jacobians()` is correct: if `res = 0 - ω - 0 = -ω`, then `ω = -res` and `∂ω/∂x = -∂res/∂x`. ✓

**One thing to verify**: The SymForce convention for `R_nominal * exp(delta)`. SymForce uses Hamilton quaternions and `Rot3.from_tangent(delta)` computes $\exp(\hat{\delta})$ as a **right** perturbation. So `R_nominal * Rot3.from_tangent(delta)` = $R_{nom} \cdot \exp(\delta)$ — this is right perturbation, consistent with `get_rotation()` which does `R_nominal @ so3_exp(delta)`. ✓

**Verdict: SymForce usage is correct.**

---

## 6. Can We Use SymForce for More Math?

**Yes — here are concrete opportunities:**

a) **The full radar chain rule** (currently hand-coded at validate_nonlinear_solver.py):
```python
J_eff_val = J_delta_radar + J_omega_radar @ J_omega_wrt_delta      # hand chain rule
J_eff_dot = J_omega_radar @ J_omega_wrt_delta_dot                  # hand chain rule
```
This is correct but involves manual matrix multiplication. We could generate a **combined** SymForce residual that takes `(v_world, R_nominal, delta, delta_dot, omega_nominal, u_sensor, ...)` as input and derives Jacobians w.r.t. `delta` and `delta_dot` directly, eliminating the need for the chain rule. The B-spline basis coefficients would still be applied outside.

b) **The accelerometer chain rule** (currently hand-coded at validate_nonlinear_solver.py):
Same story — we could make SymForce compute $\partial r_{accel} / \partial \delta$ through the full composition.

c) **The `Rot3.from_rotation_matrix` function** — this is hand-written using Shepperd's method. We could validate it against scipy's implementation as a sanity check.

**The big win**: If we create a **single SymForce residual per sensor** that includes `omega_nominal` and `delta_dot` as direct inputs, we eliminate all hand-coded chain rules. This would give us higher confidence in the Jacobians.

**Verdict: Yes. A single radar residual that takes `(v_world, R_nominal, delta, delta_dot, omega_nominal, ...)` and outputs `(res, ∂res/∂v_world, ∂res/∂delta, ∂res/∂delta_dot)` would eliminate manual chain rules.**

---

## 7. Do We Need Huber Loss in SymForce?

**No — and you shouldn't.**

Huber loss is applied as an **iteratively reweighted least squares (IRLS)** weight:
```python
w = huber_weight(r, delta=huber_delta)
sqrt_w = np.sqrt(w)
all_residuals.append(r * sqrt_w)  # weighted residual
# Jacobian rows also multiplied by sqrt_w
```

This is the standard approach: keep the residual/Jacobian computation clean and "linear", then apply the robust weight externally. If you put Huber inside SymForce, the generated Jacobians would include the non-smooth Huber derivative, which has a discontinuity at $|r| = \delta$. The IRLS formulation avoids this by treating the weight as constant within each iteration.

**Verdict: No. IRLS weighting outside SymForce is the correct approach.**

---

## Summary of Found Issues & Recommendations

| Issue | Severity | Action |
|-------|----------|--------|
| **Doppler quantization vs Huber threshold** | **DONE** | Increased `HUBER_DELTA` from 0.5 to 1.0 m/s |
| **Accel cost dominates orientation** | **HIGH** | `LAMBDA_ACCEL=0.01` is current best; 0.05 causes 7°+ ori error. Decoupling accel from ori CPs is a potential fix. |
| **-20ms IMU-MoCap offset** | **DONE** | Applied: `IMU_MOCAP_OFFSET = +0.020` shifts IMU/radar timestamps forward |
| **Gyro z-bias is real MEMS thermal bias** | **DONE** | `LOCK_BIASES=False` with split priors: accel=10.0, gyro=1.0 |
| **z-velocity bias (-0.5 to -0.65 m/s)** | **HIGH** | Root cause: poor radar elevation diversity (2 TX). Accel can't help due to ori coupling. |
| **Radar extrinsics approximate** | **MEDIUM** | ROTATION_EULER=[180,30,0] (180° roll confirmed, 30° pitch approximate). Translation [0.07,0,0] eyeballed. Calibration planned with better-resolution data. |
| **Orientation B-spline only cubic (deg 3)** | **LOW** | `ori_degree = min(3, BSPLINE_DEGREE)` caps it at 3, so angular velocity (1st derivative) is only C¹ continuous |
| **SymForce chain rule could be eliminated** | **LOW** | Create combined residuals taking `(delta, delta_dot, omega_nominal)` directly |
| **Velocity aliasing on fast bags** | **INFO** | Max 4.99 m/s — new bag with better velocity resolution planned |

The **biggest open issues** right now are: (1) the -0.5 to -0.65 m/s z-velocity bias caused by poor radar elevation diversity and accel-orientation coupling, and (2) the accel-gyro tension where increasing LAMBDA_ACCEL degrades orientation. Decoupling accel from orientation control points is the most promising architectural fix.