check if there is a radar config that also returns data when stationary
add quantization to forward and reverse model?


attitude ground truth, position fitted

---

check if the full solver.yaml params are printed into the validation plot or not. including the extrinsics at start and end

---

joint calibratoin of radar time offset and full extrinsics. maybe dynamic time-warping function to correct the USB timing jitter frame-by-frame. although, if i were to allow extrinsics and timing offset to be affected by the solver, in the moving window approach we would allow a "driftable" extrinsic/timing as the window moves, not one-off fixed value as in the global optimization

---

check if gyro has different time offset to mocap than accelerometer

---


mocap yaw als weighted factor hinzufügen für ideal magnetormeter

---

accelerometer support für pitch roll https://madflight.com/AHRS/ mahony. aus dem filter die measurement functions verwenden, also wie sensor measurement zu state korreliert (was ist deren effekt). Definition eines fehler zwischen measurement und state verstehen, eventuell rausziehen werden und als added cost term in state estimation hinzufgen.

goal is bounded attitude (if needed with mocap yaw as ideal magnetometer sim) and velocity errors. if possible with variance value, for slower and highly dynamic case.

pixhawk mahony
```
/* PARAMETERS
 weights: w_acc=0.2, w_mag=0.1, w_ext_hdg=0.1, w_gyro_bias=0.1
 max gyro bias: _bias_max=0.05 (3dps)
*/

bool AttitudeEstimatorQ::update(float dt)
{
    Quatf q_last = _q;

    // Angular rate of correction
    Vector3f corr;
    float spinRate = _gyro.length();

    if (_param_att_ext_hdg_m.get() > 0 && _ext_hdg_good) {
        if (_param_att_ext_hdg_m.get() == 1) {
            // Vision heading correction
            // Project heading to global frame and extract XY component
            Vector3f vision_hdg_earth = _q.rotateVector(_vision_hdg);
            float vision_hdg_err = wrap_pi(atan2f(vision_hdg_earth(1), vision_hdg_earth(0)));
            // Project correction to body frame
            corr += _q.rotateVectorInverse(Vector3f(0.0f, 0.0f, -vision_hdg_err)) * w_ext;
        }

        if (_param_att_ext_hdg_m.get() == 2) {
            // Mocap heading correction
            // Project heading to global frame and extract XY component
            Vector3f mocap_hdg_earth = _q.rotateVector(_mocap_hdg);
            float mocap_hdg_err = wrap_pi(atan2f(mocap_hdg_earth(1), mocap_hdg_earth(0)));
            // Project correction to body frame
            corr += _q.rotateVectorInverse(Vector3f(0.0f, 0.0f, -mocap_hdg_err)) * w_ext_hdg;
        }
    }

    if (_param_att_ext_hdg_m.get() == 0 || !_ext_hdg_good) {
        // Magnetometer correction
        // Project mag field vector to global frame and extract XY component
        Vector3f mag_earth = _q.rotateVector(_mag);
        float mag_err = wrap_pi(atan2f(mag_earth(1), mag_earth(0)) - _mag_decl);
        float gainMult = 1.0f;
        const float fifty_dps = 0.873f;

        if (spinRate > fifty_dps) {
            gainMult = math::min(spinRate / fifty_dps, 10.0f);
        }

        // Project magnetometer correction to body frame
        corr += _q.rotateVectorInverse(Vector3f(0.0f, 0.0f, -mag_err)) * w_mag * gainMult;
    }

    _q.normalize();


    // Accelerometer correction
    // Project 'k' unit vector of earth frame to body frame
    // Vector3f k = _q.rotateVectorInverse(Vector3f(0.0f, 0.0f, 1.0f));
    // Optimized version with dropped zeros
    Vector3f k(
        2.0f * (_q(1) * _q(3) - _q(0) * _q(2)),
        2.0f * (_q(2) * _q(3) + _q(0) * _q(1)),
        (_q(0) * _q(0) - _q(1) * _q(1) - _q(2) * _q(2) + _q(3) * _q(3))
    );

    // If we are not using acceleration compensation based on GPS velocity,
    // fuse accel data only if its norm is close to 1 g (reduces drift).
    const float accel_norm_sq = _accel.norm_squared();
    if (accel_norm_sq > 0.9 && accel_norm_sq < 1.1) {
        corr += (k % (_accel - _pos_acc).normalized()) * w_acc;
    }

    // Gyro bias estimation
    if (spinRate < 0.175f) {
        _gyro_bias += corr * (w_gyro_bias * dt);

        for (int i = 0; i < 3; i++) {
            _gyro_bias(i) = math::constrain(_gyro_bias(i), -_bias_max, _bias_max);
        }

    }

    // Calculate unbiased gyro rates
    _rates = _gyro + _gyro_bias;

    // Feed forward gyro
    corr += _rates;

    // Apply correction to state
    _q += _q.derivative1(corr) * dt;

    // Normalize quaternion
    _q.normalize();

    return true;
}
```

---

sliding window with frames

---

radar huber loss dependant on config
same as wraparound velocity

















Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: Add gravity-direction factor to bound attitude errors

 Context

 Orientation in the solver is primarily driven by gyroscope integration (λ_gyro=1.0). The only absolute orientation anchor is the accelerometer (λ_accel=0.01), but it's 100x weaker and its
 residual r = z_acc - R^T(a_world - g) - b_a couples position and orientation — raising λ_accel above 0.01 causes 7°+ orientation error because position spline errors leak into orientation
 updates during dynamic flight.

 Result: attitude errors can drift unboundedly. The gyro constrains angular velocity, not absolute orientation. Without a strong absolute reference, bias and noise accumulate.

 Why a Mahony-style gravity factor fixes this

 The Mahony filter's key insight: the accelerometer measures gravity direction in body frame → directly constrains roll and pitch. By extracting only the direction (normalizing out
 acceleration magnitude), we decouple from linear acceleration and thus from position errors.

 This gives us a factor that:
 - Constrains absolute roll/pitch (bounded errors)
 - Has NO Jacobian w.r.t. position control points (decoupled)
 - Down-weights automatically during high dynamics when ||a|| ≫ g

 Yaw remains unobservable from gravity alone. Radar Doppler provides partial yaw constraint through velocity projection. Full yaw bounding (MoCap heading pseudo-magnetometer) is deferred —
 it's a separable problem.

 Changes

 1. config/solver.yaml — Add gravity factor config

 After lambda_ori_reg (line 25):
 # Gravity-direction factor (absolute roll/pitch from accelerometer)
 lambda_gravity: 1.0            # Weight for gravity-direction residual; 0 = disabled
 gravity_accel_threshold: 3.0   # Gaussian trust sigma (m/s²): down-weight when ||a|| deviates from g

 2. validate_nonlinear_solver.py — Add gravity factor

 2a. compute_jacobian_analytical() — New residual block (insert at line 672, between gyro and boundary priors)

 Residual (3D per IMU sample):
 z_debiased = z_acc - b_a
 g_body_measured = normalize(z_debiased) * 9.81   # gravity direction from accel
 g_body_predicted = R(t)^T @ [0, 0, -9.81]        # gravity direction from attitude
 r = g_body_measured - g_body_predicted

 Dynamic weighting (Gaussian trust on accel norm):
 w = exp(-((||z_debiased|| - 9.81) / sigma)^2)
 - At hover (||a||≈g): w≈1.0 (full trust)
 - At 2g maneuver (||a||≈19.6): w≈exp(-9.4) ≈ 0 (no trust)
 - Skip samples where w < 1e-4 or ||z_debiased|| < 1e-6

 Jacobians (hand-derived, inline — no SymForce needed):

 - w.r.t. orientation Ω_j: ∂r/∂Ω_j = -skew(g_body_pred) @ J_R_list[j]
   - Uses evaluate_full_jacobians() (includes base knots, same as accel factor)
   - skew_symmetric() already exists at line ~88
 - w.r.t. accel bias b_a: ∂r/∂b_a = (9.81/||z_d||) * (I - ẑ_d ẑ_d^T)
   - Projection matrix perpendicular to measurement direction, scaled
 - w.r.t. position: None — this is the key decoupling

 The scaling is sqrt(lambda_gravity * w_dynamic) per sample.

 2b. Function signature — Add parameters

 Add to compute_jacobian_analytical():
 lambda_gravity: float = 0.0,
 gravity_accel_threshold: float = 3.0,

 Add same to solve_trajectory_nonlinear(), thread into _build_jacobian() closure.

 2c. Cost decomposition update

 - Store J.n_gravity = n_gravity_rows (line ~889)
 - In cost decomposition (line ~1094), add n_grav = getattr(J, 'n_gravity', 0)
 - Residual order becomes: radar | accel | gyro | **gravity** | bnd_vel | bnd_pos | ...
 - Update n_g remainder: subtract n_grav
 - Add gravity= to cost print line

 2d. Config loading in main()

 After existing lambda loads (~line 1418):
 LAMBDA_GRAVITY = _SOLVER_CFG.get('lambda_gravity', 0.0)
 GRAVITY_ACCEL_THRESHOLD = _SOLVER_CFG.get('gravity_accel_threshold', 3.0)

 Pass through to solve_trajectory_nonlinear() call (~line 1920).

 Add to config summary print block (~line 975).

 Files to modify

 - analysis/config/solver.yaml — 2 new config entries
 - analysis/validate_nonlinear_solver.py — gravity factor block, parameter threading, cost decomposition

 Verification

 1. python validate_nonlinear_solver.py circle with lambda_gravity: 0.0 — baseline (current behavior, unchanged)
 2. python validate_nonlinear_solver.py circle with lambda_gravity: 1.0 — expect reduced roll/pitch RMSE
 3. python validate_nonlinear_solver.py backflips with lambda_gravity: 1.0 — verify dynamic weighting doesn't hurt aggressive maneuvers (Gaussian trust should auto-downweight at high g)
 4. Check cost_gravity in per-iteration log decreases
 5. Verify accel bias doesn't drift significantly vs baseline
