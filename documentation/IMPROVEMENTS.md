# Potential Improvements & Ideas

**Last updated:** 2026-03-07

---

## High Priority

### 1. Calibrate Radar Extrinsics Offline
**Problem:** `ROTATION_EULER_DEG = [180, 30, 0]` and `TRANSLATION = [0.07, 0, 0]` are approximate (3D-printed mount, eyeballed offset). Possible z-offset not accounted for.  
**Approach:** Given MoCap ground truth (known body velocity and angular velocity), solve for `R_bs` and `t_bs` that minimize radar Doppler residuals across the bag.  
**Blocked by:** Current Doppler resolution (0.63 m/s) may be too coarse for fine rotation calibration. Translation calibration is impossible at this resolution (lever arm effect is ~0.1 m/s, i.e. 1/6 of a bin). **Needs better velocity resolution bag** (planned: 0.06 m/s resolution config).  
**Impact:** Removes one source of systematic error in the forward model.

### 2. Decouple Accelerometer from Orientation
**Problem:** Increasing `LAMBDA_ACCEL` above ~0.01 causes orientation to degrade. The accelerometer Jacobian has terms for both orientation and position CPs. With ~5° orientation error, gravity misalignment (~0.85 m/s²) creates systematic residuals that the optimizer "fixes" by warping orientation.  
**Approach:** Zero out `∂(accel residual)/∂(orientation CPs)` in the Jacobian. Orientation would be determined solely by gyro + radar. Position (including z-velocity) benefits from accel without orientation corruption.  
**Risk:** Orientation errors cause incorrect gravity subtraction → biased position acceleration. But this may be more manageable than the current coupling.  
**Impact:** Would allow increasing LAMBDA_ACCEL for better z-velocity constraint.

### 3. Fix z-velocity Bias
**Problem:** Persistent -0.5 to -0.65 m/s z-velocity error across bags.  
**Root cause:** Poor radar elevation diversity (IWR6843 has only 2 TX antennas). Almost all points lie near the boresight, giving one equation with two unknowns (v_x and v_z in body frame). The optimizer can trade off v_x and v_z along the boresight direction without changing radar cost.  
**Potential fixes:**
- Decouple accel from orientation (#2 above) then increase LAMBDA_ACCEL
- Add zero-mean world-z velocity prior for known level-flight segments
- Better radar config with more elevation diversity (hardware limitation)
- Calibrate extrinsics (#1 above) to remove pitch angle error contribution (~0.05 m/s per 2°)  
**Impact:** Critical for accurate 3D trajectory estimation.

---

## Medium Priority

### 4. Collect Better Resolution Radar Data
**Problem:** Current config: 16 chirps → 0.63 m/s per Doppler bin, max ±4.99 m/s. Only ~16 discrete velocity levels, aliasing on fast bags.  
**Approach:** Use `6843AOP_best_velocity.cfg` for future data collection → 0.06 m/s resolution (10× better).  
**Trade-off:** Better velocity resolution at the cost of range resolution.  
**Status:** Planned for next data collection session.

### 5. Gravity-Direction Prior
**Problem:** The accelerometer measures gravity direction in body frame, but this information is entangled with position acceleration in the current model.  
**Approach:** Add a direct orientation constraint: at low-acceleration moments (‖a_meas‖ ≈ 9.81), penalize deviation between R^T g_world and the measured accel direction. This constrains roll/pitch without coupling to position.  
**Variants:**
- Static tilt: only at near-hover timestamps
- Multirotor physics: thrust always along body +z, constraining R loosely
- Average tilt over full flight  
**Risk:** Similar to accelerometer — relies on distinguishing gravity from dynamic acceleration. During aggressive maneuvers the assumption breaks.

### 6. Combined SymForce Residuals
**Problem:** Current implementation uses hand-coded chain rules to combine SymForce Jacobians with angular velocity Jacobians (for radar and accel residuals that depend on ω).  
**Approach:** Create single SymForce residuals that take `(v_world, R_nominal, delta, delta_dot, omega_nominal, ...)` as inputs and derive all Jacobians automatically.  
**Impact:** Eliminates manual chain rule code, higher confidence in Jacobian correctness. Low priority since current chain rules are verified.

---

## Low Priority / Future

### 7. Sliding Window Real-Time Formulation
**Problem:** Current batch optimizer runs offline (~5 min for 5s of data).  
**Approach:** Fixed-duration sliding window (1–2s) that advances with new measurements. Marginalize trailing control points, warm-start from previous solution.  
**Open questions:** Marginalization strategy (Schur complement vs. drop), initialization without MoCap, bias observability in short windows.  
**Target:** 10–100 ms/window on Jetson Orin.

### 8. Higher-Degree Orientation B-spline
**Problem:** Currently degree 3 (cubic) → angular velocity is only C¹ continuous (piecewise linear derivative).  
**Approach:** Increase to degree 4 or 5. Requires more control points and increases state dimension.  
**Impact:** Smoother angular velocity representation, but may not matter given sensor noise levels.

### 9. Automatic Body Frame Detection
**Problem:** Currently `FLIPPED_BAGS` set is hard-coded. Need to manually identify which bags have flipped agiros body frame.  
**Approach:** Auto-detect from `corr(v_body_x, mean_Doppler)` sign, or read trajectory profile metadata from the bag.  
**Impact:** Quality of life improvement for processing new bags.

### 10. Model Doppler Quantization in Loss
**Problem:** 0.63 m/s quantization means a "correct" measurement can be 0.315 m/s off. Current Huber loss (δ=1.0) handles this roughly.  
**Approach:** Replace smooth residual with "flat bottom" loss where errors within ±half-bin have zero cost.  
**Impact:** Theoretically correct but likely marginal improvement over current Huber tuning.
