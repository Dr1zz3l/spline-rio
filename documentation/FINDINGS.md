# Physics Validation & Calibration Findings

**Date:** 2025-02-18 (last updated 2026-05-31)
**Script:** `analysis/validate_physics.py`, `analysis/validate_nonlinear_solver.py`, `analysis/diagnostics/diagnose_gyro.py`, `analysis/diagnostics/diagnose_doppler_sign.py`
**Status:** Foundational calibration findings; kept as reference

> **Scope note:** This document records the foundational physical/calibration
> discoveries (body-frame convention, time offsets, Doppler sign, extrinsics).
> Current RMSE results, hyperparameters, and solver config live in `CLAUDE.md`.
> §6 and §12 describe experiments run in the **older MoCap-initialized batch
> pipeline**; the current live solver is MoCap-free (P1–P3 sensor-only init).

---

## 1. Sensor Extrinsic Calibration (UPDATED: 180° ROLL MOUNT)

### Physical Mounting
User confirmed: radar mounted **upside-down** (180° roll) looking forward, 3D-printed adapter creates ~30° **downward** tilt.

Physical mount: `[180, +30, 0]` (roll, pitch, yaw) = `Rx(180°)·Ry(+30°)`.

**As-built calibrated value (2026-06-15): 27.5°, frozen.** The full-trajectory
**batch** solve self-calibrates pitch to **27.0°/27.2°** (slow/fast racing) *from the
legacy 25.5° init* — a few degrees under the 30° CAD nominal.  **The batch pitch self-cal
is init-DEPENDENT** (2026-06-24 sweep, BOTH racing bags): the converged value tracks the
init — 20°→16°, 23°→22°, 25.5°→27°, 27.5°→31°, 30°→36°, 33°→42° — diverging from a saddle
near ~23°.  So the earlier "init-independent recovery" claim was **WRONG** (it was in the
paper; now retracted in paper+report). 27° is just where the legacy 25.5° init lands, not
a uniquely recovered angle.  Freezing rests on the pitch being **weakly observable** (the
position RMSE is a shallow plateau — slow 0.19–0.33 m, fast ~flat 0.69–0.73 m across pitch
16–42°), not on recovery.  This is **frozen at 27.5°** for all sliding-window runs: the SW
likewise cannot observe a 1-DOF extrinsic per 3 s window (free pitch drifts init-dependently
to 29.5/34.7/40°).
`extrinsics.yaml` still holds **25.5°** as the *batch self-cal init only* — it was an
earlier, too-low config value, now understood to be wrong; deployed runs lock 27.5° via
`--set-ext`. (Earlier versions of this doc reported 25.5° as the answer.)

The previous value of `[0, +30, 0]` was incorrect — it did not account for the upside-down mounting. The 180° roll was discovered through radar boresight analysis.

### Extrinsic Accuracy
- **Pitch (27.5° as-built / 30° CAD nominal):** 3D-printed mount introduces ±2° uncertainty;
  **batch** pitch self-calibration recovers 27.0/27.2° and it is then frozen at 27.5° (the
  SW does not self-calibrate — it drifts). Only pitch is observable from Doppler;
  roll and yaw are locked (`optimize_pitch_only: true`).
- **Translation ([0.08, 0.02, -0.01]):** 8 cm forward, 2 cm left, 1 cm down in body frame. Updated from eyeballed 7 cm forward estimate.
- **Impact of errors:** A 2° pitch error causes ~0.05 m/s systematic Doppler error.
  Translation errors (lever arm) are significant at high angular rates — added to
  `RadarDopplerFunctor` in Phase 4b.

### Current Status
**Deployed extrinsic pitch is frozen at 27.5°** (the as-built value; batch self-cal
yields 27.0/27.2° from the legacy 25.5° init, but is init-DEPENDENT — see above, not a
unique recovery). `extrinsics.yaml` retains 25.5° only as the
batch self-cal init; racing/backflips runs lock 27.5° via `--set-ext`. For flipped bags,
additionally apply `Rz(180°)` (see Section 11).

---

## 2. Body Frame Convention (VERIFIED)

**Verified:** Drone body frame is **FLU** (x=Forward, y=Left, z=Up).

### Evidence
- IMU accelerometer during hover reads `[-0.88, 0.07, +10.04]` m/s² → z ≈ +g → **z = up** ✓  
- Body z-axis in world during hover: `[-0.02, -0.10, +0.995]` → aligned with world z (up) ✓
- Agiros quaternion represents **R_world_from_body** (confirmed by accel forward model, corr 0.83+)
- Yaw during original bag hover ≈ -81° (drone was facing roughly world -y)

---

## 3. IMU–MoCap Time Offset (VALIDATED)

### Method
Cross-correlation between MoCap-derived angular velocity and IMU gyroscope readings, sweeping lag from -200ms to +200ms.

### Results (historical — empirically retuned since)
Cross-correlation originally identified:
- **Offset: +20 ms** applied to IMU/radar timestamps (they lag MoCap by ~20 ms)
- Gyro RMSE improves: 0.18 → 0.069 rad/s (original bag), 0.42 → 0.33 rad/s (backflips)
- Per-axis breakdown: X=−20 ms, Y=−25 ms, Z=−20 ms (median=−20 ms)

**Current values** (see `analysis/config/extrinsics.yaml` — single source of truth):
- `imu_mocap_offset_sec: 0.0150` (IMU clock behind MoCap by 15 ms)
- `radar_imu_offset_sec: 0.1350` (radar processing latency behind IMU)

**Convention:** positive `imu_mocap_offset_sec` means we ADD this to IMU timestamps
before MoCap interpolation (IMU is behind MoCap).  Radar total offset =
`imu_mocap_offset - radar_imu_offset = 0.015 − 0.135 = −0.120 s`
(radar data arrives ~120 ms late relative to MoCap).

---

## 4. Gyroscope Convention (VALIDATED + BIAS CONFIRMED)

**Verified:** MoCap angular velocity is in **body frame** (not world frame). Confirmed by projection tests (direct correlation 0.40/0.78/0.83 > all rotated alternatives).

### Evidence
On all bags, body-frame RMSE ≪ world-frame RMSE:
- Original: body=0.18 vs world=0.73 rad/s
- Backflips: body=0.42 vs world=1.19 rad/s

### Gyroscope Bias (CONFIRMED REAL)
`diagnose_gyro.py` cross-validated against MoCap-derived angular velocity:
- **circle bag:** z-bias = +0.279 rad/s (+16°/s), x-corr=0.40, y-corr=0.79, z-corr=0.87
- **circle_fwd bag:** z-bias = +0.177 rad/s (+10.1°/s)
- Bias differs between flights → **MEMS thermal drift** (not a fixed offset)
- Identity axis mapping is optimal (no sign flips or axis swaps needed)
- Solver gyro z-bias estimate of 0.293 rad/s matches diagnostic 0.279 rad/s
- Best IMU-MoCap time offset for gyro: ~0ms (NOT the +20ms used for radar)

**Note:** `/angrybird2/imu` is the **drone's own Pixhawk IMU**, NOT the radar board IMU. The agiros `angular_velocity` field is body-frame Kalman-filtered true angular velocity (gyr_bias field is all zeros — not published).

---

## 5. Gravity Direction (VALIDATED)

**Verified:** `g_world = [0, 0, -9.81]` is correct.

### Evidence
- Static hover test (backflips bag, 0–15s): orientation RMSE = 2.4° using this gravity
- Accel Z correlation = 0.94 on backflips bag
- Accel Z mean offset ≈ -1.15 m/s² (consistent with small IMU bias, not sign error)

---

## 6. Accelerometer Forward Model (VALIDATED with caveats)

### Forward model: `z_imu_pred = R_body_from_world @ (a_world - g_world)`

### Source of `a_world`:

| Source | Status | Notes |
|--------|--------|-------|
| **Agiros `.acceleration` field** | ❌ NOT coordinate acceleration | Z mean = 0.000 exactly, uncorrelated with velocity differentiation. Likely commanded/reference acceleration from trajectory planner. Works accidentally on gentle flights because gravity dominates (~9.81 m/s²) and dynamics are small. |
| **Velocity differentiation** | ✅ Usable after cleaning | MoCap has near-duplicate timestamps (dt ~5µs) causing vel_diff spikes of 10,000+ m/s². After filtering (MIN_DT=1ms) + SavGol smoothing (window=15), yields reasonable results. |

### Results with cleaned vel_diff (backflips bag)

| Axis | Correlation | RMSE |
|------|-------------|------|
| X | 0.15 → 0.18 (after -20ms) | 4.68 m/s² |
| Y | 0.14 → 0.16 (after -20ms) | 4.44 m/s² |
| **Z** | **0.94 → 0.95** (after -20ms) | **3.61 m/s²** |

Z-axis validates the model. Low X/Y correlations are expected because SavGol smoothing attenuates the fast dynamics during backflips. On the original bag (gentler motion), Agiros accel gives X/Y correlations of 0.83/0.84.

---

## 7. Radar Doppler on Aggressive Flight (RESOLVED — see Section 11)

Superseded by the comprehensive Doppler sign convention analysis in Section 11.

---

## 8. MoCap Data Quality

### Near-duplicate timestamps
Both bags have MoCap samples with dt ~5µs (normal dt ~3.3ms). This causes:
- Velocity differentiation spikes of 10,000+ m/s²
- Backflips bag: 396 spikes > 100 m/s², 34 > 1000 m/s²
- Original bag: 59 spikes > 100 m/s², 10 > 1000 m/s²

**Fix:** Filter samples with dt < 1ms before differentiation, then apply SavGol smoothing (window=15).

### Doppler quantization
16 chirps per frame → Doppler resolution = V_MAX / (N_chirps/2) = 4.99 / 8 ≈ 0.624 m/s per bin. Observed as vertical lines in pred-vs-meas scatter plots. Unique measured velocity values: 15–16 per flight phase.

---

## 9. Radar Driver Coordinate Transform

The TI mmWave driver maps native coordinates to ROS FLU convention:
```
ROS X (forward)  = mmWave Y (boresight/range direction)
ROS Y (left)     = -mmWave X (negative azimuth)
ROS Z (up)       = mmWave Z (elevation)
```

No rotation compensation, no IMU compensation, no gravity compensation. The TF publisher uses an identity transform. Point cloud is in sensor body frame.

---

## 10. Available Bags Summary

| Key | File | Character | Body Frame | Extrinsics | Notes |
|-----|------|-----------|-----------|------------|-------|
| original | `2025-12-17-16-02-22.bag` | Gentle | Normal | `[0,+30,0]` | corr=+0.38, sign=69% |
| circle | `circle_2025-12-17-17-21-37.bag` | Moderate circles | Normal | `[0,+30,0]` | corr=+0.41, sign=79% |
| circle_fast | `circle_fast_2025-12-17-17-25-34.bag` | Fast circles | Normal | `[0,+30,0]` | corr=+0.13, sign=59% |
| circle_fwd | `circle_forward_2025-12-17-17-37-38.bag` | Circles + forward | **Flipped** | `[0,+30,180]` | corr=+0.35, sign=79% (after flip) |
| backflips | `backflips_2025-12-17-17-41-24.bag` | Repeated backflips | **Flipped** | `[0,+30,180]` | corr=+0.16, sign=59% (after flip) |
| loopings | `circle_fast_forward_2025-12-17-17-39-49.bag` | Fast circles + fwd | **Flipped** | `[0,+30,180]` | corr=+0.26, sign=64% (after flip) |

---

## 11. Doppler Sign Convention (RESOLVED — SymForce Sign Error Fixed)

### Summary

The SymForce radar residual used the wrong sign for `v_pred`, causing the solver to minimize the wrong objective. The fix improved solver results by ~5× (pos RMSE 2.0 → 0.4 m, vel RMSE 1.3 → 0.2 m/s).

### Two Independent Code Paths

The codebase had two Doppler forward model implementations that disagreed:

| Code path | Formula | Used by |
|---|---|---|
| `predict_doppler_velocity()` | `v_pred = -dot(u_body, v_ant)` | `validate_physics.py` |
| `radar_residual_with_jacobians()` (SymForce) | `v_pred = +dot(u_body, v_ant)` | `validate_nonlinear_solver.py` |

The SymForce residual was `v_meas - dot(u,v)`. But the TI radar convention is `v_meas = -dot(u,v)` (positive = receding), so at ground truth the residual was `-2·dot(u,v)` instead of zero.

### Diagnostic Evidence (`diagnostics/diagnose_doppler_sign.py` on `slow_racing_best_velocity`)

| Convention | Correlation with v_meas | RMSE | Huber-suppressed |
|---|---|---|---|
| `v_pred = -dot(u,v)` (correct), flip=OFF | **+0.85** | **0.83 m/s** | **4.3%** |
| `v_pred = +dot(u,v)` (SymForce), flip=ON | +0.74 | 1.10 m/s | 16.0% |
| `v_pred = +dot(u,v)` (SymForce), flip=OFF | −0.85 | 2.92 m/s | 76.4% |

### Why the Solver "Worked" Before

With the wrong sign and `flip=ON`, the yaw flip effectively negated the forward direction of travel, partially compensating for the sign error. Correlations were positive (~0.74) and 16% of points were Huber-suppressed — radar was contributing but weakly. The trajectory was steered mostly by IMU with radar providing noisy corrections.

### The Fix

`derive_jacobians_symforce.py` changed from:
```python
v_pred = u_body.dot(v_ant)          # wrong
return sf.V1(v_meas - v_pred)
```
to:
```python
v_pred = -u_body.dot(v_ant)         # TI IWR6843 convention: positive = receding
return sf.V1(v_meas - v_pred)
```

`generated_jacobians.py` was regenerated. `predict_doppler_velocity()` comment updated (negation was already correct, just undocumented).

### Yaw-Flip Status

The yaw flip (`FLIP_BODY_FRAME`) was previously applied to `circle_fwd`, `loopings`, `backflips`, `slow_racing_best_velocity`. With the sign fix:
- `slow_racing_best_velocity` confirmed to work with `--no-flip` (best results yet)
- The other flipped bags (`circle_fwd`, `loopings`, `backflips`) need re-evaluation — they may also work without the flip now, or the flip may reflect a real physical difference in the agiros body frame convention for those trajectory profiles

### Historical Note: Body Frame Flip Investigation

Earlier investigation found that `validate_physics.py` correlation improved with the flip for those bags. This was because `validate_physics.py` uses `predict_doppler_velocity` (correct `-dot` sign), so `flip=ON` was partially fixing the wrong sign via the extrinsic change — making `+dot` look better than `-dot` when evaluated with the flipped body frame. The real physics always used `-dot`.

---

## 12. Nonlinear Solver — Preconditioning & Damping Experiments (2026-02-19)

> **Pipeline note:** These experiments were run on the **older Python batch solver**
> with MoCap SLERP initialization and tangent-space perturbation `δ(t)` around the
> nominal.  The current C++ Ceres solver with cumulative SO(3) B-spline was not yet
> implemented.  Findings motivated the switch; see `RESEARCH_NOTES.md §7` for details.

### Context
Phase 3 Python solver on **backflips** bag (30 s offset, 5 s window, DT_ORI=0.1 s).
Biases locked to zero. Orientation initialized from MoCap SLERP, position from Phase 2 linear solver.
288 state variables: 117 position + 165 orientation + 6 biases (locked).

### Experiment 1: Jacobi Preconditioning (conservative damping: ×0.5 / ×5.0)

Colleague hypothesis: scale mismatch between position (meters) and orientation (radians) variables causes ill-conditioned Hessian, leading to zigzagging convergence.

**Implementation:** `M = diag(1/√diag(H))`, solve `(M·H·M)·δx' = M·b`, unscale `δx = M·δx'`.

| Metric | Without Precond | With Precond |
|--------|----------------|--------------|
| Pos RMSE | **5.6844 m** | **5.6844 m** |
| Vel RMSE | **4.1408 m/s** | **4.1408 m/s** |
| Ori RMSE | **34.7323°** | **34.7323°** |
| Rejected iters | 9/20 | 9/20 |
| Final cost | 44,639 | 44,639 |
| Final LM λ | 0.0954 | 0.0954 |
| Runtime | 2m 13s | 2m 16s |

**Result:** Identical — every iteration, cost value, update norm, and damping value matches exactly. The preconditioning is a no-op for this problem size (288 variables). The Hessian diagonal values are already at similar scales, so Jacobi scaling doesn't change the search direction.

### Experiment 2: Aggressive LM Damping (×0.1 / ×10.0)

Colleague hypothesis: conservative damping (×0.5 decrease / ×5.0 increase) causes the solver to linger in gradient-descent mode. Aggressive strategy (×0.1 / ×10.0) should snap into Gauss-Newton mode faster.

| Metric | Conservative (×0.5/×5) | Aggressive (×0.1/×10) | Aggressive + Precond |
|--------|----------------------|----------------------|---------------------|
| Pos RMSE | 5.6844 m | **5.5955 m** | **5.5955 m** |
| Vel RMSE | 4.1408 m/s | **4.1054 m/s** | **4.1054 m/s** |
| Ori RMSE | 34.7323° | **34.3186°** | **34.3186°** |
| Rejected iters | 9/20 | 10/20 | 10/20 |
| Final cost | 44,639 | **45,057** | **45,057** |
| λ trajectory | 5e-4 → 0.095 | 1e-4 → 1e-15 → 1e-7 | same |
| Runtime | 2m 13s | 2m 14s | 2m 12s |

**Result:** Marginal improvement (<2%) in RMSE metrics. The aggressive strategy drops λ to 1e-15 by iteration 10 (pure Gauss-Newton), but then gets stuck — 10 consecutive rejected steps while λ crawls back up. The solver reaches approximately the same local minimum either way.

### Root Cause Analysis

The real issue is **not** numerical conditioning or damping strategy. The solver converges to a bad local minimum where:

1. **Orientation drifts monotonically** from 0° → 34° over the first 11 iterations, then gets stuck
2. **Accelerometer cost dominates** (43,000 of 45,000 total cost = 96%) — the IMU accel model is fighting the optimizer
3. The accel residuals pull orientation away from the MoCap-initialized SLERP, but never find a better solution
4. Radar cost (841) and gyro cost (789) are small and healthy

The accelerometer forward model `z_pred = R_body_from_world @ (a_world - g)` relies on accurate coordinate acceleration from the position spline's second derivative. During backflips (angular rates ~10 rad/s, centripetal accelerations ~25 m/s²), the spline's acceleration estimate is poor, creating large residuals that dominate the cost and corrupt the orientation estimate.

### Conclusion

- **Jacobi preconditioning**: No effect (problem is not ill-conditioned at this scale)
- **Aggressive LM damping**: Marginal improvement, same local minimum
- **Bias locking**: Already implemented (LOCK_BIASES=True), prevents the bias-cheating issue colleague identified
- **Bottleneck**: Accelerometer model accuracy during extreme dynamics, not solver numerics

---

## Next Steps (Updated 2026-03-16)

1. **Re-evaluate flipped bags without yaw flip** — test `circle_fwd`, `loopings`, `backflips` with `--no-flip` now that the sign is fixed; update `config/bags.yaml` `flipped` list accordingly
2. **Investigate z-velocity bias** — persistent -0.5 to -0.65 m/s error; likely from poor radar elevation diversity (IWR6843 has only 2 TX antennas). Not caused by pitch angle error (2° → only 0.05 m/s).
3. **Calibrate radar extrinsics offline** — use MoCap ground truth to find true $R_{bs}$ and $t_{bs}$; with the sign fix the residuals are now well-behaved enough to attempt this
4. **Decouple accelerometer from orientation** — zero out $\partial(r_{accel})/\partial(\text{ori CPs})$ so accel only affects position. Would allow increasing LAMBDA_ACCEL for better z-vel without corrupting orientation.
5. **Collect new data** with better velocity resolution radar config (0.06 m/s bins instead of 0.63 m/s)
6. **Sliding window formulation** for real-time operation
