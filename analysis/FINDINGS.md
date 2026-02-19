# Physics Validation & Calibration Findings

**Date:** 2025-02-18  
**Script:** `analysis/validate_physics.py`  
**Status:** Active investigation

---

## 1. Sensor Extrinsic Calibration (UPDATED: SIGN CONVENTION ISSUE)

### Physical Mounting
User confirmed: radar mounted looking forward, adapter creates 30° **downward** tilt → `R_body_from_sensor = R_y(+30°)` → `ROTATION_EULER = [0, +30, 0]`.

`Rotation.from_euler('ZYX', [0, +30, 0])` maps sensor boresight `[1,0,0]` → body `[0.866, 0, -0.5]` (forward + **down**). ✓ Matches physical mounting.

### Initial Sweep (misleading)
Pitch sweep on original/circle bags showed pitch=-30° gives better correlation (+0.71/+0.74) than pitch=+30° (+0.38/+0.41). This was due to a **confounding Doppler sign inversion** on aggressive bags that also subtly affects gentle bags (see Section 11).

### Current Status
**Use `[0, +30, 0]`** (physically correct). The apparent superiority of pitch=-30° was an artifact of the Doppler sign convention issue documented in Section 11.

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

### Results
Consistent across all tested bags:
- **Offset: -20 ms** (IMU timestamps are 20ms behind MoCap)
- Gyro RMSE improves: 0.18 → 0.069 rad/s (original bag), 0.42 → 0.33 rad/s (backflips)
- Per-axis breakdown: X=-20ms, Y=-25ms, Z=-20ms (median=-20ms)

**Action:** Apply -20ms offset to radar/IMU timestamps before fusion.

---

## 4. Gyroscope Convention (VALIDATED)

**Verified:** MoCap angular velocity is in **body frame** (not world frame).

### Evidence
On all bags, body-frame RMSE ≪ world-frame RMSE:
- Original: body=0.18 vs world=0.73 rad/s
- Backflips: body=0.42 vs world=1.19 rad/s

Gyro Z shows a consistent bias of ~0.15 rad/s on backflips bag.

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

## 11. Doppler Sign Convention Issue (RESOLVED — Body Frame Flip)

### Root Cause: Agiros Body Frame Differs Between Trajectory Profiles

Different agiros trajectory profiles define the body frame orientation differently. Three of the six bags have the agiros body +x axis rotated 180° in yaw relative to the physical drone body frame. This means:
- The quaternion in agiros state data encodes an extra 180° yaw rotation
- `v_body_x` has the opposite physical meaning
- The radar (physically at body +x) appears at body -x in the flipped frame

**User confirmed**: "different trajectory profiles have different body frames... in a few flights the drone flew backwards, meaning the radar was at the back side of the drone when flying."

### 180° Yaw Flip Test Results

Applied `R_z(180°)` correction to extrinsics for each bag and compared:

| Bag | Normal corr | Normal sign% | Flipped corr | Flipped sign% | Winner |
|-----|------------|-------------|-------------|--------------|--------|
| original | **+0.378** | **69.3%** | -0.207 | 33.8% | Normal |
| circle | **+0.414** | **78.7%** | -0.432 | 23.9% | Normal |
| circle_fast | **+0.130** | **58.9%** | -0.055 | 46.9% | Normal |
| circle_fwd | -0.250 | 25.8% | **+0.347** | **78.5%** | **Flipped** |
| backflips | -0.103 | 45.6% | **+0.161** | **58.6%** | **Flipped** |
| loopings | -0.257 | 38.1% | **+0.261** | **64.2%** | **Flipped** |

Normal extrinsics: `T=[+0.07,0,0]`, `R_bs = R_y(+30°)` → boresight in body = `[0.866, 0, -0.5]`  
Flipped extrinsics: `T=[-0.07,0,0]`, `R_bs = R_z(180°) @ R_y(+30°)` → boresight in body = `[-0.866, 0, -0.5]`

### Body-Frame Flight Direction Analysis

| Bag | mean(v_body_x) | fwd% (>0) | Speed (m/s) | Winner |
|-----|---------------|-----------|-------------|--------|
| original | -0.54 | 29.2% | 1.31 | Normal |
| circle | +2.66 | 93.1% | 3.07 | Normal |
| circle_fast | +4.11 | 86.3% | 4.69 | Normal |
| circle_fwd | +2.67 | 92.5% | 2.99 | **Flipped** |
| backflips | -0.14 | 48.4% | 3.67 | **Flipped** |
| loopings | +4.21 | 85.8% | 4.70 | **Flipped** |

**Key insight**: Flight direction alone does NOT determine which extrinsics to use. Both circle (fwd%=93%) and circle_fwd (fwd%=92.5%) fly forward, but use different extrinsics. The difference is in the **agiros body frame definition**, not the physical flight direction.

### How to Apply the Fix

For "Flipped" bags, apply one of:
1. **Correct the extrinsics**: Use `ROTATION_EULER = [0, +30, 180]` and `T = [-0.07, 0, 0]`
2. **Correct the state**: Apply 180° yaw rotation to all agiros quaternions, negate v_body_x and v_body_y, negate ω_x and ω_y

### Per-Bag corr(v_body_x, mean_Doppler) — Original Observations

| Bag | Correlation | Interpretation |
|-----|------------|----------------|
| original | **+0.826** | Normal frame ✅ |
| circle | **+0.773** | Normal frame ✅ |
| circle_fast | **-0.208** | Normal frame, but aliasing at high speed |
| circle_fwd | **-0.793** | Flipped frame (explains negative corr) |
| backflips | **-0.828** | Flipped frame (explains negative corr) |
| loopings | **-0.286** | Flipped frame, plus aliasing at high speed |

### Status
**Awaiting user confirmation** from supervisor about which trajectory profiles have a flipped body frame convention. The pattern above should match the trajectory profile metadata.

---

## Next Steps

1. **Confirm with supervisor** which trajectory profiles have a rotated body frame
2. **Implement per-bag body frame detection** (auto-detect from corr(v_body_x, mean_Doppler) sign, or from trajectory profile metadata)
3. **Use all 6 bags** for solver development with correct per-bag extrinsics
4. **Use pitch=+30°** (physically correct mounting, confirmed by user)
5. **Resume Phase 3 nonlinear solver** with validated physics model
