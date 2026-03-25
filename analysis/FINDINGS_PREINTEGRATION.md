# IMU Preintegration Findings (2026-03-25)

## Setup

Solver: `validate_live_solver.py` with `--mocap-yaw` (MoCap heading + position boundary priors).
IMU is pre-downsampled to ~200 Hz (dynamic factor: `len(imu_data) // (DURATION * 200)`).
Both bags fly the same racing loop path — `fast_racing` is more aggressive with much higher Doppler aliasing.

**Bag timing convention** — `bags.yaml` entries are `[start_offset_sec, duration_sec]`:
- `slow_racing_best_velocity: [16.1, 25.8]` → 25.8 s of racing from t=16.1 s in the bag
- `fast_racing_best_velocity: [16.1, 18.0]` → 18.0 s of racing from t=16.1 s in the bag
- t=0..16 s in both bags: drone flies to start position and hovers

## Variants Tested

### Variant 1 — Baseline (per-sample IMU at ~200 Hz)

Both accel and gyro residuals from downsampled IMU. No preintegration.

Jacobian sizes: ~50 000 × 25 194 (slow), 36 600 × 17 589 (fast).

### Variant 2 — Preintegration ON TOP of per-sample IMU (wrong)

Preintegrated factors added in addition to the existing accel+gyro rows.
The Jacobian grows rather than shrinks. No accuracy benefit.

### Variant 3 — Preintegration REPLACES accel, gyro kept (correct)

`lambda_accel = 0.0` when `--preintegrate` is active; the accel loop is guarded:

```python
for imu_msg in (imu_data if lambda_accel > 0 else []):
    ...  # skipped entirely when preintegrating
```

Without this guard, zero-lambda accel rows are still assembled → Jacobian grows.

Per-sample gyro is kept at ~200 Hz. Rationale: the cumulative SO(3) B-spline has
`dt_ori = 0.008 s` (125 Hz knots), so there are ~11 free orientation knots per
~90 ms radar interval. Preintegration only constrains the interval endpoints; gyro
pins every intermediate knot. Removing gyro causes orientation to explode to ~8000°.

Jacobian sizes with preintegration: 35 678 × 25 194 (slow), 24 786 × 17 589 (fast).

## Results

| Variant | slow pos (m) | slow ori (°) | fast pos (m) | fast ori (°) |
|---|---|---|---|---|
| Baseline | 0.349 | 9.6 | 1.456 | 5.7 |
| + Preint on top | 0.362 | 9.0 | 1.505 | 6.6 |
| Preint replaces accel | **0.366** | **3.2** | **1.48** | **11.8** |

### Jacobian / timing effect (fast_racing)

| | Baseline | Preint replaces accel |
|---|---|---|
| Jacobian rows | 36 600 | 24 786 (−32 %) |
| Per-iteration time | ~10 s | ~4.5 s (2.2×) |
| Total solve time | ~284 s | ~86 s (3.3×) |

## Interpretation

### slow_racing — orientation improves dramatically (9.6° → 3.2°)

Removing low-weight (λ=0.01) per-sample accel **helps** slow_racing.
The per-sample accel was likely contributing noise (motor/propwash vibration) that
outweighed its information content for gentle, smooth flight where gravity provides
a strong orientation reference. The preintegrated factors give a cleaner
velocity-level constraint per radar interval.

### fast_racing — orientation degrades (5.7° → 11.8°)

For aggressive manoeuvres, rapid accelerations create tight coupling between
orientation and position dynamics at the per-sample level. One preintegrated
constraint per ~90 ms is too sparse to maintain this coupling during highly
dynamic segments.

### fast_racing position error (1.456 m) is not solved by IMU changes

The dominant error source is **Doppler aliasing**. The best_velocity radar config
has v_max = 3.136 m/s; the drone exceeds this during racing loops, producing
aliased velocity measurements. The per-frame EgoVelocityWLS solver receives
systematically wrong velocities during high-speed phases. Neither preintegration
nor per-sample IMU changes address this root cause.

`integrate_radar_velocity()` already uses IMU to unwrap aliases globally during
initialisation, but the per-frame WLS solve still receives raw (potentially
aliased) Doppler values.

## IMU-Aided Pre-Unwrapping of Radar Frames (implemented, no accuracy gain)

`preunwrap_radar_frames()` was added to `validate_live_solver.py` and wired into
`main()`.  When `--unwrap` is active it pre-processes all radar frames with the
same IMU-aided velocity propagation used in `integrate_radar_velocity()`, picks
the correct Doppler alias per point, and passes unwrapped `RadarVelocity` objects
to the solver.  The solver's in-loop unwrapping (`v_max` parameter) is kept as a
safety net.

Results with pre-unwrapping enabled (--mocap-yaw):
- fast_racing: 1.451 m / 5.68° vs baseline 1.456 m / 5.7°  — **no change**
- slow_racing: 0.367 m / 9.54° vs baseline 0.349 m / 9.6°  — **no change**
- fast_racing: 20.9% of points pre-unwrapped; slow_racing: 1.7%

**Finding**: the solver's in-loop unwrapping was already handling all aliased
points correctly.  Pre-unwrapping is robust infrastructure (protects against
cascade failures where the very first frame is aliased and the solver has no
good init velocity), but contributes nothing to accuracy.

## Root cause of fast_racing position error (1.456 m): z-velocity bias

Initial radar residuals before optimisation:
- slow_racing: mean = +0.06 m/s, std = 0.79 m/s (nearly unbiased)
- fast_racing: mean = −0.44 m/s, std = 2.74 m/s (systematic bias + 3.5× noise)

The −0.44 m/s mean matches the known z-velocity systematic bias of −0.5 to
−0.65 m/s caused by limited elevation diversity (2 TX antennas).  At faster
flight speeds the z-velocity component is larger and changes more rapidly, making
this bias more impactful.  The optimiser absorbs the bias by adjusting roll/pitch:
- fast_racing roll RMSE: 7.5°  vs slow_racing: 3.4°
- fast_racing pitch RMSE: 2.3° vs slow_racing: 1.6°

The higher per-point noise (std 2.74 vs 0.79 m/s) reflects degraded radar return
quality at high speeds (more aliased returns in ambiguous bins, lower SNR,
more multi-path).  Fixing this requires better sensor-level modeling, not
unwrapping improvements.

## Potential next directions

1. **Compensate z-velocity bias explicitly**: add a per-elevation-angle attenuation
   model or estimate the z-radar bias as an additional optimisation parameter.
   The `LAMBDA_GRAVITY` factor is already implemented but currently disabled
   (lambda_gravity=0) — enabling it with a suitable weight might provide an
   orthogonal constraint that partially counteracts the elevation bias.

2. **Discard high-elevation radar points**: returns at steep elevation angles
   contribute most to the systematic z-velocity bias.  A per-point elevation
   cutoff (e.g. |elevation| < 20°) would sacrifice some geometry diversity but
   reduce the bias-induced corruption.

3. **Estimate radar z-bias as a parameter**: add a scalar `z_bias` to the state
   (like the existing `radar_extrinsic_delta`) and let the solver absorb it.

## Archived: Old Next: IMU-Aided Per-Frame Doppler Unwrapping

Extend the existing IMU-aided unwrapping from the global init phase to every
per-frame EgoVelocityWLS call. At each radar frame, propagate an IMU-predicted
body velocity forward from the previous estimate and use it to select the correct
alias offset `k` before the WLS fit:

```
v_unwrapped = v_measured + k * 2 * v_max,   k ∈ {-1, 0, +1}
```

where `k` is chosen to minimise `|v_measured + k*2*v_max - v_imu_pred|`.

This should directly reduce the fast_racing position error by removing
the systematic alias bias from the radar velocity measurements fed to the solver.
