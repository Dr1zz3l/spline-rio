"""Verify gravity residual: codegen function vs manual computation at MoCap ground truth.

Usage (from analysis/):
    python diagnostics/verify_gravity_residual.py fast_racing_best_velocity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

import numpy as np
from scipy.interpolate import interp1d
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import quat_to_rotation_matrix
from config_loader import load_config
from codegen.generated_jacobians import gravity_residual_with_jacobians, Rot3

G_NORM = 9.81
EPSILON = 1e-7

# ── Sanity check at identity ────────────────────────────────────────────────
print("=" * 60)
print("SANITY CHECK: identity R, z_acc = [0,0,+9.81]")
print("=" * 60)

R_nom = Rot3.from_rotation_matrix(np.eye(3))
delta = np.zeros(3)
b_a = np.zeros(3)
z_acc_hover = np.array([0.0, 0.0, G_NORM])

res, _, _ = gravity_residual_with_jacobians(R_nom, delta, z_acc_hover, b_a, G_NORM, EPSILON)
print(f"  residual = {res.ravel()} (expected: [0, 0, 0])")
assert np.allclose(res, 0, atol=1e-6), f"Sanity check FAILED: residual = {res.ravel()}"
print("  PASSED\n")

# ── Load bag ─────────────────────────────────────────────────────────────────
_cfg = load_config()
BAGS = _cfg['bags']['bags']
_TIMING = _cfg['bags']['timing']
_EXT = _cfg['extrinsics']

bag_key = sys.argv[1] if len(sys.argv) > 1 else "fast_racing_best_velocity"
if bag_key not in BAGS:
    print(f"Unknown bag '{bag_key}'. Available: {list(BAGS.keys())}")
    sys.exit(1)

BAG_PATH = BAGS[bag_key]
START_OFFSET, DURATION = _TIMING.get(bag_key, [0.0, 5.0])
IMU_MOCAP_OFFSET = _EXT.get('imu_mocap_offset_sec', 0.020)

print(f"Bag: {bag_key}  ({BAG_PATH})")
print(f"Window: +{START_OFFSET}s  duration {DURATION}s")
print(f"IMU-MoCap offset: {IMU_MOCAP_OFFSET} s\n")

bag_data = load_bag_topics(BAG_PATH, verbose=False)
t_start = bag_data.start_time + START_OFFSET
t_end = t_start + DURATION

mocap = [p for p in bag_data.mocap_pose if t_start <= p.timestamp <= t_end]
imu   = [d for d in bag_data.imu_data   if t_start <= d.timestamp <= t_end]

print(f"MoCap samples: {len(mocap)}")
print(f"IMU samples:   {len(imu)}\n")

if len(mocap) < 2 or len(imu) == 0:
    print("Not enough data — check timing window in bags.yaml.")
    sys.exit(1)

# Build MoCap SLERP interpolator
mocap_times = np.array([p.timestamp for p in mocap])
mocap_quat  = np.array([p.orientation for p in mocap])  # [qx, qy, qz, qw] or [qw, qx, qy, qz]
quat_interp = interp1d(mocap_times, mocap_quat, axis=0, kind='linear',
                       bounds_error=False, fill_value='extrapolate')

# ── Per-sample comparison ─────────────────────────────────────────────────────
res_codegen_list = []
res_manual_list  = []

for d in imu:
    t = d.timestamp + IMU_MOCAP_OFFSET   # align IMU to MoCap time
    if t < mocap_times[0] or t > mocap_times[-1]:
        continue

    z_acc = np.array(d.linear_acceleration)
    a_norm = np.linalg.norm(z_acc)
    if a_norm < 0.5:   # skip near-zero (shouldn't happen)
        continue

    # MoCap ground-truth rotation
    q = quat_interp(t)
    q = q / np.linalg.norm(q)
    R_wb = quat_to_rotation_matrix(q)   # world→body or body→world depending on convention

    # --- codegen residual ---
    R_nom = Rot3.from_rotation_matrix(R_wb)
    res_cg, _, _ = gravity_residual_with_jacobians(
        R_nom, delta, z_acc, b_a, G_NORM, EPSILON
    )
    res_codegen_list.append(res_cg.ravel())

    # --- manual residual: normalize(z_acc) * g - R^T @ [0,0,+g] ---
    z_normed = (z_acc / a_norm) * G_NORM
    g_body_pred = R_wb.T @ np.array([0.0, 0.0, G_NORM])
    res_manual_list.append(z_normed - g_body_pred)

res_codegen = np.array(res_codegen_list)   # (N, 3)
res_manual  = np.array(res_manual_list)    # (N, 3)

print("=" * 60)
print("GRAVITY RESIDUAL STATS (at MoCap ground truth R)")
print("=" * 60)

for label, res in [("Codegen", res_codegen), ("Manual ", res_manual)]:
    print(f"\n  [{label}]")
    print(f"    mean (x,y,z): {res.mean(axis=0)}")
    print(f"    std  (x,y,z): {res.std(axis=0)}")
    print(f"    max |res|:    {np.abs(res).max():.4f} m/s²")

diff = res_codegen - res_manual
print(f"\n  [Difference codegen - manual]")
print(f"    max |diff|: {np.abs(diff).max():.2e}  (should be ~0)")

if np.abs(diff).max() < 1e-4:
    print("\n  MATCH: codegen and manual residuals agree.")
else:
    print("\n  WARNING: codegen and manual residuals differ!")

print()
print("Expected: mean z-residual near 0 when R is ground truth (perfect prediction).")
z_mean_cg  = res_codegen[:, 2].mean() if len(res_codegen) else float('nan')
z_mean_man = res_manual[:, 2].mean()  if len(res_manual)  else float('nan')
print(f"  z-mean codegen : {z_mean_cg:.4f} m/s²")
print(f"  z-mean manual  : {z_mean_man:.4f} m/s²")
