#!/usr/bin/env bash
# Phase 3 Batch A: headline re-benchmark with RANSAC default (now on via solver.yaml).
# Tables II (batch) + III (SW) + figure npz (--save-arrays) + extrinsic-pitch check (racing batch)
# + a determinism re-run of slow SW. Config UNCHANGED from the paper (UNIV + per-bag overrides).
set -u
cd "$(dirname "$0")/../analysis" || exit 1
U="--set marg_prior_scale=1.0 --set lambda_gyro_omega_sigma=4.0 --set lambda_gyro_omega_pow=4.0 --set omega_soft_sigma=4.0 --set accel_soft_sigma=8.0 --set radar_intensity_weight=1.0"
FAST="--set dt_pos=0.04 --set dt_ori=0.016 --set lock_extrinsics=1 --set-ext rotation_euler_deg=[180.0,27.5,0.0]"
BACK="--set dt_ori=0.008 --set lock_gyro_bias=0 --set lambda_pos_init_prior=0.5 --set radar_zbias_fixed=-1.5 --set-ext rotation_euler_deg=[180.0,27.5,0.0]"
O=../baselines/results/ransac_default; mkdir -p "$O"
PY="../.venv/bin/python3 -u"

# --- Table II: batch (pitch optimized for racing -> extrinsic self-cal check) ---
timeout 1200 $PY validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp --save-arrays --no-plot > $O/slow_batch.log 2>/dev/null
timeout 1200 $PY validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp --save-arrays --no-plot > $O/fast_batch.log 2>/dev/null
timeout 3600 $PY validate_live_solver.py backflips_best_velocity --mocap-yaw --cpp --save-arrays --no-plot > $O/back_batch.log 2>/dev/null

# --- Table III: sliding window (with --save-arrays for figures) ---
timeout 1200 $PY validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp --sliding-window $U --set max_iterations=12 --set lambda_heading=10.0 --save-arrays --no-plot > $O/slow_sw.log 2>/dev/null
timeout 1200 $PY validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp --sliding-window $U $FAST --save-arrays --no-plot > $O/fast_sw.log 2>/dev/null
timeout 3600 $PY validate_live_solver.py backflips_best_velocity --mocap-yaw --cpp --sliding-window $U $BACK --save-arrays --no-plot > $O/back_sw.log 2>/dev/null

# --- determinism: re-run slow SW, compare RESULTS to the earlier slow_ransac.log ---
timeout 1200 $PY validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp --sliding-window $U --set max_iterations=12 --set lambda_heading=10.0 --no-plot > $O/slow_sw_determ.log 2>/dev/null

echo DONE > $O/batchA_done.flag
