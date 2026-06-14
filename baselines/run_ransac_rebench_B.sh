#!/usr/bin/env bash
# Phase 3 Batch B: NEES recompute (3 bags) + held-out best-velocity flights, RANSAC default.
set -u
cd "$(dirname "$0")/../analysis" || exit 1
U="--set marg_prior_scale=1.0 --set lambda_gyro_omega_sigma=4.0 --set lambda_gyro_omega_pow=4.0 --set omega_soft_sigma=4.0 --set accel_soft_sigma=8.0 --set radar_intensity_weight=1.0"
FAST="--set dt_pos=0.04 --set dt_ori=0.016 --set lock_extrinsics=1 --set-ext rotation_euler_deg=[180.0,27.5,0.0]"
BACK="--set dt_ori=0.008 --set lock_gyro_bias=0 --set lambda_pos_init_prior=0.5 --set radar_zbias_fixed=-1.5 --set-ext rotation_euler_deg=[180.0,27.5,0.0]"
SLOW="--set max_iterations=12 --set lambda_heading=10.0"
O=../baselines/results/ransac_default; PY="../.venv/bin/python3 -u"

# --- NEES: run with nees=1, snapshot npz, evaluate ---
nees_run () {  # $1 alias  $2 extra-flags  $3 tag
  timeout 2400 $PY validate_live_solver.py "$1" --mocap-yaw --cpp --sliding-window $U $2 --set nees=1 --no-plot > "$O/nees_$3.log" 2>/dev/null
  cp ../plots/nees_last_run.npz ../plots/nees_$3.npz 2>/dev/null
  $PY nees_eval.py "$1" ../plots/nees_$3.npz > "$O/nees_${3}_eval.txt" 2>/dev/null
}
nees_run slow_racing_best_velocity "$SLOW" slow
nees_run fast_racing_best_velocity "$FAST" fast
nees_run backflips_best_velocity   "$BACK" back

# --- Held-out best-velocity flights ---
timeout 1200 $PY validate_live_solver.py fast_racing_best_velocity_no_clustering --mocap-yaw --cpp --sliding-window $U $FAST --no-plot > $O/heldout_fast2.log 2>/dev/null
timeout 1200 $PY validate_live_solver.py circle_best_velocity --mocap-yaw --cpp --sliding-window $U $SLOW --no-plot > $O/heldout_circle.log 2>/dev/null

echo DONE > $O/batchB_done.flag
