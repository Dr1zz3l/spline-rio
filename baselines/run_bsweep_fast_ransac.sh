#!/usr/bin/env bash
# VII-B b-sweep on fast_racing under RANSAC default (locked pitch 27.5).
# b=0 is the headline default (0.389); re-measure b=-0.5 and b=-1.0 for the sweep.
set -u
cd "$(dirname "$0")/../analysis" || exit 1
U="--set marg_prior_scale=1.0 --set lambda_gyro_omega_sigma=4.0 --set lambda_gyro_omega_pow=4.0 --set omega_soft_sigma=4.0 --set accel_soft_sigma=8.0 --set radar_intensity_weight=1.0"
FAST="--set dt_pos=0.04 --set dt_ori=0.016 --set lock_extrinsics=1 --set-ext rotation_euler_deg=[180.0,27.5,0.0]"
O=../baselines/results/ransac_default; PY="../.venv/bin/python3 -u"
for b in -0.5 -1.0; do
  timeout 1200 $PY validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp --sliding-window $U $FAST --set radar_zbias_fixed=$b --no-plot > "$O/fast_bsweep_b${b}.log" 2>&1
  echo "done b=$b exit=$?"
done
echo DONE > $O/bsweep_done.flag
