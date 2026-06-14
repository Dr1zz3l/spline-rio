#!/usr/bin/env bash
# Old-firmware stress bags (Dec-2025, coarse 0.6 m/s Doppler) under RANSAC default.
# Watch: RANSAC inlier 0.15 m/s vs 0.604 m/s quantization -> possible starvation.
# Logs RANSAC kept-% (printed by the prefilter) + final RMSE.
set -u
cd "$(dirname "$0")/../analysis" || exit 1
U="--set marg_prior_scale=1.0 --set lambda_gyro_omega_sigma=4.0 --set lambda_gyro_omega_pow=4.0 --set omega_soft_sigma=4.0 --set accel_soft_sigma=8.0 --set radar_intensity_weight=1.0"
O=../baselines/results/ransac_default; PY="../.venv/bin/python3 -u"
for b in circle circle_fast circle_fwd loopings backflips; do
  timeout 2400 $PY validate_live_solver.py "$b" --mocap-yaw --cpp --sliding-window $U --no-plot > "$O/oldfw_${b}.log" 2>&1
  echo "done $b exit=$?"
done
echo DONE > $O/oldfw_done.flag
