#!/usr/bin/env bash
# Re-run OUR solver on the 4 ICINS flights to capture per-axis (horiz/vert pos,
# roll/pitch/yaw ori) + whole-traj-aligned metrics for paper Table VI.
set -u
cd "$(dirname "$0")/../analysis" || exit 1
# Suppress Ceres/glog INFO spam (per-iteration tables, sparse-matrix allocs): it
# bloats logs to ~25 MB and the synchronous I/O slows the solve into timeouts.
export GLOG_minloglevel=3
UNIV="--set marg_prior_scale=1.0 --set lambda_gyro_omega_sigma=4.0 --set lambda_gyro_omega_pow=4.0 --set omega_soft_sigma=4.0 --set accel_soft_sigma=8.0 --set radar_intensity_weight=1.0"
OUT=../baselines/results/ours_icins
mkdir -p "$OUT"
FLIGHTS="${ICINS_FLIGHTS:-1 2 3 4}"
for f in $FLIGHTS; do
  echo "########## icins_flight_$f ##########"
  timeout 3600 ../.venv/bin/python3 -u validate_live_solver.py icins_flight_$f \
    --mocap-yaw --cpp --sliding-window $UNIV \
    --set lambda_heading=10.0 --set max_iterations=25 --set lock_extrinsics=1 \
    --set-ext 'rotation_euler_deg=[-178.501,-0.099,46.997]' \
    --set-ext 'translation_body_m=[0.01,0.1,0.06]' \
    --imu-hz 400 --no-plot --whole-traj-align \
    > "$OUT/flight_${f}.log" 2>&1
  echo "  exit=$? -> $OUT/flight_${f}.log"
  grep -E "RESULTS|Position RMSE|Position drift|horizontal|vertical|Velocity RMSE|Orientation RMSE|Per-axis ori|whole-traj align" "$OUT/flight_${f}.log" | tail -25
done
echo "ALL DONE"
