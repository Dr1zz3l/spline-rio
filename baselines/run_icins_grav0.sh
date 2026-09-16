#!/usr/bin/env bash
# Re-run OUR solver on the 4 ICINS flights with the gravity/tilt factor DISABLED
# (lambda_gravity=0), RANSAC front-end ON (default), to regenerate the Table VI /
# Section VI-E baseline numbers for the "no gravity anywhere" paper.
#
# Writes to a SEPARATE dir (ours_icins_grav0/) so the current gravity=0.001
# baselines (baselines/results/ours_icins/flight_*_ransac.log) stay intact for
# side-by-side comparison.
#
# Usage:  bash baselines/run_icins_grav0.sh
# (Runs all 4 flights sequentially; each takes ~30-60 min, so ~2-4 h total.)
set -u
cd "$(dirname "$0")/../analysis" || exit 1
export GLOG_minloglevel=3   # suppress Ceres/glog INFO spam (keeps logs small + fast)

UNIV="--set marg_prior_scale=1.0 --set lambda_gyro_omega_sigma=4.0 --set lambda_gyro_omega_pow=4.0 --set omega_soft_sigma=4.0 --set accel_soft_sigma=8.0 --set radar_intensity_weight=1.0"
OUT=../baselines/results/ours_icins_grav0
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
# Resumable: append (do NOT truncate), and skip flights that already have a
# complete result, so a re-run after a crash continues where it left off.
echo "ICINS re-run, lambda_gravity=0.0, RANSAC ON  (started/resumed $(date))" | tee -a "$SUMMARY"

FLIGHTS="${ICINS_FLIGHTS:-1 2 3 4}"
for f in $FLIGHTS; do
  if grep -q "whole-traj align" "$OUT/flight_${f}.log" 2>/dev/null; then
    echo "########## icins_flight_$f -- already complete, skipping ##########" | tee -a "$SUMMARY"
    continue
  fi
  echo "########## icins_flight_$f ##########" | tee -a "$SUMMARY"
  ../.venv/bin/python3 -u validate_live_solver.py icins_flight_$f \
    --mocap-yaw --cpp --sliding-window $UNIV \
    --set lambda_heading=10.0 --set max_iterations=25 --set lock_extrinsics=1 \
    --set lambda_gravity=0.0 \
    --set-ext 'rotation_euler_deg=[-178.501,-0.099,46.997]' \
    --set-ext 'translation_body_m=[0.01,0.1,0.06]' \
    --imu-hz 400 --no-plot --whole-traj-align \
    > "$OUT/flight_${f}.log" 2>&1
  ec=$?
  echo "  flight_$f exit=$ec -> $OUT/flight_${f}.log" | tee -a "$SUMMARY"
  grep -E "Position drift|horizontal|vertical|Velocity RMSE|Orientation RMSE|Per-axis ori|whole-traj align" \
    "$OUT/flight_${f}.log" | tail -12 | tee -a "$SUMMARY"
  echo "" | tee -a "$SUMMARY"
done
echo "ALL DONE  ($(date))" | tee -a "$SUMMARY"
echo "Summary written to $SUMMARY"
