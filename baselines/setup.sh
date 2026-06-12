#!/usr/bin/env bash
# External-baseline setup: Doer/Trommer RIO toolbox (ekf_rio / x_rio).
# Run from the repo root:  ./baselines/setup.sh
# Idempotent. Fetches third-party code + public datasets (NOT our rosbags).
set -euo pipefail
cd "$(dirname "$0")"

# --- 1. Baseline code (pinned for reproducibility) --------------------------
RIO_COMMIT="${RIO_COMMIT:-dcf0bc9}"  # pinned 2026-06-12 (origin/main)
if [ ! -d rio ]; then
  git clone --recursive https://github.com/christopherdoer/rio.git
fi
( cd rio && git fetch -q && git checkout -q "$RIO_COMMIT" \
  && git submodule update --init --recursive )
echo "[setup] rio @ $(cd rio && git rev-parse --short HEAD)"

# --- 2. Public baseline datasets ---------------------------------------------
# Links are listed on https://christopherdoer.github.io/datasets/
# (radar_inertial_datasets_icins_2021, multi_radar_inertial_datasets_JGN2022).
# The demo bags ship inside the rio repo's package demo_datasets folders;
# larger sets must be fetched from the page above:
mkdir -p datasets
echo "[setup] demo bags: $(find rio -name '*.bag' | wc -l) found inside rio packages"
echo "[setup] for full datasets, download from christopherdoer.github.io/datasets/ into baselines/datasets/"

# --- 3. Catkin workspace (built INSIDE the ROS1 docker of this repo) --------
# Host side only prepares the workspace layout; build happens in the container:
#   docker compose -f docker/docker-compose.yml run --rm ros1 bash -lc \
#     "cd /workspace/baselines/catkin_ws && catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release"
mkdir -p catkin_ws/src
[ -e catkin_ws/src/rio ] || ln -s "$(pwd)/rio" catkin_ws/src/rio
echo "[setup] catkin_ws prepared (build inside docker, see baselines/README.md)"
