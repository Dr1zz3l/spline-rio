# External baseline validation (ekf_rio / x_rio, Doer & Trommer)

Goal: real baseline comparison on shared data, both directions:
(A) baseline on OUR bags, (B) our system on the baseline's public datasets.

## Setup (anyone, after cloning this repo)
```bash
./baselines/setup.sh                  # clones+pins rio toolbox, prepares catkin ws
docker compose -f docker/docker-compose.yml run --rm ros1 bash -lc \
  "cd /workspace/baselines/catkin_ws && catkin init && \
   catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release"
```
Full datasets: download from https://christopherdoer.github.io/datasets/
(radar_inertial_datasets_icins_2021, multi_radar_inertial_datasets_JGN2022)
into `baselines/datasets/`. Demo bags ship inside the rio packages.

## Direction A — x_rio / ekf_rio on our bags
- Our topics: radar `/mmWaveDataHdl/RScanVelocity` (PointCloud2 fields
  x,y,z,velocity,intensity,...; see analysis/lib/rosbag_loader/RADAR_FIELDS.md),
  IMU `/angrybird2/imu`, GT `/mocap/angrybird2/pose`.
- The rio stack expects its own radar point-cloud layout (check
  rio/*/launch demo configs for expected fields: x,y,z,snr_db?,v_doppler_mps,
  noise_db — VERIFY in rio_utils/radar_ego_velocity_estimator). Write a
  minimal field-renaming relay node or bag-rewrite script in
  `baselines/adapters/` — keep it faithful (rename/reorder fields only).
- Extrinsics for our platform: rotation [180, 27.5, 0] deg (locked pitch),
  translation [0.08, 0.02, -0.01] m, body x-fwd/y-left/z-up; time offsets in
  analysis/config/extrinsics.yaml (radar_imu_offset_sec etc.).
- Tune the baseline's NOISE parameters honestly for our radar config
  (0.049 m/s Doppler bins) and document every change in this README.

## Direction B — our system on their bags
- Convert their bag topics into our loader's expectations (or extend
  analysis/lib/rosbag_loader). Run with the universal config (CLAUDE.md
  "UNIVERSAL WEIGHTING CONFIG"); grids per dynamics class.

## Fairness protocol (MANDATORY — see also report/PAPER_REPLAN.md item 3)
1. Heading: our system uses --mocap-yaw (pseudo-magnetometer). ekf_rio is
   yaw-unaided; x_rio variants may use yaw aiding (ekf_yrio). Either give the
   baseline equivalent yaw aiding, ALSO run ours without --mocap-yaw, or
   report the asymmetry explicitly. Never compare aided vs unaided silently.
2. Identical metrics for both: causal/live estimate, velocity RMSE,
   orientation RMSE, position drift-% of GT path length, start-anchored
   alignment (paper Sec. Experimental Setup). Evaluate both through
   analysis/ eval code where possible.
3. Report whatever comes out, including baseline wins.
4. Pin everything: rio commit hash, parameter files used (commit them under
   baselines/configs/), bag names + time windows (analysis/config/bags.yaml).

## Deliverables
ROADMAP.md Part 6 results table; paper Related-Work caveat replaced + results
table extended; PDF recommitted. Status: SCAFFOLD ONLY — runs not yet done.
