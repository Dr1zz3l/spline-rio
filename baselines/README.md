# External baseline validation (ekf_rio / x_rio, Doer & Trommer)

Goal: real baseline comparison on shared data, both directions:
(A) baseline on OUR bags, (B) our system on the baseline's public datasets.

## Setup (anyone, after cloning this repo)
```bash
./baselines/setup.sh   # clones+pins rio @ dcf0bc9 AND catkin_simple @ 0e62848
                       # (catkin_simple: build dep of all rio pkgs, no noetic apt pkg)
# Base ROS image (service name is iwr6843; it requires /dev/ttyUSB* so use
# plain docker for the baseline workflow):
docker compose -f docker/docker-compose.yml build
# Baselines image = base + gfortran (reve vendors ODRPACK, Fortran):
docker build -t rio-baselines -f baselines/Dockerfile .
# Build the workspace (repo mounts at /workspace; ws symlinks are relative
# for exactly this reason):
docker run --rm -v "$PWD":/workspace rio-baselines bash -lc \
  "source /opt/ros/noetic/setup.bash && cd /workspace/baselines/catkin_ws && \
   catkin init && catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release"
# Smoke test (their demo bag; expect 'Position Error 3D ... 0.23 percent'):
docker run --rm -v "$PWD":/workspace rio-baselines bash -lc \
  "source /opt/ros/noetic/setup.bash && \
   source /workspace/baselines/catkin_ws/devel/setup.bash && \
   timeout 240 roslaunch ekf_rio demo_datasets_ekf-rio_rosbag.launch \
     enable_rviz:=False do_plot:=False"
```
Full datasets: download from https://christopherdoer.github.io/datasets/
(radar_inertial_datasets_icins_2021, multi_radar_inertial_datasets_JGN2022)
into `baselines/datasets/`. Demo bags ship inside the rio packages.

## Direction A — x_rio / ekf_rio on our bags: NEGATIVE RESULT (2026-06-12)

Status: both ekf_rio and x_rio **diverge (km-scale, ~0.5·g·t² in z) on
slow_racing** despite three documented tuning rounds (configs in
baselines/configs/, every change commented). Evidence chain:

1. Adapter + conventions verified independently: rio's own TI-format path
   consumes our clouds ("Detected ti_mmwave_rospkg pcl format!"); a Python
   replication of rio's remap + our q_b_r reproduces GT body velocity from
   our radar with corr 0.87–0.96 → geometry/Doppler-sign/extrinsics correct.
2. Their demo bag through our build reproduces their result (0.23 % final
   drift on vicon_easy) → build is healthy.
3. Standalone reve on our converted bag: zero-velocity detection works at
   standstill, but ~half of all scans fail ("estimation failed") — our
   best_velocity radar config yields only ~13 points/frame (median; p10=8),
   below what reve's RANSAC/sigma gates expect (their clouds are O(100) pts).
4. ekf_rio with their stock init covariances (sigma_v = sigma_rp = 0):
   Kalman gain starts at exactly 0, vibration (our gyro PSD ≈ 2.1 deg/s/√Hz
   measured in flight vs their 0.2 default, reconfigure-clamped at 1.0)
   makes propagation overconfident, the chi²(3)@95% gate then rejects ~all
   radar updates → pure dead reckoning → blowup.
5. With honest init/process noise (round 2) the filter is sane for ~9 s,
   then a single update at the first real motion event blows up the bias
   states (b_a → −16 m/s², b_g → 475 deg/s in one step) → divergence.
   Gentler bias PSDs (round 3) → back to gate-rejection divergence.

Interpretation (for the paper): the Doer/Trommer EKF stack is built for
dense radar clouds, a hardware radar trigger, and a low-vibration IMU; our
platform (sparse 13-pt clouds @ 0.049 m/s quantization, no trigger, racing
quad vibration) violates all three. This is a platform mismatch, not a
defect of their method on their own data. Report as honest negative result;
the cross-comparison runs on THEIR datasets instead (Direction B below).

Repro commands (diverging runs, for the record):
```bash
.venv/bin/python3 baselines/adapters/convert_bag_to_rio.py slow_racing_best_velocity
docker run --rm -v "$PWD":/workspace rio-baselines bash -lc \
  "source /opt/ros/noetic/setup.bash && source /workspace/baselines/catkin_ws/devel/setup.bash && \
   timeout 150 roslaunch /workspace/baselines/configs/our_bags_ekf-rio_rosbag.launch \
     rosbag:=slow_racing_best_velocity_rio.bag out_bag:=/workspace/baselines/results/slow_racing_ekf_rio.bag"
# x_rio variant: our_bags_x-rio_rosbag.launch, out slow_racing_x_rio.bag
# standalone ego-velocity diagnosis: reve_our_bags.launch
```

### Original Direction A notes (adapter design)
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
