# Continuous-Time Radar-Inertial Odometry

6-DOF state estimation on a quadrotor from a **single-chip 60 GHz mmWave radar**
(TI IWR6843AOPEVM, 10–60 points/frame) and a **1 kHz IMU** — through aggressive
racing and attitude-tumbling (backflip) flight up to 10 rad/s. The trajectory is
a continuous-time **quintic position B-spline** + **cumulative SO(3) orientation
B-spline**, optimized by a **fixed-lag smoother with Schur-complement
marginalization** (3 s window, 0.3 s stride). Radar Doppler measures velocity
directly, so the system is a **velocity-and-orientation** source for fusion with
a position-fixing sensor (GPS/VIO/SLAM); position is reported as drift per distance.

This repo contains the **research system and the paper**, built on a ROS1 driver
for the radar.

## The paper

| | Path | Format | Length |
|---|---|---|---|
| **Journal (T-RO) — full version** | `report/main.tex` | `IEEEtran[journal]` | 12 pp |
| Conference cut | `paper/IEEE-conference-template-062824.tex` | `IEEEtran[conference]` | 10 pp |

Build: `cd report && latexmk -pdf main.tex`. Submission target is **IEEE
Transactions on Robotics**; see `report/NOTES.md` for T-RO norms and pre-submission
TODOs. The conference cut (`paper/`) is the trimmed version for ICRA/IROS-style
8-page tracks.

### Headline results (live leading edge, the estimate a deployed system emits)

| Flight | Velocity RMSE | Orientation RMSE | Position (drift) | Solve/window |
|--------|--------------:|-----------------:|-----------------:|-------------:|
| Slow racing | 0.48 m/s | 1.97° | 0.31 m (0.64%) | 0.70 s |
| Fast racing | 0.32 m/s | 2.88° | 0.40 m (0.88%) | 0.35 s |
| Backflips   | 2.35 m/s | 6.35° | 1.55 m (2.85%) | ~0.6 s |

One **dynamics-adaptive measurement-weighting law** and one configuration cover
hover → 10 rad/s, validated on held-out flights. A reve-style **RANSAC
ego-velocity prefilter** is the default radar front-end (closes an
order-of-magnitude vertical-drift gap on the public ICINS-2021 flights). The
orientation output covariance is NEES-consistent on racing.

## Repository layout

```
report/          # T-RO journal paper (main.tex) — the full version
paper/           # conference cut (10 pp)
analysis/        # Python: live RIO solver entry points, sweeps, eval, figure gen
  validate_live_solver.py     # MoCap-free init (P1-P3) + solver — main entry point
  config/                     # solver.yaml, solver_cpp.yaml, bags.yaml, extrinsics.yaml
rio_solver_cpp/  # C++ Ceres solver (batch + sliding window), Python bindings
baselines/       # external cross-validation vs Doer/Trommer EKF-RIO (ICINS-2021)
documentation/   # FINDINGS, Forward/Backward Model, RESEARCH_NOTES, ROADMAP, SW_DEVELOPMENT
mmwave_ti_ros/   # ROS1 driver for the IWR6843 radar (data capture)
docker/, scripts/, config/    # driver dev container + setup
rosbags/, plots/              # datasets and outputs (gitignored / large)
CLAUDE.md        # operational hub: how to run, current config, current results
```

## Quick start — RIO solver + analysis

```bash
cd analysis
# install deps (or use the existing .venv)
uv pip install -r requirements.txt

# Live sliding-window RIO, C++ backend (current best). UNIV = universal weighting config.
UNIV="--set marg_prior_scale=1.0 --set lambda_gyro_omega_sigma=4.0 --set lambda_gyro_omega_pow=4.0 \
  --set omega_soft_sigma=4.0 --set accel_soft_sigma=8.0 --set radar_intensity_weight=1.0"
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity \
  --mocap-yaw --cpp --sliding-window $UNIV \
  --set dt_pos=0.04 --set dt_ori=0.016 --set lock_extrinsics=1 \
  --set-ext 'rotation_euler_deg=[180.0,27.5,0.0]'
```

The C++ solver lives in `rio_solver_cpp/` (`cd build_release && cmake .. && cmake
--build . -j$(nproc)`). See **`CLAUDE.md`** for the full command set (per-bag
configs, batch vs sliding window, NEES, figures) and the current results tables.

## Reproducibility

Every number in the paper is produced by the commands below, run from a clone +
the rosbags. Runs are **deterministic** (seeded RANSAC, `numpy.default_rng(0)`);
iteration-capped configs reproduce bit-identically, full-convergence configs vary
`<2%` (multithread reduction order). The accelerometer-bias init is **sensor-only
by default** (gravity-aligned scalar correction, no external attitude); add
`--mocap-accel-bias` for the legacy MoCap-attitude seed (shifts headline live-edge
metrics by `≤5%`).

### 0. Setup
```bash
git clone https://github.com/Dr1zz3l/radar-iwr6843-driver
cd radar-iwr6843-driver
cd analysis && uv pip install -r requirements.txt && cd ..          # Python env
cd rio_solver_cpp && mkdir -p build_release && cd build_release \
  && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j"$(nproc)" && cd ../..   # C++ Ceres solver
# Place the rosbags under ./rosbags/  (aliases -> paths in analysis/config/bags.yaml)
cd analysis
UNIV="--set marg_prior_scale=1.0 --set lambda_gyro_omega_sigma=4.0 --set lambda_gyro_omega_pow=4.0 \
  --set omega_soft_sigma=4.0 --set accel_soft_sigma=8.0 --set radar_intensity_weight=1.0"
```

### 1. Headline sliding-window results (Table III)
```bash
# Slow racing -> 0.48 m/s / 1.97deg / 0.31 m (0.64%)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity \
  --mocap-yaw --cpp --sliding-window $UNIV --set max_iterations=12 --set lambda_heading=10.0

# Fast racing -> 0.32 m/s / 2.88deg / 0.40 m (0.88%)
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity \
  --mocap-yaw --cpp --sliding-window $UNIV \
  --set dt_pos=0.04 --set dt_ori=0.016 --set lock_extrinsics=1 \
  --set-ext 'rotation_euler_deg=[180.0,27.5,0.0]'

# Backflips   -> 2.35 m/s / 6.35deg / 1.55 m (2.85%)
../.venv/bin/python3 validate_live_solver.py backflips_best_velocity \
  --mocap-yaw --cpp --sliding-window $UNIV \
  --set dt_ori=0.008 --set lock_gyro_bias=0 --set lambda_pos_init_prior=0.5 \
  --set radar_zbias_fixed=-1.5 --set-ext 'rotation_euler_deg=[180.0,27.5,0.0]'
```
Each prints a settled-vs-live RMSE table (position/velocity/orientation, drift %)
and the per-segment KITTI RPE.  Append `--set nees=1` for the covariance dump
(see §3 NEES).

### 2. Batch solver + extrinsic self-calibration (Sec. VI-A)
```bash
# per-bag spline grid from bags.yaml; pitch left free -> self-cal ~27deg
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp   # 0.20 m / 1.1deg
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp   # 0.64 m / 2.5deg
```

### 3. Figures + NEES
```bash
# First re-run the Sec.1 (SW) and Sec.2 (batch) commands for ALL three bags with
#   --save-arrays --no-plot  appended (writes plots/<bag>/live_solver/*.npz), then:
cd ../report/figures
../../.venv/bin/python3 gen_combined_traj.py   # Fig. 3  trajectory overlays (MoCap vs SW live)
../../.venv/bin/python3 gen_error_time.py      # Fig. 4  position/velocity/orientation error vs time
cd ../../analysis
# Fig. 1 (pipeline) and Fig. 2 (sliding-window marginalization) are TikZ, drawn inline in main.tex.
```
```bash
# NEES output-covariance consistency (Sec. "Output Covariance Consistency"):
# append --set nees=1 to a Sec.1 command to dump per-window covariance, then evaluate:
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw --cpp \
  --sliding-window $UNIV --set max_iterations=12 --set lambda_heading=10.0 --set nees=1 --no-plot
../.venv/bin/python3 nees_eval.py slow_racing_best_velocity   # orientation/velocity NEES vs chi2(3)
```
```bash
# Supplementary (NOT paper figures — data preserved in the prior-scale table / repo):
../../.venv/bin/python3 report/figures/gen_rpe.py          # per-segment KITTI RPE vs segment length
../../.venv/bin/python3 report/figures/gen_prior_scale.py  # prior-scale sweep behind the Table VII data
```

### 4. Ablations (Table VI) and the real-time / window sweep (Table IV)
```bash
# Window-duration + per-regime real-time operating points (Table IV):
../.venv/bin/python3 sweep_sw_params.py --mode window        # window in {1.0,1.5,2.0,2.5,3.0}s
# ...or a single point, e.g. fast real-time (2.0 s window, 0.3 s stride):
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw --cpp \
  --sliding-window $UNIV --set dt_pos=0.04 --set dt_ori=0.016 --set lock_extrinsics=1 \
  --set-ext 'rotation_euler_deg=[180.0,27.5,0.0]' --set window_duration=2.0 --set window_stride=0.3

# Each Table VI row toggles ONE knob on a Sec.1/Sec.2 command:
#   IMU rate 1000 vs 200 Hz ........  --imu-hz 200
#   IMU preintegration vs raw ......  --preintegrate   (or --set use_preintegration=1)
#   Orientation reg: min-alpha .....  --set lambda_ori_accel=0.1   (default)
#                    min-omega .....  --set lambda_ori_reg=0.001 --set lambda_ori_accel=0
#   Marginalization: boundary-only .  --set marg_prior_scale=0     (vs the unscaled Schur default)
#   Extrinsic pitch ................  batch frees pitch; add --set lock_extrinsics=1 to lock at 27.5 deg

# Backflips elevation-bias (b) sweep + the RANSAC-vs-Huber rebench tables:
( cd ../baselines && ./run_bsweep_fast_ransac.sh )                        # radar_zbias_fixed sweep
( cd ../baselines && ./run_ransac_rebench_A.sh && ./run_ransac_rebench_B.sh )  # RANSAC vs Huber front-end
```

### 5. External cross-validation vs EKF-RIO baselines (Table V)
```bash
cd ../baselines
./setup.sh                 # build the Doer/Trommer ekf-rio / ekf-yrio baselines (Docker)
./run_icins_ours.sh        # our system on the public ICINS-2021 flights
./run_ransac_rebench_A.sh  # re-evaluate their EKFs under our causal metrics
cd ../analysis             # see baselines/README.md for details
```

### 6. Held-out generalization (Sec. VI-E)
```bash
# the SAME config, unmodified, on flights never used for tuning (use the Sec.1 fast/slow flags):
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity_no_clustering \
  --mocap-yaw --cpp --sliding-window $UNIV \
  --set dt_pos=0.04 --set dt_ori=0.016 --set lock_extrinsics=1 --set-ext 'rotation_euler_deg=[180.0,27.5,0.0]'
../.venv/bin/python3 validate_live_solver.py circle_best_velocity \
  --mocap-yaw --cpp --sliding-window $UNIV --set max_iterations=12 --set lambda_heading=10.0
( cd ../baselines && ./run_oldfw_ransac.sh )   # old-firmware (12x coarser Doppler) stress-tier bags
```

### 7. Build the paper
```bash
cd ../report && latexmk -pdf main.tex          # -> main.pdf (12 pp)
```

## Data capture — ROS1 driver

The radar is read by a Dockerized ROS1 driver.

```bash
# Build/run inside the dev container
cd mmwave_ti_ros/ros1_driver
catkin_make && source devel/setup.bash
roslaunch ti_mmwave_rospkg 6843AOP_velocity_3d.launch          # with RViz
# roslaunch ti_mmwave_rospkg 6843AOP_velocity_3d_headless.launch  # headless for recording

# record (another terminal)
rosbag record /mmWaveDataHdl/RScanVelocity /tf /tf_static
```

USB/GUI setup is handled by the Docker dev container (`docker/`, `scripts/`,
`.devcontainer/`). With other catkin packages present, `CATKIN_IGNORE` files keep
only `ros1_driver/` building. Detailed driver docs:
[Velocity Publisher README](mmwave_ti_ros/ros1_driver/src/ti_mmwave_rospkg/VELOCITY_PUBLISHER_README.md).

## Documentation

- **`CLAUDE.md`** — operational hub (how to run, current config, current results).
- `documentation/` — math reference (Forward/Backward Model), calibration FINDINGS,
  RESEARCH_NOTES, SW_DEVELOPMENT, ROADMAP.
- `rio_solver_cpp/README.md` — C++ solver build + Python API.
- `baselines/README.md` — external EKF-RIO cross-validation setup.

## Hardware / TI references

- Board: [IWR6843AOPEVM](https://www.ti.com/tool/IWR6843AOPEVM)
- [mmWave Demo Visualizer 3.6.0](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/)
- [Radar Toolbox 2.20.00.05](https://dev.ti.com/tirex/explore/node?node=A__ANSECEN8pUpQyDw4PbR9XQ__radar_toolbox__1AslXXD__2.20.00.05)

Platform: TI IWR6843AOPEVM mmWave radar + Pixhawk IMU on an Agiros quadrotor,
Vicon MoCap ground truth.
