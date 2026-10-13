> **rio_ros (renamed from rio_gp_ros 2026-10-13).** Two live nodes on one
> real-time runtime: `rio_spline_node.py` (spline sliding window; `roslaunch
> rio_ros rio_spline.launch`, or `radar_acro.launch` whose default backend is
> now `spline`) and `rio_gp_node.py` (GP/WNOA). The public spline release
> contains the spline node only; the GP sections below apply to the
> development tree. Newest material first: see the 2026-10-13 sections at the
> end.

# rio_ros — live GP/WNOA radar-inertial odometry node

ROS1 (noetic) wrapper around the `rio_gp` solver (branch `gp-backend`,
`documentation/GP_MIGRATION.md`). Standalone odometry: no host fusion
graph, no map, no loop closures. The estimation path is causal and
sensor-only; the only external reference is the yaw relay (mocap today,
magnetometer later — the drone has no magnetometer yet).

## What runs where

| Piece | Source of truth | Node port |
|---|---|---|
| Solver core (knots, factors, marginal, arrow solve) | `rio_solver_cpp/src/gp_solver.cpp` | pybind `rio_gp` (unchanged) |
| Knot seeding | `seed_dead_reckon=1` (IMU dead-reckon) | validated offline: parity to 4 digits on 3 bags + ICINS 4/4 |
| Base Doppler pre-unwrap | `preunwrap_radar_frames` (IMU-anchored, WLS reset) | `PreUnwrapper` (incremental port) |
| Live-window re-unwrap + RANSAC | `_gp_causal_unwrap_stride` (GP-22) | `causal_reunwrap_stride` (verbatim port) |
| Bootstrap | P1/P3 stationary init | state machine (1 s stationary -> bias + gravity attitude + relay yaw) |
| Re-arm after divergence | `gp_reset_t` experiment | watchdog `_rearm` (carry attitude + relaxed anchor 0.05) |

## Timing model (the sync answer)

Everything is keyed to **header stamps in IMU time**, never arrival time:

- radar stamps: `t -= radar_imu_offset_sec` (0.135 s, Mar-2026 calibration;
  re-calibrate per day, see FINDINGS.md)
- mocap stamps: `t -= imu_mocap_offset_sec` (0.015 s)
- a stride `(t_prev, t_prev+0.3]` closes only when the IMU watermark passes
  `t_now + margin + radar_imu_offset` — the offset term is essential: radar
  arrivals trail their corrected stamps by the offset, and without it ~45%
  of frames are lost (found in bag replay: fast diverged 38 m). Radar is
  consumed by pop, so a straggler that misses its stride is fed with the
  next one instead of being dropped
- measured sensitivity: a 20 ms radar-IMU error changes per-point Doppler
  by ~0.06 m/s, below the effective noise (GP-10) — ms-level jitter is
  invisible; the per-day constant is what matters
- duplicate (stamp-frozen) radar frames are dropped on ingestion (zero-dt
  knots would divide by zero in the WNOA prior)

## Topics

Subscribed: radar `PointCloud2` (fields x,y,z,velocity,intensity), `Imu`,
`PoseStamped` (yaw relay + position gauge at arm time).

Published:

| Topic | Type | Rate | Content |
|---|---|---|---|
| `rio_gp/odom` | `nav_msgs/Odometry` | per stride (~3.3 Hz) | live-edge pose+twist; pose cov 6x6 (p, theta-right-tangent) and body-frame velocity cov from the solver marginals |
| `rio_gp/odom_fast` | `nav_msgs/Odometry` | 100 Hz (param) | IMU dead-reckon from the last live edge (control-grade latency); covariance frozen at last stride |
| `rio_gp/state_cov` | `Float64MultiArray` | per stride | `[t, q(xyzw), p(3), v_world(3), omega_body(3), ba(3), bg(3), Sigma9(81, row-major)]` — full record for offline analysis; Sigma9 over (dp_world, dtheta_right, dv_world) |
| `rio_gp/status` | `std_msgs/String` (JSON) | per stride | state machine, GN iters, ms/update, active knots, watchdog flags, re-arm count |

Latency: the smoothed estimate trails by stride + radar offset + solve
(~0.47 s worst case); `odom_fast` hides that for consumers that need
immediacy.

## Replay validation (offline, stubbed rospy, full bags through the
## callbacks in arrival order)

Start-anchored position RMSE over the harness flight window (the node
additionally flies the 16-22 s pre-window segment the offline harness
never sees, and self-arms from the bag's stationary period):

| Bag | node replay | offline harness | note |
|---|---|---|---|
| slow racing | 0.243 | 0.189 | parity (pre-window drift + draw noise) |
| fast racing | 0.99 | 0.461 | one aliasing transient (t~24-35), self-recovers; the EMV chirp removes this section entirely |
| backflips | 1.03 | 0.690 | flips tracked; late growth = post-flight starvation outside eval |

(Table measured at the pre-GP-27 compiled-in defaults. GP-27 2026-07-17
changed the defaults to `radar_frame_sigma_c=0.4` + `lambda_heading=0.6`;
the node inherits them via GpConfig on rebuild. New offline headline:
slow 0.205, fast 0.472, backflips 0.839 (see GP_MIGRATION.md GP-27);
re-run `scripts/test_node_replay.py` for fresh parity numbers.)

No divergence, no watchdog false trips after the warmup gate, ~8 ms/update.
Replay driver: `scripts/test_node_replay.py` (runs on the HOST with the
analysis venv, no ROS needed):

```bash
../.venv/bin/python3 scripts/test_node_replay.py slow_racing_best_velocity
```

Run it after any node change; slow must stay ~0.24 and fast/backflips must
not diverge.

## Watchdog

Per stride: non-finite state/bias (hard -> re-arm), GN pinned at its cap
for N strides, solve-time budget, `sqrt(tr(Sigma_pos)/3)` threshold, radar
starvation (coasts on prior+IMU, flag only). Re-arm policy (validated
offline as the `gp_reset_t` experiment): carry attitude+position from the
propagation state, keep biases, relaxed first-knot anchor (0.05) so fresh
gravity/heading evidence can re-pull a bad carried attitude. Expect a ~5 s
orientation transient when re-arming mid-maneuver; near-baseline when
re-arming in quiet flight. Do NOT use accelerometer gravity for in-air
attitude: on a multirotor the specific force tracks the thrust axis, not
gravity (31 deg error measured in banked flight).

## Build + run

```bash
# 1. inside the noetic container: build rio_gp against the container python
bash src/rio_ros/scripts/build_rio_gp.sh /root/radar-iwr6843-driver

# 2. build the workspace
catkin_make && source devel/setup.bash

# 3. run
roslaunch rio_ros rio_gp.launch

# rosbag rehearsal (recommended before any flight):
rosbag play --clock <bag>   # node keys on header stamps; sim time fine
```

The node needs numpy only (no scipy). `rio_gp_path` in the params yaml
must point at the directory containing the container-built
`rio_gp.cpython-38-*.so`.

## Recording a full-output bag

```bash
rosbag record /mmWaveDataHdl/RScanVelocity /angrybird2/imu \
  /mocap/angrybird2/pose /rio_gp/odom /rio_gp/odom_fast \
  /rio_gp/state_cov /rio_gp/status
```

This captures raw sensors (for offline re-runs) + the live estimate with
full covariance (`state_cov`).

## Known deltas vs the offline harness (accepted)

- WLS in the pre-unwrap is a compact IRLS port (prediction-grade; alias
  selection only needs +-v_max accuracy).
- The offline harness seeds its first window from the P1/P2 init spline;
  the node uses the constant-pose init + dead-reckon seeding (parity
  validated).
- Bias RW is ON here (GP-13, benchmark-invisible offline, correct for
  long missions); the offline headline runs have it off.

## 2026-09-10: per-return weight channel + 50 Hz sub-knots (harness parity)

The node's front end now produces the harness's per-return radar weights:
`PreUnwrapper.unwrap_frame` returns (n, 9) arrays whose column 5 is the
weight the solver applies (`RadarPoint.w`): the measured per-return law for
clean returns (bearing law, DIRCOS form, s0 0.0827 / coef 0.03291) and the
measured alias down-weight for returns the unwrapper moved (`w_al` from the
frame's body rate, sigma_al 0.6 m/s); columns 6-8 (clean weight, frame alias
weight, raw Doppler) let `causal_reunwrap_stride` recompose column 5 after
the live-window re-unwrap.  `flip_hz` default 100 -> 50 (the harness's
2026-09-10 default; `GpConfig` defaults `use_point_weights`, `sigma_radar`
0.25 and `record_shadow` are inherited).  Replay (`scripts/test_node_replay.py`,
start-anchored position RMSE, IMU-time stamps): slow 0.165 m, fast 0.721 m,
backflips 0.845 m (July: 0.243 / 0.99 / 1.03), no re-arms, ~13.5 ms/update.
The replay harness takes `RIO_REPO` / `RIO_BUILD` / `RIO_DATA` so it can run
from a worktree against another checkout's build and rosbags.  The container
`.so` must be rebuilt (`scripts/build_rio_gp.sh`) to pick up the new
`GpConfig` fields before a live run.

## 2026-10-03: per-return law synced with the harness; audit against the deployed GP

The node's law was still the pre-2026-09-11 (s0 0.0827, coef 0.03291); it is
now the GP harness's (0.0829, 0.03392), and a `bearing_law: [s0, coef]`
parameter overrides it. Everything else in the GP front end matches the
harness as of 2026-09-30: yaw +1.0, gyro y-scale, alias repair, IMU
misalignment, IMU-propagated causal re-unwrap, alias down-weight
(0.095 / 4.5 / 0.6), v_max 3.136, RANSAC 0.15, clocks 0.127 / 0.005. The
2026-09-30 accelerometer lag is deployed for the SPLINE only; the GP (harness
and node) does not apply it. The GP C++ (`gp_solver.cpp`) last changed
2026-09-17 with opt-in features only, so any container `.so` built after
the GP-31 consider state (2026-09-10) is current; rebuild
(`scripts/build_rio_gp.sh`) if in doubt. Replay (deployed profile,
`worklog/artifacts/2026-10-03_node/`), position RMSE old law -> new law:
slow 0.163 -> 0.163, fast 0.552 -> 0.558, backflips 0.408 -> 0.414, fast2
0.177 -> 0.176, circle 0.482 -> 0.485 (neutral; the old arm reproduces the
2026-09-27 replay to the digit). There is NO spline node.

## 2026-10-13: two nodes (GP + spline), one real-time runtime

`worklog/2026-10-13_realtime-nodes.md`. The package now has:

| file | role |
|---|---|
| `scripts/rio_frontend.py` | shared causal front end (pre-unwrap, alias repair, weights, causal re-unwrap + RANSAC); per-node bearing law |
| `scripts/rio_runtime.py` | shared runtime: callbacks, bootstrap, stride assembly, **solver worker thread** (callbacks never wait for a solve), backlog guard + re-arm, accel-lag resampling, publishing, watchdog |
| `scripts/rio_gp_node.py` | GP backend (replay identical to the pre-refactor node) |
| `scripts/rio_spline_node.py` | **spline backend** (the deployed spline config) |
| `config/rio_spline_profile.yaml` | GENERATED from the harness by `analysis/export_spline_node_config.py`; never edit |
| `config/rio_spline_params.yaml` | spline deployment knobs (topics, real-time, watchdog) |
| `launch/rio_spline.launch` | spline node alone; `radar_acro.launch backend:=spline` with the driver |

Replay (`RIO_NODE=gp|spline`; the spline needs
`RIO_NODE_PROFILE=config/rio_spline_profile.yaml:config/rio_spline_params.yaml`;
`RIO_REPLAY_REALTIME=1` = wall-clock pace, threaded, prints latency / stride
timing; `RIO_REPLAY_DUMP=x.npz` saves the live edge + status):

```bash
cd <repo>
C=mmwave_ti_ros/ros1_driver/src/rio_ros/config
RIO_NODE=spline RIO_NODE_PROFILE=$C/rio_spline_profile.yaml:$C/rio_spline_params.yaml \
  .venv/bin/python3 mmwave_ti_ros/ros1_driver/src/rio_ros/scripts/test_node_replay.py fast_racing_best_velocity
# batch: worklog/artifacts/2026-10-03_node/replay2.sh <bag> <gp|spline>
```

Spline replay (start-anchored position m): slow 0.150, fast 0.147, fast2
0.217, backflips 0.460, circle 0.344 (GP node 0.163 / 0.558 / 0.176 / 0.414 /
0.485). Real time at 1x on this laptop: spline stride 106-111 ms median, max
230 ms; GP 42-46 ms, max 81 ms (budget 300 ms); latency data -> publish ~0.29 s
(spline) / ~0.20 s (GP), bridged by `odom_fast`.

Environment: a pixi env (`pixi.toml` at the repo root; `pixi run build-rio`
builds both modules into `rio_solver_cpp/build_pixi/lib`, `pixi run check`)
replaces the Docker container build; set `rio_gp_path` / `rio_solver_path` to
that directory. Not resolved yet (see the worklog, sec. 6).
