#!/usr/bin/env python3
"""
Plot all 3D radar returns from a ROS bag into a global map using MoCap pose.
Colors points by intensity.

Controls (Open3D):
  F = toggle yaw flip and recompute
  Q = quit
  Ctrl+C in terminal = quit
"""

import sys
import signal
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import rotation_matrix_from_euler, predict_doppler_velocity

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

# ==================== Bag Catalogue ====================
BAGS = {
    "original":     "rosbags/2025-12-17-16-02-22.bag",
    "circle":       "rosbags/circle_2025-12-17-17-21-37.bag",
    "circle_fast":  "rosbags/circle_fast_2025-12-17-17-25-34.bag",
    "circle_fwd":   "rosbags/circle_forward_2025-12-17-17-37-38.bag",
    "loopings":     "rosbags/circle_fast_forward_2025-12-17-17-39-49.bag",
    "backflips":    "rosbags/backflips_2025-12-17-17-41-24.bag",
    "test":         "rosbags/Wed_11032026_1503/slowracing_oldconfig_2026-03-11-17-18-43.bag"
}

# Bags where the agiros body frame is rotated 180 deg in yaw by default
FLIPPED_BAGS = {"loopings", "test"}

# ==================== Extrinsics ====================
ROTATION_EULER_DEG = np.array([180.0, 35.0, 0.0])  # roll, pitch, yaw
_R_base = rotation_matrix_from_euler(
    np.radians(ROTATION_EULER_DEG[0]),
    np.radians(ROTATION_EULER_DEG[1]),
    np.radians(ROTATION_EULER_DEG[2]),
)
_R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)


def build_extrinsics(flip: bool):
    if flip:
        return np.array([-0.07, 0.0, 0.0]), _R_yaw_flip @ _R_base
    else:
        return np.array([0.07, 0.0, 0.0]), _R_base.copy()


# ==================== Point cloud builder ====================
def build_point_cloud(radar_frames, mocap_slerp, pos_interp, vel_interp, omega_interp,
                      t_start, t_end, flip,
                      MIN_RANGE, MAX_RANGE, MIN_INTENSITY, MAX_DOPPLER_ERROR):
    TRANSLATION, SENSOR_ROTATION = build_extrinsics(flip)
    world_points = []
    world_intensities = []

    for frame in radar_frames:
        t = frame.timestamp
        if t < t_start or t > t_end or frame.num_points() == 0:
            continue

        pos_world     = pos_interp(t)
        vel_world     = vel_interp(t)
        omega_body    = omega_interp(t)
        R_wb          = mocap_slerp(t).as_matrix()

        points_s = np.array(frame.positions)
        pred_dopplers = predict_doppler_velocity(
            v_body_world=vel_world,
            omega_body=omega_body,
            R_world_from_body=R_wb,
            radar_positions_sensor=points_s,
            T_body_from_sensor=TRANSLATION,
            R_body_from_sensor=SENSOR_ROTATION,
        )

        for i in range(frame.num_points()):
            p_s       = frame.positions[i]
            r_val     = frame.ranges[i] if frame.ranges is not None else np.linalg.norm(p_s)
            intensity = frame.intensities[i] if frame.intensities is not None else 100.0

            if r_val < MIN_RANGE or r_val > MAX_RANGE or intensity < MIN_INTENSITY:
                continue
            if abs(frame.velocities[i] - pred_dopplers[i]) > MAX_DOPPLER_ERROR:
                continue

            p_body   = SENSOR_ROTATION @ p_s + TRANSLATION
            p_world  = R_wb @ p_body + pos_world

            # Basic room-scale sanity bounds to drop multipath ghosts
            if (p_world[0] < -8.0 or p_world[0] > 8.0 or
                    p_world[1] < -8.0 or p_world[1] > 8.0 or
                    p_world[2] < -1.0 or p_world[2] > 5.0):
                continue

            world_points.append(p_world)
            world_intensities.append(intensity)

    return np.array(world_points), np.array(world_intensities)


def colorize(intensities, cmap):
    norm = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
    return cmap(norm)[:, :3]


def main():
    print("=" * 80)
    print("PLOTTING 3D RADAR RETURNS MAP")
    print("  Open3D controls: F = toggle yaw flip | Q = quit | Ctrl+C = quit")
    print("=" * 80)

    # ==================== Configuration ====================
    bag_key  = sys.argv[1] if len(sys.argv) > 1 else "circle_fwd"
    BAG_PATH = BAGS[bag_key] if bag_key in BAGS else bag_key

    flip_state = {"flip": bag_key in FLIPPED_BAGS}
    if "--flip"    in sys.argv: flip_state["flip"] = True
    if "--no-flip" in sys.argv: flip_state["flip"] = False

    # Timing offsets (same as validate_nonlinear_solver.py)
    IMU_MOCAP_OFFSET = +0.020   # seconds
    RADAR_IMU_OFFSET = -0.019   # seconds

    # Filtering parameters
    MIN_RANGE          = 0.2
    MAX_RANGE          = 80.0
    MIN_INTENSITY      = 1.0
    MAX_DOPPLER_ERROR  = 100.0  # m/s — lower (e.g. 0.5) keeps only static returns

    print(f"Bag: {BAG_PATH}  |  flip={flip_state['flip']}")

    # ==================== Load Data ====================
    bag_data      = load_bag_topics(BAG_PATH, verbose=True)
    radar_frames  = bag_data.radar_velocity
    agiros_states = bag_data.agiros_state

    # Apply time offsets
    radar_total_offset = IMU_MOCAP_OFFSET - RADAR_IMU_OFFSET
    for f in radar_frames:
        f.timestamp += radar_total_offset

    # Filter near-duplicate MoCap timestamps (dt < 1 ms)
    filtered = [agiros_states[0]]
    for s in agiros_states[1:]:
        if s.timestamp - filtered[-1].timestamp >= 1e-3:
            filtered.append(s)
    agiros_states = filtered

    if not agiros_states or not radar_frames:
        print("Not enough data!"); return

    mocap_times        = np.array([s.timestamp         for s in agiros_states])
    mocap_positions    = np.array([s.position           for s in agiros_states])
    mocap_quats        = np.array([s.orientation        for s in agiros_states])
    mocap_vels         = np.array([s.velocity           for s in agiros_states])
    mocap_omegas       = np.array([s.angular_velocity   for s in agiros_states])

    # Drop bad MoCap samples (zero-norm quaternion = tracking lost, e.g. on the ground)
    quat_norms = np.linalg.norm(mocap_quats, axis=1)
    valid = quat_norms > 0.1
    if not np.all(valid):
        n_dropped = np.sum(~valid)
        print(f"  Dropping {n_dropped} zero-quat MoCap samples (tracking lost)")
        mocap_times     = mocap_times[valid]
        mocap_positions = mocap_positions[valid]
        mocap_quats     = mocap_quats[valid]
        mocap_vels      = mocap_vels[valid]
        mocap_omegas    = mocap_omegas[valid]

    # Fix quaternion sign flips so SLERP takes the short arc
    for i in range(1, len(mocap_quats)):
        if np.dot(mocap_quats[i], mocap_quats[i - 1]) < 0:
            mocap_quats[i] *= -1

    mocap_slerp  = Slerp(mocap_times, Rotation.from_quat(mocap_quats))

    pos_interp   = interp1d(mocap_times, mocap_positions, axis=0, kind='cubic',  fill_value='extrapolate')
    vel_interp   = interp1d(mocap_times, mocap_vels,      axis=0, kind='linear', fill_value='extrapolate')
    omega_interp = interp1d(mocap_times, mocap_omegas,    axis=0, kind='linear', fill_value='extrapolate')

    t_start, t_end = mocap_times[0], mocap_times[-1]
    traj_pts = pos_interp(mocap_times)

    # ==================== Build initial cloud ====================
    def rebuild():
        pts, inten = build_point_cloud(
            radar_frames, mocap_slerp, pos_interp, vel_interp, omega_interp,
            t_start, t_end, flip_state["flip"],
            MIN_RANGE, MAX_RANGE, MIN_INTENSITY, MAX_DOPPLER_ERROR,
        )
        print(f"  flip={flip_state['flip']}  →  {len(pts)} points")
        return pts, inten

    pts, inten = rebuild()
    if len(pts) == 0:
        print("No valid points."); return

    cmap = plt.get_cmap("jet")

    # ==================== Visualisation ====================
    if HAS_OPEN3D:
        print("Launching Open3D viewer...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colorize(inten, cmap))

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        lines = [[i, i + 1] for i in range(len(traj_pts) - 1)]
        traj_ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(traj_pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        traj_ls.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(lines))

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"Radar Map: {bag_key} | F=flip Q=quit", width=1280, height=720)
        vis.add_geometry(pcd)
        vis.add_geometry(coord)
        vis.add_geometry(traj_ls)

        # ---- F: toggle flip (debounced to avoid key-repeat firing multiple times) ----
        import time as _time
        _last_flip_t = {"t": 0.0}
        _FLIP_COOLDOWN = 1.0  # seconds — ignore repeated keydown events within this window

        def on_flip(_vis):
            now = _time.monotonic()
            if now - _last_flip_t["t"] < _FLIP_COOLDOWN:
                return False  # key repeat, ignore
            _last_flip_t["t"] = now

            flip_state["flip"] = not flip_state["flip"]
            print(f"Flipping → {flip_state['flip']}, recomputing...")
            new_pts, new_inten = rebuild()
            if len(new_pts) == 0:
                print("  No points — reverting")
                flip_state["flip"] = not flip_state["flip"]
                return False
            pcd.points = o3d.utility.Vector3dVector(new_pts)
            pcd.colors = o3d.utility.Vector3dVector(colorize(new_inten, cmap))
            _vis.update_geometry(pcd)
            _vis.poll_events()
            _vis.update_renderer()
            return False

        vis.register_key_callback(ord("F"), on_flip)

        # ---- Ctrl+C: set exit flag ----
        quit_flag = {"val": False}

        def _sigint(sig, frame):
            print("\nCtrl+C — closing...")
            quit_flag["val"] = True

        signal.signal(signal.SIGINT, _sigint)

        # Manual event loop so Python can process signals between frames
        while vis.poll_events():
            vis.update_renderer()
            if quit_flag["val"]:
                break

        vis.destroy_window()

    else:
        # ---- Fallback: save PNG ----
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')
        sc  = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=inten, cmap='jet', s=6, alpha=0.9)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title(f"Radar Map: {bag_key}  flip={flip_state['flip']}")
        fig.colorbar(sc, ax=ax, label='Intensity', pad=0.1)
        plt.tight_layout()
        out = f"radar_map_{Path(BAG_PATH).stem}.png"
        plt.savefig(out, dpi=300)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
