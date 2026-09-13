#!/usr/bin/env python3
import sys
import signal
import time
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent.parent))  # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

from radar_velocity_utils import rotation_matrix_from_euler
from config_loader import load_config

"""
Visualize radar-to-body extrinsic calibration using Open3D.

Draws:
  - Body frame at origin
  - Radar frame after extrinsic rotation + translation
  - Translation segment from body origin to radar origin

Usage:
  python analysis/viz/plot_extrinsics.py
  python analysis/viz/plot_extrinsics.py circle_fwd
"""


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def make_translation_line(t_sensor: np.ndarray) -> o3d.geometry.LineSet:
    """Create a dashed-style line from body origin to sensor origin."""
    # Subdivide into segments for a denser line appearance
    n = 20
    pts = np.linspace([0.0, 0.0, 0.0], t_sensor, n + 1)
    pairs = [[i, i + 1] for i in range(n)]
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(pts)
    line.lines = o3d.utility.Vector2iVector(np.array(pairs, dtype=np.int32))
    line.colors = o3d.utility.Vector3dVector(np.tile([0.15, 0.15, 0.15], (n, 1)))
    return line


def make_label(text: str, position: np.ndarray, scale: float = 0.004) -> o3d.geometry.TriangleMesh:
    """Create a 3D text mesh label at position (Open3D >= 0.13)."""
    label = o3d.geometry.TriangleMesh.create_text(text, depth=0.001)
    label.scale(scale, center=label.get_center())
    # After scaling, move center to desired position
    label.translate(position - label.get_center())
    label.paint_uniform_color([0.05, 0.05, 0.05])
    label.compute_vertex_normals()
    return label


def build_geometries(R_base, R_yaw_flip, t_body, flip: bool):
    """Build all geometries for the given flip state. Returns (list, t_sensor)."""
    if flip:
        R_sensor = R_yaw_flip @ R_base
        t_sensor = R_yaw_flip @ t_body
    else:
        R_sensor = R_base
        t_sensor = t_body.copy()

    t_mag = max(np.linalg.norm(t_body), 0.01)  # use base translation for consistent scaling
    frame_size = t_mag * 1.5

    body_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)

    sensor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size * 0.8)
    sensor_frame.transform(make_transform(R_sensor, t_sensor))

    translation_line = make_translation_line(t_sensor)

    body_origin = o3d.geometry.TriangleMesh.create_sphere(radius=t_mag * 0.12)
    body_origin.paint_uniform_color([0.3, 0.3, 0.3])

    radar_origin = o3d.geometry.TriangleMesh.create_sphere(radius=t_mag * 0.10)
    radar_origin.paint_uniform_color([0.1, 0.1, 0.1])
    radar_origin.translate(t_sensor)

    geoms = [body_frame, sensor_frame, translation_line, body_origin, radar_origin]

    label_offset = np.array([0.0, 0.0, t_mag * 0.4])
    label_scale = t_mag * 0.18
    try:
        geoms.append(make_label("Body",  label_offset,           scale=label_scale))
        geoms.append(make_label("Radar", t_sensor + label_offset, scale=label_scale))
    except Exception:
        pass  # create_text unavailable — labels in terminal only

    return geoms, t_sensor


def main() -> None:
    cfg = load_config()
    ext = cfg['extrinsics']
    flipped_bags = set(cfg['bags']['flipped'])

    bag_key = sys.argv[1] if len(sys.argv) > 1 else None
    flip_state = {"flip": bag_key in flipped_bags if bag_key else False}
    if "--flip"    in sys.argv: flip_state["flip"] = True
    if "--no-flip" in sys.argv: flip_state["flip"] = False

    rot_deg = ext['rotation_euler_deg']
    t_body  = np.array(ext['translation_body_m'], dtype=float)

    R_base     = rotation_matrix_from_euler(*np.radians(rot_deg))
    R_yaw_flip = rotation_matrix_from_euler(0.0, 0.0, np.pi)

    def print_state():
        geoms, t_sensor = build_geometries(R_base, R_yaw_flip, t_body, flip_state["flip"])
        t_mag = max(np.linalg.norm(t_body), 0.01)
        flip_label = " [FLIPPED]" if flip_state["flip"] else ""
        print(
            f"Extrinsics{flip_label} | rpy=[{rot_deg[0]:.0f}, {rot_deg[1]:.0f}, {rot_deg[2]:.0f}] deg "
            f"| t=[{t_body[0]:.3f}, {t_body[1]:.3f}, {t_body[2]:.3f}] m"
        )
        print(f"  Body at origin  |  Radar at {t_sensor.round(4).tolist()}")
        print(f"  Axis size: {t_mag*1.5*1000:.1f} mm  |  Translation: {np.linalg.norm(t_sensor)*1000:.1f} mm")
        return geoms

    print("Controls: F = toggle yaw flip | Ctrl+C = quit")
    geoms = print_state()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Radar Extrinsics | F=flip  Ctrl+C=quit",
        width=1280, height=900,
    )
    for g in geoms:
        vis.add_geometry(g)

    _last_flip_t = {"t": 0.0}
    _FLIP_COOLDOWN = 0.5

    def on_flip(_vis):
        now = time.monotonic()
        if now - _last_flip_t["t"] < _FLIP_COOLDOWN:
            return False
        _last_flip_t["t"] = now

        flip_state["flip"] = not flip_state["flip"]
        new_geoms = print_state()

        # Replace all geometries: remove old, add new
        _vis.clear_geometries()
        for g in new_geoms:
            _vis.add_geometry(g, reset_bounding_box=False)
        _vis.poll_events()
        _vis.update_renderer()
        return False

    vis.register_key_callback(ord("F"), on_flip)

    quit_flag = {"val": False}

    def _sigint(sig, frame):
        print("\nCtrl+C — closing...")
        quit_flag["val"] = True

    signal.signal(signal.SIGINT, _sigint)

    while vis.poll_events():
        vis.update_renderer()
        if quit_flag["val"]:
            break

    vis.destroy_window()


if __name__ == '__main__':
    main()