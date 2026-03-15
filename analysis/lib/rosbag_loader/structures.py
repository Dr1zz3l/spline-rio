"""Data structure definitions for rosbag topics."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class MocapPose:
    """Pure Vicon mocap pose data from /mocap/angrybird2/pose."""

    timestamp: float  # seconds since epoch
    position: np.ndarray  # [x, y, z] in meters
    orientation: np.ndarray  # [qx, qy, qz, qw] quaternion
    frame_id: str = "mocap"
    child_frame_id: str = "angrybird2"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "x": self.position[0],
            "y": self.position[1],
            "z": self.position[2],
            "qx": self.orientation[0],
            "qy": self.orientation[1],
            "qz": self.orientation[2],
            "qw": self.orientation[3],
        }


@dataclass
class MocapAccel:
    """Mocap twist data from /mocap/angrybird2/accel.
    
    WARNING: Despite the topic name 'accel', this is actually TwistStamped message
    containing linear and angular VELOCITY, not acceleration.
    Contains jumps in position data.
    """

    timestamp: float  # seconds since epoch
    linear_acceleration: np.ndarray  # [vx, vy, vz] in m/s - ACTUALLY VELOCITY!
    angular_velocity: Optional[np.ndarray] = None  # [wx, wy, wz] in rad/s
    frame_id: str = "world"

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "vx": self.linear_acceleration[0],  # Actually velocity
            "vy": self.linear_acceleration[1],
            "vz": self.linear_acceleration[2],
        }
        if self.angular_velocity is not None:
            data.update({
                "wx": self.angular_velocity[0],
                "wy": self.angular_velocity[1],
                "wz": self.angular_velocity[2],
            })
        return data


@dataclass
class AgirosState:
    """State estimation from Agiros pilot (Kalman smoothed MoCap).
    
    Source: /angrybird2/agiros_pilot/state (QuadState message)
    Quality: Better than odometry, higher resolution.
    Contains full state including acceleration, jerk, snap, and bias estimates.
    """

    timestamp: float
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    orientation: np.ndarray  # [qx, qy, qz, qw]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    acceleration: Optional[np.ndarray] = None  # [ax, ay, az] linear acceleration
    angular_acceleration: Optional[np.ndarray] = None  # [alpha_x, alpha_y, alpha_z]
    jerk: Optional[np.ndarray] = None  # [jx, jy, jz] derivative of acceleration
    snap: Optional[np.ndarray] = None  # [sx, sy, sz] derivative of jerk
    acc_bias: Optional[np.ndarray] = None  # [bx, by, bz] accelerometer bias
    gyr_bias: Optional[np.ndarray] = None  # [bx, by, bz] gyroscope bias
    motors: Optional[np.ndarray] = None  # [m1, m2, m3, m4] motor commands
    frame_id: str = "world"
    child_frame_id: str = "angrybird2"

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "x": self.position[0],
            "y": self.position[1],
            "z": self.position[2],
            "vx": self.velocity[0],
            "vy": self.velocity[1],
            "vz": self.velocity[2],
            "qx": self.orientation[0],
            "qy": self.orientation[1],
            "qz": self.orientation[2],
            "qw": self.orientation[3],
            "wx": self.angular_velocity[0],
            "wy": self.angular_velocity[1],
            "wz": self.angular_velocity[2],
        }
        if self.acceleration is not None:
            data.update({"ax": self.acceleration[0], "ay": self.acceleration[1], "az": self.acceleration[2]})
        if self.motors is not None and len(self.motors) >= 4:
            data.update({"motor1": self.motors[0], "motor2": self.motors[1], "motor3": self.motors[2], "motor4": self.motors[3]})
        return data


@dataclass
class AgirosOdometry:
    """Odometry estimation from Agiros pilot.
    
    Source: /angrybird2/agiros_pilot/odometry
    Quality: Lower quality than state estimate.
    """

    timestamp: float
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    orientation: np.ndarray  # [qx, qy, qz, qw]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    frame_id: str = "world"
    child_frame_id: str = "angrybird2"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "x": self.position[0],
            "y": self.position[1],
            "z": self.position[2],
            "vx": self.velocity[0],
            "vy": self.velocity[1],
            "vz": self.velocity[2],
            "qx": self.orientation[0],
            "qy": self.orientation[1],
            "qz": self.orientation[2],
            "qw": self.orientation[3],
            "wx": self.angular_velocity[0],
            "wy": self.angular_velocity[1],
            "wz": self.angular_velocity[2],
        }


@dataclass
class IMUData:
    """Raw IMU data from Pixhawk.
    
    Source: /angrybird2/imu
    """

    timestamp: float
    linear_acceleration: np.ndarray  # [ax, ay, az] in m/s^2
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    orientation: Optional[np.ndarray] = None  # [qx, qy, qz, qw] if available
    frame_id: str = "pixhawk_imu"

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "ax": self.linear_acceleration[0],
            "ay": self.linear_acceleration[1],
            "az": self.linear_acceleration[2],
            "wx": self.angular_velocity[0],
            "wy": self.angular_velocity[1],
            "wz": self.angular_velocity[2],
        }
        if self.orientation is not None:
            data.update({
                "qx": self.orientation[0],
                "qy": self.orientation[1],
                "qz": self.orientation[2],
                "qw": self.orientation[3],
            })
        return data


@dataclass
class RadarPointCloud:
    """Radar point cloud data from /ti_mmwave/radar_scan_pcl_0.
    
    This uses PCL format with mmWaveCloudType structure (x, y, z, intensity, velocity).
    Published by DataUARTHandler_pub in the driver.
    """

    timestamp: float
    positions: np.ndarray  # Nx3 array of [x, y, z] positions in meters
    velocities: Optional[np.ndarray] = None  # N array of radial velocities in m/s
    intensities: Optional[np.ndarray] = None  # N array of reflected power
    ranges: Optional[np.ndarray] = None  # N array of ranges in meters
    noise: Optional[np.ndarray] = None  # N array of noise values
    time_cpu_cycles: Optional[np.ndarray] = None  # N array of CPU cycle timestamps
    frame_number: Optional[np.ndarray] = None  # N array of frame numbers
    frame_id: str = "ti_mmwave_pcl"

    def num_points(self) -> int:
        return len(self.positions) if self.positions is not None else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "num_points": self.num_points(),
            "has_intensities": self.intensities is not None,
            "has_velocities": self.velocities is not None,
            "has_ranges": self.ranges is not None,
        }


@dataclass
class RadarVelocity:
    """Radar velocity point cloud from /mmWaveDataHdl/RScanVelocity.
    
    This is the PRIORITY 1 most important radar topic with 9 fields:
    x, y, z, velocity, intensity, range, noise, time_cpu_cycles, frame_number
    Published by velocity_cloud_pub in the driver.
    """

    timestamp: float
    positions: np.ndarray  # Nx3 array of [x, y, z] positions in meters
    velocities: Optional[np.ndarray] = None  # N array of radial velocities in m/s
    intensities: Optional[np.ndarray] = None  # N array of reflected power
    ranges: Optional[np.ndarray] = None  # N array of ranges in meters
    noise: Optional[np.ndarray] = None  # N array of noise values
    time_cpu_cycles: Optional[np.ndarray] = None  # N array of CPU cycle timestamps
    frame_number: Optional[np.ndarray] = None  # N array of frame numbers
    frame_id: str = "ti_mmwave"

    def num_points(self) -> int:
        return len(self.positions) if self.positions is not None else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "num_points": self.num_points(),
            "has_velocities": self.velocities is not None,
            "has_intensities": self.intensities is not None,
            "has_ranges": self.ranges is not None,
            "has_noise": self.noise is not None,
        }


@dataclass
class BagData:
    """Container for all extracted rosbag data."""

    mocap_pose: List[MocapPose]
    mocap_accel: List[MocapAccel]
    agiros_state: List[AgirosState]
    agiros_odometry: List[AgirosOdometry]
    imu_data: List[IMUData]
    radar_pcl: List[RadarPointCloud]
    radar_velocity: List[RadarVelocity]

    # Metadata
    bag_path: str
    start_time: float
    end_time: float
    duration: float

    def summary(self) -> Dict[str, Any]:
        """Return a summary of loaded data."""
        return {
            "bag_path": self.bag_path,
            "duration_s": self.duration,
            "mocap_pose_samples": len(self.mocap_pose),
            "mocap_accel_samples": len(self.mocap_accel),
            "agiros_state_samples": len(self.agiros_state),
            "agiros_odometry_samples": len(self.agiros_odometry),
            "imu_samples": len(self.imu_data),
            "radar_pcl_samples": len(self.radar_pcl),
            "radar_velocity_samples": len(self.radar_velocity),
        }

    def get_sync_time_bounds(self) -> Tuple[float, float]:
        """Get time bounds where all available sensors have data."""
        time_ranges = []

        if self.mocap_pose:
            time_ranges.append((self.mocap_pose[0].timestamp, self.mocap_pose[-1].timestamp))
        if self.agiros_state:
            time_ranges.append((self.agiros_state[0].timestamp, self.agiros_state[-1].timestamp))
        if self.imu_data:
            time_ranges.append((self.imu_data[0].timestamp, self.imu_data[-1].timestamp))
        if self.radar_pcl:
            time_ranges.append((self.radar_pcl[0].timestamp, self.radar_pcl[-1].timestamp))

        if not time_ranges:
            return self.start_time, self.end_time

        max_start = max(t[0] for t in time_ranges)
        min_end = min(t[1] for t in time_ranges)
        return max_start, min_end
