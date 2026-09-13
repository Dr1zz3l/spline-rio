# RosBag Loader Module

A Python module for loading and processing ROS bag files containing mmWave radar, IMU, and motion capture data for autonomous robotics odometry estimation.

## Overview

This module provides utilities to:
- **Inspect** rosbag topics and message types
- **Load** rosbag data into typed data structures
- **Convert** ROS messages into dataclasses for analysis
- **Synchronize** data from multiple sensors

## Data Sources

The module handles the following ROS topics:

### Raw MoCap Data
- **`/mocap/angrybird2/pose`** (geometry_msgs/PoseStamped)
  - Pure Vicon motion capture data
  - Position: [x, y, z] in meters
  - Orientation: [qx, qy, qz, qw] quaternion

- **`/mocap/angrybird2/accel`** (geometry_msgs/AccelStamped)
  - Linear acceleration from Vicon
  - ⚠️ **Note**: Contains position data jumps

### State Estimation (Kalman Filtered)
- **`/angrybird2/agiros_pilot/state`** (nav_msgs/Odometry)
  - Better quality than odometry
  - Higher resolution
  - Position, velocity, orientation, angular velocity

- **`/angrybird2/agiros_pilot/odometry`** (nav_msgs/Odometry)
  - Lower quality estimate
  - Alternative for comparison

### IMU Data
- **`/angrybird2/imu`** (sensor_msgs/Imu)
  - Raw IMU measurements from Pixhawk
  - Linear acceleration, angular velocity, orientation

### Radar Data
- **`/ti_mmwave/radar_scan_pcl_0`** (sensor_msgs/PointCloud2)
  - Radar point cloud format
  - Position, intensity, velocity data

- **`/mmWaveDataHdl/RScanVelocity`** (custom message)
  - Radar velocity measurements

## Installation

```bash
cd /workspace/analysis
pip install -r requirements.txt
```

Required packages:
- rosbag
- numpy
- pandas

## Usage

### 1. Inspect Rosbag Topics

```python
from rosbag_loader import inspect_bag_topics

bag_path = '/workspace/rosbags/2025-12-17-16-02-22.bag'
topics_info = inspect_bag_topics(bag_path)
```

Output:
```
Topic                                          Type                                   Count
/mocap/angrybird2/pose                         geometry_msgs/PoseStamped              5000
/angrybird2/imu                                sensor_msgs/Imu                        10000
...
```

### 2. Load All Topics

```python
from rosbag_loader import load_bag_topics

bag_data = load_bag_topics(bag_path, verbose=True)

# Access data
print(f"Loaded {len(bag_data.mocap_pose)} MoCap poses")
print(f"Loaded {len(bag_data.imu_data)} IMU samples")
print(f"Loaded {len(bag_data.radar_pcl)} radar point clouds")
```

### 3. Access Dataclass Objects

All data is loaded into strongly-typed dataclasses:

```python
# MoCap Pose
pose = bag_data.mocap_pose[0]
print(f"Position: {pose.position}")      # [x, y, z]
print(f"Orientation: {pose.orientation}") # [qx, qy, qz, qw]
print(f"Timestamp: {pose.timestamp}")

# IMU Data
imu = bag_data.imu_data[0]
print(f"Acceleration: {imu.linear_acceleration}")  # [ax, ay, az]
print(f"Gyroscope: {imu.angular_velocity}")        # [wx, wy, wz]

# Agiros State
state = bag_data.agiros_state[0]
print(f"Position: {state.position}")
print(f"Velocity: {state.velocity}")
print(f"Angular velocity: {state.angular_velocity}")
```

### 4. Convert to DataFrames

```python
import pandas as pd

# Convert to pandas for analysis
mocap_df = pd.DataFrame([pose.to_dict() for pose in bag_data.mocap_pose])
imu_df = pd.DataFrame([imu.to_dict() for imu in bag_data.imu_data])
state_df = pd.DataFrame([state.to_dict() for state in bag_data.agiros_state])

print(mocap_df.head())
```

### 5. Find Synchronized Time Range

```python
# Get time bounds where all sensors have data
t_min, t_max = bag_data.get_sync_time_bounds()

# Filter to synchronized range
sync_poses = [p for p in bag_data.mocap_pose if t_min <= p.timestamp <= t_max]
sync_imus = [i for i in bag_data.imu_data if t_min <= i.timestamp <= t_max]
```

### 6. Get Data Summary

```python
summary = bag_data.summary()
# Returns:
# {
#     'bag_path': '...',
#     'duration_s': 30.5,
#     'mocap_pose_samples': 5000,
#     'imu_samples': 10000,
#     ...
# }
```

## Data Classes

### MocapPose
```python
@dataclass
class MocapPose:
    timestamp: float          # seconds since epoch
    position: np.ndarray      # [x, y, z] in meters
    orientation: np.ndarray   # [qx, qy, qz, qw]
    frame_id: str            # "mocap"
    child_frame_id: str      # "angrybird2"
```

### AgirosState
```python
@dataclass
class AgirosState:
    timestamp: float
    position: np.ndarray        # [x, y, z]
    velocity: np.ndarray        # [vx, vy, vz]
    orientation: np.ndarray     # [qx, qy, qz, qw]
    angular_velocity: np.ndarray # [wx, wy, wz]
```

### IMUData
```python
@dataclass
class IMUData:
    timestamp: float
    linear_acceleration: np.ndarray  # [ax, ay, az]
    angular_velocity: np.ndarray     # [wx, wy, wz]
    orientation: Optional[np.ndarray] # [qx, qy, qz, qw] if available
```

### RadarPointCloud
```python
@dataclass
class RadarPointCloud:
    timestamp: float
    points: np.ndarray          # Nx3 array of [x, y, z]
    intensities: Optional[np.ndarray]
    velocities: Optional[np.ndarray]
    point_ids: Optional[np.ndarray]
```

### BagData
Container for all extracted data:
```python
@dataclass
class BagData:
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
```

## Integration with Factor Graph Odometry

The dataclasses are designed to work with factor graph optimization pipelines:

```python
# Create factors for each sensor measurement
for pose_measurement in bag_data.mocap_pose:
    # Add pose factor to graph
    factor = PoseFactor(
        timestamp=pose_measurement.timestamp,
        position=pose_measurement.position,
        orientation=pose_measurement.orientation,
    )

for imu_measurement in bag_data.imu_data:
    # Add IMU factor to graph
    factor = IMUFactor(
        timestamp=imu_measurement.timestamp,
        acceleration=imu_measurement.linear_acceleration,
        angular_velocity=imu_measurement.angular_velocity,
    )

# Find synchronized measurements for constraints
t_min, t_max = bag_data.get_sync_time_bounds()
```

## Notes on Data Quality

- **MoCap Accel**: Contains position jumps - filter or smooth if using
- **Agiros State vs Odometry**: State is higher quality and higher resolution
- **Synchronization**: Use `get_sync_time_bounds()` to find overlapping data
- **Timestamps**: All timestamps are in seconds since epoch (rosbag convention)

## Example Notebook

See [rosbag_loader_example.ipynb](./rosbag_loader_example.ipynb) for complete examples including:
- Inspecting bag contents
- Loading and accessing data
- Converting to DataFrames
- Visualization
- Synchronization analysis

## Module Structure

```
rosbag_loader/
├── __init__.py          # Module exports
├── loader.py            # Bag loading and inspection functions
└── structures.py        # Dataclass definitions
```

## Common Issues

### ImportError: No module named 'rosbag'
Install rosbag:
```bash
pip install rosbag
```

### Topic not found
Check available topics with `inspect_bag_topics()` first. Topic names must match exactly.

### Type mismatch
Some message fields may have different structures. Check the sample output from `inspect_bag_topics()`.

## Future Enhancements

- [ ] Custom message parsing for non-standard message types
- [ ] Automatic time alignment and interpolation
- [ ] Outlier detection for position jumps
- [ ] Saving/loading to HDF5 or parquet format
- [ ] Visualization utilities
