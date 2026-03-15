# Radar Point Cloud Fields

## Implementation Status: ✅ VERIFIED

The radar loader has been verified against the actual TI mmWave driver source code (`DataHandlerClass.cpp`) and successfully extracts all fields from both radar topics.

## Topics

### 1. `/mmWaveDataHdl/RScanVelocity` (PRIORITY 1 - Most Important)

**Publisher**: `velocity_cloud_pub` in DataHandlerClass.cpp (lines 797-843)

**Format**: sensor_msgs/PointCloud2 with 9 custom fields

**Fields** (as defined in driver lines 798-806):
```cpp
"x"               (FLOAT32)  - X position in meters
"y"               (FLOAT32)  - Y position in meters  
"z"               (FLOAT32)  - Z position in meters
"velocity"        (FLOAT32)  - Radial velocity in m/s
"intensity"       (FLOAT32)  - Reflected power
"range"           (FLOAT32)  - Computed range sqrt(x²+y²+z²)
"noise"           (FLOAT32)  - Noise value
"time_cpu_cycles" (UINT32)   - CPU cycle timestamp
"frame_number"    (UINT32)   - Frame number
```

**Python Access**:
```python
frame = data.radar_velocity[0]
positions = frame.positions         # Nx3 array [x, y, z]
velocities = frame.velocities       # N array (radial velocity)
intensities = frame.intensities     # N array
ranges = frame.ranges              # N array
noise = frame.noise                # N array
time_cpu_cycles = frame.time_cpu_cycles  # N array (uint32)
frame_number = frame.frame_number       # N array (uint32)
```

### 2. `/ti_mmwave/radar_scan_pcl_0`

**Publisher**: `DataUARTHandler_pub` in DataHandlerClass.cpp (line 779)

**Format**: PCL PointCloud2 with mmWaveCloudType structure (5 fields)

**Fields**:
```cpp
x         (FLOAT32) - X position in meters
y         (FLOAT32) - Y position in meters
z         (FLOAT32) - Z position in meters
intensity (FLOAT32) - Reflected power
velocity  (FLOAT32) - Radial velocity in m/s
```

**Python Access**: Same as RScanVelocity (fields are None if not present)

## Usage

### Load Data
```python
import rosbag_loader

data = rosbag_loader.load_bag_topics('/path/to/bag.bag')
print(f"Loaded {len(data.radar_velocity)} radar frames")
```

### Plot 3D Velocity Cloud
```python
fig = rosbag_loader.plot_radar_velocity_3d(
    data.radar_velocity,
    frame_idx=0,
    color_by='velocity',  # or 'intensity', 'range', 'noise'
    vmin=-5,
    vmax=5,
)
plt.show()
```

### Plot Bird's Eye View
```python
fig = rosbag_loader.plot_radar_bev(
    data.radar_velocity,
    frame_idx=0,
    color_by='velocity',
    xlim=(-5, 5),
    ylim=(-5, 5),
)
plt.show()
```

### Plot Time Series
```python
fig = rosbag_loader.plot_radar_time_series(
    data.radar_velocity,
    metric='num_points',  # or 'mean_velocity', 'mean_intensity', 'mean_range'
)
plt.show()
```

### Compare Both Topics
```python
fig = rosbag_loader.plot_radar_comparison(
    data.radar_pcl,
    data.radar_velocity,
    pcl_idx=0,
    vel_idx=0,
    color_by='velocity',
)
plt.show()
```

## Notebooks

- **`radar_visualization.ipynb`** - Comprehensive examples of all plotting functions
- **`rosbag_loader_example.ipynb`** - General loader usage
- **`inspect_messages.ipynb`** - Message structure inspection

## Driver Reference

See `/workspace/mmwave_ti_ros/ros1_driver/src/ti_mmwave_rospkg/src/DataHandlerClass.cpp`:
- Lines 797-843: velocity_cloud_msg creation and publishing
- Lines 54-55: Publisher initialization
- Line 779: PCL pointcloud publishing

## Test Results

```
✅ Successfully loads 506 radar frames from circle bag
✅ All 9 fields present in RScanVelocity
✅ Velocity range: [-1.21, -1.21] m/s
✅ Intensity range: [15.30, 15.30]
✅ Range: [0.21, 0.21] m
```
