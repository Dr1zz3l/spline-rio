# Radar Inertial Odometry

ROS workspace for developing radar-inertial odometry pipeline.

## Structure

```
radar_odometry/
├── src/                    # ROS packages
│   └── radar_inertial_odometry/  # Main odometry package
├── build/                  # Build artifacts (gitignored)
└── devel/                  # Development space (gitignored)
```

## Build

```bash
cd /workspace/radar_odometry
catkin_make
source devel/setup.bash
```

## Usage

```bash
# Launch your odometry node
roslaunch radar_inertial_odometry odometry.launch
```
