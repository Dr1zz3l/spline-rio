# IWR6843 Docker Development Environment

This project provides a Dockerized development environment for working with the IWR6843AOPEVM radar driver. The setup enables live code editing on your host machine while running ROS in a container with full USB and GUI support.

## Project Structure

```
iwr6843-docker/
├── docker/
│   ├── Dockerfile          # ROS Noetic base image with development tools
│   └── docker-compose.yml  # Container config with volume mounts and device access
├── scripts/
│   ├── setup_submodule.sh  # Initialize TI driver git submodule
│   ├── build.sh            # Build Docker image
│   └── run.sh              # Start container and enter shell
├── config/
│   └── radar_config.cfg    # Radar sensor configuration
├── mmwave_ti_ros/          # TI driver submodule (created by setup script)
├── .gitignore              # Ignore build artifacts
└── .dockerignore           # Exclude unnecessary files from Docker build
```

## Development Workflow

### VS Code Dev Containers (Recommended)

The easiest way to develop is using VS Code's Dev Containers extension:

1. **Install the Dev Containers extension** (already installed for you!)

2. **Setup the TI driver submodule**:
   ```bash
   cd iwr6843-docker
   ./scripts/setup_submodule.sh
   ```

3. **Open in container**:
   - Press `F1` or `Ctrl+Shift+P`
   - Type "Dev Containers: Reopen in Container"
   - VS Code will build and connect to the container automatically

4. **Start developing**:
   - Your terminal runs inside the container
   - IntelliSense works with ROS code
   - Extensions run in the container
   - Code changes are instant (no rebuilds needed)

5. **Build and run**:
   ```bash
   # In VS Code's integrated terminal (already inside container)
   catkin_make
   source devel/setup.bash
   roslaunch ti_mmwave_rospkg 6843_multi_3d_0.launch


   # alternatively: 

   # Use the best_velocity config
   roslaunch ti_mmwave_rospkg 6843AOP_velocity_3d.launch cfg_file:=$(rospack find ti_mmwave_rospkg)/cfg/6843AOP_best_velocity.cfg

   # Headless version
   roslaunch ti_mmwave_rospkg 6843AOP_velocity_3d_headless.launch cfg_file:=$(rospack find ti_mmwave_rospkg)/cfg/6843AOP_best_velocity.cfg
   ```

### Manual Docker Workflow (Alternative)

If you prefer not to use Dev Containers:

1. **Setup the TI driver submodule**:
   ```bash
   cd iwr6843-docker
   ./scripts/setup_submodule.sh
   ```

2. **Build the Docker image**:
   ```bash
   ./scripts/build.sh
   ```

3. **Start the development container**:
   ```bash
   ./scripts/run.sh
   ```

4. **Build the ROS workspace** (first time only):
   ```bash
   cd /root/catkin_ws
   rosdep install --from-paths src --ignore-src -r -y
   catkin_make
   source devel/setup.bash
   ```

5. **Run the radar driver**:
   ```bash
   roslaunch ti_mmwave_rospkg 6843_multi_3d_0.launch
   ```

### Develop Code

**The key advantage**: The `mmwave_ti_ros/` directory is mounted from your host into the container. This means:

- **Edit on host**: Use VS Code, vim, or any editor on your host machine to modify code in `iwr6843-docker/mmwave_ti_ros/`
- **Build in container**: Changes are immediately visible inside the container
- **Rebuild**: Run `catkin_make` in the container to rebuild after changes
- **Commit from host**: Use git on your host to commit changes to the submodule

### Commit Workflow

1. **Make changes** to files in `mmwave_ti_ros/` using your host editor
2. **Test in container**: Rebuild and test
   ```bash
   # Inside container
   catkin_make
   roslaunch ti_mmwave_rospkg <your_launch_file>
   ```
3. **Commit on host**:
   ```bash
   cd mmwave_ti_ros
   git add <files>
   git commit -m "Your commit message"
   git push
   ```

### Daily Development

To resume work:
```bash
./scripts/run.sh  # Starts container or attaches if already running
# Inside container:
source /root/catkin_ws/devel/setup.bash
catkin_make  # If you made changes
roslaunch ...
```

To stop the container:
```bash
exit  # From inside container
docker-compose -f docker/docker-compose.yml down
```

## Features

- **Live Code Editing**: Edit code on host, changes reflected immediately in container
- **USB Device Access**: Direct access to radar hardware via `/dev/bus/usb`
- **GUI Support**: X11 forwarding for RViz and other visualization tools
- **Persistent Builds**: Build artifacts cached in Docker volumes (optional)
- **Host Networking**: Seamless ROS communication with host tools

## Requirements

- Docker & Docker Compose
- Git
- X11 server (for GUI support)
  ```bash
  xhost +local:docker  # Run this on host before starting container
  ```

## Troubleshooting

**Container can't access display**:
```bash
xhost +local:docker
```

**USB device not found**:
- Ensure container is run with `privileged: true` (already set)
- Check device permissions: `ls -l /dev/ttyUSB*` or `/dev/ttyACM*`

**Code changes not visible**:
- Verify the volume mount is working: `ls /root/catkin_ws/src/mmwave_ti_ros`
- Rebuild: `catkin_make`

**Git submodule issues**:
```bash
git submodule update --init --recursive
```