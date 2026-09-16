FROM osrf/ros:noetic-desktop-full

# Install useful dev tools
RUN apt-get update && apt-get install -y \
    usbutils \
    udev \
    git \
    build-essential \
    python3-catkin-tools \
    python3-rosdep \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create catkin workspace structure
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws

# Initialize rosdep
RUN rosdep update

# Auto-source ROS + workspace
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

# Add USB device setup script
COPY scripts/setup_usb_devices.sh /usr/local/bin/setup_usb_devices.sh
RUN chmod +x /usr/local/bin/setup_usb_devices.sh

# Set working directory
WORKDIR /root/catkin_ws

# Run USB setup on container start, then open bash
CMD ["/bin/bash", "-c", "setup_usb_devices.sh || true && exec /bin/bash"]