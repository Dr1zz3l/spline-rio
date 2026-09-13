# TI mmWave ROS2 Package

### Created by Pedrhom Nafisi, updates collaborated with JuneChul Roh (Texas Instruments)
---
### Release Notes

#### ROS2 Driver v1.3.0 12 Nov 2024

* Added tracker visualization in RVIZ
* Added support for the following 6432 TLVs
  * 6432 Compressed Point Cloud TLV (TLV Type 301)
  *	6432 Tracker (TLV Type 308 and 309)
  *	Microdoppler Raw Data   (TLV Type 310)
  *	Microdoppler Feature Extraction (TLV Type 311)
  *	Classification Output Information (TLV Type 317)
* Added new launch files and configurations for 6843AOP for static object detection and stationary target detection (fineMotion)

#### ROS2 Driver v1.2.2 20 Aug 2024

* Fixed launch file issues with legacy devices

#### ROS2 Driver v1.2.1 15 Aug 2024

* Added support for 6432 ES2
* Improved 6432 configs to have better xyz positioning and velocity estimation

#### ROS2 Driver v1.2.0 12 Mar 2024

* Updated the launch files and the rviz file
* Added a common launch file IWR6843.py for all the IWR6843x launch files.
* CLI config file, command port, and data port can passed at the command line with launch arguments. 
* 6843{ISK,AOP}{Standard,Tracking,Occupancy}.py now call the common launch file.
* Updated the rviz file to display the XYZ axes.
* Updated to use the standard header message, std_msgs/Header. Note that "seq" field in the header was deprecated in ROS 2.
* Removed ti_mmwave_rospkg/src/{build, install, log} folders.

#### ROS2 Driver v1.1.1 12 Feb 2024

* Updated ISK launch file to call correct configuration file

#### ROS2 Driver v1.1.0 21 Nov 2023 

* Updated CMakeLists.txt to install the "cfg" folder.
* Path to .cfg file is now set using get_package_share_directory().
* Updated 6843*.py launch files to add a feature optionally disable rViz with an launch argument "rviz". To disable rViz, pass "rviz:=false" in the launch command.
* Updated to get rid of error message at Ctrl+C termination
* Compatible with reference to TI Robotics SDK for Jacinto Processors

#### ROS2 Driver Created v1.0.0 29 Aug 2023 

* Initially derived from TI's original ROS1 package found in TI's Resource Explorer's [Radar Toolbox](https://dev.ti.com/tirex/global?id=RADAR_TOOLBOX).

---

### Differences from original ROS1 TI's version:
1. Support ROS2 Humble as opposed to ROS1 Noetic. 
2. Has two packages, one where the service and messages are defined.
3. Number of header files used decreased substantially.
4. Launch files are all in Python.
5. Many executables across the nodes to ensure minimal change in final output.

---
### Available devices:
```
TI mmWave IWR1443BOOST EVM
TI mmWave IWR1642BOOST EVM
TI mmWave IWR1843BOOST EVM
TI mmWave IWR6843ISK EVM
TI mmWave IWR6843AOPEVM
TI mmWave IWRL6432BOOST EVM
```
---
### Quick start guide:
1. Mount mmWave Sensor if necessary and connect a micro-USB cable to host computer that has [ROS2 Humble](https://docs.ros.org/en/humble/index.html).

2. Download the mmWave [SDK](https://www.ti.com/tool/MMWAVE-SDK) or [L-SDK](https://www.ti.com/tool/MMWAVE-L-SDK) and then use [UNIFLASH](http://www.ti.com/tool/UNIFLASH) to flash a binary to your device.  
**Do not forget to change your device's MUX switches to and from flashing mode when flashing. Check the EVM Setup Operational Modes user guide in the Radar Toolbox under Hardware Guides**

3. Clone this repository into your `<workspace dir>/src`:

4. Go back to `<workspace dir>`:

```
colcon build
source install/setup.bash
```

5. Enable command and data ports on Linux:
```
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
or
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1
```
Note: If multiple sensors are used, enable additional ports `/dev/ttyACM2` and `/dev/ttyACM3`, etc. the same as this step.

6. Launch mmWave config using launch file, 6843ISK's Occupancy Detection example below:
```
ros2 launch ti_mmwave_rospkg 6843ISK_Occupancy.launch
```

Note: If you want to build your own config, use [mmWave Demo Visualizer](https://dev.ti.com/mmwavedemovisualizer) and link the launch file to the config.

7. RVIZ will automatically start.

8. ROS topics can be accessed as follows:
```
ros2 topic list
ros2 topic echo /ti_mmwave/radar_scan
```
9. ROS parameters can be accessed as follows:
```
ros2 param list
ros2 param get //ti_mmwave/max_doppler_vel
```

---
### Message format:
```
header: 
  seq: 6264
  stamp: 
    secs: 1538888235
    nsecs: 712113897
  frame_id: "ti_mmwave"   # Frame ID used for multi-sensor scenarios
point_id: 17              # Point ID of the detecting frame (Every frame starts with 0)
x: 8.650390625            # Point x coordinates in m (front from antenna)
y: 6.92578125             # Point y coordinates in m (left/right from antenna, right positive)
z: 0.0                    # Point z coordinates in m (up/down from antenna, up positive)
range: 11.067276001       # Radar measured range in m
velocity: 0.0             # Radar measured range rate in m/s
doppler_bin: 8            # Doppler bin location of the point (total bins = num of chirps)
bearing: 38.6818885803    # Radar measured angle in degrees (right positive)
intensity: 13.6172780991  # Radar measured intensity in dB
```
---
### Troubleshooting
1.
```
mmWaveCommSrv: Failed to open User serial port with error: IO Exception (13): Permission denied
mmWaveCommSrv: Power cycle the mmWave Sensor with the reset button while keeping the USB connected. Ensure the correct launch file and configuration files are being used. Close all nodes, wait 10 seconds, then relaunch the driver
```
If the instructions are followed and this still happens, check to make sure the serial port is called without superuser permission, do the following steps:
```
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```
2.
```
mmWaveQuickConfig: Command failed (mmWave sensor did not respond with 'Done')
mmWaveQuickConfig: Response: 'sensorStop
'?`????`????`???~' is not recognized as a CLI command
mmwDemo:/>'
```
When this happens, re-run the command you send to sensor. If it continues, shut down and restart the sensor.

---