# Localization Graph SLAM

This package provides a graph-based SLAM pipeline for a TurtleBot in ROS 2. It includes:

- a line extraction node based on the Split and Merge algorithm
- a localization and mapping node that fuses wheel encoder, IMU, and LiDAR data
- visualization of extracted scan points and line segments in RViz

## Package Contents

The package exposes two ROS 2 executables:

- `line_extraction`: extracts and visualizes line features from LiDAR scans
- `localization_graph_slam`: performs graph-based localization using odometry, IMU, and LiDAR line features

## Prerequisites

Before running the nodes, make sure:

- the ROS 2 workspace has been built
- the workspace has been sourced
- the TurtleBot simulation is running
- required Python and ROS 2 dependencies are installed

## Build the Workspace

From the root of your workspace:

```bash
colcon build
```

## Step 1: Launch the Simulation

Start the TurtleBot simulation with:

```bash
ros2 launch turtlebot_simulation turtlebot_hoi_circuit1.launch.py
```

### Step 2: Run Extraction line node

Run extraction line node:

```bash
ros2 run localization_graph_slam line_extraction --ros-args -p mode:=sim
```

### Step 3: Run Graph SLAM node

```bash
ros2 run localization_graph_slam localization_graph_slam --ros-args -p mode:=sim
```

## Run Modes

Both nodes support a `mode` parameter:

- `sim`: keeps simulation conventions (IMU NED->ENU conversion, LiDAR +pi flip, simulation frames/topics).
- `real`: uses real robot conventions (no IMU NED->ENU conversion, LiDAR without +pi flip, real robot frames/topics).

### Simulation mode

```bash
ros2 run localization_graph_slam line_extraction --ros-args -p mode:=sim
ros2 run localization_graph_slam localization_graph_slam --ros-args -p mode:=sim
```

### Real robot mode

```bash
ros2 run localization_graph_slam line_extraction --ros-args -p mode:=real
ros2 run localization_graph_slam localization_graph_slam --ros-args -p mode:=real
```














-----------------



cd /home/elchina/Documents/HOL_project/Graph_SLAM_Project_tomerge
source /home/elchina/Documents/HOL_project/.venv-gtsam43/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch turtlebot_simulation turtlebot_hoi_circuit1.launch.py

<!-- cd /home/elchina/Documents/HOL_project/Graph_SLAM_Project_tomerge
source /home/elchina/Documents/HOL_project/.venv-gtsam43/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python -m localization_graph_slam.line_extraction --ros-args -p mode:=sim -->

cd /home/elchina/Documents/HOL_project/Graph_SLAM_Project_tomerge
source /home/elchina/Documents/HOL_project/.venv-gtsam43/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python -m localization_graph_slam.perform_localization --ros-args -p mode:=sim

cd /home/elchina/Documents/HOL_project/Graph_SLAM_Project_tomerge
source /home/elchina/Documents/HOL_project/.venv-gtsam43/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/turtlebot/cmd_vel


-----------------

Real robot run (same style as above):

cd /home/elchina/Documents/HOL_project/Graph_SLAM_Project_tomerge
source /home/elchina/Documents/HOL_project/.venv-gtsam43/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python -m localization_graph_slam.line_extraction --ros-args -p mode:=real

cd /home/elchina/Documents/HOL_project/Graph_SLAM_Project_tomerge
source /home/elchina/Documents/HOL_project/.venv-gtsam43/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python -m localization_graph_slam.perform_localization --ros-args -p mode:=real

cd /home/elchina/Documents/HOL_project/Graph_SLAM_Project_tomerge
source /home/elchina/Documents/HOL_project/.venv-gtsam43/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/turtlebot/cmd_vel



cd /home/elchina/Documents/HOL_project/Graph_SLAM_Project_tomerge
source /home/elchina/Documents/HOL_project/.venv-gtsam43/bin/activate
source /opt/ros/jazzy/setup.bash
colcon build --packages-select localization_graph_slam
source install/setup.bash
python -m localization_graph_slam.perform_localization --ros-args -p mode:=real