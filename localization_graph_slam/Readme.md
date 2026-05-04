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
ros2 run localization_graph_slam line_extraction
```
