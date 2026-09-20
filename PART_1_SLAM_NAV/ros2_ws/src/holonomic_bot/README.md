# holonomic_bot

This package defines the holonomic robot model itself: the simulated body, URDF/Xacro description, launch files, and ROS-Gazebo bridge setup needed to place the robot into a Gazebo Harmonic world.

It is intended to be reusable across different environments and sensor stacks.

## Overview

The robot package includes:

- a Gazebo SDF model for simulation
- a URDF/Xacro version for TF and RViz compatibility
- launch files for spawning and running the robot
- a ROS bridge configuration for topic communication
- optional SLAM tooling for mapping experiments

## Prerequisites

Use ROS 2 Humble and Gazebo Harmonic.

Install the required dependencies:

```bash
sudo apt update
sudo apt install ros-humble-ros-gz ros-humble-ros-gz-bridge \
  ros-humble-robot-state-publisher ros-humble-xacro ros-humble-rviz2 \
  ros-humble-slam-toolbox ros-humble-teleop-twist-keyboard
```

## Build

From your ROS workspace root:

```bash
cd ~/your_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select holonomic_bot
source install/setup.bash
```

This example builds only the `holonomic_bot` package. If you want the full workspace built, omit the package selection flag.

## Run the robot in a world

Start a Gazebo world in one terminal:

```bash
gz sim -r /path/to/your_world.sdf
```

In another terminal, spawn the robot and start the bridge:

```bash
ros2 launch holonomic_bot spawn_robot.launch.py rviz:=true
```

Optional manual drive test:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Robot capabilities

The robot includes:

- 4-wheel mecanum-style holonomic motion
- `/cmd_vel` input
- `/odom` odometry output
- `/scan` laser scan output
- `/tf` transforms for robot frames
- `/clock` simulation-time bridge

## Current sensor set

The default robot exposes:

- `gpu_lidar` on `/scan`
- odometry on `/odom`
- TF output for `odom -> base_link` and child frames
- robot pose information through Gazebo and the ROS bridge

## Adding more sensors

The robot can be extended with additional perception hardware.

Typical steps:

1. Add the sensor to `models/holonomic_bot/model.sdf` or the Xacro/URDF description.
2. Give it a unique frame and topic name.
3. Add the bridge mapping in the relevant config file.
4. Rebuild the package.
5. Relaunch the world and robot.

Common examples:

- RGB camera
- depth camera / point cloud
- GPS
- IMU
- ultrasonic sensor
- wheel encoders

## File structure

- `models/holonomic_bot/model.sdf` — Gazebo robot model
- `urdf/holonomic_bot.urdf.xacro` — URDF for TF and RViz
- `launch/spawn_robot.launch.py` — robot spawn and bridge launcher
- `launch/slam.launch.py` — SLAM launch support
- `config/` — bridge and parameter files

## SLAM usage

Once the robot is spawned and moving:

```bash
ros2 launch holonomic_bot slam.launch.py
```

This uses the laser scan, odometry, and TF data to build a map in RViz.

## Tips

- Ensure the Gazebo world loads the required sensor plugins or the laser topic may stay empty.
- Keep the physical model and URDF dimensions synchronized when modifying geometry.
- If you launch multiple robots, give each one a unique name and update the bridge topic names.
- In headless setups, rendering may require a working OpenGL or EGL backend.

## Notes

This package is designed to be reusable across different worlds and sensor stacks. The robot is intentionally separated from the world definition so you can switch environments without changing the robot model logic.
