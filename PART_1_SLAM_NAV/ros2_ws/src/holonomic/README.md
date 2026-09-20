# holonomic

This package provides the Gazebo Harmonic warehouse environment and the ROS bridge setup used to test a holonomic robot in simulation.

It is intended as a reusable environment package for robotics research, perception, navigation, and autonomy experiments.

## Repository structure

This repository contains two main packages:

- `holonomic` — simulation world, launch files, and environment setup
- `holonomic_bot` — robot model, topic bridge configuration, and robot-specific launch files

## Prerequisites

This project is built for ROS 2 Humble with Gazebo Harmonic.

Install the required ROS tooling:

```bash
sudo apt update
sudo apt install ros-humble-ros-gz ros-humble-ros-gz-bridge \
  ros-humble-robot-state-publisher ros-humble-xacro ros-humble-rviz2 \
  ros-humble-teleop-twist-keyboard
```

## Build

From your ROS workspace root:

```bash
cd ~/your_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select holonomic
source install/setup.bash
```

This example builds only the `holonomic` package. If you want to build the robot package as well, use:

```bash
colcon build --packages-select holonomic_bot
```

## Launch the warehouse world

After sourcing the workspace overlay:

```bash
ros2 launch holonomic spawn_warehouse.launch.py
ros2 launch holonomic_bot spawn_robot.launch.py x:=1.0 y:=2.0 z:=0.11 yaw:=1.57

```

If you want to drive the robot manually:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## What this launch starts

The launch file starts the simulation stack, including:

- `robot_state_publisher`
- Gazebo Harmonic world
- robot spawn into the world
- ROS-Gazebo bridge
- `/cmd_vel` decomposition node

## Available topics and sensors

The robot and world publish and consume the following ROS topics through the bridge:

- `/cmd_vel` — robot command input
- `/joint_x_cmd` — decomposed body-frame X motion command
- `/joint_y_cmd` — decomposed body-frame Y motion command
- `/joint_yaw_cmd` — yaw-rate command
- `/odom` — odometry output
- `/tf` — TF tree and robot frames
- `/scan` — 2D LiDAR scan data
- `/imu` — IMU data
- `/clock` — simulation clock

The default environment includes:

- LiDAR: `gpu_lidar` on `/scan`
- IMU: inertial sensor on `/imu`
- Odometry: `/odom`

## Adding more sensors

Additional sensors can be added by extending the robot description and bridge configuration.

Typical workflow:

1. Add the new sensor in the Xacro/URDF or SDF model.
2. Assign a unique topic and frame name.
3. Add the corresponding bridge mapping in `config/bridge_harmonic.yaml`.
4. Rebuild the package.
5. Relaunch the simulation.

Examples:

- RGB camera -> `/camera/image_raw`
- depth camera -> `/depth_camera/points`
- ultrasonic range sensor -> `/range`

The key rule is that the Gazebo topic name and the ROS bridge topic name must match.

## Key files

- `launch/spawn_warehouse.launch.py` — main launch entrypoint
- `launch/rsp_harmonic.launch.py` — robot state publisher setup
- `description/robot_harmonic.urdf.xacro` — robot model and sensors
- `config/bridge_harmonic.yaml` — ROS-Gazebo bridge mappings
- `worlds/warehouse.sdf` — warehouse world definition

## Troubleshooting

- No sensor messages: confirm the Gazebo world is running and the bridge names match the actual topic names.
- Robot does not move: verify the `/cmd_vel` decomposer and bridge nodes are active.
- World does not render: check the local graphics stack and Gazebo Harmonic compatibility for your machine.

## Notes

This package is designed to be extended with more sensors, different static objects, or other world layouts without changing the core ROS launch flow.
