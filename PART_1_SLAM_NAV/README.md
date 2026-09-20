
# Part 1: SLAM and Autonomous Navigation

Autonomous 2D LiDAR SLAM and Nav2 point-to-point navigation for a supplied holonomic (mecanum-wheel) mobile robot using **ROS 2 Jazzy** and **Gazebo Harmonic**.

The goal of this part was to build a working mapping and autonomous navigation pipeline, including localization, obstacle avoidance, velocity control, and recovery behaviors.

> **Scope:** The mobile base was provided as-is. I did not redesign the robot or its drive system. My changes to the supplied `holonomic_bot` package are limited to integration and bug fixes, documented below. The new SLAM, localization, navigation, and supporting functionality is implemented in `holonomic_nav`.

## Demo

| Demo | Evidence |
|---|---|
| SLAM mapping | `media/slam_mapping_demo.png` |
| Autonomous navigation and obstacle avoidance | `media/nav2_costmap_demo.mp4` |
| Saved occupancy grid | `maps/warehouse_map.png` |

The navigation demo shows the robot navigating using Nav2, with costmaps and a live obstacle reroute.

## Problem Statement Coverage

| Requirement | Implementation |
|---|---|
| 2D LiDAR SLAM | `slam_toolbox` |
| Map generation and saving | Occupancy grid and serialized pose graph |
| Localization | `robot_localization` EKF during mapping; AMCL during navigation |
| Point-to-point navigation | Nav2 |
| Holonomic motion | MPPI controller with `Omni` motion model |
| Global planning | Smac 2D planner |
| Obstacle avoidance | Nav2 local costmap and MPPI |
| Recovery behaviors | Nav2 behavior server: spin, backup, and wait |
| Velocity smoothing | Nav2 velocity smoother and mecanum plugin limits |
| Teleoperation and navigation arbitration | `twist_mux` or bundled `priority_mux.py` |

## Repository Layout

```text
PART_1_SLAM_NAV/
├── scripts/
│   └── run_mapping.sh
├── ros2_ws/
│   └── src/
│       ├── holonomic/          # Supplied warehouse world and bridge
│       ├── holonomic_bot/      # Supplied robot; integration fixes only
│       └── holonomic_nav/      # SLAM, EKF, Nav2, launch/config/scripts
├── maps/
│   ├── warehouse.yaml
│   ├── warehouse.pgm
│   └── warehouse_posegraph.*
├── media/
│   ├── slam_mapping_demo.png
│   └── nav2_costmap_demo.mp4
└── report/
    └── technical report
```

## Prerequisites

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic (gz-sim 8)

Install the required packages:

```bash
sudo apt update

sudo apt install \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-xacro \
  ros-jazzy-rviz2 \
  ros-jazzy-slam-toolbox \
  ros-jazzy-robot-localization \
  ros-jazzy-nav2-bringup \
  ros-jazzy-nav2-mppi-controller \
  ros-jazzy-nav2-smac-planner \
  ros-jazzy-teleop-twist-keyboard
```

Optional:

```bash
sudo apt install ros-jazzy-twist-mux
```

If `twist_mux` is unavailable, the project includes a bundled `priority_mux.py` fallback.

### Gazebo Fuel models

The warehouse world downloads its models from Gazebo Fuel on first run and caches them under `~/.gz/fuel/`.

To download the models in advance:

```bash
for u in \
  OpenRobotics/models/Warehouse \
  OpenRobotics/models/aws_robomaker_warehouse_ShelfF_01 \
  OpenRobotics/models/aws_robomaker_warehouse_ClutteringA_01 \
  Abdurrahman/models/aws_robomaker_warehouse_ShelfD_01 \
  MovAi/models/shelf_big ; do

  gz fuel download \
    -u "https://fuel.gazebosim.org/1.0/$u" \
    -t model
done
```

An internet connection may be required on the first launch if the models are not cached.

## Build

Run the following commands from the repository root:

```bash
cd ros2_ws

source /opt/ros/jazzy/setup.bash

colcon build --symlink-install

source install/setup.bash
```

## Quick Start

The workflow consists of two stages:

1. Generate a map of the warehouse using SLAM.
2. Load the saved map and use Nav2 for autonomous navigation.

## 1. SLAM and Mapping

### Autonomous mapping

From the repository root:

```bash
./scripts/run_mapping.sh maps 450
```

This script runs the mapping pipeline, including:

- Gazebo simulation
- Robot and sensor bridges
- EKF and odometry correction
- `slam_toolbox`
- Autonomous wall-following exploration
- Map saving and post-processing

The expected map output is:

```text
maps/warehouse.yaml
maps/warehouse.pgm
maps/warehouse_posegraph.*
```

The script is intended to provide a headless, autonomous mapping workflow without requiring manual teleoperation.

### Manual mapping

Alternatively, launch the simulation and SLAM:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash

ros2 launch holonomic_nav slam_bringup.launch.py
```

Use keyboard teleoperation:

```bash
ros2 run teleop_twist_keyboard \
  teleop_twist_keyboard \
  --ros-args \
  -r cmd_vel:=/cmd_vel_key
```

Or run the bundled autonomous explorer:

```bash
ros2 run holonomic_nav explore_drive.py \
  --ros-args \
  -p duration:=450.0
```

### Save the map

After mapping, save the occupancy grid:

```bash
ros2 run nav2_map_server map_saver_cli \
  -f maps/warehouse \
  --ros-args \
  -p save_map_timeout:=20.0
```

Serialize the SLAM pose graph:

```bash
ros2 service call \
  /slam_toolbox/serialize_map \
  slam_toolbox/srv/SerializePoseGraph \
  "{filename: 'maps/warehouse_posegraph'}"
```

### Mapping odometry workaround

The supplied mecanum plugin's wheel odometry was observed to drift significantly, with errors of approximately 70% of the driven distance and occasional yaw-direction flips.

For mapping, `odom_correction_node` derives odometry from Gazebo ground truth through a `SceneBroadcaster` world topic. This corrected odometry is used by the EKF when `use_truth_odom:=true`.

This workaround improves pose consistency in simulation, but it is important to note that **the mapping pipeline is assisted by simulator ground truth**. It is not a demonstration of fully independent, sensor-only odometry.

The measured odometry behavior and correction are discussed in Section 4.1 of the technical report.

## 2. Autonomous Navigation

Once the map has been generated, launch Nav2:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash

ros2 launch holonomic_nav nav_bringup.launch.py \
  map:=$PWD/maps/warehouse.yaml
```

This launches:

- Gazebo and the robot
- AMCL localization using the saved map
- Nav2 planning and control
- Local and global costmaps
- RViz for visualization and goal selection

### Send a navigation goal

In RViz, select the **2D Goal Pose** tool and specify the desired destination and orientation.

Alternatively, publish a goal from the command line:

```bash
ros2 topic pub --once \
  /goal_pose \
  geometry_msgs/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 3.0, y: 2.0}, orientation: {w: 1.0}}}'
```

This sends a goal at approximately `(3.0, 2.0)` in the `map` frame.

### Obstacle avoidance and recovery

To demonstrate obstacle avoidance, place an object in the robot's planned path using the Gazebo GUI.

The local costmap detects the obstacle, and the MPPI controller can adjust the trajectory to navigate around it.

If a corridor is completely blocked, the Nav2 behavior server attempts recovery actions, including spinning, backing up, and waiting.

## System Architecture

```text
                     Teleoperation
                          │
                          ▼
                 ┌──────────────────┐
 Nav2 ──────────►│ Velocity /       │
                 │ Priority Mux     │
                 └────────┬─────────┘
                          │
                          ▼
                       /cmd_vel
                          │
                          ▼
                    ros_gz bridge
                          │
                          ▼
                    Gazebo robot
                          │
             ┌────────────┼────────────┐
             │            │            │
           /scan         /imu      /odom/wheel
             │            │            │
             │            └─────┬──────┘
             │                  ▼
             │          robot_localization
             │                 EKF
             │                  │
             │                  ▼
             │          odom → base_link
             │
             ├──────► slam_toolbox ─────► map → odom
             │          (mapping)              │
             │                                 ▼
             │                               /map
             │
             └──────► AMCL ──────────────► map → odom
                        (navigation)
```

During mapping, the odometry correction node provides ground-truth-derived odometry to the EKF. During navigation, AMCL provides map-based localization.

Only one node should publish the `odom → base_link` transform, while the active mapping or localization component provides `map → odom`.

## Design Choices

| Component | Implementation | Reason |
|---|---|---|
| SLAM | `slam_toolbox` (asynchronous) | ROS 2 mapping with occupancy-grid output and pose-graph support |
| Localization fusion | `robot_localization` EKF, 2D mode | Fuses configured odometry and IMU measurements |
| Global planner | Smac 2D | Grid-based planning without nonholonomic turning constraints |
| Local controller | MPPI, `Omni` motion model | Supports forward, lateral, and rotational velocity commands |
| Navigation localization | AMCL, `OmniMotionModel` | Supports localization against the saved map with a holonomic motion model |
| Footprint | Explicit 1.8 m × 1.44 m polygon | Represents the base geometry more accurately than a single circular radius |
| Recovery | Nav2 behavior server | Provides recovery actions when navigation is obstructed |
| Velocity control | Nav2 velocity smoother and drive-plugin limits | Smooths commands and respects configured motion limits |
| Command arbitration | `twist_mux` or bundled priority mux | Routes teleoperation and Nav2 commands through one shared velocity topic |

## Changes to Supplied Packages

The `holonomic` package, containing the warehouse world, is unchanged.

The supplied `holonomic_bot` package required several integration fixes to work with the navigation stack.

### `holonomic_bot`

| File | Changes |
|---|---|
| `config/bridge.yaml` | Corrected command-velocity and odometry bridge topic names |
| `config/bridge.yaml` | Added the missing IMU bridge required for sensor integration |
| `config/bridge.yaml` | Added the joint-states bridge |
| `config/bridge.yaml` | Removed the old `pose_tf → /tf` bridge to avoid competing transform publishers |
| `urdf/holonomic_bot.urdf.xacro` | Corrected wheel joint positions to match `model.sdf` |
| `urdf/holonomic_bot.urdf.xacro` | Added the missing `imu_link` frame |

The bridge topic corrections were verified using `gz topic -l`.

Raw wheel odometry is routed to `/odom/wheel`, and the EKF is configured as the publisher of `odom → base_link`.

These changes are limited to simulation and ROS integration; the supplied robot's physical design and drive system were not redesigned.

## Results and Known Limitations

### 1. Wheel odometry

The supplied mecanum plugin approximates lateral motion using a friction-based mechanism rather than physically modeling ideal mecanum wheel behavior.

Its wheel odometry exhibited substantial drift and occasional yaw-direction errors during testing.

The ground-truth-derived correction reduced the fused pose error to within a few centimeters in the tested simulation.

See Section 4.1 of the technical report for the measurement methodology and results.

### 2. Mapping quality

The generated map is usable for navigation, but it is not a perfect reconstruction of the warehouse.

The LiDAR has a range of approximately 12 m, while the warehouse is around 30 m across. Consequently, walls in open areas may not be fully observed unless the robot drives sufficiently close.

The reactive exploration path also does not produce loop closures in the demonstrated run.

See Section 4.4 of the technical report.

### 3. Simulation performance

The warehouse simulation runs slower than real time on the test machine, limiting the speed of interactive mapping and testing.

### 4. Headless operation and demo evidence

The mapping pipeline can run headlessly using the appropriate launch configuration.

However, capturing RViz and Gazebo screen recordings requires a graphical display.

The navigation costmap demo and finished map image were captured separately on a machine with a display.

## Technical Report

The accompanying technical report contains additional details about the implementation, odometry evaluation, mapping quality, and observed limitations.

See the `report/` directory.

## Summary

This project implements a complete simulation workflow for 2D LiDAR mapping and holonomic autonomous navigation using ROS 2 Jazzy and Gazebo Harmonic.

The main contribution is the integration of SLAM, localization, Nav2 planning and control, obstacle avoidance, and recovery behaviors with the supplied mecanum-wheel robot, along with the supporting fixes needed to make the system operate reliably in simulation.