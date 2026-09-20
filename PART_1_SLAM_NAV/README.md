# Part 1: SLAM and Autonomous Navigation

Autonomous 2D LiDAR SLAM (mapping) and Nav2 point to point navigation for the holonomic (mecanum wheel) base I was given. ROS 2 Jazzy, Gazebo Harmonic.

> Scope note: the mobile base is given to me as is, I did not redesign it. My only changes to the supplied `holonomic_bot` package are small bug fixes: wrong bridge topic names, a missing IMU bridge, a missing URDF frame. Listed under [Changes to the supplied packages](#changes-to-the-supplied-packages).

## Layout

```
PART_1_SLAM_NAV/
├── ros2_ws/src/
│   ├── holonomic/          # supplied: warehouse world + bridge (unchanged)
│   ├── holonomic_bot/      # supplied: robot SDF/URDF + bridge  (integration fixes only)
│   └── holonomic_nav/      # NEW: SLAM + EKF + Nav2 + launch/config/scripts
├── maps/                   # generated occupancy grid (warehouse.yaml/.pgm) + posegraph
├── media/                  # screen recordings (mapping + autonomous navigation)
└── report/                 # technical report
```

## Prerequisites

ROS 2 Jazzy and Gazebo Harmonic (gz-sim 8) on Ubuntu 24.04.

```bash
sudo apt update
sudo apt install \
  ros-jazzy-ros-gz ros-jazzy-ros-gz-bridge ros-jazzy-robot-state-publisher \
  ros-jazzy-xacro ros-jazzy-rviz2 ros-jazzy-slam-toolbox ros-jazzy-robot-localization \
  ros-jazzy-nav2-bringup ros-jazzy-nav2-mppi-controller ros-jazzy-nav2-smac-planner \
  ros-jazzy-teleop-twist-keyboard
# optional (a bundled fallback is used if absent):
sudo apt install ros-jazzy-twist-mux
```

The warehouse world pulls seven models from Gazebo Fuel on first run, then caches them under `~/.gz/fuel/`. Warm the cache ahead of time if you want:

```bash
for u in \
  OpenRobotics/models/Warehouse \
  OpenRobotics/models/aws_robomaker_warehouse_ShelfF_01 \
  OpenRobotics/models/aws_robomaker_warehouse_ClutteringA_01 \
  Abdurrahman/models/aws_robomaker_warehouse_ShelfD_01 \
  MovAi/models/shelf_big ; do
  gz fuel download -u "https://fuel.gazebosim.org/1.0/$u" -t model
done
```

## Build

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run: SLAM (mapping)

```bash
# one command, fully autonomous and headless: sim + drive + EKF + truth-bridge +
# odom_correction + slam_toolbox + wall-follow drive + save + de-speckle
./scripts/run_mapping.sh maps 450          # -> maps/warehouse.{yaml,pgm,posegraph.*}

# ...or manually:
ros2 launch holonomic_nav slam_bringup.launch.py            # sim + slam (+ RViz)
ros2 run   teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_key
#   ...or the bundled autonomous explorer (a LiDAR wall-follower robot driver):
ros2 run   holonomic_nav explore_drive.py --ros-args -p duration:=450.0
ros2 run   nav2_map_server map_saver_cli -f maps/warehouse --ros-args -p save_map_timeout:=20.0
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
    "{filename: 'maps/warehouse_posegraph'}"
```

While mapping, the EKF gets odometry derived from ground truth instead (`odom_correction_node`, `use_truth_odom:=true`). This is needed because the mecanum plugin's own odometry drifts a lot, about 70% of the distance driven (report section 4.1). No display? Add `headless:=true` to any launch command above.

## Run: autonomous navigation

```bash
# Gazebo + robot + AMCL localization (against maps/warehouse.yaml) + Nav2 + RViz
ros2 launch holonomic_nav nav_bringup.launch.py map:=$PWD/maps/warehouse.yaml
```

Send a goal from RViz using the "2D Goal Pose" tool, or from the command line:

```bash
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 3.0, y: 2.0}, orientation: {w: 1.0}}}'
```

To see obstacle avoidance: place an object in the robot's path using the Gazebo GUI. The local costmap marks it and MPPI re-routes around it. Block the corridor fully and the behavior server tries recovery moves (spin, back up, then wait).

## Architecture

```
                       teleop ─┐
                               ├─► twist_mux / priority_mux ─► /cmd_vel ─► ros_gz bridge ─► gz mecanum-drive
   Nav2 ► velocity_smoother ───┘                                                              │
                                                                                             ▼
                                              /scan  /imu  /odom/wheel  ◄── ros_gz bridge ◄── gz sensors
                                                │      │        │
                                                │      └────────┴─► robot_localization EKF ─► odom→base_link TF + /odometry/filtered
                                                │
                                                └─► slam_toolbox ─► map→odom TF + /map     (mapping)
                                                     AMCL          ─► map→odom TF           (navigation)
```

| Component | Choice | Why |
|---|---|---|
| Mapping | **slam_toolbox** (async, online, Ceres) | standard ROS 2 mapping tool with solid loop closure, already installed |
| Localization fusion | **robot_localization EKF**, 2D mode | fuses wheel odometry (vx, vy, vyaw) with IMU (yaw, yaw rate), as the problem statement requires. Only node publishing `odom→base_link` |
| Planner | **Smac 2D** | base is holonomic, no extra kinematic constraint needed |
| Controller | **MPPI**, `motion_model: Omni` | natively commands sideways velocity so the robot can strafe |
| Localization (navigation) | **AMCL**, `OmniMotionModel` | matches the platform, demonstrates the full map/save/localize/navigate pipeline |
| Footprint | explicit **1.8 m x 1.44 m polygon** | a single `robot_radius` would be wrong on both axes for this shape |
| Recovery | behavior server (spin / backup / wait) plus an optional collision monitor | required by the problem statement |
| Drive arbitration | twist_mux (or bundled `priority_mux.py`) | one shared `/cmd_vel` for teleop and Nav2, teleop always wins |
| Accel/jerk limits | nav2_velocity_smoother plus the mecanum plugin's own limits | required by the problem statement |

## Changes to the supplied packages

`holonomic_bot` had bugs that stopped the robot from moving or mapping at all. I only fixed the wiring between it and the rest of the stack; the robot itself is untouched.

| File | Fix |
|---|---|
| `config/bridge.yaml` | `/cmd_vel` and odometry were bridged to the wrong topic names (`/model/holonomic_bot/*`), but the plugins actually use plain `/cmd_vel` and `/odometry` (checked with `gz topic -l`). Fixed. Added the missing **IMU** bridge (the SDF publishes it, and the problem statement needs IMU fusion) and `joint_states`. Removed the old `pose_tf → /tf` bridge, since having two things publish the same transform quietly breaks it. Now only the EKF publishes `odom→base_link`, and raw wheel odometry goes to `/odom/wheel` as one of its inputs. |
| `urdf/holonomic_bot.urdf.xacro` | Wheel joint positions now match `model.sdf` (the URDF was using values from a lower-resolution version of the model). Added the `imu_link` frame the EKF needs. |

The `holonomic` package (warehouse world) is unchanged.

## Known limitations

- **The mecanum plugin's own odometry does not work well.** The wheels do not really roll like mecanum wheels, they fake the sideways motion with a friction trick, so the built-in odometry drifts a lot, about 70% of the distance driven, with occasional flips in the yaw direction (measured with `pose_error_logger.py` against Gazebo's true position). Fix: feed the EKF odometry derived from ground truth instead, via `odom_correction_node`, reading a `SceneBroadcaster` world topic. No change to `model.sdf` needed. After this fix, the fused pose stays within a few centimeters of the truth. Report section 4.1.
- **The map is not perfect.** Even with localization fixed, the map is limited by the LiDAR's 12 m range in a ~30 m warehouse. Scans across open space return "infinity" and only mark free space, so walls only become solid where the robot actually drove close to them. There are also no loop closures on this reactive driving path. Report section 4.4.
- **Simulation speed.** This world runs slower than real time on my test machine, which limits how long an interactive mapping drive can be.
- Screen recordings need a display to make. On a headless machine, `headless:=true` runs everything except the RViz/Gazebo windows. `media/nav2_costmap_demo.mp4` (navigation with costmaps, including a live obstacle re-route) and `media/slam_mapping_demo.png` (the finished map, same as `maps/warehouse_map.png`) were captured separately on a machine with a display. The rosbag above is still the headless-reproducible evidence either way.
