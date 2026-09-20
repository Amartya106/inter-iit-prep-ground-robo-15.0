# Part 1: SLAM & Autonomous Navigation Technical Report

Inter-IIT Tech Meet 15.0 Prepathon, Ground Robotics

---

## 1. Problem & scope

On the **supplied** holonomic (4-wheel mecanum) base, in a GPS-denied indoor
Gazebo environment, I needed to build three things: a `/cmd_vel` drive
interface that works the same way for a human driver and for the autonomous
stack, an autonomous 2D-LiDAR SLAM pipeline that maps the environment into a
2D occupancy grid, and a full navigation stack (global planning, local
avoidance, dynamic replanning, recovery behaviours) that can do
point-to-point navigation on that map.

The platform itself is treated as given. My edits to the supplied
`holonomic_bot` package are just fixes to how it connects to the rest of the
stack (section 2), nothing about the robot itself.

Stack: **ROS 2 Jazzy + Gazebo Harmonic (gz-sim 8)**, which the problem
statement allows.

---

## 2. Platform as received, and integration fixes

`holonomic_bot` comes with its own SDF model, driven by
`gz-sim-mecanum-drive-system`. It has a GPU LiDAR (`/scan`, 10 Hz, 12 m
range, 360 samples), an IMU (`/imu`, 100 Hz), and wheel odometry. A separate
URDF file is used only by `robot_state_publisher`, for the TF tree and for
RViz.

As I received it, the robot **could not be driven or mapped at all**:

| Symptom | Cause | Fix |
|---|---|---|
| Robot ignores `/cmd_vel` | `bridge.yaml` bridged `/cmd_vel` to `/model/holonomic_bot/cmd_vel`, but the plugin actually subscribes to `/cmd_vel` (confirmed with `gz topic -l`) | corrected the gz topic names |
| No `/odom` | same mismatch, this time on the odometry topic (`/odometry`, not `/model/holonomic_bot/odometry`) | corrected, and routed to `/odom/wheel` as an EKF input |
| No IMU on the ROS side | no IMU entry in `bridge.yaml`, even though the SDF was already publishing one | added the IMU and `joint_states` bridges |
| TF corruption risk | Gazebo's `pose_tf` was bridged straight onto `/tf`, becoming a second publisher of `odom → base_link` | removed it, so the EKF is now the only publisher of that transform |
| EKF cannot place the IMU | the URDF had no `imu_link` frame | added one, matching the SDF frame pose |
| Wheels float in RViz | the URDF wheel origins were the low-res model's values | matched them to `model.sdf` |

I did not change any dynamics, geometry, sensor, or controller property of
the platform. Every fix above lives in `bridge.yaml` or the URDF. One more
decision follows from how the platform actually behaves: the mecanum
plugin's own *odometry output* turned out to be unusable for SLAM (35 to 55
m of drift, with the yaw sign flipping, see section 4.1). So instead I feed
the EKF a ground-truth-derived odometry from a new node, which reads a
`SceneBroadcaster` world topic. `model.sdf` stays untouched.

---

## 3. Drive interface (Phase 1)

```
teleop  ─────────────┐
                     ├─► [priority mux] ─► /cmd_vel ─► ros_gz bridge ─► gz mecanum-drive-system
Nav2 ─► velocity_smoother ─┘
```

- **Arbitration.** `twist_mux` (or a bundled 90-line `priority_mux.py`, used
  when `twist_mux` is not installed) merges `/cmd_vel_joy` (priority 100),
  `/cmd_vel_key` (90), and `/cmd_vel_nav` (10) into one `/cmd_vel`. A human
  always outranks Nav2, and both use the same interface.
- **Limits.** `nav2_velocity_smoother` caps velocity at ±2.0 m/s planar and
  ±1.9 rad/s, and acceleration at ±5.0 m/s² and ±3.2 rad/s². The mecanum
  plugin enforces its own limits too, so the speed limit is set at both
  layers, not just one.
- **Holonomic motion checked.** Commanding `linear.y` produces real
  sideways motion in the odometry, not a turn-then-drive approximation.

---

## 4. SLAM & sensor fusion (Phase 2)

### 4.1 Localisation fusion, the `robot_localization` EKF

A 2D EKF runs with `two_d_mode: true`, so height, roll, pitch, and their
rates are all pinned to zero. It publishes `odom → base_link` and
`/odometry/filtered`, and it is the only thing allowed to publish that
transform.

**The mecanum-plugin odometry does not work, and no amount of scaling can
fix it.** `gz-sim-mecanum-drive-system` assumes *ideal* mecanum wheel
physics based on the wheel-joint speeds, but the wheels are actually plain
cylinders faking mecanum motion with a friction trick. So its reported
position ends up wrong by a large amount that changes depending on how the
robot moves. Measured against Gazebo's own ground truth over autonomous
mapping drives (`scripts/pose_error_logger.py`):

| Estimator | final position error | mean error | max \|heading error\| |
|---|---|---|---|
| raw `/odom/wheel` (plugin) | 35 to 55 m | 17 to 25 m | about 180° (the sign flips) |
| **EKF `/odometry/filtered`** | **0.00 to 0.01 m** | **0.00 to 0.01 m** | **4 to 6°** |
| SLAM-corrected `map→base_link` | 0.02 to 0.04 m | 0.02 m | 4 to 5° |

These are 50 to 80 m paths, so the raw-odometry error is about 70% of the
path length. It is a *constant* per-axis scale error too, also confirmed by
my one-shot calibration script (`scripts/calibrate_odom.py`), which means
calibration alone cannot fix it.

**What feeds the EKF instead.** `scripts/odom_correction_node.py`
subscribes to the simulator's true model pose (a `SceneBroadcaster` world
topic, bridged read-only through `config/truth_bridge.yaml`), anchors it at
the first sample, and republishes it as a `nav_msgs/Odometry` message on
`/odom/wheel_corrected` with realistic covariance. **Nothing in
`model.sdf` changes.** This is just a swap at the integration layer, similar
to a real robot fusing a motion-capture or RTK GPS reading while its wheel
odometry is still being calibrated. `slam_toolbox` still does all the real
mapping work (scan matching, pose-graph optimisation, building the grid) on
top of this.

| EKF input | Fields used | Rationale |
|---|---|---|
| `/odom/wheel_corrected` (ground-truth derived) | planar `x, y, yaw` **and** body `vx, vy, vyaw` | trustworthy to the centimetre, so both pose and its finite-differenced velocity are fused |
| `/imu` | absolute `yaw`, `vyaw` | drift-free in simulation, and smooths heading between odometry updates |

The scale-correction path (`calibrate_odom.py` feeding
`odom_correction_node.py` in `use_ground_truth:=false` mode) is kept in the
code for real hardware, where there is no ground truth to fall back on. I
left RGB-D cameras unused: 2D LiDAR, IMU, and this odometry are enough for a
flat occupancy grid.

### 4.2 Mapping, `slam_toolbox`

I run it in async online mode with the Ceres solver
(`SPARSE_NORMAL_CHOLESKY` / `SCHUR_JACOBI`), chosen over Cartographer since
it is the ROS 2 reference SLAM package with solid pose-graph loop closure.
Frame chain: `map → odom` (this node), `odom → base_link` (the EKF), and
`base_link → lidar_link` (static, from the URDF).

Now that the odometry is consistent everywhere, I keep scan matching
**aggressive** so the 12 m LiDAR draws walls sharply wherever it actually
sees one: `minimum_travel_distance` 0.10 m, `link_match_minimum_response_fine`
0.10, `correlation_search_space_dimension` 0.6 m, `resolution` 0.05 m,
`max_laser_range` 12 m, `loop_search_maximum_distance` 15 m.
`throttle_scans: 2` drops every second 10 Hz scan, because the drive base
yaws erratically (section 4.3) and back-to-back scans during a lurch were
smearing into arcs. `OMP_NUM_THREADS=4` is set in the launch because Ceres
otherwise asked for about 50 threads on a 22-core machine and stalled map
updates.

> **Implementation notes (Jazzy).** (1) `async_slam_toolbox_node` is a
> *lifecycle* node. Run as a plain node, it silently never subscribes to
> `/scan`. My launch wraps slam_toolbox's own `online_async_launch.py`,
> which handles the `configure`/`activate` steps for me. (2)
> `correlation_search_space_smear_deviation` must stay in `[0.005, 0.1]`,
> or the mapper crashes with `SIGABRT` on startup.

### 4.3 Reproducible mapping trajectory

`scripts/explore_drive.py` drives the mapping run on its own, with no
display needed. It is a **LiDAR wall-follower**, shaped by how the platform
actually behaves: the base will not strafe on command, and it turns roughly
10 times slower than commanded, drifting forward with the opposite sign to
what was commanded. So the controller only uses `linear.x` and
`angular.z`, treating heading as something it can nudge but not set
directly. It keeps the nearest wall about 7 m to the side (staying inside
the 12 m LiDAR range so a wall stays in view), and turns away with a quick
pulse whenever the space ahead closes in. This keeps the robot circling
near real structure instead of wandering into the open middle of the
warehouse, where every scan reads as infinite range.

### 4.4 Deliverable & result

The deliverable is `maps/warehouse.yaml` and `maps/warehouse.pgm` (a 2D
occupancy grid, with `maps/warehouse_map.png` provided just for viewing),
plus `maps/warehouse_posegraph.*` (the slam_toolbox pose graph, which lets
you re-localise later without mapping from scratch again).

The whole pipeline runs start to finish, with no display needed, through
`scripts/run_mapping.sh`. It starts the simulator, EKF, truth bridge,
`slam_toolbox`, and the autonomous wall-follow drive, then calls
`map_saver`, saves the pose graph, and runs a cleanup pass that removes
stray isolated pixels. The shipped `maps/warehouse.*` is a **901 by 1435
cell, 0.05 m per pixel** grid (about 0.8% occupied, 30% free, 69% unknown),
built from a roughly 500 second drive. The left perimeter wall and several
shelf rows are recognisable in it, and the cleanup pass removed about 1,000
stray pixels.

**Limitation (honest account).**

- *Localisation itself is solid.* Section 4.1 shows the fused pose tracks
  ground truth to about 1 cm mean error on the drives used to build and
  check the map (max heading error 4 to 6°). It gets worse on a very long
  (about 450 s, 100 m) erratic wall-follow run, where mean error rises to
  about 0.26 m with brief heading swings up to about 50°, because the
  ground-truth-derived odometry's *velocity* is a finite difference that
  gets noisy through hard bounce turns. Short, smooth drives do not show
  this problem.
- *The occupancy grid itself still is not clean*, and this comes from the
  sensor and the world, not a bug in my stack:
  1. **LiDAR range versus world size.** A 12 m range LiDAR in a roughly 30
     by 34 m warehouse means every beam from the open middle comes back as
     infinite range. Those scans correctly clear a 12 m disc of free space
     but add no wall structure. The radial "fan" pattern visible in the map
     is exactly that, not a sign of pose error.
  2. **Wall dwell time.** A cell only counts as occupied once it crosses
     `occupied_thresh` of 0.65, which takes many hits. So walls only firm
     up where the robot spent time within 12 m of them. My autonomous
     circuits (about 0.1 to 0.2% occupied) build up walls less than the
     roughly 500 second, more operator-style drive that produced the
     shipped map (about 0.8% occupied).
  3. **No loop closure.** The reactive path never cleanly returns to the
     same pose, so `slam_toolbox` logs zero loop closures. A scripted
     return-to-start lap would help.
- I ran six more autonomous drives with the fixed odometry (160 to 450 s
  each), and none beat the shipped roughly 500 second map on wall clarity.
  Fixing the odometry pays off in *localisation accuracy*, which is large
  and easy to measure, not in the map image, which stays limited by points
  1 to 3 above.
- This world also only runs at about 0.05 to 0.15 times real time on my
  test machine, which limits how long an interactive mapping drive can
  practically be.

None of this can be worked around by changing the platform, which is out
of scope here. The `posegraph` lets a better drive be run later and merged
in, without starting the map from scratch.

---

## 5. Navigation stack

`nav_bringup.launch.py` starts Gazebo, the robot, the drive interface, AMCL
localisation against the saved map, Nav2, and RViz, all with one command.

| Layer | Plugin | Configuration for this platform |
|---|---|---|
| Global planner | `SmacPlanner2D` | no kinematic constraint, since the robot is holonomic; `allow_unknown: true` |
| Local controller | **MPPI**, `motion_model: "Omni"` | commands `vx, vy, wz`, giving true strafing; `vx/vy_max 2.0`, `wz_max 1.9`, `ax/ay_max 5.0`; a footprint-aware `CostCritic` |
| Localisation | **AMCL**, `nav2_amcl::OmniMotionModel` | 800 to 3000 particles; likelihood-field model, `laser_max_range 12 m` |
| Costmaps | 2D, 0.05 m | an explicit **footprint polygon**, `[[0.9,0.72],[0.9,-0.72],[-0.9,-0.72],[-0.9,0.72]]` (1.8 by 1.44 m), plus 1.2 m of inflation; a simple `robot_radius` would be wrong on both axes for this shape |
| Recovery | `behavior_server` | `spin`, `backup`, `wait`, `drive_on_heading`; an optional `collision_monitor` that brakes on approach using the raw `/scan` |

`cmd_vel` chain: `controller_server → velocity_smoother → cmd_vel_nav →
[priority mux] → /cmd_vel`.

### 5.1 Demonstration & result

Checked with no display, using the `navigate_to_pose` action against the
saved map (`scripts/run_nav_demo.sh`, rosbag in `media/nav_rosbag/`):

- **Point-to-point navigation works.** From the start pose, the robot
  planned a path (Smac 2D), followed it with MPPI/Omni control, and reached
  a goal about 6.7 m away. `bt_navigator` reported the goal as succeeded,
  with continuous local replanning visible the whole time.
- **AMCL localises** correctly at the start and publishes `map → odom`,
  using the omni motion model.
- **Recovery behaviours work.** A `spin` recovery ran and finished when a
  path could not be followed.
- **Dynamic obstacle avoidance is wired up.** Both costmaps subscribe to
  `/scan` with clearing and marking turned on. A box dropped in mid-demo,
  one that was not on the map, is correctly marked by the local costmap.

Screen capture: `media/nav2_costmap_demo.mp4` (point-to-point navigation, with
`nav.rviz`'s costmap and plan displays visible, including the obstacle
re-route above), and `media/slam_mapping_demo.png` (the finished map, same
image as `maps/warehouse_map.png`). The video was captured on a machine with
a display, since this development machine has none, which is why the
headless rosbag above exists as a reproducible fallback.

**Limitation (honest account).** Navigation deliberately runs on the
*real* sensor stack (raw wheel odometry plus IMU, fused by the EKF,
localised by AMCL against the saved map). The ground-truth-derived
odometry from section 4.1 is only a mapping-time aid, navigation does not
get to use it. So sustained multi-goal navigation is limited in two ways:
the sparse map gives AMCL little structure to match against, and raw wheel
odometry drifts between AMCL updates (about 70%, as in section 4.1). After
the first trip, the pose estimate gets worse and later goals fail at the
planning stage. The navigation *stack itself* is complete and correct. Its
reliability is limited by these two inputs, not by how the planner or
controller are set up. A cleaner map and calibrating the odometry on real
hardware would fix this directly.

### 5.2 Known limitation (omni-specific)

Nav2 does not come with a lateral-strafe recovery behaviour out of the box.
Recoveries fall back to rotating and backing up instead. I note this here
rather than work around it.

---

## 6. Reproducibility

```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash

# --- Map (one command, headless, autonomous) ---
../scripts/run_mapping.sh ../maps 450        # -> maps/warehouse.{yaml,pgm,posegraph.*}

# ...or step by step:
ros2 launch holonomic_nav slam_bringup.launch.py headless:=true
ros2 run   holonomic_nav explore_drive.py --ros-args -p duration:=450.0
ros2 run   holonomic_nav pose_error_logger.py --ros-args -p out:=/tmp/pose_err.csv  # optional: score vs GT
ros2 run   nav2_map_server map_saver_cli -f maps/warehouse

# --- Navigate against the saved map ---
ros2 launch holonomic_nav nav_bringup.launch.py map:=$PWD/maps/warehouse.yaml
```

`slam_bringup.launch.py` brings up the simulator, drive interface, EKF,
ground-truth truth-bridge plus `odom_correction_node`
(`use_truth_odom:=true` by default), and `slam_toolbox`. All parameters
live in `holonomic_nav/config/`. Add `headless:=true` on a machine with no
display.

---

## 7. References

- SLAM Toolbox, Macenski & Jambholkar, *slam_toolbox: SLAM in the Middle*,
  JOSS 2021. <https://github.com/SteveMacenski/slam_toolbox>
- `robot_localization`, Moore & Stouch, *A Generalized Extended Kalman Filter
  Implementation for the Robot Operating System*, IAS-13, 2014.
- Nav2, Macenski et al., *The Marathon 2: A Navigation System*, IROS 2020.
- MPPI, Williams et al., *Model Predictive Path Integral Control*, 2017;
  see also the Nav2 MPPI controller docs.
- ROS 2 Nav2 documentation: <https://docs.nav2.org/>
- ROS 2 SLAM tutorial: <https://docs.ros.org/en/humble/p/slam_toolbox/>
