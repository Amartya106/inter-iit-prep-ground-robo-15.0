#!/bin/bash
# Minimal Nav2 demo: one known-good point-to-point run, then the same route
# with an unmapped obstacle dropped on it -> global replan + local avoidance.
source /opt/ros/jazzy/setup.bash
source /storage/Ground-Robo-Prepathon/submission/PART_1_SLAM_NAV/ros2_ws/install/setup.bash
WS=/storage/Ground-Robo-Prepathon/submission/PART_1_SLAM_NAV/ros2_ws
LOG=/storage/Ground-Robo-Prepathon/submission/.work
cd $WS
rm -rf $LOG/nav_bag
setsid ros2 launch holonomic_nav nav_bringup.launch.py headless:=true > $LOG/nav_bringup.log 2>&1 &
NAV_PG=$!
teardown() {
  kill -9 -$NAV_PG 2>/dev/null
  for n in "gz sim -r" "gz_tools_vendor/bin/gz sim" ekf_node parameter_bridge priority_mux \
    controller_server planner_server bt_navigator behavior_server amcl map_server \
    velocity_smoother lifecycle_manager smoother_server waypoint_follower \
    "robot_state_publisher --ros-args --params-file /tmp" "ros2 bag record"; do
    for p in $(pgrep -f "$n" 2>/dev/null); do kill -9 $p 2>/dev/null; done
  done
}
trap teardown EXIT

for i in $(seq 1 90); do
  timeout 3 ros2 action list 2>/dev/null | grep -q navigate_to_pose && \
  timeout 4 ros2 run tf2_ros tf2_echo map odom 2>/dev/null | grep -q Translation && \
  { echo ">>> Nav2 up (~$((i*2))s)"; break; }
  sleep 2
done
sleep 5
setsid ros2 bag record -o $LOG/nav_bag /plan /local_plan /cmd_vel /cmd_vel_nav /amcl_pose \
  /odometry/filtered /scan /map /tf /tf_static > $LOG/nav_bag.log 2>&1 &

send() {  # x y label
  echo ">>> $3 -> ($1,$2)  $(date +%T)"
  timeout 150 ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
    "{pose: {header: {frame_id: map}, pose: {position: {x: $1, y: $2}, orientation: {w: 1.0}}}}" \
    2>&1 | grep -E "status:|error_code" | tail -2
}

send  2.0  1.0  "RUN-1 point-to-point (clean map)"
sleep 2
send -1.0  7.0  "RUN-1b return to start"
sleep 2
echo ">>> dropping unmapped obstacle on the route at (0.7, 4.0)"
ros2 run ros_gz_sim create -world warehouse -name box_obs -x 0.7 -y 4.0 -z 0.6 \
  -string '<?xml version="1.0"?><sdf version="1.9"><model name="box_obs"><static>true</static><link name="l"><collision name="c"><geometry><box><size>1.2 1.2 1.2</size></box></geometry></collision><visual name="v"><geometry><box><size>1.2 1.2 1.2</size></box></geometry></visual></link></model></sdf>' \
  > $LOG/nav_spawn.log 2>&1
sleep 4
send  2.0  1.0  "RUN-2 same route, must detour around the box"

echo "=== bt_navigator / recovery outcomes ==="
grep -aE "Begin navigating|Goal succeeded|Goal failed|Running spin|Running backup|Running wait|spin completed|Passing new path" $LOG/nav_bringup.log | tail -25
for p in $(pgrep -f "ros2 bag record"); do kill -9 $p; done
echo ">>> DONE"
