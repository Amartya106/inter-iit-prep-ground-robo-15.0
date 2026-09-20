"""
Nav2 stack for the holonomic base: planner (Smac 2D), controller (MPPI/Omni),
smoother, behaviors (spin/backup/wait/drive_on_heading/assisted_teleop),
BT navigator, waypoint follower, velocity_smoother.

cmd_vel chain:
    controller_server  --(remap cmd_vel -> cmd_vel_raw)-->
    velocity_smoother   (in: cmd_vel_raw, out: cmd_vel_nav) -->
    twist_mux           (in: cmd_vel_nav (+ teleop, higher priority), out: cmd_vel)
    -> ros_gz bridge -> gz mecanum-drive-system

velocity_smoother enforces the acceleration / jerk profile; twist_mux keeps
manual teleop able to pre-empt the autonomous stack at any moment.

An optional nav2_collision_monitor layer (approach-braking on raw /scan) can be
enabled with use_collision_monitor:=true; it is wired in as its own lifecycle
group so the default path stays simple.

Run after warehouse_sim.launch.py + localization.launch.py, or use
nav_bringup.launch.py.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_nav = get_package_share_directory("holonomic_nav")
    params_file = os.path.join(pkg_nav, "config", "nav2_params.yaml")

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_collision_monitor = LaunchConfiguration("use_collision_monitor")

    common = [params_file, {"use_sim_time": use_sim_time}]

    core_lifecycle_nodes = [
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
        "velocity_smoother",
    ]

    core_nodes = [
        Node(
            package="nav2_controller", executable="controller_server", name="controller_server",
            output="screen", parameters=common,
            remappings=[("cmd_vel", "cmd_vel_raw")],
        ),
        Node(
            package="nav2_smoother", executable="smoother_server", name="smoother_server",
            output="screen", parameters=common,
        ),
        Node(
            package="nav2_planner", executable="planner_server", name="planner_server",
            output="screen", parameters=common,
        ),
        Node(
            package="nav2_behaviors", executable="behavior_server", name="behavior_server",
            output="screen", parameters=common,
        ),
        Node(
            package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator",
            output="screen", parameters=common,
        ),
        Node(
            package="nav2_waypoint_follower", executable="waypoint_follower",
            name="waypoint_follower", output="screen", parameters=common,
        ),
        Node(
            package="nav2_velocity_smoother", executable="velocity_smoother",
            name="velocity_smoother", output="screen", parameters=common,
            remappings=[("cmd_vel", "cmd_vel_raw"), ("cmd_vel_smoothed", "cmd_vel_nav")],
        ),
        Node(
            package="nav2_lifecycle_manager", executable="lifecycle_manager",
            name="lifecycle_manager_navigation", output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": core_lifecycle_nodes,
            }],
        ),
    ]

    collision_monitor_nodes = [
        Node(
            package="nav2_collision_monitor", executable="collision_monitor",
            name="collision_monitor", output="screen", parameters=common,
            condition=IfCondition(use_collision_monitor),
        ),
        Node(
            package="nav2_lifecycle_manager", executable="lifecycle_manager",
            name="lifecycle_manager_collision_monitor", output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": ["collision_monitor"],
            }],
            condition=IfCondition(use_collision_monitor),
        ),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("use_collision_monitor", default_value="false"),
        ]
        + core_nodes
        + collision_monitor_nodes
    )
