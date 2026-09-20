"""
One command: Gazebo + robot + drive interface + AMCL localization + Nav2 + RViz.

Requires a saved map (default: <pkg>/maps/warehouse.yaml).

    ros2 launch holonomic_nav nav_bringup.launch.py
    ros2 launch holonomic_nav nav_bringup.launch.py map:=/abs/path/warehouse.yaml

Send goals from RViz ("2D Goal Pose") or:
    ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped '{...}'
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_nav = get_package_share_directory("holonomic_nav")
    launch_dir = os.path.join(pkg_nav, "launch")
    default_map = os.path.join(pkg_nav, "maps", "warehouse.yaml")
    nav_rviz = os.path.join(pkg_nav, "rviz", "nav.rviz")

    map_yaml = LaunchConfiguration("map")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    yaw = LaunchConfiguration("yaw")
    use_collision_monitor = LaunchConfiguration("use_collision_monitor")

    return LaunchDescription([
        DeclareLaunchArgument("map", default_value=default_map),
        DeclareLaunchArgument("x", default_value="-1.0"),
        DeclareLaunchArgument("y", default_value="7.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("use_collision_monitor", default_value="false"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, "warehouse_sim.launch.py")),
            # Navigation runs on the *real* sensor stack: raw wheel odometry +
            # IMU, fused by the EKF, localised by AMCL against the saved map.
            # The ground-truth-derived odometry (use_truth_odom) is a
            # mapping-only aid -- using it here would make AMCL trivial.
            launch_arguments={"use_rviz": "false", "use_truth_odom": "false",
                              "x": x, "y": y, "yaw": yaw}.items(),
        ),
        # Give Gazebo + TF a few seconds before localization / Nav2 come up.
        TimerAction(period=6.0, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(launch_dir, "localization.launch.py")),
                launch_arguments={"map": map_yaml}.items(),
            ),
        ]),
        TimerAction(period=9.0, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(launch_dir, "navigation.launch.py")),
                launch_arguments={"use_collision_monitor": use_collision_monitor}.items(),
            ),
            Node(
                package="rviz2", executable="rviz2", name="rviz2",
                arguments=["-d", nav_rviz],
                parameters=[{"use_sim_time": True}],
                output="screen",
            ),
        ]),
    ])
