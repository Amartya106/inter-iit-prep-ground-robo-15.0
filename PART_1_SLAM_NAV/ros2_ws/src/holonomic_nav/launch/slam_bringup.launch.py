"""
One command: Gazebo + robot + drive interface + slam_toolbox + RViz.

Use this to build the map. Drive with:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard \
        --ros-args -r cmd_vel:=/cmd_vel_key

Then save:
    ros2 run nav2_map_server map_saver_cli -f ~/warehouse \
        --ros-args -p save_map_timeout:=10000.0

    ros2 launch holonomic_nav slam_bringup.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_nav = get_package_share_directory("holonomic_nav")
    launch_dir = os.path.join(pkg_nav, "launch")

    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    yaw = LaunchConfiguration("yaw")

    return LaunchDescription([
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, "warehouse_sim.launch.py")),
            launch_arguments={"use_rviz": "true", "x": x, "y": y, "yaw": yaw}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, "online_slam.launch.py")),
        ),
    ])
