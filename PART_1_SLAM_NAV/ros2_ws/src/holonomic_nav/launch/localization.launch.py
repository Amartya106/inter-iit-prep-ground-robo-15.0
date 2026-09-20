"""
Map server + AMCL (Omni motion model) + lifecycle manager.

Localizes the robot against a previously saved map. Run after
warehouse_sim.launch.py and before navigation.launch.py, or use
nav_bringup.launch.py which wires all three together.

    ros2 launch holonomic_nav localization.launch.py map:=/abs/path/to/warehouse.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_nav = get_package_share_directory("holonomic_nav")
    params_file = os.path.join(pkg_nav, "config", "nav2_params.yaml")
    default_map = os.path.join(pkg_nav, "maps", "warehouse.yaml")

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("map", default_value=default_map),
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[params_file, {"use_sim_time": use_sim_time, "yaml_filename": map_yaml}],
        ),
        Node(
            package="nav2_amcl",
            executable="amcl",
            name="amcl",
            output="screen",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_localization",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": ["map_server", "amcl"],
            }],
        ),
    ])
