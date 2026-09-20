"""
slam_toolbox in async online mapping mode.

In ROS 2 Jazzy `async_slam_toolbox_node` is a *lifecycle* node -- it must be
explicitly configured + activated or it silently never subscribes to /scan.
Rather than re-implement that dance, this wraps slam_toolbox's own
`online_async_launch.py` (which handles the lifecycle transitions) and just
points it at our tuned params file.

Run this AFTER warehouse_sim.launch.py (robot spawned, /scan + TF live), then
drive a loop with explore_drive.py or teleop:

    ros2 launch holonomic_nav online_slam.launch.py
    ros2 run holonomic_nav explore_drive.py --ros-args -p duration:=180.0

Save the finished map with:

    ros2 run nav2_map_server map_saver_cli -f <pkg>/maps/warehouse \
        --ros-args -p save_map_timeout:=20.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_nav = get_package_share_directory("holonomic_nav")
    pkg_slam = get_package_share_directory("slam_toolbox")
    default_params = os.path.join(pkg_nav, "config", "slam_toolbox.yaml")

    slam_params_file = LaunchConfiguration("slam_params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument("slam_params_file", default_value=default_params),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        # slam_toolbox's Ceres backend otherwise asks for ~50 threads on this
        # 22-core box (logged: "num_threads: 50 exceeds maximum available"),
        # which oversubscribes the CPU and stalls map updates during the run.
        SetEnvironmentVariable("OMP_NUM_THREADS", "4"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_slam, "launch", "online_async_launch.py")
            ),
            launch_arguments={
                "slam_params_file": slam_params_file,
                "use_sim_time": use_sim_time,
            }.items(),
        ),
    ])
