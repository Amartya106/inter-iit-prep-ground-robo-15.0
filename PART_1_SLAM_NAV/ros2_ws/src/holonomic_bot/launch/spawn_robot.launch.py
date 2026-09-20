"""
Spawns holonomic_bot into a Gazebo Harmonic world that is ALREADY RUNNING
(e.g. `gz sim your_world.sdf`), starts robot_state_publisher for TF, and
starts the ros_gz_bridge so /cmd_vel, /odom, /tf, /scan and /clock all show
up on the ROS 2 side.

Usage:
    # terminal 1
    gz sim -r /path/to/your_world.sdf

    # terminal 2
    ros2 launch holonomic_bot spawn_robot.launch.py

Useful args:
    ros2 launch holonomic_bot spawn_robot.launch.py x:=1.0 y:=2.0 yaw:=1.57 rviz:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('holonomic_bot')

    xacro_file = os.path.join(pkg_share, 'urdf', 'holonomic_bot.urdf.xacro')
    sdf_file = os.path.join(pkg_share, 'models', 'holonomic_bot', 'model.sdf')
    bridge_yaml = os.path.join(pkg_share, 'config', 'bridge.yaml')
    rviz_config = os.path.join(pkg_share, 'config', 'holonomic_bot.rviz')

    robot_name = LaunchConfiguration('robot_name')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    z = LaunchConfiguration('z')
    yaw = LaunchConfiguration('yaw')
    use_sim_time = LaunchConfiguration('use_sim_time')
    start_rviz = LaunchConfiguration('rviz')

    declare_args = [
        DeclareLaunchArgument('robot_name', default_value='holonomic_bot'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
    ]

    # Lets Gazebo find the model by `model://holonomic_bot` too, not just -file
    set_resource_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(pkg_share, 'models') + ':' +
        os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    )

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]), value_type=str
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-file', sdf_file,
            '-name', robot_name,
            '-allow_renaming', 'false',
            '-x', x, '-y', y, '-z', z, '-Y', yaw,
        ],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{'config_file': bridge_yaml, 'use_sim_time': use_sim_time}],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription(
        declare_args + [
            set_resource_path,
            robot_state_publisher,
            spawn_robot,
            bridge,
            rviz,
        ]
    )
