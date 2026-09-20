import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_holonomic = get_package_share_directory('holonomic')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(pkg_holonomic, 'worlds', 'warehouse.sdf')
    bridge_config = os.path.join(pkg_holonomic, 'config', 'bridge_harmonic.yaml')


    resource_path_parent = os.path.dirname(pkg_holonomic)
    fuel_model_root = os.path.expanduser('~/.gz/fuel/fuel.gazebosim.org/openrobotics')
    fuel_model_path = os.path.join(fuel_model_root, 'models')
    fuel_world_path = os.path.join(fuel_model_root, 'worlds')
    existing_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    resource_entries = [
        p for p in [
            resource_path_parent,
            fuel_model_path,
            fuel_world_path,
            existing_resource_path,
        ] if p
    ]
    gz_resource_path = os.pathsep.join(dict.fromkeys(resource_entries))
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=gz_resource_path
    )
    
    # robot_state_publisher (publishes /robot_description + TF for the static links)
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(pkg_holonomic, 'launch', 'rsp_harmonic.launch.py')]
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Gazebo Harmonic server + GUI, loading the warehouse world
    # (the Warehouse Fuel model is fetched/cached automatically on first run)
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items()
    )

    # ROS2 <-> Gazebo Harmonic topic bridge (cmd_vel components, odom, tf, scan, imu, clock)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
        output='screen'
    )

    # Splits /cmd_vel into the three virtual-joint velocity commands
    # cmd_vel_decompose = Node(
    #     package='holonomic',
    #     executable='cmd_vel_decompose.py',
    #     output='screen'
    # )

    return LaunchDescription([
        set_gz_resource_path,
        rsp,
        gz_sim,
        bridge,
        # cmd_vel_decompose,
    ])
