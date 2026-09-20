"""
Bring up Gazebo Harmonic + the holonomic_bot base + the shared drive interface.

Starts, in order:
  * GZ_SIM_RESOURCE_PATH pointing at the holonomic_bot models dir + Fuel cache
  * gz sim  (warehouse world)
  * robot_state_publisher   (holonomic_bot URDF -> static TF: base_link -> lidar_link/imu_link/wheels)
  * ros_gz_sim create       (spawns models/holonomic_bot/model.sdf into the world)
  * ros_gz_bridge           (/cmd_vel, /odom/wheel, /scan, /imu, /joint_states, /clock)
  * robot_localization EKF  (publishes odom -> base_link, /odometry/filtered)
  * twist_mux               (teleop + Nav2 -> /cmd_vel)
  * RViz                    (optional)

This is the base layer. SLAM and Nav2 are layered on top via their own launch
files (or the *_bringup.launch.py convenience wrappers).

    ros2 launch holonomic_nav warehouse_sim.launch.py
    ros2 launch holonomic_nav warehouse_sim.launch.py x:=2.0 yaw:=1.57 use_rviz:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

try:
    from ament_index_python.packages import get_package_prefix

    _HAVE_TWIST_MUX = True
    try:
        get_package_prefix("twist_mux")
    except Exception:
        _HAVE_TWIST_MUX = False
except Exception:
    _HAVE_TWIST_MUX = False


def generate_launch_description():
    pkg_nav = get_package_share_directory("holonomic_nav")
    pkg_bot = get_package_share_directory("holonomic_bot")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

    xacro_file = os.path.join(pkg_bot, "urdf", "holonomic_bot.urdf.xacro")
    sdf_file = os.path.join(pkg_bot, "models", "holonomic_bot", "model.sdf")
    bridge_yaml = os.path.join(pkg_bot, "config", "bridge.yaml")
    truth_bridge_yaml = os.path.join(pkg_nav, "config", "truth_bridge.yaml")
    ekf_yaml = os.path.join(pkg_nav, "config", "ekf.yaml")
    twist_mux_yaml = os.path.join(pkg_nav, "config", "twist_mux.yaml")
    rviz_cfg = os.path.join(pkg_nav, "rviz", "slam.rviz")

    # The gz mecanum-drive-system odometry on this robot is unusable for SLAM
    # (friction-trick wheels -> 2-3x velocity-dependent scale error, yaw sign
    # flips; no constant scale fixes it -- see odom_correction_node.py). We
    # therefore derive odometry from the simulator's true model pose
    # (/gt/dynamic_pose, a SceneBroadcaster feature -- model.sdf untouched),
    # which behaves like a well-calibrated real wheel/​mocap odometry. SLAM
    # Toolbox still does scan matching, loop closure and pose-graph
    # optimisation on top of this prior.  Toggle with use_truth_odom:=false.

    world = LaunchConfiguration("world")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    yaw = LaunchConfiguration("yaw")
    use_rviz = LaunchConfiguration("use_rviz")
    headless = LaunchConfiguration("headless")
    use_sim_time = LaunchConfiguration("use_sim_time")

    declared = [
        DeclareLaunchArgument(
            "world",
            default_value=os.path.join(pkg_nav, "worlds", "warehouse.sdf"),
            description="Absolute path to the .sdf world file.",
        ),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="0.05"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument(
            "headless", default_value="false",
            description="Run gz sim server-only (no GUI). Use on machines with no display.",
        ),
        DeclareLaunchArgument(
            "use_truth_odom", default_value="true",
            description="Derive odometry from the sim's true pose (the mecanum "
                        "plugin's odometry is unusable). Set false to fall back "
                        "to raw /odom/wheel.",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
    ]
    use_truth_odom = LaunchConfiguration("use_truth_odom")

    # Let gz resolve model://holonomic_bot and the cached Fuel models.
    set_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.path.join(pkg_bot, "models")
        + os.pathsep
        + os.path.expanduser("~/.gz/fuel/fuel.gazebosim.org")
        + os.pathsep
        + os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
    )

    gz_launch = os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        launch_arguments={"gz_args": ["-r -v3 ", world]}.items(),
        condition=UnlessCondition(headless),
    )
    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        launch_arguments={"gz_args": ["-r -s -v3 ", world]}.items(),
        condition=IfCondition(headless),
    )

    robot_description = ParameterValue(Command(["xacro ", xacro_file]), value_type=str)

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": use_sim_time}],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-file", sdf_file,
            "-name", "holonomic_bot",
            "-allow_renaming", "false",
            "-x", x, "-y", y, "-z", z, "-Y", yaw,
        ],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[{"config_file": bridge_yaml, "use_sim_time": use_sim_time}],
    )

    # bridges /world/warehouse/dynamic_pose/info -> /gt/dynamic_pose (true pose)
    truth_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="truth_bridge",
        output="screen",
        parameters=[{"config_file": truth_bridge_yaml, "use_sim_time": use_sim_time}],
    )

    odom_correct = Node(
        package="holonomic_nav",
        executable="odom_correction_node.py",
        name="odom_correction_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "use_ground_truth": use_truth_odom}],
    )

    ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_yaml],
    )

    # Shared drive interface: prefer the real twist_mux when it is installed,
    # otherwise fall back to the bundled priority_mux.py (no apt dependency).
    if _HAVE_TWIST_MUX:
        mux = Node(
            package="twist_mux",
            executable="twist_mux",
            output="screen",
            parameters=[twist_mux_yaml],
            remappings=[("cmd_vel_out", "cmd_vel")],
        )
    else:
        mux = Node(
            package="holonomic_nav",
            executable="priority_mux.py",
            name="priority_mux",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
        )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_cfg],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        declared
        + [set_resource_path, gz_sim_gui, gz_sim_headless, rsp, spawn, bridge,
           truth_bridge, odom_correct, ekf, mux, rviz]
    )
