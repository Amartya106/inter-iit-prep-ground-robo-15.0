cmake_minimum_required(VERSION 3.8)
project(holonomic_nav)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  config
  worlds
  rviz
  maps
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/priority_mux.py
  scripts/explore_drive.py
  scripts/pose_error_logger.py
  scripts/odom_correction_node.py
  scripts/calibrate_odom.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
