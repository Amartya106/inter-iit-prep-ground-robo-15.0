cmake_minimum_required(VERSION 3.8)
project(holonomic_bot)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  urdf
  models
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
