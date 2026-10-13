cmake_minimum_required(VERSION 3.0.2)
project(rio_ros)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS scripts/rio_gp_node.py scripts/rio_spline_node.py
  scripts/rio_runtime.py scripts/rio_frontend.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
