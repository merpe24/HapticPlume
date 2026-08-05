cmake_minimum_required(VERSION 3.8)
project(haptic_plume_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(gz_msgs_vendor REQUIRED)
find_package(gz-msgs REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim REQUIRED)
find_package(gz_transport_vendor REQUIRED)
find_package(gz-transport REQUIRED)

# Copy necessary files to designated location
install(
  DIRECTORY config launch worlds
  DESTINATION share/${PROJECT_NAME}
)

# Add the gz-sim system plugin
add_library(pose_mirror SHARED
  src/pose_mirror.cpp
)

target_include_directories(pose_mirror PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Specify dependencies for the targets
target_link_libraries(pose_mirror PRIVATE
  gz-msgs::core
  gz-plugin::core
  gz-sim::core
  gz-transport::core
)

# Install the plugin
install(TARGETS
  pose_mirror
  DESTINATION lib
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # The world and obstacles.yaml are two hand-maintained copies of one geometry
  ament_add_pytest_test(test_world_maches_obstacles
    test/test_world_matches_obstacles.py
  )
endif()

ament_package()
