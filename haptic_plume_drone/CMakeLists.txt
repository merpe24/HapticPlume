cmake_minimum_required(VERSION 3.8)
project(haptic_plume_drone)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(haptic_plume_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# Copy necessary files to designated location
install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# Obstacle geometry library, built seperately fron the node
add_library(obstacle_field SHARED
  src/obstacle_field.cpp
)

target_include_directories(obstacle_field PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(obstacle_field PUBLIC
  Eigen3::Eigen
  yaml-cpp::yaml-cpp
)

# Add the node executables
add_executable(drone_kinematics_node
  src/drone_kinematics_node.cpp
)

# Specify dependenices for the target
target_link_libraries(drone_kinematics_node obstacle_field)

ament_target_dependencies(drone_kinematics_node
  rclcpp
  geometry_msgs
  nav_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_geometry_msgs
  haptic_plume_interfaces
)

# Install the executables
install(TARGETS
  drone_kinematics_node
  DESTINATION lib/${PROJECT_NAME}
)

# Export the library so haptic_plume_haptics can link it
install(TARGETS obstacle_field
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 yaml-cpp)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_obstacle_field test/test_obstacle_field.cpp)
  target_link_libraries(test_obstacle_field obstacle_field)
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
