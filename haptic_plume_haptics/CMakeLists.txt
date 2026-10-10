cmake_minimum_required(VERSION 3.8)
project(haptic_plume_haptics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(haptic_plume_drone REQUIRED)
find_package(Eigen3 REQUIRED)

# Copy necessary files to designated location
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Cue math library. One target for all three cues: they share the force budget
# constants and the composer links them as a unit.
add_library(haptic_cues SHARED
  src/cues/prf.cpp
  src/cues/well.cpp
  src/cues/heaviness.cpp
)

target_include_directories(haptic_cues PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# obstacle_field is linked, never reimplemented: the PRF and the kinematic clamp
# must not disagree about where a pipe is (D1).
target_link_libraries(haptic_cues PUBLIC
  haptic_plume_drone::obstacle_field
  Eigen3::Eigen
)

# Export the library so the Phase C composer and Phase D falcon node can link it
install(TARGETS haptic_cues
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
ament_export_dependencies(haptic_plume_drone Eigen3)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_prf test/test_prf.cpp)
  target_link_libraries(test_prf haptic_cues)
  ament_add_gtest(test_well test/test_well.cpp)
  target_link_libraries(test_well haptic_cues)
  ament_add_gtest(test_heaviness test/test_heaviness.cpp)
  target_link_libraries(test_heaviness haptic_cues)
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
