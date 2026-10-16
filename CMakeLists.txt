cmake_minimum_required(VERSION 3.10)
project(drone_interceptor)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(quadrotor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
)

# =============================================================================
# Main ROS2 Node
# =============================================================================

# Relative IMM-UKF version: IMM(3 models)-UKF target estimator + PN/IBVS controller.
# No acados / codegen step — links only libm.
add_executable(interceptor_node_relative_imm_ukf
    src/interceptor_node_relative_imm_ukf.cpp
)

ament_target_dependencies(interceptor_node_relative_imm_ukf
    rclcpp
    geometry_msgs
    nav_msgs
    vision_msgs
    quadrotor_msgs
    sensor_msgs
    std_msgs
    mavros_msgs
)

target_link_libraries(interceptor_node_relative_imm_ukf
    m
)

install(TARGETS
  interceptor_node_relative_imm_ukf
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install parameter files
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

# =============================================================================
# Regression tests for the header-only math used by the node
# (plain main() binaries; run via ctest / colcon test)
# =============================================================================
if(BUILD_TESTING)
    enable_testing()
    add_executable(guidance_limits_test test/guidance_limits_test.cpp)
    target_link_libraries(guidance_limits_test m)
    add_test(NAME guidance_limits_test COMMAND guidance_limits_test)

    add_executable(camera_pose_history_test test/camera_pose_history_test.cpp)
    add_test(NAME camera_pose_history_test COMMAND camera_pose_history_test)
endif()

ament_package()
