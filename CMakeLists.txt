cmake_minimum_required(VERSION 3.10)
project(drone_interceptor)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# add_compile_options(-Wno-unused-parameter -Wno-unused-variable -Wno-class-memaccess)

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

find_package(acados REQUIRED)
# get_target_property(BLASFEO_INCLUDE_DIR 
#   blasfeo INTERFACE_INCLUDE_DIRECTORIES)
# get_target_property(HPIPM_INCLUDE_DIR 
#   hpipm INTERFACE_INCLUDE_DIRECTORIES)

# =============================================================================
# ACADOS Solver Integration
# =============================================================================
# Add the path to the ACADOS-generated C code.
# You MUST run the python script `drone_model.py` or `drone_model_quaternion.py` first to generate this directory.
set(ACADOS_GENERATED_CODE_DIR "${PROJECT_SOURCE_DIR}/prototype/c_generated_code")

# Include ACADOS header files
include_directories(
    ${ACADOS_GENERATED_CODE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${BLASFEO_INCLUDE_DIR}
    ${HPIPM_INCLUDE_DIR}
)

# Add the ACADOS libraries that were generated
# add_library(acados_solver STATIC
#     ${ACADOS_GENERATED_CODE_DIR}/acados_solver_drone_interceptor_mpc.c
#     ${ACADOS_GENERATED_CODE_DIR}/acados_sim_solver_drone_interceptor_mpc.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_model/drone_interceptor_mpc_model.c
# )
# target_link_libraries(acados_solver PUBLIC ${ACADOS_LIBRARIES})
# If you are using external libraries like HPIPM, you might need to link them.
# For this example, we assume they are handled by the ACADOS installation environment.
# You may need to add find_package(BLASFEO) and find_package(HPIPM)
# and link against them: target_link_libraries(acados_solver PUBLIC blasfeo hpipm)

# =============================================================================
# Main ROS2 Nodes
# =============================================================================

# Original Euler angle version
# add_executable(interceptor_node 
#     src/interceptor_node.cpp

#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_model/drone_interceptor_mpc_expl_ode_fun.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_model/drone_interceptor_mpc_expl_vde_forw.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_model/drone_interceptor_mpc_expl_vde_adj.c

#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_cost/drone_interceptor_mpc_cost_y_0_fun.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_cost/drone_interceptor_mpc_cost_y_0_fun_jac_ut_xt.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_cost/drone_interceptor_mpc_cost_y_0_hess.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_cost/drone_interceptor_mpc_cost_y_fun.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_cost/drone_interceptor_mpc_cost_y_fun_jac_ut_xt.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_cost/drone_interceptor_mpc_cost_y_hess.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_cost/drone_interceptor_mpc_cost_y_e_fun.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_cost/drone_interceptor_mpc_cost_y_e_fun_jac_ut_xt.c
#     ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_cost/drone_interceptor_mpc_cost_y_e_hess.c

#     ${ACADOS_GENERATED_CODE_DIR}/acados_solver_drone_interceptor_mpc.c
# )

# ament_target_dependencies(interceptor_node
#     rclcpp
#     geometry_msgs
#     nav_msgs
#     vision_msgs
#     quadrotor_msgs
# )

# target_link_directories(interceptor_node PRIVATE
#     ${ACADOS_GENERATED_CODE_DIR}
# )

# # Link our nodes against the ACADOS solver library
# target_link_libraries(interceptor_node
#     # acados_solver_drone_interceptor_mpc
#     acados
#     blasfeo
#     hpipm
#     m
# )

    
# Quaternion version
add_executable(interceptor_node_quaternion
    src/interceptor_node_base.cpp
    src/interceptor_node_quaternion.cpp

    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_model/drone_interceptor_mpc_quat_expl_ode_fun.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_model/drone_interceptor_mpc_quat_expl_vde_forw.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_model/drone_interceptor_mpc_quat_expl_vde_adj.c

    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_cost/drone_interceptor_mpc_quat_cost_y_0_fun.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_cost/drone_interceptor_mpc_quat_cost_y_0_fun_jac_ut_xt.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_cost/drone_interceptor_mpc_quat_cost_y_0_hess.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_cost/drone_interceptor_mpc_quat_cost_y_fun.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_cost/drone_interceptor_mpc_quat_cost_y_fun_jac_ut_xt.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_cost/drone_interceptor_mpc_quat_cost_y_hess.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_cost/drone_interceptor_mpc_quat_cost_y_e_fun.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_cost/drone_interceptor_mpc_quat_cost_y_e_fun_jac_ut_xt.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_mpc_quat_cost/drone_interceptor_mpc_quat_cost_y_e_hess.c

    ${ACADOS_GENERATED_CODE_DIR}/acados_solver_drone_interceptor_mpc_quat.c
)

ament_target_dependencies(interceptor_node_quaternion
    rclcpp
    geometry_msgs
    nav_msgs
    vision_msgs
    quadrotor_msgs
    sensor_msgs
    std_msgs
    mavros_msgs
)

target_link_directories(interceptor_node_quaternion PRIVATE
    ${ACADOS_GENERATED_CODE_DIR}
)

target_link_libraries(interceptor_node_quaternion
    acados
    blasfeo
    hpipm
    m
)


# Dual Quaternion version (NONLINEAR_LS cost)
add_executable(interceptor_node_dq
    src/interceptor_node_base.cpp
    src/interceptor_node_dq.cpp

    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_model/drone_interceptor_dq_expl_ode_fun.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_model/drone_interceptor_dq_expl_vde_forw.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_model/drone_interceptor_dq_expl_vde_adj.c

    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_cost/drone_interceptor_dq_cost_y_0_fun.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_cost/drone_interceptor_dq_cost_y_0_fun_jac_ut_xt.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_cost/drone_interceptor_dq_cost_y_0_hess.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_cost/drone_interceptor_dq_cost_y_fun.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_cost/drone_interceptor_dq_cost_y_fun_jac_ut_xt.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_cost/drone_interceptor_dq_cost_y_hess.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_cost/drone_interceptor_dq_cost_y_e_fun.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_cost/drone_interceptor_dq_cost_y_e_fun_jac_ut_xt.c
    ${ACADOS_GENERATED_CODE_DIR}/drone_interceptor_dq_cost/drone_interceptor_dq_cost_y_e_hess.c

    ${ACADOS_GENERATED_CODE_DIR}/acados_solver_drone_interceptor_dq.c
)

ament_target_dependencies(interceptor_node_dq
    rclcpp
    geometry_msgs
    nav_msgs
    vision_msgs
    quadrotor_msgs
    sensor_msgs
    std_msgs
    mavros_msgs
)

target_link_directories(interceptor_node_dq PRIVATE
    ${ACADOS_GENERATED_CODE_DIR}
)

target_link_libraries(interceptor_node_dq
    acados
    blasfeo
    hpipm
    m
)


install(TARGETS
#   interceptor_node
  interceptor_node_quaternion
  interceptor_node_dq
  DESTINATION lib/${PROJECT_NAME} 
)

# install(FILES
#   ${ACADOS_GENERATED_CODE_DIR}/libacados_solver_drone_interceptor_mpc.so
#   DESTINATION lib/
# )

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

ament_package()
