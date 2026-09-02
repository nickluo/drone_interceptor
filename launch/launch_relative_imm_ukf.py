#!/usr/bin/env python3

"""
Launch file for the relative IMM-UKF based drone interceptor node.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('drone_interceptor')
    
    # Path to the parameters file
    config_file = os.path.join(pkg_dir, 'config', 'relative_imm_ukf_params.yaml')

    # Declare a launch argument to switch between simulation and real mode
    use_simulation_arg = DeclareLaunchArgument(
        'use_simulation',
        default_value='False',
        description='Set to True to run in simulation mode, False for real hardware.'
    )

    # Radar relay mid-course guidance (see the radar block in relative_imm_ukf_params.yaml)
    use_radar_arg = DeclareLaunchArgument(
        'use_radar',
        default_value='False',
        description='Set to True to consume the radar relay feed for mid-course guidance.'
    )

    # Opt-in re-intercept switch (defaults to the yaml value, false). Exposed as a launch arg so a
    # test/harness can enable it without editing the yaml: radar_reintercept_enable:=true.
    reintercept_arg = DeclareLaunchArgument(
        'radar_reintercept_enable',
        default_value='False',
        description='Override radar.reintercept_enable (re-engage radar after a terminal vision-loss miss).'
    )

    interceptor_node = Node(
        package='drone_interceptor',
        executable='interceptor_node_relative_imm_ukf',
        name='interceptor_node_relative_imm_ukf',
        output='screen',
        parameters=[
            config_file,
            {'use_simulation': LaunchConfiguration('use_simulation')},
            {'radar.enable': LaunchConfiguration('use_radar')},
            {'radar.reintercept_enable': LaunchConfiguration('radar_reintercept_enable')}
        ],
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'INFO']
    )

    return LaunchDescription([
        use_simulation_arg,
        reintercept_arg,
        use_radar_arg,
        interceptor_node
    ])
