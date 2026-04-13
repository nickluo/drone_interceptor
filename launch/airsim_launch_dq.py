#!/usr/bin/env python3

"""
Launch file for the dual-quaternion-based drone interceptor node (AirSim).
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_dir = get_package_share_directory('drone_interceptor')
    config_file = os.path.join(pkg_dir, 'config', 'airsim_params_dq.yaml')

    use_simulation_arg = DeclareLaunchArgument(
        'use_simulation',
        default_value='False',
        description='Set to True to run in simulation mode, False for real hardware.'
    )

    interceptor_node = Node(
        package='drone_interceptor',
        executable='interceptor_node_dq',
        name='interceptor_node_dq',
        output='screen',
        parameters=[
            config_file,
            {'use_simulation': LaunchConfiguration('use_simulation')}
        ],
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'INFO']
    )
    
    return LaunchDescription([
        use_simulation_arg,
        interceptor_node
    ])
