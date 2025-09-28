#!/usr/bin/env python3

"""
Launch file for the quaternion-based drone interceptor node.
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
    config_file = os.path.join(pkg_dir, 'config', 'interceptor_params_quaternion.yaml')

    # Declare a launch argument to switch between simulation and real mode
    use_simulation_arg = DeclareLaunchArgument(
        'use_simulation',
        default_value='False',
        description='Set to True to run in simulation mode, False for real hardware.'
    )

    interceptor_node = Node(
        package='drone_interceptor',
        executable='interceptor_node_quaternion',
        name='interceptor_node_quaternion',
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
