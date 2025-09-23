#!/usr/bin/env python3

"""
Launch file for the quaternion-based drone interceptor node.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('drone_interceptor')
    
    # Path to the parameters file
    config_file = os.path.join(pkg_dir, 'config', 'interceptor_params_quaternion.yaml')
    
    return LaunchDescription([
        Node(
            package='drone_interceptor',
            executable='interceptor_node_quaternion',
            name='interceptor_node_quaternion',
            output='screen',
            parameters=[config_file],
            arguments=['--ros-args', '--log-level', 'INFO']
        )
    ])
