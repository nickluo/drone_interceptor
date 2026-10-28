#!/usr/bin/env python3

"""
Launch file for the relative IMM-UKF based drone interceptor node.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _parse_extra_params(raw):
    """Parse "k=v,k=v" into a node parameter dict. Values are inferred: true/false -> bool,
    float syntax -> double, otherwise the raw string."""
    out = {}
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise RuntimeError(f"extra_params entry '{item}' is not key=value")
        k, v = item.split('=', 1)
        k, v = k.strip().strip("'\""), v.strip().strip("'\"")
        if v.lower() in ('true', 'false'):
            out[k] = (v.lower() == 'true')
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _launch_setup(context):
    extra_raw = LaunchConfiguration('extra_params').perform(context).strip()
    extra = _parse_extra_params(extra_raw) if extra_raw else {}

    interceptor_node = Node(
        package='drone_interceptor',
        executable='interceptor_node_relative_imm_ukf',
        name='interceptor_node_relative_imm_ukf',
        output='screen',
        parameters=[
            LaunchConfiguration('parameters'),
            {'use_simulation': LaunchConfiguration('use_simulation')},
            {'radar.enable': LaunchConfiguration('use_radar')},
            {'radar.reintercept_enable': LaunchConfiguration('radar_reintercept_enable')},
        ] + ([extra] if extra else []),
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'INFO']
    )
    return [interceptor_node]


def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('drone_interceptor')

    # Path to the parameters file
    config_file = os.path.join(pkg_dir, 'config', 'relative_imm_ukf_params.yaml')
    parameters_arg = DeclareLaunchArgument(
        'parameters',
        default_value=config_file,
        description='Relative IMM-UKF parameters file (defaults to the packaged profile).'
    )

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

    # Arbitrary node parameter overrides ("k=v,k=v"; float/bool/string inferred) applied AFTER
    # the yaml file — used by run_scenario.sh to raise max_tilt to 95 deg for the dive/bench-rig
    # scenarios of the quaternion attitude layer.
    extra_params_arg = DeclareLaunchArgument(
        'extra_params',
        default_value='',
        description='Comma-separated node parameter overrides k=v (float/bool/string), '
                    'e.g. "controller.max_tilt_deg=95.0,controller.max_tilt_terminal_deg=95.0"')

    reintercept_arg = DeclareLaunchArgument(
        'radar_reintercept_enable',
        default_value='False',
        description='Override radar.reintercept_enable (re-engage radar after a terminal vision-loss miss).'
    )

    return LaunchDescription([
        parameters_arg,
        use_simulation_arg,
        reintercept_arg,
        use_radar_arg,
        extra_params_arg,
        OpaqueFunction(function=_launch_setup)
    ])
