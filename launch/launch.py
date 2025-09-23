import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Get package directory
    package_dir = get_package_share_directory('drone_interceptor')
    
    # Path to parameter file
    params_file = os.path.join(package_dir, 'config', 'interceptor_params.yaml')
    
    # Declare a launch argument to switch between simulation and real mode
    use_simulation_arg = DeclareLaunchArgument(
        'use_simulation',
        default_value='False',
        description='Set to True to run in simulation mode, False for real hardware.'
    )

    # Define the main interceptor node
    interceptor_node = Node(
        package='drone_interceptor',
        executable='interceptor_node',
        name='interceptor_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            params_file,
            {'use_simulation': LaunchConfiguration('use_simulation')}
        ],
        # remappings=[
        #     ('/fpv/odom', '/mavros/local_position/odom'),
        # ]
    )

    return LaunchDescription([
        use_simulation_arg,
        interceptor_node
    ])
