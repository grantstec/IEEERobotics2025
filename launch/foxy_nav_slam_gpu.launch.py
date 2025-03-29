import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package directories
    ieee_robotics_share = get_package_share_directory('ieee_robotics')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')
    
    # URDF file
    urdf_file = os.path.join(ieee_robotics_share, 'urdf', 'robot.urdf')
    
    # Config files
    ekf_config_file = os.path.join(ieee_robotics_share, 'config', 'ekf_debug.yaml')
    
    # Use Foxy-specific Nav2 parameters file
    nav2_params_file = os.path.join(ieee_robotics_share, 'config', 'foxy_nav2_params.yaml')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    autostart = LaunchConfiguration('autostart', default='true')
    use_gpu = LaunchConfiguration('use_gpu', default='false')  # Start with GPU disabled for testing
    
    return LaunchDescription([
        # Launch Arguments
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('use_gpu', default_value='false'),
        
        # Xavier NX Performance Mode
        ExecuteProcess(
            cmd=['bash', '-c', 'sudo nvpmodel -m 0 && sudo jetson_clocks'],
            output='screen'
        ),
        
        # Core Robot Components with improved TF settings
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': ParameterValue(Command(['xacro ', urdf_file]), value_type=str),
                'publish_frequency': 50.0,     # Higher frequency for TF
                'ignore_timestamp': True,      # Improve TF stability
                'use_tf_static': True          # Ensure static transforms are reliable
            }]
        ),
        
        # Hardware Interface Nodes
        Node(
            package='ieee_robotics',
            executable='arduino_bridge',
            name='arduino_bridge',
            parameters=[{'port': '/dev/ttyACM1'}]
        ),
        
        Node(
            package='ieee_robotics',
            executable='teensy_bridge',
            name='teensy_bridge',
            parameters=[{'port': '/dev/ttyACM0'}]
        ),
        
        Node(
            package='ieee_robotics',
            executable='wheel_odometry',
            name='wheel_odometry'
        ),
        
        # EKF with explicit rate
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter',
            output='screen',
            parameters=[ekf_config_file]
        ),
        
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher'
        ),
        
        Node(
            package='ieee_robotics',
            executable='twist_to_motors',
            name='twist_to_motors'
        ),
        
        # SLAM - RPLidar with explicit settings matching minimal_nav
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',  # Keep this for Foxy
            name='rplidar',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'serial_baudrate': 115200,
                'frame_id': 'lidar_link',      # Match the URDF frame exactly
                'angle_compensate': True,
                'scan_mode': 'Boost',
                'scan_frequency': 10.0         # Explicit frequency setting
            }]
        ),
        
        # SLAM Toolbox with improved TF settings
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'base_frame': 'base_link',
                'odom_frame': 'odom',
                'map_frame': 'map',
                'resolution': 0.05,
                'max_laser_range': 8.0,        # Reduced from 10.0 to match minimal_nav
                'transform_publish_period': 0.05,  # Increased from 0.02
                'tf_buffer_duration': 30.0,    # Longer TF buffer like minimal_nav
                'map_update_interval': 5.0,    # Reduced update frequency
                'threads': 4,                  # Keep multithreading but limit threads
                'use_multithread': True,
                'debug_logging': True,         # Enable for troubleshooting
                'throttle_scans': 1,           # Process every scan
                'ceres_linear_solver': 'SPARSE_NORMAL_CHOLESKY',
                'ceres_preconditioner': 'SCHUR_JACOBI',
                'solver_plugin': 'solver_plugins::CeresSolver'
            }]
        ),
        
        # GPU-Accelerated Processing (disabled initially)
        Node(
            package='ieee_robotics',
            executable='gpu_image_processor',
            name='gpu_image_processor',
            parameters=[{'use_gpu': use_gpu}],
            output='screen'
        ),
        
        # Navigation - with longer delay to ensure SLAM is properly initialized
        TimerAction(
            period=10.0,  # Increased from 5.0 to 10.0 seconds
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(os.path.join(
                        nav2_bringup_share, 'launch', 'navigation_launch.py'
                    )),
                    launch_arguments={
                        'use_sim_time': use_sim_time,
                        'params_file': nav2_params_file,
                        'autostart': autostart,
                        # Foxy-specific options
                        'use_lifecycle_mgr': 'true',
                        'map_subscribe_transient_local': 'true'
                    }.items()
                ),
            ]
        ),
        
        # RViz visualization - with longer delay after Navigation is started
        TimerAction(
            period=15.0,  # Increased from 7.0 to 15.0 seconds
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=[
                        '-d', os.path.join(nav2_bringup_share, 'rviz', 'nav2_default_view.rviz'),
                        '--enable-ogre-thread'  # Keep GPU acceleration for RViz
                    ],
                    output='screen'
                ),
            ]
        ),
    ])