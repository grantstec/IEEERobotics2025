#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import TimerAction
from pathlib import Path


def generate_launch_description():
    ieee_robotics_share = get_package_share_directory('ieee_robotics')
    
    # Use your existing URDF
    urdf_file = os.path.join(
        ieee_robotics_share,
        'urdf',
        'robot.urdf'
    )
    
    # Use a modified EKF configuration
    ekf_debug_file = os.path.join(
        ieee_robotics_share,
        'config',
        'ekf_debug.yaml'
    )
    
    return LaunchDescription([
        # 1. Setup TF tree first with higher publish rate
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': open(urdf_file, 'r').read(),
                'publish_frequency': 50.0,  # Higher frequency for better TF
                'ignore_timestamp': True,
                'use_tf_static': True
            }]
        ),
        
        # 2. Hardware interfaces
        Node(
            package='ieee_robotics',
            executable='arduino_bridge',
            name='arduino_bridge',
            parameters=[{'port': '/dev/ttyACM1'}]
        ),
        
        Node(
            package='ieee_robotics',
            executable='wheel_odometry',
            name='wheel_odometry'
        ),

        Node(
            package='ieee_robotics',
            executable='twist_to_motors',
            name='twist_to_motors',
        ),
        
        # 3. Fixed LIDAR setup with explicit Boost mode
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',  # Foxy uses this executable
            name='rplidar',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'serial_baudrate': 115200,
                'frame_id': 'lidar_link',  # Must match the URDF
                'angle_compensate': True,
                'scan_mode': 'Boost'    # Force to Boost mode   
            }]
        ),
        
        # 4. Simplified SLAM with more debugging output
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
                'resolution': 0.025,
                'max_laser_range': 8.0,
                'transform_publish_period': 0.05,  # Faster transform publishing
                'tf_buffer_duration': 30.0,        # Longer TF buffer
                'debug_logging': True,
                'throttle_scans': 1                # Process every scan
            }]
        ),
        
        # 5. Hardware Switch Node
        Node(
            package='ieee_robotics',
            executable='hardware_switch',
            name='hardware_switch',
            output='screen',
            parameters=[{
                'gpio_pin': 12,           # Using pin 12 (GPIO79)
                'active_high': True,      # Switch is active high
                'debounce_time': 0.05,    # 50ms debounce time
                'publish_rate': 10.0      # Publish at 10Hz
            }]
        ),
        
        # 6. Teensy IMU Bridge (replaces original teensy_bridge)
        Node(
            package='ieee_robotics',
            executable='teensy_imu_bridge',
            name='teensy_imu_bridge',
            output='screen',
            parameters=[{
                'port': '/dev/ttyACM0',
                'baudrate': 115200,
                'timeout': 1.0
            }]
        ),
        
        # 7. Fire Detection State Machine
        Node(
            package='ieee_robotics',
            executable='fire_detection_state_machine',
            name='fire_detection_state_machine',
            output='screen',
            parameters=[{
                'goal_active': False  # Will be controlled by teensy_bridge_adapter
            }]
        ),
        
        # 8. Competition Round Management System
        Node(
            package='ieee_robotics',
            executable='round_state_manager',
            name='round_state_manager',
            output='screen',
            parameters=[{
                'initial_round': 1,
                'enable_manual_control': True,
                'data_dir': str(Path.home() / 'competition_data')
            }]
        ),
        
        # 9. Teensy bridge adapter
        Node(
            package='ieee_robotics',
            executable='teensy_bridge_adapter',
            name='teensy_bridge_adapter',
            output='screen'
        ),
        
        # 10. Navigation strategy
        Node(
            package='ieee_robotics',
            executable='competition_navigation_strategy',
            name='competition_navigation_strategy',
            output='screen',
            parameters=[{
                'goal_reached_distance': 0.05,
                'auto_return_to_start': True,
                'auto_end_round': True
            }]
        )
    ])