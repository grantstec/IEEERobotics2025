from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    # Declare launch arguments for flexibility
    frame_id = LaunchConfiguration('frame_id', default='lidar_link')
    
    return LaunchDescription([
        # Launch argument for frame ID
        DeclareLaunchArgument(
            'frame_id', 
            default_value='lidar_link',
            description='Frame ID for the LiDAR'
        ),
        
        # SLAM Toolbox - Foxy-compatible parameters
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
                'resolution': 0.05,  # 5cm grid resolution
                'max_laser_range': 10.0,  # A1M8's max range
                'transform_publish_period': 0.02,
                'map_update_interval': 5.0,  
                'use_scan_matching': True,
                
                # Additional debugging parameters
                'debug_logging': True,
                'throttle_scans': 1,
                'transform_timeout': 0.2,
                'tf_buffer_duration': 10.0
            }]
        ),
        
        # RPLIDAR - Use rplidar_composition for Foxy
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'serial_baudrate': 115200,
                'frame_id': frame_id,  # Use the configurable frame ID
                'angle_compensate': True,
                'scan_mode': 'Standard',
                
                # Additional parameters for troubleshooting
                'inverted': False,
                'angle_compensate_multiple': False
            }]
        )
    ])