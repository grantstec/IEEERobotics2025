import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    urdf_file = os.path.join(
        get_package_share_directory('ieee_robotics'),
        'urdf',
        'robot.urdf'
    )




    return LaunchDescription([
        # Start the robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': ParameterValue(Command(['xacro ', urdf_file]), value_type=str)
            }]
        ),

        Node(
            package='ieee_robotics',
            executable='arduino_bridge',
            name='arduino_bridge',
            parameters=[{'serial_port': '/dev/ttyACM1'}]
        ),
        Node(
            package='ieee_robotics',
            executable='teensy_bridge',
            name='teensy_bridge',
            parameters=[{'serial_port': '/dev/ttyACM0'}]
        ),
        Node(
            package='ieee_robotics',
            executable='wheel_odometry',
            name='wheel_odometry'
        ),
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter',
            output='screen',
            parameters=[os.path.join(
                get_package_share_directory('ieee_robotics'),
                'config',
                'ekf.yaml'
            )]
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

    ])