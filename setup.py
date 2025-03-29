from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ieee_robotics'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/urdf', glob('urdf/*.urdf')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/scripts', glob('scripts/*.sh')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='IEEE Firefighting Robot Package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'fire_detection_node = ieee_robotics.fire_detection_node:main',
            'arduino_bridge = ieee_robotics.arduino_bridge:main',
            'teensy_bridge = ieee_robotics.teensy_bridge:main',
            'twist_to_motors = ieee_robotics.twist_to_motors:main',
            'wheel_odometry = ieee_robotics.wheel_odometry:main',
            'gpu_image_processor = ieee_robotics.gpu_image_processor:main',
            'system_monitor = ieee_robotics.system_monitor:main',
            # Add missing executables:
            'hardware_switch = ieee_robotics.hardwareswitch:main',
            'teensy_imu_bridge = ieee_robotics.teensy_imu_bridge:main',
            'fire_detection_state_machine = ieee_robotics.fire_detection_state_machine:main',
            'round_state_manager = ieee_robotics.round_tracker:main',
            'teensy_bridge_adapter = ieee_robotics.teensy_bridge_adapter:main',
            'competition_navigation_strategy = ieee_robotics.navigation_strategy:main',
        ],
    },
)