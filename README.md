## IEEE r5 robotics comp 2025

### Current Status
- Configuring navigation stack parameters for optimal SLAM and navigation at same time. 
- Compute porblems with rpi5, moving to Jetson Xavier currently for more compute power. 
- Teensy IMU -> RPI/Jetson ROS :white_check_mark:
- Teensy IR -> RPI/Jetson ROS :white_check_mark:
- RPI/Jetson ROS -> Arduino Mega Steppers :white_check_mark:
- Arduino Mega Steppers -> RPI/Jetson ROS :white_check_mark:

### Implementaitons
- ros2 navigation stack
- rplidar a1m8
- teensy with bno055 imu and ir camera mlx90640 processing
- custom arduino mega with stepper motor setup
- SLAM
- Robot localization ekf

### Old picture of robot (not updated with new jetson xavier)

![IMG_5884](https://github.com/user-attachments/assets/1aaf2bca-31b3-4c13-8cd2-39206f177370)




