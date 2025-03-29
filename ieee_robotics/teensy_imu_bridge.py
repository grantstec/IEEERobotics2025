#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
from std_msgs.msg import String
import serial
import time

class TeensyIMUBridge(Node):
    """
    Node to handle the Teensy serial connection and process IMU data.
    This is a simplified version of the original teensy_bridge.py that
    only handles IMU data and fire detection signals.
    """
    
    def __init__(self):
        super().__init__('teensy_imu_bridge')
        
        # Parameters
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('timeout', 1.0)
        
        # Publishers
        self.imu_pub = self.create_publisher(Imu, 'imu/data_raw', 10)
        self.fire_detection_pub = self.create_publisher(String, 'teensy/fire_detection', 10)
        
        # Serial connection
        self.serial = None
        self.connect_serial()
        
        # Timer for processing serial data
        self.create_timer(0.01, self.process_serial)
        
        self.get_logger().info("Teensy IMU Bridge initialized")

    def connect_serial(self):
        """Connect to the Teensy serial port"""
        try:
            self.serial = serial.Serial(
                port=self.get_parameter('port').value,
                baudrate=self.get_parameter('baudrate').value,
                timeout=self.get_parameter('timeout').value
            )
            self.get_logger().info(f"Connected to Teensy on {self.get_parameter('port').value}")
            return True
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to connect to Teensy: {str(e)}")
            self.serial = None
            return False

    def process_serial(self):
        """Process serial data from the Teensy"""
        # If not connected, try to reconnect
        if self.serial is None:
            if not self.connect_serial():
                # If reconnection failed, wait and try again next time
                return
        
        try:
            # Check if there is data waiting
            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('utf-8').strip()
                
                if line.startswith('IMU:'):
                    self.process_imu_data(line)
                elif line.startswith('FIRE:'):
                    self.process_fire_data(line)
        except Exception as e:
            self.get_logger().error(f"Error processing serial data: {str(e)}")
            # Close the connection and try to reconnect next time
            try:
                self.serial.close()
            except:
                pass
            self.serial = None

    def process_imu_data(self, line):
        """Process IMU data from the Teensy"""
        try:
            _, data = line.split(':')
            values = list(map(float, data.split(',')))
            
            if len(values) != 10:
                self.get_logger().warn(f"Invalid IMU data format: expected 10 values, got {len(values)}")
                return
            
            # Create and publish IMU message
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'imu_link'
            
            # Set orientation quaternion [w, x, y, z]
            # To this:
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = values[:4]
            
            # Set angular velocity [x, y, z]
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = values[4:7]
            
            # Set linear acceleration [x, y, z]
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = values[7:]
            
            # Publish
            self.imu_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error parsing IMU data: {str(e)}")

    def process_fire_data(self, line):
        """Process fire detection data from the Teensy"""
        # Just forward the raw fire detection string
        msg = String()
        msg.data = line
        self.fire_detection_pub.publish(msg)
        self.get_logger().debug(f"Published fire detection data: {line}")

def main(args=None):
    rclpy.init(args=args)
    node = TeensyIMUBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.serial is not None:
            node.serial.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()