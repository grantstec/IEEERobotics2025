#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import serial
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import time

class VelocityArduinoBridge(Node):
    def __init__(self):
        super().__init__('velocity_arduino_bridge')
        
        # Parameters
        self.declare_parameter('port', '/dev/ttyACM1')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('timeout', 1.0)
        self.declare_parameter('debug_output', False)  # Control verbosity
        
        # Steps per revolution and wheel radius for accurate math
        self.declare_parameter('microsteps', 16)
        self.declare_parameter('steps_per_rev', 200)
        self.declare_parameter('wheel_radius', 0.045)
        
        # Get parameter values
        self.microsteps = self.get_parameter('microsteps').value
        self.steps_per_rev = self.get_parameter('steps_per_rev').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.debug_output = self.get_parameter('debug_output').value
        
        # Calculate steps per radian
        self.steps_per_rad = (self.steps_per_rev * self.microsteps) / (2 * math.pi)
        
        # Only log configuration once at startup
        self.get_logger().info(f"Motor config: {self.steps_per_rev} steps/rev, {self.microsteps} microsteps, radius: {self.wheel_radius}m")
        
        # QoS profile 
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        
        # Publishers/Subscribers
        self.step_pub = self.create_publisher(JointState, 'wheel_steps', qos)
        self.wheel_cmd_sub = self.create_subscription(Twist, 'wheel_commands', self.wheel_cmd_callback, qos)
        
        # Serial connection
        self.serial = None
        self.connect_serial()
        self.read_timer = self.create_timer(0.01, self.read_serial)
        
        # Command tracking - keep track of previous values to avoid duplicates
        self.last_left_speed = 0.0
        self.last_right_speed = 0.0
        self.prev_left_speed = 0.0  # For detecting changes
        self.prev_right_speed = 0.0  # For detecting changes
        
        # Create command sender timer
        self.send_timer = self.create_timer(0.1, self.send_command)

    def connect_serial(self):
        try:
            self.serial = serial.Serial(
                port=self.get_parameter('port').value,
                baudrate=self.get_parameter('baudrate').value,
                timeout=self.get_parameter('timeout').value
            )
            self.get_logger().info(f"Connected to Arduino at {self.get_parameter('port').value}")
            # Wait for Arduino to reset after serial connection
            time.sleep(2.0)
            # Flush serial buffer
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
        except Exception as e:
            self.get_logger().error(f"Serial connection failed: {str(e)}")

    def wheel_cmd_callback(self, msg):
        """Handle wheel commands from twist_to_motors node"""
        # Extract left/right wheel angular velocities (rad/s)
        left_angular_vel = msg.linear.x
        right_angular_vel = msg.linear.y
        
        # Convert angular velocity (rad/s) to steps/s
        left_steps_per_sec = left_angular_vel * self.steps_per_rad
        right_steps_per_sec = right_angular_vel * self.steps_per_rad
        
        # Store for sending
        self.last_left_speed = left_steps_per_sec
        self.last_right_speed = right_steps_per_sec

    def send_command(self):
        """Send velocity commands to Arduino at fixed rate, but only when changed"""
        if not self.serial or not self.serial.is_open:
            return
            
        # Only send commands if the values have changed significantly
        if (abs(self.last_left_speed - self.prev_left_speed) > 0.1 or 
            abs(self.last_right_speed - self.prev_right_speed) > 0.1):
            
            try:
                # Send velocity command
                cmd_str = f"VEL:{self.last_left_speed:.1f},{self.last_right_speed:.1f}\n"
                self.serial.write(cmd_str.encode())
                
                # Log non-zero commands only if debug enabled and motors are not stopped
                if self.debug_output and (abs(self.last_left_speed) > 0.1 or abs(self.last_right_speed) > 0.1):
                    self.get_logger().debug(f"Motor cmd: L={self.last_left_speed:.1f}, R={self.last_right_speed:.1f}")
                
                # Remember speeds for change detection
                self.prev_left_speed = self.last_left_speed
                self.prev_right_speed = self.last_right_speed
                
            except Exception as e:
                self.get_logger().error(f"Failed to send command: {str(e)}")

    def read_serial(self):
        """Read position feedback from Arduino"""
        if not self.serial or not self.serial.is_open:
            return
            
        try:
            while self.serial.in_waiting > 0:
                line = self.serial.readline().decode('utf-8').strip()
                
                # Handle critical messages from Arduino (convert others to debug)
                if line.startswith("MEGA:"):
                    if "Error" in line or "Failed" in line or "fault" in line.lower():
                        self.get_logger().error(f"Arduino: {line[5:]}")
                    elif self.debug_output:
                        self.get_logger().debug(f"Arduino: {line[5:]}")
                
                # Process position updates silently
                elif line.startswith("STEPS:"):
                    try:
                        _, data = line.split(':')
                        left, right = map(int, data.split(','))
                        
                        # Create and publish joint state message
                        msg = JointState()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.name = ['left_wheel', 'right_wheel']
                        msg.position = [float(left), float(right)]
                        
                        self.step_pub.publish(msg)
                    except ValueError:
                        self.get_logger().warn(f"Invalid data: {line}")
                
                # Ignore all other output to reduce noise
                
        except Exception as e:
            self.get_logger().error(f"Serial read error: {str(e)}")
            # Try to reconnect
            self.connect_serial()

    def destroy_node(self):
        """Clean shutdown"""
        if self.serial and self.serial.is_open:
            # Send stop command before closing
            try:
                self.serial.write(b"VEL:0.0,0.0\n")
                time.sleep(0.1)  # Brief delay to allow command to be sent
                self.serial.close()
            except:
                pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VelocityArduinoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()