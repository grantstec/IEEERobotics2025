#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class TwistToMotors(Node):
    def __init__(self):
        super().__init__('twist_to_motors')
        
        # Declare parameters for easier tuning
        self.declare_parameter('wheel_radius', 0.045)  # meters
        self.declare_parameter('wheel_base', 0.21)     # meters
        
        # Get parameter values
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_base = self.get_parameter('wheel_base').value
        
        # Log the configuration
        self.get_logger().info(f"Wheel radius: {self.wheel_radius}m, Wheel base: {self.wheel_base}m")
        
        # Subscribe to cmd_vel from teleop or navigation stack
        self.sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.twist_callback,
            10
        )
        
        # Publish wheel velocity commands to arduino_bridge
        self.wheel_pub = self.create_publisher(Twist, 'wheel_commands', 10)

    def twist_callback(self, msg):
        # Get linear and angular velocities from the Twist message
        linear_x = msg.linear.x    # m/s
        angular_z = msg.angular.z  # rad/s
        
        # Differential drive conversion
        # CORRECTED: Changed signs to fix left/right inversion
        # For positive angular_z (counterclockwise rotation), right wheel should be faster
        left = (linear_x - (angular_z * self.wheel_base / 2)) / self.wheel_radius   # rad/s
        right = (linear_x + (angular_z * self.wheel_base / 2)) / self.wheel_radius  # rad/s
        
        # Debug output
        self.get_logger().debug(f"Twist in: linear={linear_x} m/s, angular={angular_z} rad/s")
        self.get_logger().debug(f"Wheel out: left={left} rad/s, right={right} rad/s")
        
        # Create and publish command
        cmd = Twist()
        cmd.linear.x = left   # Left wheel angular velocity (rad/s)
        cmd.linear.y = right  # Right wheel angular velocity (rad/s)
        self.wheel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = TwistToMotors()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()