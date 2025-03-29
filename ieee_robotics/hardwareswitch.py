#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import time

class HardwareSwitch(Node):
    """
    Node to read a hardware switch from GPIO and publish its state to ROS.
    """
    
    def __init__(self):
        super().__init__('hardware_switch')
        
        # Parameters
        self.declare_parameter('gpio_pin', 12)
        self.declare_parameter('active_high', True)
        self.declare_parameter('publish_rate', 1.0)  # Slower rate for less noise
        
        # Get parameters
        self.gpio_pin = self.get_parameter('gpio_pin').value
        self.active_high = self.get_parameter('active_high').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # State variables
        self.switch_state = False  # Default to OFF
        
        # Publisher for switch state
        self.switch_pub = self.create_publisher(Bool, 'hardware/start_switch', 10)
        
        # Add a dedicated subscriber
        self.cmd_sub = self.create_subscription(
            Bool,
            'hardware/set_switch', 
            self.set_switch_callback,
            10
        )
        
        # Also accept messages on the hardware/start_switch topic
        self.state_sub = self.create_subscription(
            Bool,
            'hardware/start_switch',
            self.switch_state_callback, 
            10
        )
        
        # Create timer for publishing state at regular intervals
        self.create_timer(1.0/self.publish_rate, self.publish_state)
        
        # Print initial instructions
        self.get_logger().info("=== VIRTUAL HARDWARE SWITCH INITIALIZED ===")
        self.get_logger().info(f"Current switch state: {'ON' if self.switch_state else 'OFF'}")
        self.get_logger().info("To turn switch ON:  ros2 topic pub -1 /hardware/set_switch std_msgs/Bool '{data: true}'")
        self.get_logger().info("To turn switch OFF: ros2 topic pub -1 /hardware/set_switch std_msgs/Bool '{data: false}'")
        self.get_logger().info("================================================")
    
    def set_switch_callback(self, msg):
        """Callback for dedicated control topic"""
        new_state = msg.data
        if new_state != self.switch_state:
            self.switch_state = new_state
            self.get_logger().info(f"*** SWITCH STATE CHANGED TO: {'ON' if new_state else 'OFF'} ***")
    
    def switch_state_callback(self, msg):
        """Callback for the actual switch state topic"""
        new_state = msg.data
        if new_state != self.switch_state:
            self.switch_state = new_state
            self.get_logger().info(f"*** SWITCH STATE CHANGED TO: {'ON' if new_state else 'OFF'} ***")
    
    def publish_state(self):
        """Publish the current switch state"""
        msg = Bool()
        msg.data = self.switch_state
        self.switch_pub.publish(msg)
    
    def read_switch(self):
        """Virtual read - just return the stored state"""
        return self.switch_state

def main(args=None):
    rclpy.init(args=args)
    node = HardwareSwitch()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()