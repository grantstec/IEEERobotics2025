#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import time
import Jetson.GPIO as GPIO

class HardwareSwitch(Node):
    """
    Node to read a hardware switch from GPIO and publish its state to ROS.
    """
    
    def __init__(self):
        super().__init__('hardware_switch')
        
        # Parameters
        self.declare_parameter('gpio_pin', 12)  # Using pin 12 (GPIO79) by default
        self.declare_parameter('active_high', True)  # Switch is active high
        self.declare_parameter('debounce_time', 0.05)  # 50ms debounce time
        self.declare_parameter('publish_rate', 10.0)  # Publish at 10Hz by default
        
        # Get parameters
        self.gpio_pin = self.get_parameter('gpio_pin').value
        self.active_high = self.get_parameter('active_high').value
        self.debounce_time = self.get_parameter('debounce_time').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Initialize GPIO
        self.setup_gpio()
        
        # Publisher for switch state
        self.switch_pub = self.create_publisher(Bool, 'hardware/start_switch', 10)
        
        # Last known switch state
        self.last_state = self.read_switch()
        self.last_change_time = time.time()
        
        # Timer to periodically publish the current switch state
        period = 1.0 / self.publish_rate
        self.create_timer(period, self.publish_switch_state)
        
        self.get_logger().info(f"Hardware switch initialized on GPIO pin {self.gpio_pin}")
        self.get_logger().info(f"Current switch state: {'ON' if self.last_state else 'OFF'}")

    def setup_gpio(self):
        """Set up the GPIO pin for the switch"""
        # Use BOARD pin numbering (physical pin numbers)
        GPIO.setmode(GPIO.BOARD)
        
        # Set up the pin with appropriate pull-up/down
        if self.active_high:
            # For active high, use pull-down resistor
            GPIO.setup(self.gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            self.get_logger().info("Switch configured as active-high with pull-down")
        else:
            # For active low, use pull-up resistor
            GPIO.setup(self.gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self.get_logger().info("Switch configured as active-low with pull-up")

    def read_switch(self):
        """Read the current switch state with proper logic"""
        # Read the physical pin state
        pin_state = GPIO.input(self.gpio_pin)
        
        # Apply logic based on active_high setting
        if self.active_high:
            return bool(pin_state)  # Return True if HIGH, False if LOW
        else:
            return not bool(pin_state)  # Return True if LOW, False if HIGH
    
    def publish_switch_state(self):
        """Read and publish the current switch state with debouncing"""
        current_time = time.time()
        current_state = self.read_switch()
        
        # Check if state has changed and debounce time has passed
        if (current_state != self.last_state and 
            current_time - self.last_change_time > self.debounce_time):
            
            # Update the last change time
            self.last_change_time = current_time
            
            # Log the state change
            if current_state:
                self.get_logger().info("Switch turned ON")
            else:
                self.get_logger().info("Switch turned OFF")
        
        # Always update last_state 
        self.last_state = current_state
        
        # Publish the current state
        msg = Bool()
        msg.data = current_state
        self.switch_pub.publish(msg)

    def destroy_node(self):
        """Clean up GPIO on shutdown"""
        GPIO.cleanup(self.gpio_pin)
        super().destroy_node()

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