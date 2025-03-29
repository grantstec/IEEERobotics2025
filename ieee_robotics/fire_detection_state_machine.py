#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from tf_transformations import quaternion_from_euler
from std_msgs.msg import Bool, String
import time
import math

class FireDetectionStateMachine(Node):
    """
    Node to implement the fire detection state machine.
    This handles the rotation sequence to find fire and publishes navigation goals.
    """
    
    # State machine constants
    STATE_INITIAL = 0
    STATE_CHECK_0_DEGREES = 1
    STATE_CHECK_45_DEGREES_LEFT = 2
    STATE_CHECK_90_DEGREES = 3
    STATE_RESET = 4

    def __init__(self):
        super().__init__('fire_detection_state_machine')
        
        # Parameters
        self.declare_parameter('goal_active', False)  # Default to inactive
        
        # State variables
        self.goal_active = self.get_parameter('goal_active').value
        self.switch_activated = False
        self.current_state = self.STATE_INITIAL
        self.last_state_change_time = 0
        self.current_fire_detection = None
        
        # Subscribe to goal_active control from the adapter
        self.goal_active_sub = self.create_subscription(
            Bool, 
            'teensy/goal_active', 
            self.goal_active_callback, 
            10
        )
        
        # Subscribe to fire detection from the IMU bridge
        self.fire_detection_sub = self.create_subscription(
            String,
            'teensy/fire_detection',
            self.fire_detection_callback,
            10
        )
        
        # Subscribe to the switch state
        self.switch_sub = self.create_subscription(
            Bool,
            'hardware/start_switch',
            self.switch_callback,
            10
        )
        
        # Define fire destination points (same as original)
        self.final_destinations = [
            self.create_goal_pose(2.0, 0.0, 0),
            self.create_goal_pose(2.0, 2.0, 45),
            self.create_goal_pose(0.0, 2.0, 90)
        ]
        
        # Publish to the same goal_pose topic as the original for compatibility
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        
        # Timer for state machine processing
        self.create_timer(0.1, self.process_state_machine)
        self.get_logger().info("Fire detection state machine initialized - waiting for switch activation")

    def goal_active_callback(self, msg):
        """Handle goal_active state changes from the adapter"""
        previous_state = self.goal_active
        self.goal_active = msg.data
        
        if self.goal_active and not previous_state:
            self.get_logger().info("Fire detection state machine ENABLED")
        elif not self.goal_active and previous_state:
            self.get_logger().info("Fire detection state machine DISABLED")
            
        # If both switch and goal_active become true, reset state machine
        if self.goal_active and self.switch_activated:
            self.current_state = self.STATE_INITIAL
            self.last_state_change_time = time.time()
            self.get_logger().info("Starting fire detection scan sequence")

    def switch_callback(self, msg):
        """Handle switch state changes"""
        previous_state = self.switch_activated
        self.switch_activated = msg.data
        
        # Log switch activation
        if self.switch_activated and not previous_state:
            self.get_logger().info("Switch ACTIVATED")
            
            # If the state machine is already enabled, start it now
            if self.goal_active:
                self.current_state = self.STATE_INITIAL
                self.last_state_change_time = time.time()
                self.get_logger().info("Starting fire detection scan sequence")
        
        elif not self.switch_activated and previous_state:
            self.get_logger().info("Switch DEACTIVATED")

    def fire_detection_callback(self, msg):
        """Process fire detection data from the Teensy"""
        if msg.data.startswith('FIRE:'):
            self.current_fire_detection = '1' in msg.data

    def create_goal_pose(self, x, y, angle=0.0):
        """Create a PoseStamped message with the given coordinates and angle"""
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, math.radians(angle)))
        return goal

    def process_state_machine(self):
        """Process the fire detection state machine"""
        # Only run if both goal_active and switch_activated are true
        if not self.goal_active or not self.switch_activated:
            return
            
        # Skip if no fire detection data available
        if self.current_fire_detection is None:
            return
            
        current_time = time.time()
        
        if self.current_state == self.STATE_INITIAL:
            # Start by looking straight ahead (0 degrees)
            self.send_rotation_goal(0)
            self.current_state = self.STATE_CHECK_0_DEGREES
            self.last_state_change_time = current_time
            self.get_logger().info("Starting fire detection scan at 0 degrees")

        elif self.current_state == self.STATE_CHECK_0_DEGREES:
            # Wait for 5 seconds at each position
            if current_time - self.last_state_change_time > 5:
                if self.current_fire_detection:
                    # Fire detected at 0 degrees
                    self.publish_fire_location(0)
                else:
                    # No fire, try 45 degrees left
                    self.send_rotation_goal(45)
                    self.current_state = self.STATE_CHECK_45_DEGREES_LEFT
                    self.last_state_change_time = current_time
                    self.get_logger().info("No fire detected at 0 degrees, checking 45 degrees")
        
        elif self.current_state == self.STATE_CHECK_45_DEGREES_LEFT:
            if current_time - self.last_state_change_time > 5:
                if self.current_fire_detection:
                    # Fire detected at 45 degrees
                    self.publish_fire_location(45)
                else:
                    # No fire, try 90 degrees left
                    self.send_rotation_goal(90)
                    self.current_state = self.STATE_CHECK_90_DEGREES
                    self.last_state_change_time = current_time
                    self.get_logger().info("No fire detected at 45 degrees, checking 90 degrees")

        elif self.current_state == self.STATE_CHECK_90_DEGREES:
            if current_time - self.last_state_change_time > 5:
                if self.current_fire_detection:
                    # Fire detected at 90 degrees
                    self.publish_fire_location(90)
                else:
                    # No fire found, reset to 0 degrees
                    self.send_rotation_goal(0)
                    self.current_state = self.STATE_RESET
                    self.last_state_change_time = current_time
                    self.get_logger().info("No fire detected at 90 degrees, resetting to 0 degrees")

        elif self.current_state == self.STATE_RESET:
            if current_time - self.last_state_change_time > 5:
                self.get_logger().warn("No fire detected in full scan")
                self.goal_active = False
                # Could add more complex retry behavior here if needed

    def send_rotation_goal(self, angle):
        """Send a rotation goal to turn the robot to the specified angle"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'base_link'
        goal_msg.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, math.radians(angle)))
        self.goal_pub.publish(goal_msg)
        self.get_logger().debug(f"Sent rotation goal: {angle} degrees")

    def publish_fire_location(self, angle):
        """Publish the fire location goal based on the detected angle"""
        # Select the appropriate predefined destination based on angle
        goal_msg = self.final_destinations[{0:0, 45:1, 90:2}.get(angle, 0)]
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f"Fire detected at {angle} degrees! Publishing goal position.")
        
        # Since we've detected the fire, disable the state machine
        self.goal_active = False

def main(args=None):
    rclpy.init(args=args)
    node = FireDetectionStateMachine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()