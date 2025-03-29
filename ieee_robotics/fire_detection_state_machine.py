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
    STATE_MOVE_TO_START_POSITION = 0
    STATE_CHECK_0_DEGREES = 1
    STATE_CHECK_MINUS_45_DEGREES = 2
    STATE_CHECK_MINUS_90_DEGREES = 3
    STATE_MOVE_TO_CENTER = 4
    STATE_CENTER_CHECK_PLUS_45_DEGREES = 5
    STATE_CENTER_CHECK_MINUS_45_DEGREES = 6
    STATE_CENTER_CHECK_MINUS_135_DEGREES = 7
    STATE_WAIT_AT_FIRE = 8        # New state for waiting at fire location
    STATE_RETURN_TO_START = 9     # New state for returning to start
    STATE_RESET = 10

    def __init__(self):
        super().__init__('fire_detection_state_machine')
        
        # Parameters
        self.declare_parameter('goal_active', False)         # Default to inactive
        self.declare_parameter('wait_time', 12.0)            # Time to wait at each scan position
        self.declare_parameter('rotation_wait_time', 3.0)    # Time to wait for rotation to complete
        self.declare_parameter('fire_wait_time', 5.0)        # Time to wait at fire before return
        self.declare_parameter('start_position_x', 0.3)      # Starting position X
        self.declare_parameter('start_position_y', -0.2)     # Starting position Y
        self.declare_parameter('center_position_x', 1.1)     # Center position X
        self.declare_parameter('center_position_y', -1.1)    # Center position Y
        
        # State variables
        self.goal_active = self.get_parameter('goal_active').value
        self.wait_time = self.get_parameter('wait_time').value
        self.rotation_wait_time = self.get_parameter('rotation_wait_time').value
        self.fire_wait_time = self.get_parameter('fire_wait_time').value
        self.start_position_x = self.get_parameter('start_position_x').value
        self.start_position_y = self.get_parameter('start_position_y').value
        self.center_position_x = self.get_parameter('center_position_x').value
        self.center_position_y = self.get_parameter('center_position_y').value
        self.switch_activated = False
        self.current_state = self.STATE_MOVE_TO_START_POSITION
        self.last_state_change_time = 0
        self.current_fire_detection = None
        self.is_running = False               # Track if state machine is currently running
        self.goal_sent_time = 0               # Track when a goal was sent for timeout monitoring
        self.waiting_for_rotation = False     # Track if we're waiting for rotation
        self.scanning_started_time = 0        # When actual scanning started after rotation
        self.fire_detected_location = None    # Store which fire location was detected
        
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
        
        # Define fire destination points based on the angles
        self.fire_destinations = {
            "0_degrees": self.create_goal_pose(1.8, 0.0, 0),
            "minus_45_degrees": self.create_goal_pose(2.0, -2.0, -48),
            "minus_90_degrees": self.create_goal_pose(0.0, -1.8, -95),
            "center_plus_45": self.create_goal_pose(1.8, 0.0, 45),
            "center_minus_45": self.create_goal_pose(2.0, -2.0, -45),
            "center_minus_135": self.create_goal_pose(0.0, -1.8, -135)
        }
        
        # Publish to the same goal_pose topic as the original for compatibility
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        
        # Timer for state machine processing - run at a faster rate for more responsive state transitions
        self.create_timer(0.05, self.process_state_machine)
        self.get_logger().info("Fire detection state machine initialized - waiting for switch activation")
        self.get_logger().info(f"Using wait time of {self.wait_time} seconds at each position")
        self.get_logger().info(f"Using rotation wait time of {self.rotation_wait_time} seconds")
        self.get_logger().info(f"Using fire wait time of {self.fire_wait_time} seconds")

    def goal_active_callback(self, msg):
        """Handle goal_active state changes from the adapter"""
        previous_state = self.goal_active
        self.goal_active = msg.data
        
        if self.goal_active and not previous_state:
            self.get_logger().info("Fire detection state machine ENABLED")
        elif not self.goal_active and previous_state:
            self.get_logger().info("Fire detection state machine DISABLED")
            
        # If both switch and goal_active become true, reset state machine
        if self.goal_active and self.switch_activated and not self.is_running:
            self.current_state = self.STATE_MOVE_TO_START_POSITION
            self.last_state_change_time = time.time()
            self.is_running = True  # Set running flag
            self.get_logger().info("Starting fire detection sequence with initial positioning")

    def switch_callback(self, msg):
        """Handle switch state changes"""
        previous_state = self.switch_activated
        self.switch_activated = msg.data
        
        # Log switch activation
        if self.switch_activated and not previous_state:
            self.get_logger().info("Switch ACTIVATED")
            
            # If the state machine is already enabled, start it now
            if self.goal_active and not self.is_running:
                self.current_state = self.STATE_MOVE_TO_START_POSITION
                self.last_state_change_time = time.time()
                self.is_running = True
                self.get_logger().info("Starting fire detection sequence with initial positioning")
        
        elif not self.switch_activated and previous_state:
            self.get_logger().info("Switch DEACTIVATED")
            self.is_running = False

    def fire_detection_callback(self, msg):
        """Process fire detection data from the Teensy"""
        if msg.data.startswith('FIRE:'):
            # Parse '1' or '0' from the FIRE: message
            detection_value = msg.data.split(':')[1].strip()
            new_detection = (detection_value == '1')
            
            # Log when fire state changes (especially if fire is detected)
            if new_detection != self.current_fire_detection:
                if new_detection:
                    self.get_logger().info("FIRE DETECTED by thermal sensor!")
                # Uncomment if you want to log when fire is no longer detected
                # else:
                #     self.get_logger().info("No fire detected")
                    
            self.current_fire_detection = new_detection

    def create_goal_pose(self, x, y, angle=0.0):
        """Create a PoseStamped message with the given coordinates and angle"""
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        quat = quaternion_from_euler(0, 0, math.radians(angle))
        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]
        return goal

    def process_state_machine(self):
        """Process the fire detection state machine"""
        # Only run if both goal_active and switch_activated are true
        if not self.goal_active or not self.switch_activated:
            if self.is_running:
                self.is_running = False
                self.get_logger().info("Fire detection state machine stopped")
            return
            
        # Skip if no fire detection data available yet and we're in an active scanning state
        if self.current_fire_detection is None and self.current_state > self.STATE_MOVE_TO_CENTER:
            return
            
        current_time = time.time()
        
        # Handle state transitions
        if self.current_state == self.STATE_MOVE_TO_START_POSITION:
            # Move to starting position
            self.send_position_goal(self.start_position_x, self.start_position_y, 0)
            self.current_state = self.STATE_CHECK_0_DEGREES
            self.last_state_change_time = current_time
            self.goal_sent_time = current_time
            self.waiting_for_rotation = False  # We're waiting for position, not rotation
            self.get_logger().info(f"Moving to starting position ({self.start_position_x}, {self.start_position_y})")

        elif self.current_state == self.STATE_CHECK_0_DEGREES:
            # First wait for navigation to complete (5 seconds should be plenty)
            if not self.waiting_for_rotation and current_time - self.goal_sent_time > 5.0:
                self.get_logger().info("Reached starting position, starting scan at 0 degrees")
                # Mark the start of the actual scanning time
                self.scanning_started_time = current_time
                self.waiting_for_rotation = True
            
            # Now that we're in position, check for fire detection for the specified wait time
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at 0 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                if self.current_fire_detection:
                    # Fire detected at 0 degrees - immediately send goal
                    self.fire_detected_location = "0_degrees"
                    self.publish_fire_location("0_degrees")
                    self.current_state = self.STATE_WAIT_AT_FIRE
                    self.last_state_change_time = current_time
                    self.get_logger().info("FIRE DETECTED at 0 degrees! Navigating to fire location")
                else:
                    # No fire, try -45 degrees
                    self.send_rotation_goal(-45)
                    self.current_state = self.STATE_CHECK_MINUS_45_DEGREES
                    self.last_state_change_time = current_time
                    self.goal_sent_time = current_time
                    self.waiting_for_rotation = False  # Reset to wait for rotation
                    self.get_logger().info("No fire detected at 0 degrees, rotating to -45 degrees")
            
        elif self.current_state == self.STATE_CHECK_MINUS_45_DEGREES:
            # First wait for rotation to complete
            if not self.waiting_for_rotation and current_time - self.goal_sent_time > self.rotation_wait_time:
                self.get_logger().info("Rotation to -45 degrees complete, starting scan")
                # Mark the start of the actual scanning time
                self.scanning_started_time = current_time
                self.waiting_for_rotation = True
            
            # Now that we're rotated, check for fire detection
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at -45 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                if self.current_fire_detection:
                    # Fire detected at -45 degrees - immediately send goal
                    self.fire_detected_location = "minus_45_degrees"
                    self.publish_fire_location("minus_45_degrees")
                    self.current_state = self.STATE_WAIT_AT_FIRE
                    self.last_state_change_time = current_time
                    self.get_logger().info("FIRE DETECTED at -45 degrees! Navigating to fire location")
                else:
                    # No fire, try -90 degrees
                    self.send_rotation_goal(-90)
                    self.current_state = self.STATE_CHECK_MINUS_90_DEGREES
                    self.last_state_change_time = current_time
                    self.goal_sent_time = current_time
                    self.waiting_for_rotation = False  # Reset to wait for rotation
                    self.get_logger().info("No fire detected at -45 degrees, rotating to -90 degrees")

        elif self.current_state == self.STATE_CHECK_MINUS_90_DEGREES:
            # First wait for rotation to complete
            if not self.waiting_for_rotation and current_time - self.goal_sent_time > self.rotation_wait_time:
                self.get_logger().info("Rotation to -90 degrees complete, starting scan")
                # Mark the start of the actual scanning time
                self.scanning_started_time = current_time
                self.waiting_for_rotation = True
            
            # Now that we're rotated, check for fire detection
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at -90 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                if self.current_fire_detection:
                    # Fire detected at -90 degrees - immediately send goal
                    self.fire_detected_location = "minus_90_degrees"
                    self.publish_fire_location("minus_90_degrees")
                    self.current_state = self.STATE_WAIT_AT_FIRE
                    self.last_state_change_time = current_time
                    self.get_logger().info("FIRE DETECTED at -90 degrees! Navigating to fire location")
                else:
                    # No fire found, move to center position
                    self.send_position_goal(self.center_position_x, self.center_position_y, 0)
                    self.current_state = self.STATE_MOVE_TO_CENTER
                    self.last_state_change_time = current_time
                    self.goal_sent_time = current_time
                    self.waiting_for_rotation = False  # Reset for navigation
                    self.get_logger().info(f"No fire detected in initial scan, moving to center position ({self.center_position_x}, {self.center_position_y})")

        elif self.current_state == self.STATE_MOVE_TO_CENTER:
            # Wait for navigation to complete
            if not self.waiting_for_rotation and current_time - self.goal_sent_time > 5.0:  # 5 seconds to reach center
                # Now start the center scanning sequence with +45 degrees
                self.send_rotation_goal(45)
                self.current_state = self.STATE_CENTER_CHECK_PLUS_45_DEGREES
                self.last_state_change_time = current_time
                self.goal_sent_time = current_time
                self.waiting_for_rotation = False  # Reset for new rotation
                self.get_logger().info("Reached center position, rotating to +45 degrees")

        elif self.current_state == self.STATE_CENTER_CHECK_PLUS_45_DEGREES:
            # First wait for rotation to complete
            if not self.waiting_for_rotation and current_time - self.goal_sent_time > self.rotation_wait_time:
                self.get_logger().info("Rotation to +45 degrees complete, starting scan")
                # Mark the start of the actual scanning time
                self.scanning_started_time = current_time
                self.waiting_for_rotation = True
            
            # Now that we're rotated, check for fire detection
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at +45 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                if self.current_fire_detection:
                    # Fire detected at center +45 degrees - immediately send goal
                    self.fire_detected_location = "center_plus_45"
                    self.publish_fire_location("center_plus_45")
                    self.current_state = self.STATE_WAIT_AT_FIRE
                    self.last_state_change_time = current_time
                    self.get_logger().info("FIRE DETECTED at center +45 degrees! Navigating to fire location")
                else:
                    # No fire, try -45 degrees from center
                    self.send_rotation_goal(-45)
                    self.current_state = self.STATE_CENTER_CHECK_MINUS_45_DEGREES
                    self.last_state_change_time = current_time
                    self.goal_sent_time = current_time
                    self.waiting_for_rotation = False  # Reset for new rotation
                    self.get_logger().info("No fire detected at +45 degrees, rotating to -45 degrees")

        elif self.current_state == self.STATE_CENTER_CHECK_MINUS_45_DEGREES:
            # First wait for rotation to complete
            if not self.waiting_for_rotation and current_time - self.goal_sent_time > self.rotation_wait_time:
                self.get_logger().info("Rotation to -45 degrees complete, starting scan")
                # Mark the start of the actual scanning time
                self.scanning_started_time = current_time
                self.waiting_for_rotation = True
            
            # Now that we're rotated, check for fire detection
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at -45 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                if self.current_fire_detection:
                    # Fire detected at center -45 degrees - immediately send goal
                    self.fire_detected_location = "center_minus_45"
                    self.publish_fire_location("center_minus_45")
                    self.current_state = self.STATE_WAIT_AT_FIRE
                    self.last_state_change_time = current_time
                    self.get_logger().info("FIRE DETECTED at center -45 degrees! Navigating to fire location")
                else:
                    # No fire, try -135 degrees from center
                    self.send_rotation_goal(-135)
                    self.current_state = self.STATE_CENTER_CHECK_MINUS_135_DEGREES
                    self.last_state_change_time = current_time
                    self.goal_sent_time = current_time
                    self.waiting_for_rotation = False  # Reset for new rotation
                    self.get_logger().info("No fire detected at -45 degrees, rotating to -135 degrees")

        elif self.current_state == self.STATE_CENTER_CHECK_MINUS_135_DEGREES:
            # First wait for rotation to complete
            if not self.waiting_for_rotation and current_time - self.goal_sent_time > self.rotation_wait_time:
                self.get_logger().info("Rotation to -135 degrees complete, starting scan")
                # Mark the start of the actual scanning time
                self.scanning_started_time = current_time
                self.waiting_for_rotation = True
            
            # Now that we're rotated, check for fire detection
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at -135 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                if self.current_fire_detection:
                    # Fire detected at center -135 degrees - immediately send goal
                    self.fire_detected_location = "center_minus_135"
                    self.publish_fire_location("center_minus_135")
                    self.current_state = self.STATE_WAIT_AT_FIRE
                    self.last_state_change_time = current_time
                    self.get_logger().info("FIRE DETECTED at center -135 degrees! Navigating to fire location")
                else:
                    # No fire found in entire scan sequence, reset
                    self.send_rotation_goal(0)
                    self.current_state = self.STATE_RESET
                    self.last_state_change_time = current_time
                    self.waiting_for_rotation = False
                    self.get_logger().warn("No fire detected in any scan position! Resetting to 0 degrees.")

        elif self.current_state == self.STATE_WAIT_AT_FIRE:
            # Wait at the fire location for 5 seconds
            if current_time - self.last_state_change_time > self.fire_wait_time:
                self.get_logger().info(f"Waited at fire location for {self.fire_wait_time} seconds, returning to start")
                self.current_state = self.STATE_RETURN_TO_START
                self.last_state_change_time = current_time
                
                # Return to start position
                self.send_position_goal(self.start_position_x, self.start_position_y, 0)

        elif self.current_state == self.STATE_RETURN_TO_START:
            # Wait for return navigation to complete (10 seconds should be enough)
            if current_time - self.last_state_change_time > 10.0:
                self.get_logger().info("Returned to start position, fire detection sequence complete")
                self.goal_active = False
                self.is_running = False

        elif self.current_state == self.STATE_RESET:
            if current_time - self.last_state_change_time > 5:
                self.get_logger().warn("Fire detection scan complete - no fire found")
                self.goal_active = False
                self.is_running = False

    def send_position_goal(self, x, y, angle=0):
        """Send a position goal to move the robot to the specified position and orientation"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        
        quat = quaternion_from_euler(0, 0, math.radians(angle))
        goal_msg.pose.orientation.x = quat[0]
        goal_msg.pose.orientation.y = quat[1]
        goal_msg.pose.orientation.z = quat[2]
        goal_msg.pose.orientation.w = quat[3]
        
        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f"Sent position goal: ({x}, {y}) at {angle} degrees")

    def send_rotation_goal(self, angle):
        """Send a rotation goal to turn the robot to the specified angle"""
        # For rotation goals, we send map-relative angles
        # This ensure the robot turns to the absolute angle relative to the map
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        
        # Keep the current position, just change orientation
        if self.current_state < self.STATE_MOVE_TO_CENTER:
            # Use start position
            goal_msg.pose.position.x = self.start_position_x
            goal_msg.pose.position.y = self.start_position_y
        else:
            # Use center position
            goal_msg.pose.position.x = self.center_position_x
            goal_msg.pose.position.y = self.center_position_y
        
        quat = quaternion_from_euler(0, 0, math.radians(angle))
        goal_msg.pose.orientation.x = quat[0]
        goal_msg.pose.orientation.y = quat[1]
        goal_msg.pose.orientation.z = quat[2]
        goal_msg.pose.orientation.w = quat[3]
        
        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f"Sent rotation goal: {angle} degrees")

    def publish_fire_location(self, fire_location_key):
        """Publish the fire location goal based on the detected position"""
        if fire_location_key in self.fire_destinations:
            goal_msg = self.fire_destinations[fire_location_key]
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            self.goal_pub.publish(goal_msg)
            self.get_logger().info(f"FIRE DETECTED at {fire_location_key}! Publishing goal position immediately.")
        else:
            self.get_logger().error(f"Unknown fire location key: {fire_location_key}")

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