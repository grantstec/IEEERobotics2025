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
    STATE_WAIT_AT_FIRE = 8       
    STATE_GOTO_ORIGIN = 9        # Renamed for clarity
    STATE_CHECK_CORNER_1 = 10    # New states for checking corners
    STATE_CHECK_CORNER_2 = 11
    STATE_CHECK_CORNER_3 = 12
    STATE_RESET = 13             # Moved to end

    def __init__(self):
        super().__init__('fire_detection_state_machine')
        
        # Parameters
        self.declare_parameter('goal_active', False)         # Default to inactive
        self.declare_parameter('wait_time', 2.0)            # Time to wait at each scan position
        self.declare_parameter('rotation_wait_time', 3.0)    # Time to wait for rotation to complete
        self.declare_parameter('fire_wait_time', 5.0)       # Time to wait at fire before return
        self.declare_parameter('corner_wait_time', 4.0)      # Time to wait at each corner
        self.declare_parameter('start_position_x', 0.3)      # Starting position X
        self.declare_parameter('start_position_y', -0.2)     # Starting position Y
        self.declare_parameter('center_position_x', 1.0)     # Center position X
        self.declare_parameter('center_position_y', -1.0)    # Center position Y
        
        # State variables
        self.goal_active = self.get_parameter('goal_active').value
        self.wait_time = self.get_parameter('wait_time').value
        self.rotation_wait_time = self.get_parameter('rotation_wait_time').value
        self.fire_wait_time = self.get_parameter('fire_wait_time').value
        self.corner_wait_time = self.get_parameter('corner_wait_time').value
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
        self.origin_sent = False              # Flag to track if origin command was sent
        self.corner_check_started = False     # Flag to track if we've started corner checking
        
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
            "0_degrees": self.create_goal_pose(1.6, -0.3, -5),
            "minus_45_degrees": self.create_goal_pose(1.6, -1.6, -50),
            "minus_90_degrees": self.create_goal_pose(0.3, -1.6, -95),
            "center_plus_45": self.create_goal_pose(1.6, -0.25, 45),
            "center_minus_45": self.create_goal_pose(1.6, -1.6, -45),
            "center_minus_135": self.create_goal_pose(0.3, -1.6, -135)
        }
        
        # Define corner positions to check if no fire found in initial scan
        self.corners = [
            {"name": "Corner 1", "x": 1.6, "y": -0.3, "angle": 0},    # Top right
            {"name": "Corner 2", "x": 1.6, "y": -1.6, "angle": -45},  # Bottom right
            {"name": "Corner 3", "x": 0.3, "y": -1.6, "angle": -90}   # Bottom left
        ]
        
        # Publish to the same goal_pose topic as the original for compatibility
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        
        # Timer for state machine processing - run at a faster rate for more responsive state transitions
        self.create_timer(0.05, self.process_state_machine)
        self.get_logger().info("Fire detection state machine initialized - waiting for switch activation")
        self.get_logger().info(f"Using wait time of {self.wait_time} seconds at each position")
        self.get_logger().info(f"Using rotation wait time of {self.rotation_wait_time} seconds")
        self.get_logger().info(f"Using fire wait time of {self.fire_wait_time} seconds")
        self.get_logger().info(f"Will return to origin (0,0) after waiting at fire location")
        self.get_logger().info(f"Added corner scanning if no fire detected in main scan")

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
        """Process fire detection data from the Teensy and immediately respond if fire detected"""
        if msg.data.startswith('FIRE:'):
            # Parse '1' or '0' from the FIRE: message
            detection_value = msg.data.split(':')[1].strip()
            new_detection = (detection_value == '1')
            
            # Only process changes in detection state
            if new_detection != self.current_fire_detection:
                self.current_fire_detection = new_detection
                
                # If fire is newly detected, immediately respond regardless of state
                if new_detection and self.is_running:
                    self.get_logger().info("???? FIRE DETECTED by thermal sensor! Immediately navigating to corresponding location")
                    
                    # Determine which position we're in and navigate directly to corresponding corner
                    # We're now prioritizing immediate corner navigation
                    
                    # CRITICAL CHANGE: Detect current state and navigate to appropriate corner
                    if self.current_state == self.STATE_CHECK_0_DEGREES:
                        self.get_logger().info("FIRE DETECTED in 0 degrees scan!")
                        corner_idx = 0  # Corner 1 (Top right)
                        self.navigate_to_fire_corner(corner_idx)
                        return
                        
                    elif self.current_state == self.STATE_CHECK_MINUS_45_DEGREES:
                        self.get_logger().info("FIRE DETECTED in -45 degrees scan!")
                        corner_idx = 1  # Corner 2 (Bottom right)
                        self.navigate_to_fire_corner(corner_idx)
                        return
                        
                    elif self.current_state == self.STATE_CHECK_MINUS_90_DEGREES:
                        self.get_logger().info("FIRE DETECTED in -90 degrees scan!")
                        corner_idx = 2  # Corner 3 (Bottom left)
                        self.navigate_to_fire_corner(corner_idx)
                        return
                        
                    elif self.current_state == self.STATE_CENTER_CHECK_PLUS_45_DEGREES:
                        self.get_logger().info("FIRE DETECTED in center +45 degrees scan!")
                        corner_idx = 0  # Corner 1 (Top right)
                        self.navigate_to_fire_corner(corner_idx)
                        return
                        
                    elif self.current_state == self.STATE_CENTER_CHECK_MINUS_45_DEGREES:
                        self.get_logger().info("FIRE DETECTED in center -45 degrees scan!")
                        corner_idx = 1  # Corner 2 (Bottom right)
                        self.navigate_to_fire_corner(corner_idx)
                        return
                        
                    elif self.current_state == self.STATE_CENTER_CHECK_MINUS_135_DEGREES:
                        self.get_logger().info("FIRE DETECTED in center -135 degrees scan!")
                        corner_idx = 2  # Corner 3 (Bottom left)
                        self.navigate_to_fire_corner(corner_idx)
                        return
                        
                    # Handle direct corner detection
                    elif self.current_state == self.STATE_CHECK_CORNER_1:
                        self.get_logger().info(f"FIRE DETECTED at {self.corners[0]['name']}!")
                        self.navigate_to_fire_corner(0)
                        return
                        
                    elif self.current_state == self.STATE_CHECK_CORNER_2:
                        self.get_logger().info(f"FIRE DETECTED at {self.corners[1]['name']}!")
                        self.navigate_to_fire_corner(1)
                        return
                        
                    elif self.current_state == self.STATE_CHECK_CORNER_3:
                        self.get_logger().info(f"FIRE DETECTED at {self.corners[2]['name']}!")
                        self.navigate_to_fire_corner(2)
                        return
                        
                    # If we're in an unknown state but detected fire, default to corner 1
                    else:
                        self.get_logger().info(f"FIRE DETECTED in unknown state {self.current_state}!")
                        self.navigate_to_fire_corner(0)  # Default to corner 1
                        return

    def navigate_to_fire_corner(self, corner_idx):
        """Immediately navigate to the specified corner when fire is detected"""
        if corner_idx < 0 or corner_idx >= len(self.corners):
            self.get_logger().error(f"Invalid corner index: {corner_idx}")
            return
            
        corner = self.corners[corner_idx]
        
        # Log with high visibility to confirm navigation is happening
        self.get_logger().info("="*50)
        self.get_logger().info(f"?????? NAVIGATING TO FIRE AT {corner['name']}: ({corner['x']}, {corner['y']})")
        self.get_logger().info("="*50)
        
        # Send the position goal multiple times to ensure it's received
        for i in range(10):  # Send 10 times to be extra sure
            self.send_position_goal(corner['x'], corner['y'], corner['angle'])
            time.sleep(0.05)  # Short delay between publishes
        
        # Update state machine
        self.fire_detected_location = f"corner_{corner_idx+1}"
        self.current_state = self.STATE_WAIT_AT_FIRE
        self.last_state_change_time = time.time()
        self.waiting_for_rotation = False
        
        # Additional safety - force publish extra goal after a short delay
        # This ensures navigation system receives the goal
        self.create_timer(0.5, lambda: self.send_position_goal(corner['x'], corner['y'], corner['angle']), oneshot=True)

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
            if not self.waiting_for_rotation and current_time - self.goal_sent_time > 8.0:
                self.get_logger().info("Reached starting position, starting scan at 0 degrees")
                # Mark the start of the actual scanning time
                self.scanning_started_time = current_time
                self.waiting_for_rotation = True
            
            # Now that we're in position, wait for the scanning time
            # We don't check for fire here anymore - that's done in the callback
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at 0 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                # No fire detected after full wait time, move to next position
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
            
            # Now that we're rotated, wait for the scanning time
            # We don't check for fire here anymore - that's done in the callback
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at -45 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                # No fire detected after full wait time, move to next position
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
            
            # Now that we're rotated, wait for the scanning time
            # We don't check for fire here anymore - that's done in the callback
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at -90 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                # No fire detected after full wait time, move to center position
                self.send_position_goal(self.center_position_x, self.center_position_y, 0)
                self.current_state = self.STATE_MOVE_TO_CENTER
                self.last_state_change_time = current_time
                self.goal_sent_time = current_time
                self.waiting_for_rotation = False  # Reset for navigation
                self.get_logger().info(f"No fire detected in initial scan, moving to center position ({self.center_position_x}, {self.center_position_y})")

        elif self.current_state == self.STATE_MOVE_TO_CENTER:
            # Wait for navigation to complete
            if not self.waiting_for_rotation and current_time - self.goal_sent_time > 8.0:  # 8 seconds to reach center
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
            
            # Now that we're rotated, wait for the scanning time
            # We don't check for fire here anymore - that's done in the callback
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at +45 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                # No fire detected after full wait time, move to next position
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
            
            # Now that we're rotated, wait for the scanning time
            # We don't check for fire here anymore - that's done in the callback
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at -45 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                # No fire detected after full wait time, move to next position
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
            
            # Now that we're rotated, wait for the scanning time
            # We don't check for fire here anymore - that's done in the callback
            if self.waiting_for_rotation and current_time - self.scanning_started_time > self.wait_time:
                self.get_logger().info(f"Completed scan at -135 degrees after {current_time - self.scanning_started_time:.1f} seconds")
                
                # CHANGED: Instead of going to RESET, go to the first corner
                self.send_position_goal(self.corners[0]["x"], self.corners[0]["y"], self.corners[0]["angle"])
                self.current_state = self.STATE_CHECK_CORNER_1
                self.last_state_change_time = current_time
                self.goal_sent_time = current_time
                self.corner_check_started = False
                self.get_logger().info(f"No fire detected in center scan, moving to {self.corners[0]['name']} at ({self.corners[0]['x']}, {self.corners[0]['y']})")

        elif self.current_state == self.STATE_CHECK_CORNER_1:
            # First wait for navigation to complete
            if not self.corner_check_started and current_time - self.goal_sent_time > 8.0:
                self.get_logger().info(f"Reached {self.corners[0]['name']}, checking for fire")
                self.corner_check_started = True
                self.scanning_started_time = current_time
            
            # Now check for fire detection for the corner wait time
            # Fire detection is handled in fire_detection_callback
            if self.corner_check_started and current_time - self.scanning_started_time > self.corner_wait_time:
                self.get_logger().info(f"Completed check at {self.corners[0]['name']} after {current_time - self.scanning_started_time:.1f} seconds")
                
                # Move to the next corner
                self.send_position_goal(self.corners[1]["x"], self.corners[1]["y"], self.corners[1]["angle"])
                self.current_state = self.STATE_CHECK_CORNER_2
                self.last_state_change_time = current_time
                self.goal_sent_time = current_time
                self.corner_check_started = False
                self.get_logger().info(f"No fire detected at {self.corners[0]['name']}, moving to {self.corners[1]['name']} at ({self.corners[1]['x']}, {self.corners[1]['y']})")

        elif self.current_state == self.STATE_CHECK_CORNER_2:
            # First wait for navigation to complete
            if not self.corner_check_started and current_time - self.goal_sent_time > 8.0:
                self.get_logger().info(f"Reached {self.corners[1]['name']}, checking for fire")
                self.corner_check_started = True
                self.scanning_started_time = current_time
            
            # Now check for fire detection for the corner wait time
            # Fire detection is handled in fire_detection_callback
            if self.corner_check_started and current_time - self.scanning_started_time > self.corner_wait_time:
                self.get_logger().info(f"Completed check at {self.corners[1]['name']} after {current_time - self.scanning_started_time:.1f} seconds")
                
                # Move to the next corner
                self.send_position_goal(self.corners[2]["x"], self.corners[2]["y"], self.corners[2]["angle"])
                self.current_state = self.STATE_CHECK_CORNER_3
                self.last_state_change_time = current_time
                self.goal_sent_time = current_time
                self.corner_check_started = False
                self.get_logger().info(f"No fire detected at {self.corners[1]['name']}, moving to {self.corners[2]['name']} at ({self.corners[2]['x']}, {self.corners[2]['y']})")

        elif self.current_state == self.STATE_CHECK_CORNER_3:
            # First wait for navigation to complete
            if not self.corner_check_started and current_time - self.goal_sent_time > 8.0:
                self.get_logger().info(f"Reached {self.corners[2]['name']}, checking for fire")
                self.corner_check_started = True
                self.scanning_started_time = current_time
            
            # Now check for fire detection for the corner wait time
            # Fire detection is handled in fire_detection_callback
            if self.corner_check_started and current_time - self.scanning_started_time > self.corner_wait_time:
                self.get_logger().info(f"Completed check at {self.corners[2]['name']} after {current_time - self.scanning_started_time:.1f} seconds")
                
                # If we get here, we've checked all corners and found no fire
                self.current_state = self.STATE_RESET
                self.last_state_change_time = current_time
                self.get_logger().warn("No fire detected in any position or corner! Ending search.")

        elif self.current_state == self.STATE_WAIT_AT_FIRE:
            # Wait at the fire location for specified time
            if current_time - self.last_state_change_time > self.fire_wait_time:
                self.get_logger().info(f"Waited at fire location for {self.fire_wait_time} seconds, now going to origin (0,0)")
                
                # Move to the next state
                self.current_state = self.STATE_GOTO_ORIGIN
                self.last_state_change_time = current_time
                self.origin_sent = False  # Reset flag
                
                # We'll send the navigation command in the next state
                self.get_logger().info("STATE CHANGED: Moving to GOTO_ORIGIN state")

        elif self.current_state == self.STATE_GOTO_ORIGIN:
            # If we haven't sent the origin goal yet, send it now
            if not self.origin_sent:
                # CRITICAL: Navigate to origin (0,0) instead of start position
                self.send_position_goal(0.0, 0.0, 0)
                self.origin_sent = True
                self.get_logger().info("?? Sent navigation command to origin (0,0)")
                self.goal_sent_time = current_time
            
            # Wait for origin navigation to complete (allow enough time)
            if current_time - self.goal_sent_time > 10.0:
                self.get_logger().info("? Reached origin position (0,0), fire detection sequence complete")
                self.goal_active = False
                self.is_running = False

        elif self.current_state == self.STATE_RESET:
            if current_time - self.last_state_change_time > 5:
                self.get_logger().warn("Fire detection scan complete - no fire found anywhere")
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
        
        # Send the goal multiple times to ensure it's received
        for i in range(3):
            self.goal_pub.publish(goal_msg)
            time.sleep(0.1)  # Short delay between publishes
            
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
            
            # Log the exact coordinates we're sending
            self.get_logger().info(f"?? IMMEDIATE NAVIGATION to fire at {fire_location_key}")
            self.get_logger().info(f"Goal position: x={goal_msg.pose.position.x:.2f}, y={goal_msg.pose.position.y:.2f}")
            
            # Publish the goal multiple times to ensure it's received
            for i in range(5):  # Increased to 5 times
                self.goal_pub.publish(goal_msg)
                time.sleep(0.1)  # Slightly longer delay
            
            # Update to make sure we won't rescan since we're navigating to fire
            self.waiting_for_rotation = False
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