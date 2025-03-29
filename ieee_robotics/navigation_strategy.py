#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import json
import math
import threading
import time

class CompetitionNavigationStrategy(Node):
    """
    Node to implement round-specific navigation strategies for the competition.
    Integrates with round_state_manager to handle different behaviors in each round.
    """
    
    def __init__(self):
        super().__init__('competition_navigation_strategy')
        
        # Parameters
        self.declare_parameter('goal_reached_distance', 0.05)  # Distance in meters to consider goal reached
        self.declare_parameter('auto_return_to_start', True)  # Automatically return to start after reaching fire
        self.declare_parameter('auto_end_round', True)      # Automatically end the round when back at start
        self.declare_parameter('backup_distance', 0.3)      # Distance to back up to hook in meters
        self.declare_parameter('backup_speed', 0.1)         # Speed for backing up in m/s
        self.declare_parameter('forward_distance', 0.5)     # Distance to move forward after latching in meters
        self.declare_parameter('forward_speed', 0.1)        # Speed for moving forward in m/s

        # Add position tracking variables for hook operations
        self.backup_start_position = None
        self.forward_start_position = None
        
        # State variables
        self.current_round = 1
        self.round_active = False
        self.is_navigating = False
        self.fire_location = None
        self.has_fire_location = False
        self.goal_reached = False
        self.returning_to_start = False
        self.start_position = None  # Will store the starting position
        self.current_position = None  # Will track current robot position
        self.goal_reached_distance = self.get_parameter('goal_reached_distance').value
        self.auto_return_to_start = self.get_parameter('auto_return_to_start').value
        self.auto_end_round = self.get_parameter('auto_end_round').value
        
        # Movement parameters
        self.backup_distance = self.get_parameter('backup_distance').value
        self.backup_speed = self.get_parameter('backup_speed').value
        self.forward_distance = self.get_parameter('forward_distance').value
        self.forward_speed = self.get_parameter('forward_speed').value
        
        # Add switch state
        self.switch_activated = False
        
        # Add flags for round completion
        self.round1_destination_reached = False
        self.round2_destination_reached = False
        self.round3_destination_reached = False
        self.mission_complete = False
        
        # Round 3 specific flags for the hook latching sequence
        self.hose_point_reached = False
        self.fire_nav_started = False
        self.round3_approach_hook_done = False
        self.round3_face_away_done = False
        self.round3_backing_up_done = False
        self.round3_moving_forward_done = False
        self.hook_position = None
        self.approach_position = None
        
        # For direct motion control
        self.start_motion_time = None
        self.is_backing_up = False
        self.is_moving_forward = False
        
        # Navigation control lock (for thread safety)
        self.nav_lock = threading.Lock()
        
        # Subscribers
        self.round_sub = self.create_subscription(
            Int32, 
            'competition/current_round', 
            self.round_callback, 
            10
        )
        
        self.status_sub = self.create_subscription(
            String,
            'competition/status',
            self.status_callback,
            10
        )
        
        self.fire_location_sub = self.create_subscription(
            PoseStamped,
            'competition/fire_location',
            self.fire_location_callback,
            10
        )
        
        # Add odometry subscription to track position
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )
        
        # Add subscription for manual return-to-start commands
        self.return_to_start_sub = self.create_subscription(
            Bool,
            'competition/return_to_start',
            self.return_to_start_callback,
            10
        )
        
        # Add switch subscriber
        self.switch_sub = self.create_subscription(
            Bool,
            'hardware/start_switch',
            self.switch_callback,
            10
        )
        
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.fire_detected_pub = self.create_publisher(Bool, 'fire_detection/active', 10)
        
        # Add publisher for ending rounds
        self.end_round_pub = self.create_publisher(String, 'competition/end_round', 10)
        
        # Timer for strategy execution
        self.strategy_timer = self.create_timer(1.0, self.execute_strategy)
        
        # Add timer for goal tracking
        self.goal_tracking_timer = self.create_timer(0.5, self.check_goal_status)
        
        # Add timer for direct motion control
        self.motion_control_timer = self.create_timer(0.1, self.motion_control_callback)
        
        # Initialize
        self.get_logger().info("Competition Navigation Strategy initialized")
        self.get_logger().info(f"Auto return to start: {'Enabled' if self.auto_return_to_start else 'Disabled'}")
        self.get_logger().info(f"Auto end round: {'Enabled' if self.auto_end_round else 'Disabled'}")
        self.get_logger().info(f"Goal reached distance: {self.goal_reached_distance}m")
        self.get_logger().info("Waiting for switch activation to begin navigation")
    
    def switch_callback(self, msg):
        """Handle switch state changes"""
        previous_state = self.switch_activated
        self.switch_activated = msg.data
        
        if self.switch_activated and not previous_state:
            self.get_logger().info(f"Start switch ACTIVATED in Round {self.current_round}")
            
            # Reset navigation state when switch is activated
            with self.nav_lock:
                self.is_navigating = False
                
                # Reset the round-specific completion flags when starting a new navigation
                if self.current_round == 1:
                    self.round1_destination_reached = False
                elif self.current_round == 2:
                    self.round2_destination_reached = False
                elif self.current_round == 3:
                    self.round3_destination_reached = False
                    self.hose_point_reached = False
                    self.fire_nav_started = False
                    self.round3_approach_hook_done = False
                    self.round3_face_away_done = False
                    self.round3_backing_up_done = False
                    self.round3_moving_forward_done = False
        
        elif not self.switch_activated and previous_state:
            self.get_logger().info("Start switch DEACTIVATED")
            
            # Stop any ongoing navigation when switch is deactivated
            with self.nav_lock:
                self.is_navigating = False
                
                # Stop any direct motion control
                if self.is_backing_up or self.is_moving_forward:
                    self.stop_robot()
    
    def round_callback(self, msg):
        """Handle round updates"""
        self.current_round = msg.data
        self.get_logger().info(f"Round updated to: {self.current_round}")
        
        # Reset state when round changes
        with self.nav_lock:
            self.is_navigating = False
            self.goal_reached = False
            self.returning_to_start = False
            self.mission_complete = False
            self.hose_point_reached = False
            self.fire_nav_started = False
            self.round3_approach_hook_done = False
            self.round3_face_away_done = False
            self.round3_backing_up_done = False 
            self.round3_moving_forward_done = False
            
            # Stop any direct motion control
            if self.is_backing_up or self.is_moving_forward:
                self.stop_robot()
                self.is_backing_up = False
                self.is_moving_forward = False
            
            # Reset the round-specific completion flags
            if self.current_round == 1:
                self.round1_destination_reached = False
            elif self.current_round == 2:
                self.round2_destination_reached = False
            elif self.current_round == 3:
                self.round3_destination_reached = False
            
            # Configure fire detection based on round
            self.configure_fire_detection(self.current_round)
    
    def status_callback(self, msg):
        """Process competition status updates"""
        try:
            status = json.loads(msg.data)
            self.round_active = status.get('active', False)
            self.has_fire_location = status.get('has_fire_location', False)
            
            if status.get('round') != self.current_round:
                self.current_round = status.get('round')
                self.get_logger().info(f"Round updated via status to: {self.current_round}")
                
                # Reset state when round changes
                with self.nav_lock:
                    self.is_navigating = False
                    self.goal_reached = False
                    self.returning_to_start = False
                    self.mission_complete = False
                    self.hose_point_reached = False
                    self.fire_nav_started = False
                    self.round3_approach_hook_done = False
                    self.round3_face_away_done = False
                    self.round3_backing_up_done = False
                    self.round3_moving_forward_done = False
                    
                    # Stop any direct motion control
                    if self.is_backing_up or self.is_moving_forward:
                        self.stop_robot()
                        self.is_backing_up = False
                        self.is_moving_forward = False
                    
                    # Reset the round-specific completion flags
                    if self.current_round == 1:
                        self.round1_destination_reached = False
                    elif self.current_round == 2:
                        self.round2_destination_reached = False
                    elif self.current_round == 3:
                        self.round3_destination_reached = False
                
                # Configure fire detection based on updated round
                self.configure_fire_detection(self.current_round)
            
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse status message")
    
    def fire_location_callback(self, msg):
        """Store fire location when received"""
        self.fire_location = msg
        self.has_fire_location = True
        self.get_logger().info(f"Received fire location at: ({msg.pose.position.x}, {msg.pose.position.y})")
    
    def odom_callback(self, msg):
        """Store current robot position for goal tracking"""
        self.current_position = msg.pose.pose
        
        # Store the start position if not already set
        if self.start_position is None:
            self.start_position = msg.pose.pose
            self.get_logger().info(f"Start position set: ({self.start_position.position.x:.2f}, {self.start_position.position.y:.2f})")
            
            # Calculate hook position when we know the start position
            if self.hook_position is None:
                self.calculate_hook_position()
            
    def return_to_start_callback(self, msg):
        """Handle manual return-to-start commands"""
        if msg.data and self.start_position is not None:
            self.get_logger().info("Received manual command to return to start")
            # Set flags
            self.goal_reached = True
            self.returning_to_start = True
            # Navigate back to start
            self.navigate_to_start()
    
    def calculate_hook_position(self):
        """Calculate the hook position based on the starting position"""
        if self.start_position is None:
            return
            
        # Hook is 46 inches (1.1684 meters) in +x direction and 4 inches (0.1016 meters) in +y from start
        hook_x = self.start_position.position.x + 1.1684
        hook_y = self.start_position.position.y + 0.1016
        hook_z = self.start_position.position.z
        
        # Store hook position
        self.hook_position = {
            'x': hook_x,
            'y': hook_y,
            'z': hook_z,
            'orientation': self.start_position.orientation  # Same orientation as start
        }
        
        # Calculate approach position - 1 foot in front of hook
        approach_x = hook_x - 0.3048  # 1 foot = 0.3048 meters
        approach_y = hook_y
        
        # Store approach position
        self.approach_position = {
            'x': approach_x,
            'y': approach_y,
            'z': hook_z,
            'orientation': self.start_position.orientation  # Same orientation as start
        }
        
        self.get_logger().info(f"Hook position set: ({hook_x:.2f}, {hook_y:.2f})")
        self.get_logger().info(f"Approach position set: ({approach_x:.2f}, {approach_y:.2f})")
    
    def check_goal_status(self):
        """Check if we've reached the current goal"""
        # Skip if we're not navigating or no current position
        if not self.round_active or self.current_position is None:
            return
            
        # Skip if we've already determined we reached the goal
        if self.goal_reached and not self.returning_to_start:
            return
        
        # For Round 3, handle the hook latching sequence
        if self.current_round == 3:
            # Make sure hook position is calculated
            if self.hook_position is None:
                self.calculate_hook_position()
                return
                
            # Handle approach to hook position
            if not self.round3_approach_hook_done and not self.is_backing_up and not self.is_moving_forward:
                # Check if we're at the approach position
                dx = self.current_position.position.x - self.approach_position['x']
                dy = self.current_position.position.y - self.approach_position['y']
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance <= self.goal_reached_distance:
                    self.get_logger().info(f"Approach position reached! Distance: {distance:.2f}m")
                    self.round3_approach_hook_done = True
                    
                    # Pause briefly to stabilize
                    time.sleep(1.0)
                    
                    # Now turn to face away from hook
                    self.turn_away_from_hook()
                    return
            
            # Handle turning to face away from hook
            elif self.round3_approach_hook_done and not self.round3_face_away_done and not self.is_backing_up and not self.is_moving_forward:
                # Check if we're facing 180 degrees from the start orientation
                # This is a simplified check - we should ideally compare quaternions
                # For now, we'll assume the turning action has completed
                # No need to check here - the goal pose will handle this
                
                # Start backing up once we've turned
                self.round3_face_away_done = True
                self.get_logger().info("Turned to face away from hook, starting to back up")
                
                # Pause briefly to stabilize
                time.sleep(1.0)
                
                # Start backing up
                self.back_up_to_hook()
                return
            
            # Handle moving forward after backing up
            elif self.round3_face_away_done and self.round3_backing_up_done and not self.round3_moving_forward_done and not self.is_backing_up and not self.is_moving_forward:
                self.get_logger().info("Hose should be latched, starting to move forward")
                
                # Start moving forward
                self.move_forward_from_hook()
                return
                
            # Handle moving to fire location
            elif self.round3_moving_forward_done and not self.fire_nav_started:
                self.get_logger().info("Hook latching complete, navigating to fire location")
                
                # Start navigating to fire
                self.fire_nav_started = True
                self.navigate_to_fire_location()
                return
        
        # Check distance to fire location (our goal)
        if self.has_fire_location and self.fire_location is not None and not self.returning_to_start:
            # Calculate distance to goal
            dx = self.current_position.position.x - self.fire_location.pose.position.x
            dy = self.current_position.position.y - self.fire_location.pose.position.y
            distance_to_goal = math.sqrt(dx*dx + dy*dy)
            
            # Check if we've reached the goal
            if distance_to_goal <= self.goal_reached_distance:
                if not self.goal_reached:
                    self.goal_reached = True
                    self.get_logger().info(f"Goal reached! Distance: {distance_to_goal:.2f}m")
                    
                    # Set round-specific destination reached flag
                    if self.current_round == 1:
                        self.round1_destination_reached = True
                        self.get_logger().info("Round 1 destination reached!")
                    elif self.current_round == 2:
                        self.round2_destination_reached = True
                        self.get_logger().info("Round 2 destination reached!")
                    elif self.current_round == 3:
                        self.round3_destination_reached = True
                        self.get_logger().info("Round 3 destination reached!")
                    
                    # If auto return is enabled, start returning to start in all rounds
                    if self.auto_return_to_start:
                        self.get_logger().info("Starting return to start position...")
                        # Wait briefly to let the robot stabilize
                        time.sleep(2.0)
                        self.returning_to_start = True
                        self.navigate_to_start()
        
        # Check if we've reached the start position (when returning)
        elif self.returning_to_start and self.start_position is not None:
            # Calculate distance to start
            dx = self.current_position.position.x - self.start_position.position.x
            dy = self.current_position.position.y - self.start_position.position.y
            distance_to_start = math.sqrt(dx*dx + dy*dy)
            
            # Check if we've reached the start
            if distance_to_start <= self.goal_reached_distance:
                self.returning_to_start = False
                self.get_logger().info(f"Returned to start! Distance: {distance_to_start:.2f}m")
                
                # Check if the round goal was achieved and we should end the round
                round_completed = False
                if self.current_round == 1 and self.round1_destination_reached:
                    round_completed = True
                elif self.current_round == 2 and self.round2_destination_reached:
                    round_completed = True
                elif self.current_round == 3 and self.round3_destination_reached:
                    round_completed = True
                
                if round_completed:
                    self.mission_complete = True
                    self.get_logger().info("Mission complete! All goals accomplished for this round.")
                    
                    # Auto-end the round if enabled
                    if self.auto_end_round:
                        self.end_current_round()
                
                # Reset for potential next round
                self.goal_reached = False
    
    def motion_control_callback(self):
        """Handle direct motion control for backing up and moving forward using odometry feedback"""
        # Only proceed if we need to handle direct motion
        if not self.is_backing_up and not self.is_moving_forward:
            return
            
        # Get current time
        current_time = time.time()
        
        # Handle backing up
        if self.is_backing_up and self.backup_start_position is not None:
            # Calculate elapsed time (for logging and minimum time check)
            elapsed_time = current_time - self.start_motion_time
            
            # Calculate actual distance traveled using odometry
            dx = self.current_position.position.x - self.backup_start_position['x']
            dy = self.current_position.position.y - self.backup_start_position['y']
            distance_traveled = math.sqrt(dx*dx + dy*dy)
            
            # We also want to ensure a minimum backup time for reliable latching
            min_backup_time = 3.0  # Minimum 3 seconds of backing up
            
            # If we've traveled far enough AND exceeded minimum time, stop backing up
            if distance_traveled >= self.backup_distance and elapsed_time >= min_backup_time:
                self.stop_robot()
                self.is_backing_up = False
                self.round3_backing_up_done = True
                self.get_logger().info(f"Finished backing up {distance_traveled:.2f}m after {elapsed_time:.2f}s")
                return
                    
            # Otherwise, continue backing up
            twist = Twist()
            twist.linear.x = -self.backup_speed  # Negative for backing up
            self.cmd_vel_pub.publish(twist)
        
        # Handle moving forward
        elif self.is_moving_forward and self.forward_start_position is not None:
            # Calculate elapsed time (for logging and minimum time check)
            elapsed_time = current_time - self.start_motion_time
            
            # Calculate actual distance traveled using odometry
            dx = self.current_position.position.x - self.forward_start_position['x']
            dy = self.current_position.position.y - self.forward_start_position['y']
            distance_traveled = math.sqrt(dx*dx + dy*dy)
            
            # We also want to ensure a minimum forward time
            min_forward_time = 2.0  # Minimum 2 seconds of moving forward
            
            # If we've traveled far enough AND exceeded minimum time, stop moving forward
            if distance_traveled >= self.forward_distance and elapsed_time >= min_forward_time:
                self.stop_robot()
                self.is_moving_forward = False
                self.round3_moving_forward_done = True
                self.get_logger().info(f"Finished moving forward {distance_traveled:.2f}m after {elapsed_time:.2f}s")
                return
                    
            # Otherwise, continue moving forward
            twist = Twist()
            twist.linear.x = self.forward_speed  # Positive for moving forward
            self.cmd_vel_pub.publish(twist)
    
    def stop_robot(self):
        """Stop the robot's motion"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Robot stopped")
    
    def back_up_to_hook(self):
        """Send velocity commands to back up into the hook using odometry feedback"""
        # Store starting position for distance calculation
        self.backup_start_position = {
            'x': self.current_position.position.x,
            'y': self.current_position.position.y
        }
        
        # Set up for backing up
        self.is_backing_up = True
        self.start_motion_time = time.time()
        
        # Start backing up
        twist = Twist()
        twist.linear.x = -self.backup_speed  # Negative for backing up
        self.cmd_vel_pub.publish(twist)
        
        self.get_logger().info(f"Starting to back up at {self.backup_speed}m/s for {self.backup_distance}m")
    
    def move_forward_from_hook(self):
        """Send velocity commands to move forward after latching using odometry feedback"""
        # Store starting position for distance calculation
        self.forward_start_position = {
            'x': self.current_position.position.x,
            'y': self.current_position.position.y
        }
        
        # Set up for moving forward
        self.is_moving_forward = True
        self.start_motion_time = time.time()
        
        # Start moving forward
        twist = Twist()
        twist.linear.x = self.forward_speed  # Positive for moving forward
        self.cmd_vel_pub.publish(twist)
        
        self.get_logger().info(f"Starting to move forward at {self.forward_speed}m/s for {self.forward_distance}m")
    
    def turn_away_from_hook(self):
        """Rotate 180 degrees to face away from the hook"""
        # Create a goal pose to turn 180 degrees
        turn_goal = PoseStamped()
        turn_goal.header.stamp = self.get_clock().now().to_msg()
        turn_goal.header.frame_id = "map"
        
        # Use the approach position coordinates
        turn_goal.pose.position.x = self.approach_position['x']
        turn_goal.pose.position.y = self.approach_position['y']
        turn_goal.pose.position.z = self.approach_position['z']
        
        # Calculate orientation 180 degrees from start orientation
        # This is a simplified approach - we should ideally use quaternion math
        # For now, we'll negate the z component and w component to rotate 180 degrees
        q = self.start_position.orientation
        turn_goal.pose.orientation.x = q.x
        turn_goal.pose.orientation.y = q.y
        turn_goal.pose.orientation.z = -q.z  # Negate z component
        turn_goal.pose.orientation.w = -q.w  # Negate w component
        
        # Publish the goal
        self.goal_pub.publish(turn_goal)
        self.get_logger().info("Turning to face away from hook")
    
    def approach_hook_position(self):
        """Navigate to a position 1 foot in front of the hook, facing it"""
        if self.approach_position is None:
            self.get_logger().warn("No approach position available")
            return
            
        # Create a goal pose for the approach position
        approach_goal = PoseStamped()
        approach_goal.header.stamp = self.get_clock().now().to_msg()
        approach_goal.header.frame_id = "map"
        
        # Use the calculated approach position
        approach_goal.pose.position.x = self.approach_position['x']
        approach_goal.pose.position.y = self.approach_position['y']
        approach_goal.pose.position.z = self.approach_position['z']
        
        # Use the starting orientation
        approach_goal.pose.orientation = self.start_position.orientation
        
        # Publish the goal
        self.goal_pub.publish(approach_goal)
        self.get_logger().info(f"Navigating to hook approach position: ({self.approach_position['x']:.2f}, {self.approach_position['y']:.2f})")
    
    def end_current_round(self):
        """Send signal to end the current round"""
        self.get_logger().info(f"Automatically ending Round {self.current_round}")
        
        # Create and publish end round message
        msg = String()
        msg.data = "end"
        self.end_round_pub.publish(msg)
        
        # Give time for the round state manager to process the end signal
        time.sleep(0.5)
    
    def configure_fire_detection(self, round_num):
        """Configure fire detection system based on current round"""
        fire_detection_active = False
        
        if round_num == 1:
            # In Round 1, we actively search for fire
            fire_detection_active = True
            self.get_logger().info("Round 1: Fire detection ENABLED")
        else:
            # In Rounds 2 and 3, we use saved fire location
            fire_detection_active = False
            self.get_logger().info(f"Round {round_num}: Fire detection DISABLED, using saved location")
        
        # Publish fire detection configuration
        msg = Bool()
        msg.data = fire_detection_active
        self.fire_detected_pub.publish(msg)
    
    def execute_strategy(self):
        """Execute the appropriate navigation strategy for the current round"""
        # Only proceed if the round is active AND the switch is activated
        if not self.round_active or not self.switch_activated:
            return
        
        # Use a lock to prevent race conditions
        with self.nav_lock:
            if self.is_navigating:
                # Already executing a navigation task
                return
            
            # Mark as navigating to prevent concurrent execution
            self.is_navigating = True
        
        try:
            # Execute round-specific strategy
            if self.current_round == 1:
                self.execute_round1_strategy()
            elif self.current_round == 2:
                self.execute_round2_strategy()
            elif self.current_round == 3:
                self.execute_round3_strategy()
            else:
                self.get_logger().warn(f"Unknown round: {self.current_round}")
        finally:
            # Reset navigation flag when done
            with self.nav_lock:
                self.is_navigating = False
    
    def execute_round1_strategy(self):
        """Execute Round 1 strategy: Let fire detection state machine handle it"""
        # In Round 1, fire detection is handled by the teensy_bridge and fire_detection_state_machine
        # The fire detection state machine will send goal poses when it detects fire
        # Nothing to do here - just wait for the fire location
        self.get_logger().debug("Round 1: Awaiting fire detection from the fire detection state machine")
    
    def navigate_to_start(self):
        """Navigate back to the starting position"""
        if self.start_position is None:
            self.get_logger().warn("No start position available to navigate to")
            return
        
        # Create a goal pose to return to start
        start_goal = PoseStamped()
        start_goal.header.stamp = self.get_clock().now().to_msg()
        start_goal.header.frame_id = "map"
        
        # Use the stored start position
        start_goal.pose.position.x = self.start_position.position.x
        start_goal.pose.position.y = self.start_position.position.y
        start_goal.pose.position.z = self.start_position.position.z
        
        # Use the original orientation (how the robot started)
        start_goal.pose.orientation = self.start_position.orientation
        
        # Publish the start position as a goal
        self.goal_pub.publish(start_goal)
        self.get_logger().info(f"Navigating back to start: ({start_goal.pose.position.x:.2f}, {start_goal.pose.position.y:.2f})")
        
    def execute_round2_strategy(self):
        """Execute Round 2 strategy: Navigate to fire location from Round 1"""
        if not self.has_fire_location or self.fire_location is None:
            self.get_logger().warn("Round 2: No fire location available, cannot navigate")
            return
        
        # Skip if we've already reached the goal
        if self.goal_reached:
            return
        
        self.get_logger().info("Executing Round 2 strategy: Navigating to saved fire location")
        
        # Simply publish the saved fire location directly to goal_pose
        # This will be handled by Nav2
        self.goal_pub.publish(self.fire_location)
    
    def execute_round3_strategy(self):
        """Execute Round 3 strategy: Navigate to hook, latch hose, navigate to fire"""
        if not self.has_fire_location or self.fire_location is None:
            self.get_logger().warn("Round 3: No fire location available, cannot navigate")
            return
        
        # If we're already returning to start, don't interfere
        if self.returning_to_start:
            return
            
        # If we've already reached the fire, don't resend goals
        if self.goal_reached:
            return
            
        # Make sure hook position is calculated
        if self.hook_position is None and self.start_position is not None:
            self.calculate_hook_position()
            return
        elif self.hook_position is None:
            self.get_logger().warn("No start position available, cannot calculate hook position")
            return
            
        # Handle each step of the Round 3 sequence
        
        # Step 1: Navigate to approach position (1 foot in front of hook)
        if not self.round3_approach_hook_done:
            self.get_logger().info("Round 3: Navigating to hook approach position")
            self.approach_hook_position()
            return
            
        # Step 2: Turn to face away from hook (handled in check_goal_status)
        # Step 3: Back up to hook (handled in check_goal_status)
        # Step 4: Move forward from hook (handled in check_goal_status)
        
        # Step 5: Navigate to fire location after hook latching is complete
        if self.round3_moving_forward_done and not self.fire_nav_started:
            self.get_logger().info("Round 3: Hook latching complete, navigating to fire location")
            self.navigate_to_fire_location()
    
    def navigate_to_fire_location(self):
        """Navigate to the saved fire location"""
        if self.fire_location is None:
            self.get_logger().warn("No fire location available to navigate to")
            return
        
        # Publish the fire location as a navigation goal
        self.goal_pub.publish(self.fire_location)
        self.get_logger().info(f"Navigating to fire at: ({self.fire_location.pose.position.x}, {self.fire_location.pose.position.y})")
    
    def navigate_to_point(self, pose):
        """Navigate to a specified pose"""
        if pose is None:
            self.get_logger().warn("Invalid navigation point")
            return
        
        # Publish the pose as a navigation goal
        self.goal_pub.publish(pose)
        self.get_logger().info(f"Navigating to point: ({pose.pose.position.x}, {pose.pose.position.y})")

def main(args=None):
    rclpy.init(args=args)
    node = CompetitionNavigationStrategy()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()