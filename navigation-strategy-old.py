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
        
        # Add switch state
        self.switch_activated = False
        
        # Add flags for round completion
        self.round1_destination_reached = False
        self.round2_destination_reached = False
        self.round3_destination_reached = False
        self.mission_complete = False
        self.hose_point_reached = False
        self.fire_nav_started = False
        
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
        
        elif not self.switch_activated and previous_state:
            self.get_logger().info("Start switch DEACTIVATED")
            
            # Stop any ongoing navigation when switch is deactivated
            with self.nav_lock:
                self.is_navigating = False
    
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
            
    def return_to_start_callback(self, msg):
        """Handle manual return-to-start commands"""
        if msg.data and self.start_position is not None:
            self.get_logger().info("Received manual command to return to start")
            # Set flags
            self.goal_reached = True
            self.returning_to_start = True
            # Navigate back to start
            self.navigate_to_start()
    
    def check_goal_status(self):
        """Check if we've reached the current goal"""
        # Skip if we're not navigating or no current position
        if not self.round_active or self.current_position is None:
            return
            
        # Skip if we've already determined we reached the goal
        if self.goal_reached and not self.returning_to_start:
            return
        
        # For Round 3, check if we've reached the hose location
        if self.current_round == 3 and self.hose_point_reached and not self.fire_nav_started and not self.goal_reached:
            # Calculate the distance to a point 4 feet in front of the start
            hose_x = self.start_position.position.x + 1.22  # 4 feet in meters
            hose_y = self.start_position.position.y
            
            dx = self.current_position.position.x - hose_x
            dy = self.current_position.position.y - hose_y
            distance_to_hose = math.sqrt(dx*dx + dy*dy)
            
            # If we've reached the hose point, proceed to fire
            if distance_to_hose <= self.goal_reached_distance:
                self.get_logger().info(f"Hose point reached! Distance: {distance_to_hose:.2f}m")
                self.get_logger().info("Pausing briefly at hose location...")
                time.sleep(2.0)  # Pause for hose latching
                
                # Now navigate to the fire
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
        """Execute Round 3 strategy: Navigate to hose release point, then to fire"""
        if not self.has_fire_location or self.fire_location is None:
            self.get_logger().warn("Round 3: No fire location available, cannot navigate")
            return
        
        # If we're already returning to start, don't interfere
        if self.returning_to_start:
            return
            
        # If we've already reached the fire, don't resend goals
        if self.goal_reached:
            return
            
        # If we're already navigating to fire, don't interfere
        if self.fire_nav_started:
            return
            
        # Only navigate to hose point if we haven't done it yet
        if not self.hose_point_reached:
            self.get_logger().info("Round 3: Navigating to hose release point")
            
            # Create the hose release point 4 feet (1.22 meters) in front of starting position
            if self.start_position is not None:
                hose_point = PoseStamped()
                hose_point.header.frame_id = "map"
                hose_point.header.stamp = self.get_clock().now().to_msg()
                
                # 4 feet straight ahead from starting position
                hose_point.pose.position.x = self.start_position.position.x + 1.22  # 4 feet in meters
                hose_point.pose.position.y = self.start_position.position.y
                hose_point.pose.position.z = self.start_position.position.z
                
                # Use the same orientation as the starting position
                hose_point.pose.orientation = self.start_position.orientation
                
                # Navigate to the hose release point
                self.navigate_to_point(hose_point)
                
                # Set flag to remember we've started navigation to the hose point
                self.hose_point_reached = True
    
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