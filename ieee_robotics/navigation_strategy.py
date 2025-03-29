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
    Enhanced navigation strategy with improved goal handling and state management
    """
    
    def __init__(self):
        super().__init__('competition_navigation_strategy')
        
        # Parameters with more robust defaults
        self.declare_parameter('goal_reached_distance', 0.1)  # Tighter tolerance
        self.declare_parameter('goal_orientation_tolerance', 0.2)
        self.declare_parameter('timeout_duration', 60.0)  # 60-second timeout
        self.declare_parameter('auto_return_to_start', True)
        self.declare_parameter('auto_end_round', True)
        
        # Get parameter values
        self.goal_reached_distance = self.get_parameter('goal_reached_distance').value
        self.goal_orientation_tolerance = self.get_parameter('goal_orientation_tolerance').value
        self.timeout_duration = self.get_parameter('timeout_duration').value
        self.auto_return_to_start = self.get_parameter('auto_return_to_start').value
        self.auto_end_round = self.get_parameter('auto_end_round').value
        
        # Enhanced state tracking
        self.current_goal = None
        self.goal_start_time = None
        self.goal_timeout_timer = None
        
        # State variables with more explicit tracking
        self.current_round = 1
        self.round_active = False
        self.goal_active = False
        self.goal_reached = False
        self.navigation_in_progress = False
        
        # Location tracking
        self.start_position = None
        self.current_position = None
        self.fire_location = None
        
        # Synchronization primitives
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
        
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )
        
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Create timer for goal monitoring
        self.goal_monitor_timer = self.create_timer(0.5, self.monitor_goal_progress)
        
        self.get_logger().info("Enhanced Navigation Strategy Initialized")
    
    def round_callback(self, msg):
        """Handle round updates with improved state management"""
        with self.nav_lock:
            self.current_round = msg.data
            self.reset_navigation_state()
            self.get_logger().info(f"Round updated to: {self.current_round}")
    
    def status_callback(self, msg):
        """Robust status handling"""
        try:
            status = json.loads(msg.data)
            with self.nav_lock:
                self.round_active = status.get('active', False)
                
                # Reset if round changes
                if status.get('round', self.current_round) != self.current_round:
                    self.reset_navigation_state()
                    self.current_round = status.get('round', self.current_round)
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse status message")
    
    def reset_navigation_state(self):
        """Comprehensive state reset"""
        self.navigation_in_progress = False
        self.goal_active = False
        self.goal_reached = False
        self.current_goal = None
        
        # Cancel any existing goal timeout timer
        if self.goal_timeout_timer:
            self.goal_timeout_timer.cancel()
            self.goal_timeout_timer = None
    
    def fire_location_callback(self, msg):
        """Updated fire location handling"""
        with self.nav_lock:
            self.fire_location = msg
            self.get_logger().info(f"Fire location updated: ({msg.pose.position.x}, {msg.pose.position.y})")
    
    def odom_callback(self, msg):
        """Store current position for goal tracking"""
        with self.nav_lock:
            self.current_position = msg.pose.pose
            
            # Store start position if not set
            if self.start_position is None:
                self.start_position = msg.pose.pose
    
    def monitor_goal_progress(self):
        """Enhanced goal monitoring with timeout"""
        with self.nav_lock:
            # Skip if no active goal or no current position
            if not self.goal_active or self.current_position is None or self.current_goal is None:
                return
            
            # Check for goal timeout
            if self.goal_start_time and time.time() - self.goal_start_time > self.timeout_duration:
                self.get_logger().warn(f"Goal timeout after {self.timeout_duration} seconds")
                self.reset_navigation_state()
                return
            
            # Check goal proximity
            goal_x = self.current_goal.pose.position.x
            goal_y = self.current_goal.pose.position.y
            current_x = self.current_position.position.x
            current_y = self.current_position.position.y
            
            # Compute distance to goal
            distance = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
            
            # Check if goal is reached
            if distance <= self.goal_reached_distance:
                self.get_logger().info(f"Goal reached! Distance: {distance}")
                self.goal_reached = True
                self.goal_active = False
                self.navigation_in_progress = False
                
                # Optional: automatic return or round progression logic here
                if self.auto_return_to_start:
                    self.navigate_to_start()
    
    def navigate_to_fire(self):
        """Navigate to fire location with robust goal handling"""
        with self.nav_lock:
            if not self.fire_location:
                self.get_logger().warn("No fire location available")
                return
            
            # Prepare goal
            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'
            goal.pose = self.fire_location.pose
            
            # Set goal tracking variables
            self.current_goal = goal
            self.goal_start_time = time.time()
            self.goal_active = True
            self.navigation_in_progress = True
            
            # Publish goal
            self.goal_pub.publish(goal)
            self.get_logger().info(f"Navigating to fire: ({goal.pose.position.x}, {goal.pose.position.y})")
    
    def navigate_to_start(self):
        """Return to start position with similar robust handling"""
        with self.nav_lock:
            if not self.start_position:
                self.get_logger().warn("No start position available")
                return
            
            # Prepare goal
            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'
            goal.pose = self.start_position
            
            # Set goal tracking variables
            self.current_goal = goal
            self.goal_start_time = time.time()
            self.goal_active = True
            self.navigation_in_progress = True
            
            # Publish goal
            self.goal_pub.publish(goal)
            self.get_logger().info(f"Returning to start: ({goal.pose.position.x}, {goal.pose.position.y})")
    
    def execute_strategy(self):
        """Round-specific navigation strategy"""
        with self.nav_lock:
            # Check preconditions
            if not self.round_active or self.navigation_in_progress:
                return
            
            # Round 1: Fire Detection (passive)
            if self.current_round == 1:
                pass  # Fire detection handled by state machine
            
            # Round 2: Navigate to Fire Location
            elif self.current_round == 2:
                if self.fire_location:
                    self.navigate_to_fire()
            
            # Round 3: Special Sequence (hook, navigation)
            elif self.current_round == 3:
                # Implement complex Round 3 sequence here
                pass

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