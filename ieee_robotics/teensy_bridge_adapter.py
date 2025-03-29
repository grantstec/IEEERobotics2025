#!/usr/bin/env python3
"""
Adapter node to enhance the existing teensy_bridge.py with round-specific behavior.
This node works alongside the existing teensy_bridge without modifying it directly.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String, Bool
from geometry_msgs.msg import PoseStamped
import json

class TeensyBridgeAdapter(Node):
    """
    Node that adds round-specific behavior to the teensy_bridge system.
    Disables fire detection in rounds 2 and 3, and manages goal sending.
    """
    
    def __init__(self):
        super().__init__('teensy_bridge_adapter')
        
        # State variables
        self.current_round = 1
        self.round_active = True
        self.bypass_teensy_fire_detection = False
        self.switch_activated = True
        
        # Parameter to control fire detection behavior
        self.declare_parameter('enable_fire_detection_round1', True)
        self.enable_fire_detection_round1 = self.get_parameter('enable_fire_detection_round1').value
        
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
        
        self.saved_fire_sub = self.create_subscription(
            PoseStamped,
            'competition/fire_location',
            self.saved_fire_callback,
            10
        )
        
        # Monitor goal_pose from teensy_bridge in Round 1 (for logging only)
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_pose_callback,
            10
        )

        # Monitor switch state
        self.switch_sub = self.create_subscription(
            Bool,
            'hardware/start_switch',
            self.switch_callback,
            10
        )
        
        # Publishers
        # This will connect to the 'goal_active' parameter in teensy_bridge.py
        # to enable/disable the fire detection state machine
        self.teensy_goal_active_pub = self.create_publisher(
            Bool, 
            'teensy/goal_active', 
            10
        )
        
        # Initialize by updating teensy's state
        # Only run this once at startup, not on a timer
        self.update_teensy_state()
        
        self.get_logger().info("Teensy Bridge Adapter initialized")
        self.get_logger().info(f"Fire detection in Round 1: {'ENABLED' if self.enable_fire_detection_round1 else 'DISABLED'}")
    
    def round_callback(self, msg):
        """Handle round updates"""
        new_round = msg.data
        
        if new_round != self.current_round:
            self.current_round = new_round
            self.get_logger().info(f"Round updated to: {self.current_round}")
            self.update_teensy_state()
    
    def status_callback(self, msg):
        """Process competition status updates"""
        try:
            status = json.loads(msg.data)
            self.round_active = status.get('active', False)
            
            # Update round if needed
            if status.get('round') != self.current_round:
                self.current_round = status.get('round')
                self.get_logger().info(f"Round updated via status to: {self.current_round}")
                self.update_teensy_state()
                
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse status message")
    
    def saved_fire_callback(self, msg):
        """
        Handle saved fire location data from round manager.
        This is not needed for core functionality but kept for monitoring.
        """
        if self.current_round > 1:
            self.get_logger().debug(f"Received saved fire location for Round {self.current_round}")
            
    def goal_pose_callback(self, msg):
        """
        Monitor goal poses from teensy_bridge for logging purposes.
        Helps track when teensy sends goals in Round 1.
        """
        if self.current_round == 1:
            self.get_logger().info(f"Received goal pose: ({msg.pose.position.x}, {msg.pose.position.y})")
    
    def switch_callback(self, msg):
        """Monitor hardware switch state"""
        self.switch_activated = msg.data
        
        # Update teensy state when switch changes
        if self.switch_activated:
            self.update_teensy_state()
    
    def update_teensy_state(self):
        """Update teensy_bridge's state based on current round"""
        # ENABLE fire detection in Round 1 - to match competition_navigation_strategy
        teensy_active = True  # Always enable for debugging
        
        # Create message to control teensy's goal_active parameter
        msg = Bool()
        msg.data = teensy_active
        
        # Publish to teensy - once, not repeatedly
        self.teensy_goal_active_pub.publish(msg)
        
        # Log less verbosely - only first time or when there's a change
        if not hasattr(self, 'last_state') or self.last_state != teensy_active:
            self.last_state = teensy_active
            if teensy_active:
                self.get_logger().info("Enabled fire detection for Round 1")
            else:
                self.get_logger().info(f"Disabled fire detection for Round {self.current_round}")

def main(args=None):
    rclpy.init(args=args)
    node = TeensyBridgeAdapter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()