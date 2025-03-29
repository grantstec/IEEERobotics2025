#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
from geometry_msgs.msg import PoseStamped, Point
import os
import json
import yaml
from pathlib import Path
import threading
import time

class RoundStateManager(Node):
    """
    Node to track competition round state and manage fire location data between rounds.
    
    Functionality:
    - Keeps track of the current competition round
    - Saves fire location from Round 1
    - Provides the stored fire location for Round 2
    - Manages transitions between rounds
    """
    
    def __init__(self):
        super().__init__('round_state_manager')
        
        # Parameters
        self.declare_parameter('data_dir', str(Path.home() / 'competition_data'))
        self.declare_parameter('initial_round', 1)
        self.declare_parameter('enable_manual_control', True)
        
        # Get parameters
        self.data_dir = Path(self.get_parameter('data_dir').value)
        self.current_round = self.get_parameter('initial_round').value
        self.enable_manual_control = self.get_parameter('enable_manual_control').value
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # File paths
        self.state_file = self.data_dir / 'round_state.json'
        self.fire_location_file = self.data_dir / 'fire_location.yaml'
        
        # State variables
        self.fire_location = None
        self.round_active = False
        self.fire_detected = False
        
        # Load previous state if it exists
        self.load_state()
        
        # Publishers
        self.round_pub = self.create_publisher(Int32, 'competition/current_round', 10)
        self.status_pub = self.create_publisher(String, 'competition/status', 10)
        self.fire_location_pub = self.create_publisher(PoseStamped, 'competition/fire_location', 10)
        
        # Subscribers
        self.round_sub = self.create_subscription(
            Int32, 
            'competition/set_round', 
            self.round_callback, 
            10
        )
        
        # Subscribe directly to goal_pose from teensy_bridge in Round 1
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_pose_callback,
            10
        )
        
        # Keep this as backup for other fire detection methods
        self.fire_detection_sub = self.create_subscription(
            PoseStamped,
            'fire_detection/location',
            self.fire_detection_callback,
            10
        )
        
        self.start_round_sub = self.create_subscription(
            String,
            'competition/start_round',
            self.start_round_callback,
            10
        )
        
        self.end_round_sub = self.create_subscription(
            String,
            'competition/end_round',
            self.end_round_callback,
            10
        )
        
        # Timer for periodic status updates
        self.create_timer(1.0, self.publish_status)
        
        # Initialize current round state
        self.update_round_state(self.current_round)
        
        self.get_logger().info(f"Round State Manager initialized in Round {self.current_round}")
        self.get_logger().info(f"Data directory: {self.data_dir}")
        
        if self.fire_location:
            self.get_logger().info(f"Loaded saved fire location: {self.fire_location}")
        else:
            self.get_logger().info("No saved fire location found")
        
        # Display controls if manual control is enabled
        if self.enable_manual_control:
            self.get_logger().info("\n" + "-"*50 + 
                "\nManual Control Commands (via 'ros2 topic pub'):" +
                "\n- Set round: 'competition/set_round' (Int32)" +
                "\n- Start round: 'competition/start_round' (String, message: 'start')" +
                "\n- End round: 'competition/end_round' (String, message: 'end')" +
                "\n" + "-"*50)
    
    def load_state(self):
        """Load saved state and fire location from files"""
        # Load round state
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_round = state.get('current_round', self.current_round)
                    self.get_logger().info(f"Loaded round state: {self.current_round}")
            except Exception as e:
                self.get_logger().error(f"Error loading round state: {str(e)}")
        
        # Load fire location
        if self.fire_location_file.exists():
            try:
                with open(self.fire_location_file, 'r') as f:
                    fire_data = yaml.safe_load(f)
                    if fire_data and 'fire_location' in fire_data:
                        self.fire_location = fire_data['fire_location']
                        self.get_logger().info(f"Loaded fire location: {self.fire_location}")
            except Exception as e:
                self.get_logger().error(f"Error loading fire location: {str(e)}")
    
    def save_state(self):
        """Save current round state to file"""
        try:
            state = {
                'current_round': self.current_round,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            self.get_logger().info(f"Saved round state: {self.current_round}")
        except Exception as e:
            self.get_logger().error(f"Error saving round state: {str(e)}")
    
    def save_fire_location(self, location):
        """Save fire location to file"""
        try:
            # Format location for saving
            fire_data = {
                'fire_location': {
                    'x': location['x'],
                    'y': location['y'],
                    'z': location['z'],
                    'orientation': location['orientation'],
                    'detected_in_round': self.current_round,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            with open(self.fire_location_file, 'w') as f:
                yaml.dump(fire_data, f, default_flow_style=False)
                
            self.get_logger().info(f"Saved fire location: {location}")
        except Exception as e:
            self.get_logger().error(f"Error saving fire location: {str(e)}")
    
    def round_callback(self, msg):
        """Handle round change requests"""
        requested_round = msg.data
        
        if requested_round < 1 or requested_round > 3:
            self.get_logger().error(f"Invalid round number: {requested_round}. Must be 1-3.")
            return
        
        if requested_round != self.current_round:
            self.get_logger().info(f"Changing from Round {self.current_round} to Round {requested_round}")
            self.update_round_state(requested_round)
    
    def start_round_callback(self, msg):
        """Handle round start signal"""
        if msg.data.lower() == "start":
            self.round_active = True
            self.get_logger().info(f"Round {self.current_round} started")
            
            # Reset fire detection for Round 1
            if self.current_round == 1:
                self.fire_detected = False
            
            # Publish the current status
            self.publish_status()
            
            # If in Round 2 and we have a saved fire location, publish it
            if self.current_round == 2 and self.fire_location:
                self.publish_fire_location()
    
    def end_round_callback(self, msg):
        """Handle round end signal"""
        if msg.data.lower() == "end":
            self.round_active = False
            self.get_logger().info(f"Round {self.current_round} ended")
            
            # Auto-increment round if not manually controlled
            if not self.enable_manual_control and self.current_round < 3:
                next_round = self.current_round + 1
                self.get_logger().info(f"Auto-advancing to Round {next_round}")
                self.update_round_state(next_round)
            
            # Publish the current status
            self.publish_status()
    
    def goal_pose_callback(self, msg):
        """Handle goal poses from teensy_bridge (Round 1)"""
        # Only process goal poses in Round 1
        if self.current_round != 1 or not self.round_active:
            return
        
        if not self.fire_detected:  # Only save the first detection
            # Extract location coordinates from the message
            location = {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z,
                'orientation': {
                    'x': msg.pose.orientation.x,
                    'y': msg.pose.orientation.y,
                    'z': msg.pose.orientation.z,
                    'w': msg.pose.orientation.w
                }
            }
            
            self.fire_location = location
            self.fire_detected = True
            
            # Save the fire location to file
            self.save_fire_location(location)
            
            self.get_logger().info(f"Fire location received from teensy_bridge: {location}")
            
            # Acknowledge detection with status update
            self.publish_status()
    
    def fire_detection_callback(self, msg):
        """Handle fire detection results from other sources"""
        # Only process fire detections in Round 1
        if self.current_round != 1 or not self.round_active:
            return
        
        if not self.fire_detected:  # Only save the first detection
            # Extract location coordinates from the message
            location = {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z,
                'orientation': {
                    'x': msg.pose.orientation.x,
                    'y': msg.pose.orientation.y,
                    'z': msg.pose.orientation.z,
                    'w': msg.pose.orientation.w
                }
            }
            
            self.fire_location = location
            self.fire_detected = True
            
            # Save the fire location to file
            self.save_fire_location(location)
            
            self.get_logger().info(f"Fire detected and location saved: {location}")
            
            # Acknowledge detection with status update
            self.publish_status()
    
    def update_round_state(self, new_round):
        """Update the round state and save it"""
        self.current_round = new_round
        self.round_active = False  # Reset to inactive when changing rounds
        
        # Reset fire detection flag if entering Round 1
        if new_round == 1:
            self.fire_detected = False
        
        # Save the updated state
        self.save_state()
        
        # Publish the new round
        round_msg = Int32()
        round_msg.data = self.current_round
        self.round_pub.publish(round_msg)
        
        # Update status
        self.publish_status()
    
    def publish_status(self):
        """Publish current competition status"""
        status_info = {
            'round': self.current_round,
            'active': self.round_active,
            'fire_detected': self.fire_detected,
            'has_fire_location': self.fire_location is not None,
            'timestamp': self.get_clock().now().to_msg().sec
        }
        
        # Create status message
        status_msg = String()
        status_msg.data = json.dumps(status_info)
        self.status_pub.publish(status_msg)
        
        # Also publish current round
        round_msg = Int32()
        round_msg.data = self.current_round
        self.round_pub.publish(round_msg)
    
    def publish_fire_location(self):
        """Publish the saved fire location for navigation"""
        if not self.fire_location:
            self.get_logger().warn("No fire location available to publish")
            return
        
        # Create pose message from saved location
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"  # Assuming location is in map frame
        
        pose_msg.pose.position.x = self.fire_location['x']
        pose_msg.pose.position.y = self.fire_location['y']
        pose_msg.pose.position.z = self.fire_location['z']
        
        # Use saved orientation if available
        if 'orientation' in self.fire_location:
            pose_msg.pose.orientation.x = self.fire_location['orientation']['x']
            pose_msg.pose.orientation.y = self.fire_location['orientation']['y']
            pose_msg.pose.orientation.z = self.fire_location['orientation']['z']
            pose_msg.pose.orientation.w = self.fire_location['orientation']['w']
        else:
            # Default orientation (facing the fire)
            pose_msg.pose.orientation.w = 1.0
        
        # Publish the fire location
        self.fire_location_pub.publish(pose_msg)
        self.get_logger().info(f"Published fire location: {self.fire_location}")

def main(args=None):
    rclpy.init(args=args)
    node = RoundStateManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
