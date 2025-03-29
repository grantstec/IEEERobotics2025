#!/usr/bin/env python3
"""
GPU-Accelerated Image Processor Node for ROS 2
This node leverages the Jetson Xavier NX's GPU to process images faster.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os

# Ensure environment is correctly set
os.environ['OPENCV_DNN_CUDA'] = 'ON'

class GPUImageProcessor(Node):
    """
    ROS 2 node that uses GPU acceleration for image processing tasks.
    This helps offload CPU work to the GPU on the Jetson Xavier NX.
    """
    
    def __init__(self):
        super().__init__('gpu_image_processor')
        
        # Declare parameters
        self.declare_parameter('use_gpu', True)  # Enable/disable GPU acceleration
        self.declare_parameter('process_scan', True)  # Process LiDAR scans
        self.declare_parameter('visualization', True)  # Enable visualizations
        
        # Get parameters
        self.use_gpu = self.get_parameter('use_gpu').value
        self.process_scan = self.get_parameter('process_scan').value
        self.visualization = self.get_parameter('visualization').value
        
        # Check if CUDA is available
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if self.use_gpu and not self.cuda_available:
            self.get_logger().warn("GPU acceleration requested but CUDA is not available. Falling back to CPU.")
            self.use_gpu = False
            
        # Log status
        if self.use_gpu:
            self.get_logger().info("Using GPU acceleration via CUDA")
        else:
            self.get_logger().info("Using CPU for processing")
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize filters and processing tools
        self.initialize_processing_tools()
        
        # Create publishers and subscribers for image processing
        if self.process_scan:
            self.scan_sub = self.create_subscription(
                LaserScan,
                '/scan',
                self.scan_callback,
                10
            )
            
            self.processed_scan_pub = self.create_publisher(
                LaserScan,
                '/processed_scan',
                10
            )
        
        # Visualization publishers
        if self.visualization:
            self.vis_pub = self.create_publisher(
                Image,
                '/processed_visualization',
                10
            )
            
        # Performance monitoring
        self.processing_times = []
        self.create_timer(5.0, self.report_performance)
        
        self.get_logger().info("GPU Image Processor initialized")
    
    def initialize_processing_tools(self):
        """Initialize processing tools for GPU or CPU usage."""
        if self.use_gpu and self.cuda_available:
            # For GPU processing
            self.gpu_stream = cv2.cuda.Stream()
            
            # No need to create a filter object in advance
            # We'll use cv2.cuda.GaussianFilter directly in the processing
            self.get_logger().info("Successfully initialized GPU processing")
        
        # Compute lookup tables for scan processing
        self.compute_trigonometry_tables()
    
    def compute_trigonometry_tables(self):
        """Pre-compute trigonometry tables for faster scan processing."""
        angles = np.linspace(0, 2*np.pi, 360)
        self.sin_table = np.sin(angles)
        self.cos_table = np.cos(angles)
    
    def scan_callback(self, msg):
        """Process LiDAR scan data using GPU acceleration if available."""
        start_time = time.time()
        
        # Convert scan to numpy array for processing
        ranges = np.array(msg.ranges, dtype=np.float32)
        
        # Replace infinities with max range
        ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)
        
        if self.use_gpu and self.cuda_available:
            # GPU-accelerated processing
            try:
                # Create CUDA arrays
                d_ranges = cv2.cuda_GpuMat()
                d_ranges.upload(ranges.reshape(-1, 1))
                
                # Apply GPU-based Gaussian filter (instead of bilateral filter)
                # Create the filter on-the-fly
                d_filtered = cv2.cuda.createGaussianFilter(
                    cv2.CV_32FC1, cv2.CV_32FC1, (5, 5), 1.5
                ).apply(d_ranges)
                
                # Download results
                ranges_filtered = d_filtered.download().flatten()
                
                # Use vectorized NumPy operations for coordinate conversion
                angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges_filtered))
                x_coords = ranges_filtered * np.cos(angles)
                y_coords = ranges_filtered * np.sin(angles)
                
            except cv2.error as e:
                self.get_logger().error(f"GPU processing error: {str(e)}")
                ranges_filtered = ranges  # Fall back to original data
                angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges_filtered))
                x_coords = ranges_filtered * np.cos(angles)
                y_coords = ranges_filtered * np.sin(angles)
        else:
            # CPU-based processing
            try:
                ranges_filtered = cv2.GaussianBlur(
                    ranges.reshape(-1, 1), (5, 5), 1.5
                ).flatten()
            except cv2.error as e:
                self.get_logger().error(f"CPU processing error: {str(e)}")
                ranges_filtered = ranges
                
            # Use vectorized NumPy operations for coordinate conversion
            angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges_filtered))
            x_coords = ranges_filtered * np.cos(angles)
            y_coords = ranges_filtered * np.sin(angles)
        
        # Create new scan message with filtered data
        processed_msg = LaserScan()
        processed_msg.header = msg.header
        processed_msg.angle_min = msg.angle_min
        processed_msg.angle_max = msg.angle_max
        processed_msg.angle_increment = msg.angle_increment
        processed_msg.time_increment = msg.time_increment
        processed_msg.scan_time = msg.scan_time
        processed_msg.range_min = msg.range_min
        processed_msg.range_max = msg.range_max
        processed_msg.ranges = ranges_filtered.tolist()
        
        # Publish processed scan
        self.processed_scan_pub.publish(processed_msg)
        
        # Visualize if enabled
        if self.visualization:
            self.visualize_scan(x_coords, y_coords, ranges_filtered)
        
        # Record processing time
        elapsed = time.time() - start_time
        self.processing_times.append(elapsed)
    
    def visualize_scan(self, x_coords, y_coords, ranges):
        """Create and publish a visualization of the processed scan."""
        try:
            # Create a visual representation of the scan
            resolution = 0.05  # 5cm per pixel
            map_size = 400     # 20m x 20m map
            center = map_size // 2
            
            if self.use_gpu and self.cuda_available:
                # GPU-based visualization (where possible)
                # We still need to use CPU for drawing as OpenCV drawing functions aren't GPU accelerated
                vis_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
                
                # Convert coordinates to pixel positions
                pixel_x = (x_coords / resolution).astype(int) + center
                pixel_y = (y_coords / resolution).astype(int) + center
                
                # Keep only points within the map
                valid_idx = (
                    (pixel_x >= 0) & (pixel_x < map_size) & 
                    (pixel_y >= 0) & (pixel_y < map_size)
                )
                
                # Draw points
                for x, y, r in zip(pixel_x[valid_idx], pixel_y[valid_idx], ranges[valid_idx]):
                    # Color based on distance (red: close, blue: far)
                    color_val = min(255, int(r * 25.5))  # Scale to 0-255
                    color = (color_val, 0, 255 - color_val)  # Red to blue gradient
                    cv2.circle(vis_map, (x, y), 2, color, -1)
                
                # Draw robot at center
                cv2.circle(vis_map, (center, center), 5, (0, 255, 0), -1)
                
                # Upload and apply final GPU processing
                d_vis_map = cv2.cuda_GpuMat()
                d_vis_map.upload(vis_map)
                
                # Apply Gaussian blur using GPU
                d_blurred = cv2.cuda.createGaussianFilter(
                    cv2.CV_8UC3, cv2.CV_8UC3, (3, 3), 1.5
                ).apply(d_vis_map)
                
                # Download the final image
                vis_map = d_blurred.download()
                
            else:
                # CPU-based visualization
                vis_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
                
                # Convert coordinates to pixel positions
                pixel_x = (x_coords / resolution).astype(int) + center
                pixel_y = (y_coords / resolution).astype(int) + center
                
                # Keep only points within the map
                valid_idx = (
                    (pixel_x >= 0) & (pixel_x < map_size) & 
                    (pixel_y >= 0) & (pixel_y < map_size)
                )
                
                # Draw points
                for x, y, r in zip(pixel_x[valid_idx], pixel_y[valid_idx], ranges[valid_idx]):
                    # Color based on distance (red: close, blue: far)
                    color_val = min(255, int(r * 25.5))  # Scale to 0-255
                    color = (color_val, 0, 255 - color_val)  # Red to blue gradient
                    cv2.circle(vis_map, (x, y), 2, color, -1)
                
                # Draw robot at center
                cv2.circle(vis_map, (center, center), 5, (0, 255, 0), -1)
                
                # Apply some visual enhancements
                vis_map = cv2.GaussianBlur(vis_map, (3, 3), 1.5)
            
            # Add some text with info
            cv2.putText(vis_map, f"GPU: {'On' if self.use_gpu else 'Off'}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            avg_time = np.mean(self.processing_times[-10:]) if self.processing_times else 0
            cv2.putText(vis_map, f"Proc: {avg_time*1000:.1f}ms", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert to ROS Image and publish
            img_msg = self.bridge.cv2_to_imgmsg(vis_map, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "map"
            self.vis_pub.publish(img_msg)
            
        except Exception as e:
            self.get_logger().error(f"Visualization error: {str(e)}")
    
    def report_performance(self):
        """Report processing performance statistics."""
        if not self.processing_times:
            return
            
        # Calculate statistics
        times = np.array(self.processing_times[-100:])  # Last 100 frames
        avg_time = np.mean(times) * 1000  # ms
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        fps = 1.0 / np.mean(times)
        
        self.get_logger().info(
            f"Performance Stats - "
            f"Avg: {avg_time:.2f}ms, Min: {min_time:.2f}ms, Max: {max_time:.2f}ms, FPS: {fps:.1f}, "
            f"GPU: {'Enabled' if self.use_gpu else 'Disabled'}"
        )
        
        # Clear old times to avoid memory buildup
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-100:]


def main(args=None):
    rclpy.init(args=args)
    node = GPUImageProcessor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()