#!/usr/bin/env python3
"""
System Monitor Node for Jetson Xavier NX
This node monitors CPU, GPU, and memory usage and publishes it to ROS topics.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import subprocess
import re
import threading
import time
import psutil
import os

class SystemMonitor(Node):
    """
    ROS 2 node that monitors system resources on the Jetson Xavier NX.
    """
    
    def __init__(self):
        super().__init__('system_monitor')
        
        # Create publishers
        self.cpu_pub = self.create_publisher(Float32MultiArray, '/system/cpu_usage', 10)
        self.mem_pub = self.create_publisher(Float32MultiArray, '/system/memory_usage', 10)
        self.gpu_pub = self.create_publisher(Float32MultiArray, '/system/gpu_usage', 10)
        self.system_info_pub = self.create_publisher(String, '/system/info', 10)
        
        # Create timers
        self.create_timer(1.0, self.publish_cpu_usage)
        self.create_timer(1.0, self.publish_memory_usage)
        self.create_timer(2.0, self.publish_gpu_usage)
        self.create_timer(10.0, self.publish_system_info)
        
        # Track if we're running on a Jetson
        self.is_jetson = os.path.exists('/sys/devices/platform/host1x')
        
        if self.is_jetson:
            self.get_logger().info("Running on Jetson platform")
        else:
            self.get_logger().info("Not running on Jetson platform, some metrics will be unavailable")
        
        # Start tegrastats in a separate thread if on Jetson
        self.tegrastats_data = {}
        if self.is_jetson:
            self.tegrastats_thread = threading.Thread(target=self.run_tegrastats)
            self.tegrastats_thread.daemon = True
            self.tegrastats_thread.start()
    
    def run_tegrastats(self):
        """
        Run tegrastats command in a background thread and parse its output.
        This provides detailed GPU, CPU, and memory usage for Jetson devices.
        """
        try:
            process = subprocess.Popen(['sudo', 'tegrastats'], stdout=subprocess.PIPE, text=True)
            
            while True:
                output = process.stdout.readline()
                if output:
                    # Parse GR3D (GPU) usage
                    gpu_match = re.search(r'GR3D_FREQ (\d+)%', output)
                    if gpu_match:
                        self.tegrastats_data['gpu'] = int(gpu_match.group(1))
                    
                    # Parse RAM usage
                    ram_match = re.search(r'RAM (\d+)/(\d+)MB', output)
                    if ram_match:
                        used = int(ram_match.group(1))
                        total = int(ram_match.group(2))
                        self.tegrastats_data['ram_used'] = used
                        self.tegrastats_data['ram_total'] = total
                    
                    # Parse CPU usage
                    cpu_matches = re.findall(r'CPU\[\d+\] (\d+)%', output)
                    if cpu_matches:
                        self.tegrastats_data['cpu'] = [int(usage) for usage in cpu_matches]
                
                time.sleep(0.5)  # Don't hog the CPU
                
        except Exception as e:
            self.get_logger().error(f"Tegrastats error: {str(e)}")
    
    def publish_cpu_usage(self):
        """Publish CPU usage information."""
        msg = Float32MultiArray()
        
        if self.is_jetson and 'cpu' in self.tegrastats_data:
            # Use tegrastats data for Jetson
            msg.data = [float(cpu) for cpu in self.tegrastats_data['cpu']]
        else:
            # Use psutil for other platforms
            cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            msg.data = [float(cpu) for cpu in cpu_percent]
        
        self.cpu_pub.publish(msg)
        
        # Log if any CPU core is above 80%
        if any(cpu > 80.0 for cpu in msg.data):
            self.get_logger().warn(f"High CPU usage detected: {max(msg.data):.1f}%")
    
    def publish_memory_usage(self):
        """Publish memory usage information."""
        msg = Float32MultiArray()
        
        if self.is_jetson and 'ram_used' in self.tegrastats_data and 'ram_total' in self.tegrastats_data:
            # Use tegrastats data for Jetson
            used = self.tegrastats_data['ram_used']
            total = self.tegrastats_data['ram_total']
            percent = (used / total) * 100.0
            msg.data = [float(used), float(total), float(percent)]
        else:
            # Use psutil for other platforms
            memory = psutil.virtual_memory()
            msg.data = [float(memory.used / 1024 / 1024), float(memory.total / 1024 / 1024), float(memory.percent)]
        
        self.mem_pub.publish(msg)
        
        # Log if memory usage is above 80%
        if msg.data[2] > 80.0:
            self.get_logger().warn(f"High memory usage detected: {msg.data[2]:.1f}%")
    
    def publish_gpu_usage(self):
        """Publish GPU usage information."""
        msg = Float32MultiArray()
        
        if self.is_jetson and 'gpu' in self.tegrastats_data:
            # Use tegrastats data for Jetson
            gpu_usage = float(self.tegrastats_data['gpu'])
            msg.data = [gpu_usage]
            
            # Log GPU usage for visibility
            self.get_logger().info(f"GPU Usage: {gpu_usage:.1f}%")
        else:
            # Not on Jetson, no GPU data available
            msg.data = [0.0]
        
        self.gpu_pub.publish(msg)
    
    def publish_system_info(self):
        """Publish general system information."""
        try:
            info_str = "System Info:\n"
            
            # Get CPU temperature
            if self.is_jetson:
                try:
                    with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
                        temp = float(f.read().strip()) / 1000
                        info_str += f"CPU Temperature: {temp:.1f}°C\n"
                except:
                    info_str += "CPU Temperature: N/A\n"
            else:
                info_str += "CPU Temperature: N/A (Not on Jetson)\n"
            
            # Get uptime
            uptime_seconds = float(open('/proc/uptime').read().split()[0])
            days, remainder = divmod(uptime_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            info_str += f"System Uptime: {int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            
            # Get load average
            load1, load5, load15 = os.getloadavg()
            info_str += f"Load Average: {load1:.2f}, {load5:.2f}, {load15:.2f}\n"
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            info_str += f"Disk Usage: {disk.percent}% ({disk.used / (1024**3):.1f}GB/{disk.total / (1024**3):.1f}GB)\n"
            
            # ROS 2 specific info
            info_str += f"ROS Distro: {os.environ.get('ROS_DISTRO', 'unknown')}\n"
            info_str += f"RMW Implementation: {os.environ.get('RMW_IMPLEMENTATION', 'unknown')}\n"
            
            # Publish the info
            msg = String()
            msg.data = info_str
            self.system_info_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing system info: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = SystemMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()