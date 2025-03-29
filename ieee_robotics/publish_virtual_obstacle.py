import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import struct
import math

class VirtualObstaclePublisher(Node):
    def __init__(self):
        super().__init__('virtual_obstacle_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, '/virtual_obstacles', 10)
        self.publish_obstacle()

    def publish_obstacle(self):
        # Define the center and radius of the obstacle
        center_x = 1.3
        center_y = -0.25
        radius = 0.07
        num_points = 40  # Number of points to approximate the circle

        # Generate points along the circumference of the circle
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append([x, y, 0.0])

        # Create a PointCloud2 message
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'  # Frame ID of the costmap
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 12  # 3 fields * 4 bytes each
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True

        # Pack the points into binary format
        buffer = []
        for point in points:
            buffer.append(struct.pack('fff', *point))
        msg.data = b''.join(buffer)

        # Publish the message
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published virtual obstacle at (center_x={center_x}, center_y={center_y}, radius={radius})")

def main(args=None):
    rclpy.init(args=args)
    virtual_obstacle_publisher = VirtualObstaclePublisher()
    rclpy.spin_once(virtual_obstacle_publisher)  # Spin once to publish the obstacle
    virtual_obstacle_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()