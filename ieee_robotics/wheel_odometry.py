import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class WheelOdometry(Node):
    def __init__(self):
        super().__init__('wheel_odometry')  
        # In Foxy, declare parameters individually
        self.declare_parameter('wheel_radius', 0.045)
        self.declare_parameter('base_width', 0.21)
        self.declare_parameter('steps_per_rev', 3200)
        
        # QoS profile for Foxy
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        
        # Subscriber
        self.subscription = self.create_subscription(
            JointState,
            'wheel_steps',
            self.step_callback,
            qos
        )
        
        # Publisher
        self.odom_pub = self.create_publisher(Odometry, 'wheel_odom', qos)
        
        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # State variables
        self.prev_left = 0
        self.prev_right = 0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self.get_logger().info("Wheel Odometry Node Started")

    def step_callback(self, msg):
        try:
            current_left = msg.position[msg.name.index('left_wheel')]
            current_right = msg.position[msg.name.index('right_wheel')]
        except ValueError:
            self.get_logger().warn("Invalid joint names in message")
            return

        # Calculate delta steps
        d_left = current_left - self.prev_left
        d_right = current_right - self.prev_right
        
        # Convert steps to meters - parameter access is different in Foxy
        wheel_circumference = 2 * math.pi * self.get_parameter('wheel_radius').value
        meters_per_step = wheel_circumference / self.get_parameter('steps_per_rev').value
        
        d_left_m = d_left * meters_per_step
        d_right_m = d_right * meters_per_step
        
        # Calculate odometry
        linear = (d_left_m + d_right_m) / 2
        angular = (d_right_m - d_left_m) / self.get_parameter('base_width').value
        
        self.x += linear * math.cos(self.theta)
        self.y += linear * math.sin(self.theta)
        self.theta += angular
        
        # Normalize theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Publish odometry
        self.publish_odometry(msg.header.stamp)
        self.publish_tf(msg.header.stamp)
        
        self.prev_left = current_left
        self.prev_right = current_right

    def publish_odometry(self, stamp):
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.orientation.z = math.sin(self.theta / 2)
        odom_msg.pose.pose.orientation.w = math.cos(self.theta / 2)
        
        # Pose covariance (adjust based on your system)
        odom_msg.pose.covariance = [
            0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.01, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.03
        ]
        
        self.odom_pub.publish(odom_msg)

    def publish_tf(self, stamp):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link'
        
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.rotation.z = math.sin(self.theta / 2)
        tf_msg.transform.rotation.w = math.cos(self.theta / 2)
        
        self.tf_broadcaster.sendTransform(tf_msg)

def main(args=None):
    rclpy.init(args=args)
    node = WheelOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()