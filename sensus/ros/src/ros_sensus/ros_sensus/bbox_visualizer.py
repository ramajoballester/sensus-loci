import rclpy
from rclpy.node import Node
from vision_msgs.msg import BoundingBox3D
from geometry_msgs.msg import Pose, Vector3
import random

class BoundingBox3DPublisher(Node):

    def __init__(self):
        super().__init__('bounding_box_publisher')
        self.publisher_ = self.create_publisher(BoundingBox3D, 'bounding_box_topic', 10)
        self.timer = self.create_timer(1, self.publish_bounding_box)

    def publish_bounding_box(self):
        msg = BoundingBox3D()
        msg.center.position.x = random.uniform(-10, 10)
        msg.center.position.y = random.uniform(-10, 10)
        msg.center.position.z = random.uniform(-10, 10)
        msg.size.x = random.uniform(0, 10)
        msg.size.y = random.uniform(0, 10)
        msg.size.z = random.uniform(0, 10)
        
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing bounding box:\n{}'.format(msg))

def main(args=None):
    rclpy.init(args=args)
    bounding_box_publisher = BoundingBox3DPublisher()
    rclpy.spin(bounding_box_publisher)
    bounding_box_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
