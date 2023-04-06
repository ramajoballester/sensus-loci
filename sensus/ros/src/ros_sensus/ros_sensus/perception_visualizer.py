import rclpy
import os
import json
import sensus

from std_msgs.msg import Header
from rclpy.node import Node
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray
from perception_interfaces.msg import Detection

class Visualizer(Node):
    '''
    Visualizer class

    Subscribes to a Detection topic and publishes a BoundingBox3DArray message

    Parameters
    ----------
    None

    Attributes
    ----------
    cfg : dict
        Dictionary containing the configuration parameters
    detection_subscriber : rclpy.subscription.Subscription
        Subscriber to Detection topic. Set in cfg['detection_topic']
    viz_publisher : rclpy.publisher.Publisher
        Publisher to BoundingBox3DArray topic. Set in cfg['visualization_topic']
    
    Methods
    -------
    visualize_detections(msg: perception_interfaces.msg.Detection)
        Callback function for the Detection subscriber. It converts the 
        BoundinBox3DHypothesis from Detection to BoundingBox3DArray 
        and publishes it
    '''

    def __init__(self):
        '''
        Visualizer class constructor

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        super().__init__('object_visualizer')
        config_path = os.path.join(sensus.__path__[0],
                                   'ros', 'src', 'ros_sensus', 'config', 'ros',
                                   'perception_manager_config.json')
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.detection_subscriber = self.create_subscription(
            Detection,
            self.cfg['detection_topic'],
            self.visualize_detections, 10)
        self.viz_publisher = self.create_publisher(
            BoundingBox3DArray,
            self.cfg['visualization_topic'], 10)


    def visualize_detections(self, msg):
        '''
        Callback function for the Detection subscriber. It converts the
        BoundinBox3DHypothesis from Detection to BoundingBox3DArray 
        and publishes it

        Parameters
        ----------
        msg : perception_interfaces.msg.Detection
            Detection message
        
        Returns
        -------
        None
        '''

        # Create BoundingBox3DArray message
        bboxes = BoundingBox3DArray()
        bboxes.header = Header()
        bboxes.header.stamp = self.get_clock().now().to_msg()

        # Convert Detection ROS message to BoundingBox3DArray
        for i in range(len(msg.hypothesis.bbox_hypothesis)):
            bbox = BoundingBox3D()
            bbox.center = msg.hypothesis.bbox_hypothesis[i].bounding_box.center.pose
            bbox.size = msg.hypothesis.bbox_hypothesis[i].bounding_box.size
            bboxes.boxes.append(bbox)

        # Publish message
        self.viz_publisher.publish(bboxes)


def main():
    rclpy.init()
    node = Visualizer()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error('Error publishing BoundingBox3DArray message: {}'.format(e))
        if e == KeyboardInterrupt:
            node.get_logger().info('Shutting down perception visualizer')
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
