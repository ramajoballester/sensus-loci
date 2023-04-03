import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point

class BoundingBoxVisualizer(Node):

    def __init__(self):
        super().__init__('bounding_box_visualizer')
        self.publisher = self.create_publisher(MarkerArray, 'bounding_box_markers', 10)

    def create_marker(self, bbox):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = bbox[0]
        marker.pose.position.y = bbox[1]
        marker.pose.position.z = bbox[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = bbox[3]
        marker.scale.y = bbox[4]
        marker.scale.z = bbox[5]
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        return marker

    def publish_marker(self, bbox):
        marker_array = MarkerArray()
        marker_array.markers.append(self.create_marker(bbox))
        self.publisher.publish(marker_array)


def main():
    rclpy.init()
    node = BoundingBoxVisualizer()
    bbox = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0]
    node.publish_marker(bbox)
    rclpy.spin(node)



if __name__ == '__main__':
    main()