import rclpy
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
import numpy as np


def publish_bounding_boxes(node):
    # Create publisher for BoundingBox3DArray
    publisher = node.create_publisher(BoundingBox3DArray, 'bounding_boxes', 10)

    # Define header for BoundingBox3DArray message
    header = Header()
    header.stamp = node.get_clock().now().to_msg()
    header.frame_id = 'map'

    # Create empty list to hold BoundingBox3D messages
    boxes = []

    # Generate random bounding boxes and append to boxes list
    result_ros = [[  1.6682, -37.9861,  -2.6241,   4.3703,   1.7161,   1.5333,   3.2115],
            [  0.4471,  23.5651,  -2.7133,   4.2763,   1.6689,   1.5844,  -1.5486],
            [ 66.7484, -37.9560,  -2.6043,   4.2618,   1.6901,   1.5347,   2.6354],
            [ 64.0491, -29.3706,  -2.6038,   4.5148,   1.6910,   1.5218,   4.3003],
            [ 59.8636,  37.4965,  -2.5408,   4.4287,   1.7143,   1.5376,   2.0700],
            [ 65.4579, -25.9227,  -2.3259,   4.1568,   1.6307,   1.4564,   2.8197],
            [  1.5115,  38.0660,  -2.7265,   4.1277,   1.6456,   1.5130,   3.0918],
            [  0.4457, -24.9556,  -2.4891,   4.4855,   1.6773,   1.5584,   1.5373],
            [ 66.8613, -20.3428,  -2.5316,   4.1281,   1.6367,   1.4983,   4.3373],
            [ 70.0768,  -8.6076,  -2.4863,   3.9550,   1.6057,   1.5017,   4.5547],
            [ 67.8440, -35.8827,  -2.4773,   4.2053,   1.6244,   1.4792,   2.7898],
            [ 63.2876, -33.6338,  -2.4852,   4.1945,   1.6562,   1.4963,   2.6599],
            [ 34.3747, -15.4040,  -0.8330,   4.2352,   1.6472,   1.5045,   2.8870],
            [ 58.9799, -23.8835,  -2.5292,   4.3683,   1.6999,   1.4977,   2.9430],
            [ 68.7469, -15.8206,  -2.5538,   4.5310,   1.6784,   1.5144,   4.4730],
            [  1.8188, -34.2937,  -2.6546,   4.2469,   1.6146,   1.4301,   3.2200],
            [ 60.7744,  26.4245,  -2.6090,   4.8026,   1.7432,   1.5551,   1.9572],
            [ 56.6543, -35.5691,  -2.3679,   4.1845,   1.6281,   1.4994,   2.5852]]

    for result in result_ros:
        box = BoundingBox3D()

        # Set center pose
        center = Pose()
        center.position = Point(x=result[0], y=result[1], z=result[2] - 6)
        center.orientation = Quaternion(x=0.0, y=0.0, z=np.sin(result[6]/2), w=np.cos(result[6]/2))
        box.center = center

        # Set size vector
        size = Vector3()
        size.x = result[3]
        size.y = result[4]
        size.z = result[5]
        box.size = size

        boxes.append(box)

    # Create BoundingBox3DArray message
    bbox_array = BoundingBox3DArray()
    bbox_array.header = header
    bbox_array.boxes = boxes

    # Publish message
    publisher.publish(bbox_array)

    node.get_logger().info('Published BoundingBox3DArray message')


def main():
    rclpy.init()
    node = rclpy.create_node('bounding_box_publisher')
    try:
        publish_bounding_boxes(node)
    except Exception as e:
        node.get_logger().error('Error publishing BoundingBox3DArray message: {}'.format(e))

    rclpy.spin(node)
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()


if __name__ == '__main__':
    main()
