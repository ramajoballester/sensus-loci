import rclpy
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from rclpy.node import Node
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np

import os
from mmdetection3d import checkpoints
from sensus import configs as sensus_configs
from sensus.utils.data_converter import pc2pc_object
from mmdet3d.apis import inference_detector, init_model
from mmdet3d.utils import register_all_modules

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')
        self.subscriber = self.create_subscription(
            PointCloud2,
            '/roundabout/velodyne_64_points',
            self.process_pointcloud,
            10)
        self.publisher = self.create_publisher(BoundingBox3DArray, '/detections', 10)

        register_all_modules()
        model_cfg = os.path.join(sensus_configs.__path__[0],
            'second/second_hv_secfpn_8xb6-80e_kitti-3d-3class-ros.py')
        checkpoint_path = os.path.join(checkpoints.__path__[0],
            'hv_second_secfpn_6x8_80e_kitti-3d-3class_20210831_022017-ae782e87.pth')
        device = 'cuda:0'
        self.model = init_model(model_cfg, checkpoint_path, device=device)
        self.z_offset = 4.0


    def process_pointcloud(self, msg):
        self.get_logger().info('Received PointCloud2 message')

        # Extract point cloud data
        pc_buffer = np.frombuffer(msg.data,
                            dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)],
                            count=msg.width*msg.height, offset=0)
        pc_ros = pc_buffer.view(dtype=np.float32).reshape(pc_buffer.shape[0], -1)
        # ! Trick to increase the height of the point cloud
        pc_ros[:, 2] = pc_ros[:, 2] + self.z_offset

        pc_object_ros, _ = pc2pc_object(pc_ros.flatten(), self.model.cfg.test_pipeline)
        result_ros, _ = inference_detector(self.model, pc_object_ros)

        self.publish_bounding_boxes(result_ros.pred_instances_3d.bboxes_3d.tensor.tolist())



    def publish_bounding_boxes(self, result_ros):
        # Define header for BoundingBox3DArray message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'velodyne_64'

        bboxes = []

        for result in result_ros:
            box = BoundingBox3D()

            # Set center pose
            center = Pose()
            center.position = Point(x=result[0], y=result[1], z=result[2] - self.z_offset + result[5]/2)
            center.orientation = Quaternion(x=0.0, y=0.0, z=np.sin(result[6]/2), w=np.cos(result[6]/2))
            box.center = center

            # Set size vector
            size = Vector3()
            size.x = result[3]
            size.y = result[4]
            size.z = result[5]
            box.size = size

            bboxes.append(box)

        # Create BoundingBox3DArray message
        bbox_array = BoundingBox3DArray()
        bbox_array.header = header
        bbox_array.boxes = bboxes

        # Publish message
        self.publisher.publish(bbox_array)
        self.get_logger().info('Published BoundingBox3DArray message')


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error('Error publishing BoundingBox3DArray message: {}'.format(e))

    rclpy.spin(node)
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()


if __name__ == '__main__':
    main()
