import os
import rclpy
import numpy as np
import sensor_msgs
import std_msgs
import mmdet3d
import sensus
import json

from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from perception_interfaces.msg import (Detection, 
                                       ClassTypeHypothesis,
                                       BoundingBox3DHypothesis,
                                       VelocityHypothesis)

from mmdetection3d import checkpoints, configs
from sensus import configs as sensus_configs
from sensus.utils.data_converter import pc2pc_object
from mmdet3d.apis import inference_detector, init_model
from mmdet3d.utils import register_all_modules


class ObjectDetector(Node):
    ''' 
    ObjectDetector class

    Subscribes to a PointCloud2 topic and publishes a Detection message

    Parameters
    ----------
    None

    Attributes
    ----------
    cfg : dict
        Dictionary containing the configuration parameters
    pc_subscriber : rclpy.node.Node
        Subscriber to PointCloud2 topic
    detection_publisher : rclpy.node.Node
        Publisher to Detection topic
    model : mmdet3d.models.detectors
        MMDetection3D object detection model
    z_offset : float
        Height offset to increase the height of the point cloud
    detection_id : int
        Unique detection ID (incremental integer)
    
    Methods
    -------
    process_pointcloud(msg: sensor_msgs.msg.PointCloud2)
        Callback function for the PointCloud2 subscriber. It processes the point cloud and publishes a Detection message
    process_detections(detection_preds: mmdet3d.structures.det3d_data_sample.Det3DDataSample,
                          header: std_msgs.msg.Header)
        Processes the mmdetection3d model output and returns a Detection message
    create_detection_msg(bbox_pred: list, score_pred: float, label_pred: int)
        Creates and returns ClassTypeHypothesis, BoundingBox3DHypothesis, and VelocityHypothesis messages 

    '''


    def __init__(self):
        '''
        ObjectDetector class constructor
        
        Parameters
        ----------
        None

        Returns
        -------
        None      
        '''

        super().__init__('object_detector')
        config_path = os.path.join(sensus.__path__[0],
                                   'ros', 'src', 'ros_sensus', 'config', 'ros',
                                   'perception_manager_config.json')
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.pc_subscriber = self.create_subscription(
            PointCloud2,
            self.cfg['pointcloud_topic'],
            self.process_pointcloud,
            10)
        self.detection_publisher = self.create_publisher(Detection, self.cfg['detection_topic'], 10)

        # Register all mmdetection3d modules
        register_all_modules()
        # model_cfg = os.path.join(sensus_configs.__path__[0],
        #     'second/second_hv_secfpn_8xb6-80e_kitti-3d-3class-ros.py')
        model_cfg = os.path.join(configs.__path__[0],
            self.cfg['detection_model_cfg'])
        # checkpoint_path = os.path.join(checkpoints.__path__[0],
        #     'hv_second_secfpn_6x8_80e_kitti-3d-3class_20210831_022017-ae782e87.pth')
        checkpoint_path = os.path.join(checkpoints.__path__[0],
            self.cfg['detection_model_checkpoint'])
        device = self.cfg['device']
        self.model = init_model(model_cfg, checkpoint_path, device=device)
        self.z_offset = 4.0
        self.detection_id = 0


    def process_pointcloud(self, msg: sensor_msgs.msg.PointCloud2):
        '''
        Callback function for the PointCloud2 subscriber. Receives a PointCloud2 message, processes it, and publishes a Detection message

        Parameters
        ----------
        msg : sensor_msgs.msg.PointCloud2
            PointCloud2 message
        
        Returns
        -------
        None
        '''

        self.get_logger().info('Received PointCloud2 message')

        # Extract point cloud data
        pc_buffer = np.frombuffer(msg.data,
                            dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)],
                            count=msg.width*msg.height, offset=0)
        pc_ros = pc_buffer.view(dtype=np.float32).reshape(pc_buffer.shape[0], -1)
        # ! Increase the height of the point cloud (to make it similar to NuScenes dataset)
        pc_ros[:, 2] = pc_ros[:, 2] + self.z_offset

        pc_object_ros, _ = pc2pc_object(pc_ros.flatten(), self.model.cfg.test_pipeline)
        detection_preds, _ = inference_detector(self.model, pc_object_ros)

        detection = self.process_detections(detection_preds, msg.header)

        # Publish message
        self.detection_publisher.publish(detection)
        self.get_logger().info('Published BoundingBox3DArray message')


    def process_detections(self, detection_preds: mmdet3d.structures.det3d_data_sample.Det3DDataSample, header: std_msgs.msg.Header):
        '''
        Processes the mmdetection3d model output and returns a Detection message

        Parameters
        ----------
        detection_preds : mmdet3d.structures.det3d_data_sample.Det3DDataSample
            mmdetection3d model output
        header : std_msgs.msg.Header
            PointCloud2 message header
        
        Returns
        -------
        detection : Detection
            Detection message        
        '''

        # Define header for Detection message
        detection = Detection()
        detection.header = Header()
        detection.header.stamp = self.get_clock().now().to_msg()
        detection.header.frame_id = header.frame_id
        self.detection_id += 1
        detection.id = self.detection_id

        type_list = []
        bbox_list = []
        vel_list = []
        
        for i, bbox_pred in enumerate(detection_preds.pred_instances_3d.bboxes_3d.tensor.tolist()):
            type_hyp, bbox_hyp, vel_hyp = self.create_detection_msg(
                bbox_pred,
                detection_preds.pred_instances_3d.scores_3d[i].item(),
                detection_preds.pred_instances_3d.labels_3d[i].item()
            )
            type_list.append(type_hyp)
            bbox_list.append(bbox_hyp)
            vel_list.append(vel_hyp)

        detection.hypothesis.type_hypothesis = type_list
        detection.hypothesis.bbox_hypothesis = bbox_list
        detection.hypothesis.velocity_hypothesis = vel_list

        return detection

    
    def create_detection_msg(self, bbox_pred: list, score_pred: float, label_pred: int):
        '''
        Creates and returns a ClassTypeHypothesis, BoundingBox3DHypothesis, and VelocityHypothesis message

        Parameters
        ----------
        bbox_pred : list
            Bounding box prediction in format [x, y, z, w, l, h, yaw, vx, vy] 
        score_pred : float
            Score prediction
        label_pred : int
            Label prediction
        
        Returns
        -------
        type_hyp : ClassTypeHypothesis
            ClassTypeHypothesis message
        bbox_hyp : BoundingBox3DHypothesis
            BoundingBox3DHypothesis message
        vel_hyp : VelocityHypothesis
            VelocityHypothesis message     
        '''


        # Class Type hypothesis
        type_hyp = ClassTypeHypothesis()

        # * self.model.cfg.class_names in CenterPoint with NuScenes dataset
        # ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        #     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        type_hyp.type.type = int(self.cfg['type_converter'][self.model.cfg.class_names[label_pred]][0])
        type_hyp.type.class_detection = int(self.cfg['type_converter'][self.model.cfg.class_names[label_pred]][1])
        # ! Assume same score for all hypotheses
        type_hyp.score = score_pred


        # Bounding box hypothesis
        bbox_hyp = BoundingBox3DHypothesis()
        center = Pose()
        # ! Trick to restore the original height of the bounding box
        center.position = Point(x=bbox_pred[0],
                                y=bbox_pred[1],
                                z=bbox_pred[2] - self.z_offset + bbox_pred[5]/2)
        center.orientation = Quaternion(x=0.0,
                                        y=0.0,
                                        z=np.sin(bbox_pred[6]/2),
                                        w=np.cos(bbox_pred[6]/2))
        bbox_hyp.bounding_box.center.pose = center
        bbox_hyp.score = score_pred

        size = Vector3()
        size.x = bbox_pred[3]
        size.y = bbox_pred[4]
        size.z = bbox_pred[5]
        bbox_hyp.bounding_box.size = size


        # Velocity hypothesis
        vel_hyp = VelocityHypothesis()
        vel_hyp.velocity.x = bbox_pred[7]
        vel_hyp.velocity.y = bbox_pred[8]
        vel_hyp.velocity.z = 0.0
        vel_hyp.score = score_pred

        return type_hyp, bbox_hyp, vel_hyp



def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error('Error publishing Detection message: {}'.format(e))
        if e == KeyboardInterrupt:
            node.get_logger().info('Shutting down perception manager')
            node.destroy_node()
            rclpy.shutdown()



if __name__ == '__main__':
    main()
