import rclpy
from sensor_msgs.msg import PointCloud2
import pickle


def main():
    rclpy.init()
    node = rclpy.create_node('PointCloudPublisher')
    try:
        with open('/home/breaststroker/alvaro/sensus-loci/sensus/data/lidar.pickle', 'rb') as f:
            data = pickle.load(f)
            publisher = node.create_publisher(PointCloud2, 'point_cloud', 10)
            data.header.stamp = node.get_clock().now().to_msg()
            data.header.frame_id = 'map'
            publisher.publish(data)
    except Exception as e:
        node.get_logger().error('Error publishing Point Cloud message: {}'.format(e))

    
    rclpy.spin(node)

    

if __name__ == '__main__':
    main()