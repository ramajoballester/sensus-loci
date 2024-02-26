import open3d as o3d
import mmcv
import numpy as np
from mmengine import load

from mmdet3d.visualization import Det3DLocalVisualizer

if __name__ == '__main__':
    # o3d.visualization.webrtc_server.enable_webrtc()
    # cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    # cube_red.compute_vertex_normals()
    # cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    # o3d.visualization.draw(cube_red)

    # info_file = load('mmdetection3d/demo/data/kitti/000008.pkl')
    # points = np.fromfile('mmdetection3d/demo/data/kitti/000008.bin', dtype=np.float32)
    info_file = load('sensus/data/DAIR-V2X/cooperative-vehicle-infrastructure-kittiformat/infrastructure-side/dair_infos_train.pkl')
    points = np.fromfile('sensus/data/DAIR-V2X/cooperative-vehicle-infrastructure-kittiformat/infrastructure-side/training/velodyne/000009.bin', dtype=np.float32)

    points = points.reshape(-1, 4)[:, :3]
    lidar2img = np.array(info_file['data_list'][0]['images']['CAM2']['lidar2img'], dtype=np.float32)

    visualizer = Det3DLocalVisualizer(save_dir='.')
    # img = mmcv.imread('mmdetection3d/demo/data/kitti/000008.png')
    img = mmcv.imread('sensus/data/DAIR-V2X/cooperative-vehicle-infrastructure-kittiformat/infrastructure-side/training/image_2/000009.jpg')
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.set_image(img)
    visualizer.draw_points_on_image(points, lidar2img)
    visualizer.show()
