import mmcv
import numpy as np
from mmengine import load

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import CameraInstance3DBoxes

def flip_horiz_roty(bboxes):
    x = np.pi - bboxes[:, 6]
    x = np.arctan2(np.sin(x), np.cos(x))
    bboxes[:, 6] = x
    return bboxes

# info_file = load('demo/data/kitti/000008.pkl')
info_file = load('data/DAIR-V2X/cooperative-vehicle-infrastructure-kittiformat/infrastructure-side/dair_infos_train.pkl')
cam2img = np.array(info_file['data_list'][230]['images']['CAM2']['cam2img'], dtype=np.float32)
bboxes_3d = []
for instance in info_file['data_list'][230]['instances']:
    bboxes_3d.append(instance['bbox_3d'])
gt_bboxes_3d = np.array(bboxes_3d, dtype=np.float32)
gt_bboxes_3d = flip_horiz_roty(gt_bboxes_3d)
gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d)
input_meta = {'cam2img': cam2img}

visualizer = Det3DLocalVisualizer()

# img = mmcv.imread('demo/data/kitti/000008.png')
img = mmcv.imread('data/DAIR-V2X/cooperative-vehicle-infrastructure-kittiformat/infrastructure-side/training/image_2/000288.jpg')
img = mmcv.imconvert(img, 'bgr', 'rgb')
visualizer.set_image(img)
# project 3D bboxes to image
visualizer.draw_proj_bboxes_3d(gt_bboxes_3d, input_meta)
visualizer.show()