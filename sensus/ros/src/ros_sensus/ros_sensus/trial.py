import pickle
import os
import sensus

import open3d as o3d
import numpy as np

from sensus.utils.data_converter import pc2pc_object
from open3d.web_visualizer import draw
from mmdetection3d import data, demo, configs, checkpoints
from sensus import configs as sensus_configs

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.utils import register_all_modules



register_all_modules()


# build the model from a config file and a checkpoint file
# model_cfg = os.path.join(configs.__path__[0],
#     'second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py')
model_cfg = os.path.join(sensus_configs.__path__[0],
    'second/second_hv_secfpn_8xb6-80e_kitti-3d-3class-ros.py')
checkpoint_path = os.path.join(checkpoints.__path__[0],
    'hv_second_secfpn_6x8_80e_kitti-3d-3class_20210831_022017-ae782e87.pth')
pcd_path = os.path.join(demo.__path__[0],
    'data/kitti/000008.bin')
device = 'cuda:0'
model = init_model(model_cfg, checkpoint_path, device=device)