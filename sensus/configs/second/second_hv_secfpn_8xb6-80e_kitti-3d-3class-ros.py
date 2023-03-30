_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../../../mmdetection3d/configs/_base_/schedules/cyclic-40e.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py',
]
