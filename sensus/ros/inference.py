import os
import numpy as np

from argparse import ArgumentParser
from mmdet3d.apis import inference_detector, init_model
from mmdet3d.core.points import get_points_type
from mmdetection3d import data, demo, configs, checkpoints
from sensus.utils.data_converter import pc2pc_object

# Supress warnings
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')

    args = parser.parse_args()

    ## Substitute the following lines with the ros lidar message
    pcd_path = os.path.join(demo.__path__[0],
        'data/kitti/kitti_000008.bin')
    # pcd_path = os.path.join(data.__path__[0],
    #     'nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin')
    
    with open(pcd_path, 'rb') as f:
        points = np.fromfile(f, dtype=np.float32, count=-1)
    ################################
    points = points.reshape((-1, 4))
    print(points.shape)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    points[:, 0] = x
    points[:, 1] = y
    points[:, 2] = z

    points.flatten()
    
    print('{:.3f}, {:.3f}, {:.3f}'.format(x.mean(), y.mean(), z.mean()))

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    pc_object = pc2pc_object(points, model.cfg.data.test.pipeline)

    # test with a single point cloud
    result, _ = inference_detector(model, pc_object)
    try:
        print(len(result[0]['boxes_3d']))
    except Exception as e:
        print(e)

    print(result)


if __name__ == '__main__':
    main()



