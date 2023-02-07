import os
import numpy as np

from argparse import ArgumentParser
from mmdet3d.apis import inference_detector, init_model
import sensus
from sensus.utils.data_converter import pc2pc_object
from mmdet3d.utils import register_all_modules
from mmdetection3d import data, demo, configs, checkpoints


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

    # register all modules in mmdet3d into the registries
    register_all_modules()

    ## Substitute the following lines with the ros lidar message
    pcd_path = os.path.join(os.path.dirname(sensus.__path__[0]),
        args.pcd)
    with open(pcd_path, 'rb') as f:
        points = np.fromfile(f, dtype=np.float32, count=-1)
    ################################

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    pc_object, pc = pc2pc_object(points, model.cfg.test_pipeline)
    print(pc_object)

    # test with a single point cloud
    result, data = inference_detector(model, pc)
    print(result)


if __name__ == '__main__':
    main()



