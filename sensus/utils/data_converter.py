import numpy as np
from mmdet3d.core.points import get_points_type
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile


def pc2pc_object(pc, pipeline):
    pipeline_dict = {**pipeline[0]}
    pipeline_dict.pop('type')
    pc_loader = LoadPointsFromFile(**pipeline_dict)

    pc = pc.reshape(-1, pc_loader.load_dim)
    pc = pc[:, pc_loader.use_dim]
    attribute_dims = None

    if pc_loader.shift_height:
        floor_height = np.percentile(pc[:, 2], 0.99)
        height = pc[:, 2] - floor_height
        pc = np.concatenate(
            [pc[:, :3],
                np.expand_dims(height, 1), pc[:, 3:]], 1)
        attribute_dims = dict(height=3)

    if pc_loader.use_color:
        assert len(pc_loader.use_dim) >= 6
        if attribute_dims is None:
            attribute_dims = dict()
        attribute_dims.update(
            dict(color=[
                pc.shape[1] - 3,
                pc.shape[1] - 2,
                pc.shape[1] - 1,
            ]))

    points_class = get_points_type(pc_loader.coord_type)
    pc_object = points_class(
        pc, points_dim=pc.shape[-1], attribute_dims=attribute_dims)

    return pc_object