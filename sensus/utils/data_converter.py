import numpy as np
from mmdet3d.structures.points import get_points_type
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile


def pc2pc_object(pc, pipeline):
    """
    Convert point cloud to PointCloud2 object :class:`sensus.tools.Dummy`

    Parameters
    ----------
    pc : np.ndarray
        Point cloud in numpy array format
    pipeline : list[dict]
        Pipeline configuration

    Returns
    -------
    pc_object : PointCloud2
        Point cloud in PointCloud2 object format
    pc : np.ndarray
        Point cloud in numpy array format    
    """
    
    pipeline_dict = {**pipeline[0]}
    assert pipeline_dict['type'] == 'LoadPointsFromFile'
    pipeline_dict.pop('type')
    pc_loader = LoadPointsFromFile(**pipeline_dict)

    # ROS point cloud has shape (n, 4) (x, y, z, intensity)
    pc = pc.reshape(-1, 4)
    # ! If the point cloud dimension is larger than 4, add zero channels
    # NuScenes point cloud dimension is 5 (x, y, z, intensity, ring_index)
    if pc_loader.load_dim > 4:
        pc = np.hstack((pc, np.zeros((pc.shape[0], pc_loader.load_dim - pc.shape[1]), dtype=pc.dtype)))

    pc = pc[:, pc_loader.use_dim]
    
    if pc_loader.norm_intensity:
        assert len(pc_loader.use_dim) >= 4, \
            f'When using intensity norm, expect used dimensions >= 4, got {len(pc_loader.use_dim)}'  # noqa: E501
        pc[:, 3] = np.tanh(pc[:, 3])
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

    return pc_object, pc