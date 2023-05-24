point_cloud_range = [-100, -75, -10, 100, 75, 60]
class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=['Car', 'Pedestrian', 'Cyclist'])
dataset_type = 'KittiDataset'
data_root = '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/'
input_modality = dict(use_lidar=True, use_camera=True)
data_prefix = dict(pts='training/velodyne', img='training/image_2')
backend_args = None
use_dim = 4
load_dim = 4
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root=
            '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/',
            info_path=
            '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/kitti_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(Car=5, Cyclist=5, Pedestrian=5)),
            classes=['Car', 'Pedestrian', 'Cyclist'],
            sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                backend_args=None),
            backend_args=None)),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-100, -75, -10, 100, 75, 60]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-100, -75, -10, 100, 75, 60]),
    dict(type='ObjectNameFilter', classes=['Car', 'Pedestrian', 'Cyclist']),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D')
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='KittiDataset',
        data_root=
        '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/',
        ann_file='kitti_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='ObjectSample',
                db_sampler=dict(
                    data_root=
                    '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/',
                    info_path=
                    '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/kitti_dbinfos_train.pkl',
                    rate=1.0,
                    prepare=dict(
                        filter_by_difficulty=[-1],
                        filter_by_min_points=dict(
                            Car=5, Cyclist=5, Pedestrian=5)),
                    classes=['Car', 'Pedestrian', 'Cyclist'],
                    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
                    points_loader=dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=4,
                        use_dim=4,
                        backend_args=None),
                    backend_args=None)),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-100, -75, -10, 100, 75, 60]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-100, -75, -10, 100, 75, 60]),
            dict(
                type='ObjectNameFilter',
                classes=['Car', 'Pedestrian', 'Cyclist']),
            dict(type='PointShuffle'),
            dict(
                type='Pack3DDetInputs',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        metainfo=dict(classes=['Car', 'Pedestrian', 'Cyclist']),
        test_mode=False,
        data_prefix=dict(pts='training/velodyne', img='training/image_2'),
        box_type_3d='LiDAR',
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root=
        '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/',
        ann_file='kitti_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                backend_args=None),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D')
                ]),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        metainfo=dict(classes=['Car', 'Pedestrian', 'Cyclist']),
        modality=dict(use_lidar=True, use_camera=True),
        data_prefix=dict(pts='training/velodyne', img='training/image_2'),
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root=
        '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/',
        ann_file='kitti_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                backend_args=None),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D')
                ]),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        metainfo=dict(classes=['Car', 'Pedestrian', 'Cyclist']),
        modality=dict(use_lidar=True, use_camera=True),
        test_mode=True,
        data_prefix=dict(pts='training/velodyne', img='training/image_2'),
        box_type_3d='LiDAR',
        backend_args=None))
val_evaluator = dict(
    type='KittiMetric',
    ann_file=
    '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/kitti_infos_val.pkl',
    metric='bbox',
    backend_args=None)
test_evaluator = dict(
    type='KittiMetric',
    ann_file=
    '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/kitti_infos_val.pkl',
    metric='bbox',
    backend_args=None)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
voxel_size = [0.4, 0.4, 8]
model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            voxel_size=[0.4, 0.4, 8],
            max_voxels=(30000, 40000),
            point_cloud_range=[-100, -75, -10, 100, 75, 60])),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        legacy=False,
        point_cloud_range=[-100, -75, -10, 100, 75, 60]),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(512, 512)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=384,
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist'])
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-200, -150, -20, 200, 150, 50],
            max_num=500,
            score_threshold=0.3,
            out_size_factor=4,
            voxel_size=[0.4, 0.4],
            code_size=9,
            pc_range=[-100, -75]),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.4, 0.4, 8],
            out_size_factor=1,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            point_cloud_range=[-100, -75, -10, 100, 75, 60])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-200, -150, -20, 200, 150, 50],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.3,
            pc_range=[-100, -75],
            out_size_factor=4,
            voxel_size=[0.4, 0.4],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))
lr = 0.0001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=0.001,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=1e-08,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.8947368421052632,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=20)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False, base_batch_size=32)
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'work_dirs/main_config/epoch_20.pth'
resume = False
db_sampler = dict(
    data_root='/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/',
    info_path=
    '/home/javier/datasets/DAIR/single-infrastructure-side-mmdet/kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Cyclist=5, Pedestrian=5)),
    classes=['Car', 'Pedestrian', 'Cyclist'],
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    backend_args=None)
launcher = 'none'
work_dir = './work_dirs/main_config'
