_base_ = [
    '../_base_/datasets/dair-mono3d.py', '../_base_/models/smoke.py',
    '../_base_/default_runtime.py'
]

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='RandomShiftScale', shift_scale=(0.2, 0.4), aug_prob=0.3),
    dict(type='AffineResize', img_scale=(1920, 1080), down_ratio=4),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='AffineResize', img_scale=(1920, 1080), down_ratio=4),
    dict(type='Pack3DDetInputs', keys=['img'])
]
# img_scale=(1280, 384)

train_dataloader = dict(
    batch_size=4, num_workers=4, dataset=dict(pipeline=train_pipeline))
# batch = 8
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# training schedule for 6x
max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[50],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2.5e-4),
    clip_grad=None)

find_unused_parameters = True
