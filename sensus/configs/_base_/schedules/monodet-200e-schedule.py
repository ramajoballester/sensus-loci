
lr = 0.000225

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler
    # During the first 16 epochs, learning rate increases from 0 to lr * 10
    # during the next 24 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        eta_min=lr * 1e-4,
        begin=0,
        end=200,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    dict(
        type='CosineAnnealingMomentum',
        T_max=200,
        eta_min=1,
        begin=0,
        end=200,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Runtime settings，training schedule for 40e
# Although the max_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual max epoch
# number could be Nx40
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=20)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (6 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=48)
