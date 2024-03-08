_base_ = [
    '../_base_/models/second_hv_secfpn_dair-vehicle.py',
    '../_base_/datasets/dair-vehicle-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

train_cfg = dict(by_epoch=True, val_interval=2)
val_cfg = dict()
test_cfg = dict()