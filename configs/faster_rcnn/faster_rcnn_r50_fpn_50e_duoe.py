_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/duoe_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
classes = ('holothurian', 'echinus', 'scallop', 'starfish')

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=4)))

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
# cfg_dict = './faster_rcnn_r50_fpn_1x_duor.py'
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=5)  # The save interval is 1
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='EvalHook', by_epoch=False),
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='MMDetWandbHook',# The Wandb logger is also supported, It requires `wandb` to be installed.
             interval=50,
             init_kwargs={'project': "DUO-Detection", # Project name in WandB
                          'name': 'faster_rcnn_r50_fpn_50e_duoe'},
             log_checkpoint=True,
             num_eval_images=100,
             bbox_score_thr=0.3,
             ),
    ])

# overwrite schedule
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[20, 48])
runner = dict(type='EpochBasedRunner', max_epochs=50)