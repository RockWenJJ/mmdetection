_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/duor_detection.py',
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

checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=1)  # The save interval is 1
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
        # dict(type='MMDetWandbHook', by_epoch=False, # The Wandb logger is also supported, It requires `wandb` to be installed.
        #      init_kwargs={'entity': "wenjj", # The entity used to log on Wandb
        #                   'project': "DUO-Detection", # Project name in WandB
        #                  }), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
        # # MMDetWandbHook is mmdet implementation of WandbLoggerHook. ClearMLLoggerHook, DvcliveLoggerHook, MlflowLoggerHook, NeptuneLoggerHook, PaviLoggerHook, SegmindLoggerHook are also supported based on MMCV implementation.
    ])