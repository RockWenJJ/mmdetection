_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/duor_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
classes = ('holothurian', 'echinus', 'scallop', 'starfish')

model = dict(
    backbone=dict(
        deep_stem=False),
    roi_head=dict(
        bbox_head=dict(num_classes=4)))

# data = dict(
#     train=dict(classes=classes),
#     val=dict(classes=classes),
#     test=dict(classes=classes))
# cfg_dict = './faster_rcnn_r50_fpn_1x_duor.py'
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=5)  # The save interval is 1
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='EvalHook', by_epoch=False),
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='MMDetWandbHook',  # The Wandb logger is also supported, It requires `wandb` to be installed.
             interval=50,
             init_kwargs={'project': "DuoDetGray",  # Project name in WandB
                          'name': 'duor_gray'},
             log_checkpoint=False,
             num_eval_images=100,
             bbox_score_thr=0.3,
             ),
    ])

# overwrite schedule
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# default decay ratio: gamma:0.1, min_lr: None
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[150, 180])
runner = dict(type='EpochBasedRunner', max_epochs=200)


# overwrite dataset config
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/duo_resized_gray/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (512, 288)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomCrop',
         crop_type='absolute',
         crop_size=crop_size,
         recompute_bbox=True,
         allow_negative_crop=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=crop_size, keep_ratio=False),
            # dict(type='CenterCrop',
            #      crop_type='absolute',
            #      crop_size=crop_size,
            #      recompute_bbox=True,
            #      allow_negative_crop=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_test_crop256x256.json',
        img_prefix=data_root + 'images/test_crop256x256/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_test_crop256x256.json',
        img_prefix=data_root + 'images/test_crop256x256/',
        pipeline=test_pipeline))
evaluation = dict(interval=2, metric='bbox', classwise=True)