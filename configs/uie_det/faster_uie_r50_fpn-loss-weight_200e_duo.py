_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/duor_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
classes = ('holothurian', 'echinus', 'scallop', 'starfish')

model = dict(
    type='FasterUIE',
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=len(classes),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=10.0),
            loss_bbox=dict(type='L1Loss', loss_weight=10.0))),
    uie_neck = dict(
        type='Decoder',
        in_chs0=(2048, 1024, 512),
        in_chs1=(1024, 512, 256),
        out_chs=(1024, 512, 256),
        kernel_sizes=(3, 3, 3),
        strides=(1, 1, 1),
        paddings=(1, 1, 1),
        reflect_padding=True,
        instance_norm=True,
        out_indices=(0, 1, 2),
        concat=True,
        upsample_cfg=dict(type='UpsampleLayer', bilinear=False),
    ),
    uie_head=dict(
        type='BaseSwinHead',
        in_ch=256,
        instance_norm=False,
        loss_mse_cfg=dict(type='MSELoss', loss_weight=1.),
        loss_ssim_cfg=dict(type='SSIMLoss', loss_weight=0.1),
        loss_cos_cfg=dict(type='CosineLoss', loss_weight=1.)
    )
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='UIEWandbLoggerHook',
             interval=50,
             vis_interval=2000,
             log_checkpoint=False,
             log_checkpoint_metadata=True,
             init_kwargs=dict(project='UieDet',
                              name='faster_uie-r50_fpn-loss-weight_200e_duo')
             )
    ])

# overwrite optimizer
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
optimizer_config = dict(grad_clip=None)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[150, 180])
runner = dict(type='EpochBasedRunner', max_epochs=200)


# overwrite dataset config
dataset_type = 'UwCocoDataset'
data_root = 'data/duo_resized/'
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'cl_img']),
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
            dict(type='ImageToTensor', keys=['img', 'cl_img']),
            # dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'cl_img']),
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

checkpoint_config = dict(interval=5)
evaluation = dict(type='UieDetEvalHook', interval=1, metric='bbox', classwise=True)