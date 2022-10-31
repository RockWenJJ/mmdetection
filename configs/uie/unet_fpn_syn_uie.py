_base_=['../_base_/models/unet_encoder_fpn_base.py',
        '../_base_/datasets/syrea_uie.py',
        '../_base_/schedules/schedule_20e.py',
        '../_base_/default_runtime.py']

model = dict(
    type='UNet',
    multi_scales=True
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='UIEWandbLoggerHook',
             interval=50,
             vis_interval=1000,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             init_kwargs=dict(project='SyreaNetUIE',
                              name='unet_fpn_syn_uie_221026')
             )
    ])

# overwrite schedule
# optimizer
# optimizer = dict(type='Adam', lr=0.02, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# default decay ratio: gamma:0.1, min_lr: None
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[5, 70, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)

# overwrite dataset config
# dataset settings
dataset_type = 'SynDataset'
data_root = './data/synthesis/'
# real_dataset_type = 'UWDataset'
# real_root = './data/real/'
# img_norm_cfg = dict(
#     mean=[68.48, 125.32, 126.41], std=[34.50, 40.80, 41.77], to_rgb=True)
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
# syn_cfg = dict(coef_path='./data/coeffs.json', rand=False, num=1)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSynthesisFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=[(32, 32), (64, 64), (128, 128), (256, 256)],
         flip=False,
         transforms=[
             dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
             dict(type='RandomFlip', flip_ratio=0.5),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='SyreaFormatBundle'),
             dict(type='Collect', keys=['img', 'input'])
         ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSynthesisFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=[(32, 32), (64, 64), (128, 128), (256, 256)],
         flip=False,
         transforms=[
             dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
             dict(type='RandomFlip', flip_ratio=0.),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='SyreaFormatBundle'),
             dict(type='Collect', keys=['img', 'input'])
         ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root+'train_infos.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test_infos.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'test_infos.json',
        img_prefix=data_root,
        pipeline=test_pipeline)
)

# checkpoint_config = dict(interval=5)
evaluation = dict(type='UieEvalHook', interval=2)