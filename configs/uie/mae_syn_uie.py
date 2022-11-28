_base_=['../_base_/models/mae_vit_base_patch16.py',
        '../_base_/datasets/syrea_uie.py',
        '../_base_/schedules/schedule_20e.py',
        '../_base_/default_runtime.py']

# model = dict(
#     type='UNet',
#     multi_scales=False
# )

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='UIEWandbLoggerHook',
             interval=50,
             vis_interval=1000,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             init_kwargs=dict(project='MaeUIE',
                              name='mae_syn_uie_20221128')
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
    step=[5, 98])
runner = dict(type='EpochBasedRunner', max_epochs=100)

# overwrite dataset config
# dataset settings
dataset_type = 'SynDataset'
data_root = './data/synthesis/'
real_dataset_type = 'UWDataset'
real_root = './data/real/'
# img_norm_cfg = dict(
#     mean=[68.48, 125.32, 126.41], std=[34.50, 40.80, 41.77], to_rgb=True)
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
# syn_cfg = dict(coef_path='./data/coeffs.json', rand=False, num=1)

img_scale = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSynthesisFromFile'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomNoise', ratio=0.8, noise_types=['gaussian', 'poisson']),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SyreaFormatBundle'),
    dict(type='Collect', keys=['img', 'input'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSynthesisFromFile'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='SyreaFormatBundle'),
    dict(type='Collect', keys=['img', 'input'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                  'img_shape', 'img_norm_cfg'))
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root+'train_infos.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test_infos.json',
        img_prefix=data_root,
        pipeline=val_pipeline),
    test=dict(
        type=real_dataset_type,
        ann_file=real_root+'test_infos.json',
        img_prefix=real_root,
        pipeline=test_pipeline)
)

checkpoint_config = dict(interval=5)
evaluation = dict(type='UieEvalHook', interval=2)