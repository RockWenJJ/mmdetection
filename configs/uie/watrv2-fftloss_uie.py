_base_ = [
    '../_base_/datasets/syrea_uie.py',
    '../_base_/schedules/schedule_20e.py',
    '../_base_/default_runtime.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='WaTrV2'
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='UIEWandbLoggerHook',
             interval=50,
             vis_interval=2000,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             init_kwargs=dict(project='ICCV2023',
                              name='watrv2-fftloss_uie')
             )
    ])

# overwrite schedule
# optimizer
# optimizer = dict(type='Adam', lr=0.02, weight_decay=0.0001)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.01, norm_type=2))
optimizer_config = dict(grad_clip=None)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# default decay ratio: gamma:0.1, min_lr: None
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[50, 80])
runner = dict(type='EpochBasedRunner', max_epochs=100)

# overwrite dataset config
# dataset settings
dataset_type = 'SynBackDataset'
data_root = './data/synthesis/'
real_dataset_type = 'UWDataset'
real_root = './data/real/'
# img_norm_cfg = dict(
#     mean=[68.48, 125.32, 126.41], std=[34.50, 40.80, 41.77], to_rgb=True)
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
# syn_cfg = dict(coef_path='./data/coeffs.json', rand=False, num=1)

img_scale = (128, 128)  # (620, 460) (w, h)
crop_size = (128, 128)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSynthesisFromFile'),
    dict(type='LoadBackFromFile'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    # dict(type='RandomCrop',
    #      crop_type='absolute',
    #      crop_size=crop_size,
    #      recompute_bbox=True,
    #      allow_negative_crop=True),
    dict(type='RandomNoise', ratio=0.8, noise_types=['gaussian', 'poisson']),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SyreaFormatBundle'),
    dict(type='Collect', keys=['img', 'input', 'back', 'target'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSynthesisFromFile'),
    dict(type='LoadBackFromFile'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    # dict(type='CenterCrop',
    #      crop_type='absolute',
    #      crop_size=crop_size,
    #      recompute_bbox=True,
    #      allow_negative_crop=True),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='SyreaFormatBundle'),
    dict(type='Collect', keys=['img', 'input', 'back', 'target'])
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
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_infos.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test_infos.json',
        img_prefix=data_root,
        pipeline=val_pipeline),
    test=dict(
        type=real_dataset_type,
        ann_file=real_root + 'test_infos.json',
        img_prefix=real_root,
        pipeline=test_pipeline)
)

checkpoint_config = dict(interval=2)
evaluation = dict(type='UieEvalHook', interval=2)