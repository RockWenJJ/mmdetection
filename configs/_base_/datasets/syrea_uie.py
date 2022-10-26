# dataset settings
dataset_type = 'SyreaDataset'
data_root = './data/syrea/'
# img_norm_cfg = dict(
#     mean=[68.48, 125.32, 126.41], std=[34.50, 40.80, 41.77], to_rgb=True)
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
syn_cfg = dict(coef_path='./data/coeffs.json', rand=False, num=1)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Synthesize', **syn_cfg),
    dict(type='SyreaFormatBundle'),
    dict(type='Collect', keys=['img', 'syn_img0', 'syn_num', 'depth'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
    # dict(type='RandomFlip', flip_ratio=0.5)
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root+'train_infos.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    # val=dict(
    #     type=dataset_type,
    #     img_prefix=data_root+'val/',
    #     pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     img_prefix=data_root+'test/',
    #     pipeline=test_pipeline)
)