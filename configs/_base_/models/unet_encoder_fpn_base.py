# model settings
model = dict(
    type='UNet',
    backbone=dict(
        type='Encoder',
        in_chs=(3, 64, 128, 256, 512),
        out_chs=(64, 128, 256, 512, 512),
        kernel_sizes=(3, 3, 3, 3, 3),
        strides=(1, 1, 1, 1, 1),
        paddings=(1, 1, 1, 1, 1),
        reflect_padding=True,
        instance_norm=True,
        out_indices=(0, 1, 2, 3, 4),
        avg_down=False
    ),
    neck=dict(
        type='FeaturePyramid',
        decode_cfg=dict(
            type='Decoder',
            in_chs0=(512, 256, 128, 64),
            in_chs1=(512, 256, 128, 64),
            out_chs=(256, 128, 64, 64),
            kernel_sizes=(3, 3, 3, 3),
            strides=(1, 1, 1, 1),
            paddings=(1, 1, 1, 1),
            reflect_padding=True,
            instance_norm=True,
            out_indices=(0, 1, 2, 3),
            concat=True,
            upsample_cfg=dict(type='UpsampleLayer', bilinear=False),
        ),
        in_chs=(256, 128, 64, 64), # the same with output channels from Decoder
        out_ch=64,
        kernel_sizes=(3, 3, 3, 3),
        strides=(1, 1, 1, 1),
        paddings=(1, 1, 1, 1),
        reflect_padding=True,
        instance_norm=True,
        out_indices=(0, 1, 2, 3)
    ),
    head=dict(
        type='BaseHead',
        in_ch=64,
        loss_mse_cfg=dict(type='MSELoss', loss_weight=10.),
        loss_ssim_cfg=dict(type='SSIMLoss', loss_weight=1.),
    ),
    multi_scales=True,
)