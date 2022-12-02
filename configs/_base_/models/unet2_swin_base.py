# model settings
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='UNet2',
    encoder=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decoder=dict(
        type='Decoder',
        in_chs0=(768, 576, 384),
        in_chs1=(384, 192, 96),
        out_chs=(576, 384, 256),
        kernel_sizes=(3, 3, 3),
        strides=(1, 1, 1),
        paddings=(1, 1, 1),
        reflect_padding=False, # align with SwinTransformer
        instance_norm=False, # align with SwinTransformer
        out_indices=(0, 1, 2),
        concat=True,
        upsample_cfg=dict(type='UpsampleLayer', bilinear=False),
        out_cfg=dict(
            type='BaseSwinHead',
            in_ch=256,
            loss_mse_cfg=dict(type='MSELoss', loss_weight=10.),
            loss_ssim_cfg=dict(type='SSIMLoss', loss_weight=1.),
            loss_cos_cfg=dict(type='CosineLoss', loss_weight=10.)
        )
    ),
    decoder_back=dict(
        type='Decoder',
        in_chs0=(768, 576, 384),
        in_chs1=(384, 192, 96),
        out_chs=(576, 384, 256),
        kernel_sizes=(3, 3, 3),
        strides=(1, 1, 1),
        paddings=(1, 1, 1),
        reflect_padding=False, # align with SwinTransformer
        instance_norm=False, # align with SwinTransformer
        out_indices=(0, 1, 2),
        concat=True,
        upsample_cfg=dict(type='UpsampleLayer', bilinear=False),
        out_cfg=dict(
            type='BaseSwinHead',
            in_ch=256,
            loss_mse_cfg=dict(type='MSELoss', loss_weight=10.),
            loss_ssim_cfg=dict(type='SSIMLoss', loss_weight=1.),
            loss_cos_cfg=dict(type='CosineLoss', loss_weight=10.)
        )
    ),
    multi_scales=False
)