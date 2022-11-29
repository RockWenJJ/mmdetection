model = dict(
    type='MaskedAutoencoderViT',
    img_size=224,
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4
)
