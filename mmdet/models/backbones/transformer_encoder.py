import warnings
from collections import OrderedDict
from copy import deepcopy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import FFN, build_dropout

from ...utils import get_root_logger
from ..utils.transformer import PatchMerging, PatchEmbed
from ..builder import BACKBONES

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros( d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe.unsqueeze(0)

# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerBlockSequence(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 pos_embed,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.blocks = ModuleList()
        for i in range(depth):
            block = TransformerBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                pos_embed=pos_embed,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                init_cfg=None
            )
            self.blocks.append(block)
            
        self.downsample = downsample
    
    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)
            
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

class TransformerBlock(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 pos_embed,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.pos_embed = pos_embed
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=drop_rate)
        head_embed_dims = embed_dims // num_heads
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None
        )
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        
    
    def embed_pos(self, x, pos):
        return x if pos is None else x + pos
    
    def forward(self, x, hw_shape):
        
        #TODO: reshape self.pos
        # interpolate pos_embed
        pos_embed = F.interpolate(self.pos_embed, mode='bilinear',size= hw_shape)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        
        identity = x
        
        q = k = self.embed_pos(x, pos_embed)
        x = self.attn(q, k, value=x)[0]
        x = x + identity
        x = self.norm1(x)
        
        identity = x
        x = self.ffn(x, identity=identity)
        x = self.norm2(x)
        
        return x
        


@BACKBONES.register_module()
class TransformerEncoder(BaseModule):
    
    def __init__(self,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 mlp_ratio=4,
                 depths=(2, 3, 3, 2),
                 strides=(4, 2, 2, 2),
                 num_heads=(3, 3, 6, 3),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 scale_invar_pos_embed=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        
        super(TransformerEncoder, self).__init__(init_cfg=init_cfg)
        
        num_layers = len(depths)
        self.num_layers = num_layers
        self.out_indices = out_indices
        self.scale_invar_pos_embed = scale_invar_pos_embed
        
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)
        
        # positional embedding
        
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        
        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers -1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            # add positional embedding
            # pos_embed = nn.Parameter(torch.zeros(1, in_channels, 18, 18))
            pos_embed =  nn.Parameter(positionalencoding2d(in_channels, 18, 18))
            setattr(self, f'pos_embed{i}', pos_embed)
            
            stage = TransformerBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                pos_embed=pos_embed,
                downsample=downsample,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels
            
        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
                
    
    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            # for i in range(self.num_layers):
            #     trunc_normal_(getattr(self, f'pos_embed{i}'), std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
            
    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)
        
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        
        return outs


@BACKBONES.register_module()
class TransformerEncoder2(BaseModule):
    
    def __init__(self,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 mlp_ratio=4,
                 depths=(2, 2, 2, 2),
                 strides=(4, 2, 2, 2),
                 num_heads=(3, 3, 6, 3),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 scale_invar_pos_embed=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        
        super().__init__(init_cfg=init_cfg)
        
        num_layers = len(depths)
        self.num_layers = num_layers
        self.out_indices = out_indices
        self.scale_invar_pos_embed = scale_invar_pos_embed
        
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)
        
        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 18, 18))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        
        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None
            
            # add positional embedding
            # pos_embed = nn.Parameter(torch.zeros(1, in_channels, 18, 18))
            # setattr(self, f'pos_embed{i}', pos_embed)
            
            stage = TransformerBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                pos_embed=self.pos_embed,
                downsample=downsample,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels
        
        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
    
    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            trunc_normal_(self.pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
    
    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)
        
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        
        return outs
            