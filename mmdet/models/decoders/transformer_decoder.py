import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmcv.runner import BaseModule, ModuleList
from mmdet.models.utils import *
from mmdet.models.builder import DECODERS, build_layer, build_neck, build_head

from ...utils import get_root_logger
from ..backbones.transformer_encoder import TransformerBlockSequence, TransformerBlock
from ..necks.swin_neck import PatchExpanding

@DECODERS.register_module()
class TransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims=768,
                 mlp_ratio=4,
                 depths=(2, 3, 3, 2),
                 strides=(2, 2, 2, 2),
                 num_heads=(3, 6, 3, 3),
                 out_indices=(3,),
                 qkv_bias=True,
                 qkv_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 scale_invar_pos_embed=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 out_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.num_layers = len(depths)
        self.out_indices = out_indices
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        self.scale_invar_pos_embed = scale_invar_pos_embed
        
        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(self.num_layers):
            # if i < self.num_layers - 1:
            upsample = PatchExpanding(
                in_channels=in_channels,
                scale=2,
                norm_cfg=norm_cfg
            )
                
            pos_embed = nn.Parameter(torch.zeros(1, in_channels, 18, 18))
            setattr(self, f'pos_embed{i}', pos_embed)
            
            stage = TransformerBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                pos_embed=pos_embed,
                downsample=upsample,
                init_cfg=None)
            self.stages.append(stage)
            if upsample:
                in_channels = upsample.out_channels
        
        if out_cfg is not None:
            self.out_head = build_head(out_cfg)
            self.add_module('out_head', self.out_head)
        else:
            self.out_head = None

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for i in range(self.num_layers):
                trunc_normal_(getattr(self, f'pos_embed{i}'), std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                    
    def forward(self, xs):
        
        x = xs.pop(0)
        hw_shape = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, _, _ = stage(x, hw_shape)
            if len(xs) > 0:
                x0 = xs.pop(0)
                x = x + x0.flatten(2).transpose(1, 2)
            if i in self.out_indices:
                B, L, C = x.shape
                H, W = hw_shape
                x = x.transpose(1, 2).view(B, C, H, W)
                outs.append(x)
        
        if self.out_head is None:
            return tuple(outs)
        else: # return final outputs if out_head is not None
            head_outs = []
            # if self.multi_scales:
            #     for i, x in enumerate(outs):
            #         head_outs.append(self.out_head(x))
            # else:
            head_outs.append(self.out_head(outs[-1]))
        
        return tuple(head_outs)