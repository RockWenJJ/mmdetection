# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES, build_layer, build_head
from ..utils.ckpt_convert import swin_converter
from ..utils.transformer import PatchEmbed, PatchMerging
from ..backbones.conv_trans_encoder import WindowMSA, ShiftWindowMSA, SwinBlock, SwinBlockSequence, ConvBlock
from ..necks.swin_neck import PatchExpanding

def patch_split(x, hw_shape, shuffle=False):
    '''patch split function to split a feature map into cnn and transformer features'''
    B, L, C = x.shape
    H, W = hw_shape
    
    assert L == H * W, 'input feature has wrong size'
    assert H % 2 == 0 and W % 2 == 0, f"input size ({H}*{W}) are not even"
    
    x = x.view(B, H, W, C)
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    
    x = torch.cat([x0, x1, x2, x3], -1)
    
    assert x.shape[-1] == 4 * C, f"error when doing the spliting"
    
    if shuffle:
        idx = torch.randperm(x.shape[-1])
        x = x[..., idx]
    
    feat_trans = x[..., :2 * C].view(B, -1, 2 * C)
    feat_cnn = x[..., 2 * C:].permute(0, 3, 1, 2)
    hw_shape = feat_cnn.shape[-2:]
    
    return feat_trans, feat_cnn, hw_shape
    


@BACKBONES.register_module()
class ConvTransformerDecoder(BaseModule):
    """ Conv-Transformer Decoder that utilizes both transformer and CNN features

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """
    
    def __init__(self,
                 in_channels=96*2**4,
                 embed_dims=96*2**4,
                 patch_size=2,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 6, 2, 2),
                 num_heads=(24, 12, 6, 3),
                 strides=(2, 2, 2, 2),
                 out_indices=(3,),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 out_cfg=None,
                 init_cfg=None):
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        
        super().__init__(init_cfg=init_cfg)
        
        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        
        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'
        
        # self.patch_embed = PatchEmbed(
        #     in_channels=in_channels,
        #     embed_dims=embed_dims,
        #     conv_type='Conv2d',
        #     kernel_size=patch_size,
        #     stride=strides[0],
        #     norm_cfg=norm_cfg if patch_norm else None,
        #     init_cfg=None)
        
        # if self.use_abs_pos_embed:
        #     patch_row = pretrain_img_size[0] // patch_size
        #     patch_col = pretrain_img_size[1] // patch_size
        #     self.absolute_pos_embed = nn.Parameter(
        #         torch.zeros((1, embed_dims, patch_row, patch_col)))
        
        # self.drop_after_pos = nn.Dropout(p=drop_rate)
        
        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        
        self.stages = ModuleList()
        # in_channels = embed_dims * 2
        for i in range(num_layers):
            swin_block = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=None,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            
            cnn_block = ConvBlock(
                in_ch=in_channels,
                out_ch=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                reflect_padding=True,
                instance_norm=True)
            
            expand = PatchExpanding(in_channels)
            
            stage = ModuleList()
            stage.append(swin_block)
            stage.append(cnn_block)
            stage.append(expand)
            self.stages.append(stage)
            
            in_channels = in_channels // 2
            
        if out_cfg is not None:
            self.out_head = build_head(out_cfg)
            self.add_module('out_head', self.out_head)
        else:
            self.out_head = None
        
        # self.num_features = [int(embed_dims  // (2 ** i)) for i in range(num_layers)]
        # # Add a norm layer for each output
        # for i in out_indices:
        #     layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
        #     layer_name = f'norm{i}'
        #     self.add_module(layer_name, layer)
    
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
    
    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)
            
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
            
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
            
            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()
            
            # load state_dict
            self.load_state_dict(state_dict, False)
    
    def forward(self, xs):
        # x, hw_shape = self.patch_embed(x)
        
        outs = []
        x = xs.pop(0)
        for i, stage in enumerate(self.stages):
            hw_shape = x.shape[-2:]
            x_trans = x.flatten(2).transpose(1, 2)
            swin_block, conv_block, expand = stage[0], stage[1], stage[2]
            x_trans, hw_shape, out, out_hw_shape = swin_block(x_trans, hw_shape)  # process transformer feature
            x_cnn = conv_block(x)  # process cnn feature
            # x_trans_view = x_trans.view(-1, *hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            x_cnn_view = x_cnn.flatten(2).transpose(1, 2)
            
            # x = torch.concat([x_trans, x_cnn_view], -1)
            x = x_trans + x_cnn_view
            
            x, hw_shape = expand(x, hw_shape)
            x = x.view(-1, *hw_shape, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
            
            if len(xs) > 0:
                x0 = xs.pop(0)
                x = x + x0
            out = x
            
            if i in self.out_indices:
                outs.append(out)
        
        if self.out_head is None:
            return tuple(outs)
        else:
            head_outs = []
            head_outs.append(self.out_head(outs[-1]))
        
        return tuple(head_outs)
