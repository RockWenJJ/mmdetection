import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule
from ..utils import *
from ..builder import NECKS, build_layer, build_neck


@NECKS.register_module()
class Decoder(BaseModule):
    def __init__(self,
                 in_chs,
                 out_chs,
                 kernel_sizes,
                 strides,
                 paddings,
                 reflect_padding,
                 instance_norm,
                 out_indices,
                 concat=True,
                 upsample_cfg=None,
                 ):
        super(Decoder, self).__init__()
        self.out_indices = out_indices
        self.concat = concat
        self.upsample_layers = []
        self.conv_layers = []
        conv_type = 'CIR' if instance_norm else 'CBR'
        for i, (in_ch, out_ch, k, s, p) in enumerate(zip(in_chs, out_chs, kernel_sizes, strides, paddings)):
            conv_layer = []
            upsample_cfg['in_ch'] = in_ch
            upsample_layer = build_layer(upsample_cfg)
            if concat:
                in_ch *= 2
            conv_cfg = {'type': conv_type,
                        'in_ch': in_ch,
                        'out_ch': out_ch,
                        'kernel_size': k,
                        'stride': s,
                        'padding': p,
                        'reflect_padding': reflect_padding
                        }
            conv_layer.append(build_layer(conv_cfg))
            conv_cfg['in_ch'] = out_ch
            conv_layer.append(build_layer(conv_cfg))
            conv_layer = nn.Sequential(*conv_layer)
            
            self.upsample_layers.append(upsample_layer)
            self.conv_layers.append(conv_layer)
            
            upsample_name = f'decode{i}_up'
            conv_name = f'decode{i}_conv'
            self.add_module(upsample_name, upsample_layer)
            self.add_module(conv_name, conv_layer)
    
    def forward(self, xs):
        outs = []
        x = xs.pop(0)
        for i, (upsample, conv) in enumerate(zip(self.upsample_layers, self.conv_layers)):
            x = upsample(x)
            if self.concat:
                x = torch.concat([x, xs[i]], dim=1)
            else:
                x = x + xs[i]
            x = conv(x)
            if i in self.out_indices:
                outs.append(x)
        
        return x, outs


@NECKS.register_module()
class FeaturePyramid(BaseModule):
    def __init__(self,
                 in_chs,
                 out_ch,
                 kernel_sizes,
                 strides,
                 paddings,
                 reflect_padding,
                 instance_norm,
                 out_indices=None,
                 decode_cfg=None):
        super().__init__()
        self.out_indices = out_indices
        self.decoder = build_neck(decode_cfg)
        self.convs = []
        conv_type = 'CIR' if instance_norm else 'CBR'
        for i, (in_ch, k, s, p) in enumerate(zip(in_chs, kernel_sizes, strides, paddings)):
            conv_cfg = {'type': conv_type,
                        'in_ch': in_ch,
                        'out_ch': out_ch,
                        'kernel_size': k,
                        'stride': s,
                        'padding': p,
                        'reflect_padding': reflect_padding
                        }
            conv = build_layer(conv_cfg)
            self.convs.append(conv)
            conv_name = f'decode_subconv{i}'
            self.add_module(conv_name, conv)
    
    def forward(self, xs):
        _, xs = self.decoder(xs)
        outs = []
        for i, (xi, conv) in enumerate(zip(xs, self.convs)):
            x = conv(xi)
            if i in self.out_indices:
                outs.append(x)
        return x, outs
