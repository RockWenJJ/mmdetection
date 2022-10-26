import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule
from ..utils import *
from ..builder import BACKBONES, build_layer


@BACKBONES.register_module()
class Encoder(BaseModule):
    def __init__(self,
                 in_chs,
                 out_chs,
                 kernel_sizes,
                 strides,
                 paddings,
                 reflect_padding,
                 instance_norm,
                 out_indices,
                 avg_down=False):
        super(Encoder, self).__init__()
        self.out_indices = out_indices
        self.layers = []
        conv_type = 'CIR' if instance_norm else 'CBR'
        ds_in_ch = in_chs[0]  # just for initialization
        for i, (in_ch, out_ch, k, s, p) in enumerate(zip(in_chs, out_chs, kernel_sizes, strides, paddings)):
            sub_layer = []
            if i != 0:
                sub_layer.append(DownSampleLayer(avg_down, ds_in_ch))
            layer_cfg = {'type': conv_type,
                         'in_ch': in_ch,
                         'out_ch': out_ch,
                         'kernel_size': k,
                         'stride': s,
                         'padding': p,
                         'reflect_padding': reflect_padding
                         }
            sub_layer.append(build_layer(layer_cfg))
            layer_cfg['in_ch'] = out_ch
            sub_layer.append(build_layer(layer_cfg))
            
            sub_layer = nn.Sequential(*sub_layer)
            self.layers.append(sub_layer)
            layer_name = f'encode{i}'
            self.add_module(layer_name, sub_layer)
            
            ds_in_ch = out_ch
    
    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return x, outs