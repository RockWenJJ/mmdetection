import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule
from ..builder import LAYERS, build_layer


@LAYERS.register_module()
class CBR(BaseModule):
    '''A Conv+BN+ReLU Block'''
    
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 padding,
                 reflect_padding=False
                 ):
        super(CBR, self).__init__()
        if not reflect_padding:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(kernel_size // 2),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.conv(x)


@LAYERS.register_module()
class CIR(BaseModule):
    '''A Conv+IN+ReLU Block'''
    
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 padding,
                 reflect_padding=False):
        super(CIR, self).__init__()
        if not reflect_padding:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(kernel_size // 2),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.conv(x)


@LAYERS.register_module()
class DownSampleLayer(BaseModule):
    '''Downsampling layer.'''
    
    def __init__(self,
                 avg_down,
                 in_ch=None):
        super(DownSampleLayer, self).__init__()
        if avg_down:
            self.downsample = nn.AvgPool2d(2)
        else:
            assert in_ch is not None
            self.downsample = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=0)
            )
    
    def forward(self, x):
        return self.downsample(x)


@LAYERS.register_module()
class UpsampleLayer(BaseModule):
    def __init__(self,
                 in_ch=None,
                 bilinear=False):
        super(UpsampleLayer, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.up(x)
    
@LAYERS.register_module()
class PixelshuffleLayer(BaseModule):
    def __init__(self, in_ch=None, up_scale=2):
        super().__init__()
        self.up = nn.PixelShuffle(up_scale)
    
    def forward(self, x):
        return self.up(x)