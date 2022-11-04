import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule
from ..utils import *
from ..builder import HEADS, build_layer, build_head, build_loss

@HEADS.register_module()
class BaseHead(BaseModule):
    def __init__(self,
                 in_ch,
                 out_ch=3,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 instance_norm=True,
                 loss_mse_cfg=dict(type='MSELoss', loss_weight=1.0),
                 loss_ssim_cfg=None):
        super(BaseHead, self).__init__()
        conv_type = 'CIR' if instance_norm else 'CBR'
        conv_cfg = {'type': conv_type,
                    'in_ch': in_ch,
                    'out_ch': out_ch,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding}
        # self.head = build_layer(conv_cfg)
        self.head = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size)
        self.add_module('head', self.head)
        
        self.loss_mse = build_loss(loss_mse_cfg)
        self.loss_ssim = build_loss(loss_ssim_cfg)
    
    def forward(self, x):
        return self.head(x)
    
    def loss_single(self, pred, gt):
        loss_mse = self.loss_mse(pred, gt)
        loss_ssim = self.loss_ssim(pred, gt)
        return loss_mse, loss_ssim
    
    def loss(self, preds, gts, img_metas):
        loss_mse, loss_ssim = self.loss_single(preds, gts)
        
        return dict(loss_mse=loss_mse, loss_ssim=loss_ssim)


@HEADS.register_module()
class BaseSwinHead(BaseModule):
    def __init__(self,
                 in_ch,
                 out_ch=3,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 instance_norm=True,
                 up_scale=4,
                 loss_mse_cfg=dict(type='MSELoss', loss_weight=1.0),
                 loss_ssim_cfg=None,
                 loss_cos_cfg=None):
        super().__init__()
        conv_type = 'CIR' if instance_norm else 'CBR'
        conv_cfg = {'type': conv_type,
                    'in_ch': in_ch,
                    'out_ch': out_ch,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding}
        # self.head = build_layer(conv_cfg)
        self.pre = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size)
        self.up = nn.PixelShuffle(up_scale)
        
        self.head = nn.Conv2d(in_ch//(up_scale*up_scale), out_ch, kernel_size=kernel_size)
        self.add_module('pre', self.pre)
        self.add_module('up', self.up)
        self.add_module('head', self.head)
        
        self.loss_mse = build_loss(loss_mse_cfg)
        self.loss_ssim = build_loss(loss_ssim_cfg)
        self.loss_cos = build_loss(loss_cos_cfg)
    
    def forward(self, x):
        x = self.up(self.pre(x))
        return self.head(x)
    
    def loss_single(self, pred, gt):
        loss_mse = self.loss_mse(pred, gt)
        loss_ssim = self.loss_ssim(pred, gt) if self.loss_ssim is not None else 0
        loss_cos = self.loss_cos(pred, gt)
        return loss_mse, loss_ssim, loss_cos
    
    def loss(self, preds, gts, img_metas):
        loss_mse, loss_ssim, loss_cos = self.loss_single(preds, gts)
        
        return dict(loss_mse=loss_mse, loss_ssim=loss_ssim, loss_cos=loss_cos)


@HEADS.register_module()
class BaseSwinHead2(BaseModule):
    def __init__(self,
                 in_ch,
                 out_ch=3,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 instance_norm=True,
                 up_scale=4,
                 loss_mse_cfg=dict(type='MSELoss', loss_weight=1.0),
                 loss_ssim_cfg=None):
        super().__init__()
        conv_type = 'CIR' if instance_norm else 'CBR'
        conv_cfg = {'type': conv_type,
                    'in_ch': in_ch,
                    'out_ch': out_ch,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding}
        # self.head = build_layer(conv_cfg)
        self.pre = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size)
        self.up = nn.PixelShuffle(up_scale)
        self.conv = nn.Conv2d(in_ch // (up_scale * up_scale), in_ch // (up_scale * up_scale), kernel_size=kernel_size)
        self.up2 = nn.PixelShuffle(2)
        
        
        self.head = nn.Conv2d(in_ch // (up_scale * up_scale * 4), out_ch, kernel_size=kernel_size)
        
        self.add_module('pre', self.pre)
        self.add_module('up', self.up)
        self.add_module('conv', self.conv)
        self.add_module('up2', self.up2)
        self.add_module('head', self.head)
        
        self.loss_mse = build_loss(loss_mse_cfg)
        self.loss_ssim = build_loss(loss_ssim_cfg)
    
    def forward(self, x):
        x = self.up2(self.conv(self.up(self.pre(x))))
        return self.head(x)
    
    def loss_single(self, pred, gt):
        loss_mse = self.loss_mse(pred, gt)
        loss_ssim = self.loss_ssim(pred, gt)
        return loss_mse, loss_ssim
    
    def loss(self, preds, gts, img_metas):
        loss_mse, loss_ssim = self.loss_single(preds, gts)
        
        return dict(loss_mse=loss_mse, loss_ssim=loss_ssim)


@HEADS.register_module()
class BaseMultiConvHead(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 strides,
                 paddings,
                 instance_norm=True,
                 loss_mse_cfg=dict(type='MSELoss', loss_weight=1.0),
                 loss_ssim_cfg=None):
        super().__init__()
        conv_type = 'CIR' if instance_norm else 'CBR'
        assert len(in_channels) == len(out_channels), 'in_channels length must be the same of out_channels length'
        self.layers = nn.ModuleList()
        for i, (in_ch, out_ch, k, s, p) in enumerate(zip(in_channels, out_channels, kernel_sizes, strides, paddings)):
            if i < len(in_channels)-1:
                conv_cfg = {'type': conv_type,
                            'in_ch': in_ch,
                            'out_ch': out_ch,
                            'kernel_size': k,
                            'stride': s,
                            'padding': p}
                self.layers.append(build_layer(conv_cfg))
            else:
                self.layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
            
        self.add_module('layers', self.layers)
        
        self.loss_mse = build_loss(loss_mse_cfg)
        self.loss_ssim = build_loss(loss_ssim_cfg)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def loss_single(self, pred, gt):
        loss_mse = self.loss_mse(pred, gt)
        loss_ssim = self.loss_ssim(pred, gt)
        return loss_mse, loss_ssim
    
    def loss(self, preds, gts, img_metas):
        loss_mse, loss_ssim = self.loss_single(preds, gts)
        
        return dict(loss_mse=loss_mse, loss_ssim=loss_ssim)