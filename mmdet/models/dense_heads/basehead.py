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