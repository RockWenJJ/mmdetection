import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum

from collections import OrderedDict
from mmcv.runner import BaseModule
from ..builder import DETECTORS, build_loss, build_layer


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

@DETECTORS.register_module()
class UNetB(BaseModule):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.apply(self._init_weights)

        self.l1_loss = build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.ssim_loss = build_loss(dict(type='SSIMLoss', loss_weight=1.0))

        self.multi_scales = False

    def forward_img(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return tuple([out])
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img, img_metas, return_loss=True, **kwargs):
    
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0]), {}
    
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, img_metas, **kwargs):
        assert 'input' in kwargs
        if self.multi_scales:
            assert isinstance(img, (list, tuple)), "when multi_scale, "
        input_img = kwargs['input'][-1] if self.multi_scales else kwargs['input']
    
        xs = self.forward_img(input_img)
    
        losses = dict()
        images_dict = dict()
        images_dict['input'] = input_img
    
        if self.multi_scales:
            raise NotImplementedError
        else:
            images_dict['predict'] = xs[-1]
            images_dict['target'] = img
            losses.update(self.loss(xs[-1], img, img_metas))
    
        return losses, images_dict

    def forward_test(self, img, img_metas=None, **kwargs):
        input_img = img
        xs = self.forward_img(input_img)
        return xs[-1]

    def forward_dummy(self, img):
        '''Used for computing network flops and convert to onnx models'''
        xs = self.forward_img(img)
        return xs[-1]

    def train_step(self, data, optimizer):
        losses, images_dict = self(**data)
        loss, log_vars = self._parse_losses(losses)
    
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']), images=images_dict)
    
        return outputs

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
    
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
    
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            # if dist.is_available() and dist.is_initialized():
            #     loss_value = loss_value.data.clone()
            #     dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
    
        return loss, log_vars

    def loss_single(self, pred, gt):
        loss_l1 = self.l1_loss(pred, gt)
        loss_ssim = self.ssim_loss(pred, gt)
        
        return loss_l1, loss_ssim

    def loss(self, preds, gts, img_metas, suffix=None):
    
        loss_l1, loss_ssim = self.loss_single(preds, gts)
    
        if suffix is None:
            return dict(loss_l1=loss_l1, loss_ssim=loss_ssim)
        else:
            loss_dict = dict()
            loss_dict[f'loss_l1_{suffix}'] = loss_l1
            loss_dict[f'loss_ssim_{suffix}'] = loss_ssim
            return loss_dict