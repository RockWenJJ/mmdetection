import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numbers
import numpy as np
from einops import rearrange

import math
import numpy as np
import time
from torch import einsum

from collections import OrderedDict
from mmcv.runner import BaseModule
from ..builder import DETECTORS, build_loss, build_layer


##########################################################################
def window_partition(x, window_size: int, h, w):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - w % window_size) % window_size
    pad_b = (window_size - h % window_size) % window_size
    x = F.pad(x, [pad_l, pad_r, pad_t, pad_b])
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    H = H + pad_b
    W = W + pad_r
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    windows = F.pad(x, [pad_l, -pad_r, pad_t, -pad_b])
    return windows


##########################################################################


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## FFN
class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()
        
        hidden_features = int(dim * 3)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x


class LocalTransformerBlock(nn.Module):
    def __init__(self, dim, window_size, shift_size, bias):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        
        self.qkv_conv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
    
    def window_partitions(self, x, window_size: int):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size(M)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def create_mask(self, x):
        
        n, c, H, W = x.shape
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = self.window_partitions(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    
    def forward(self, x):
        shortcut = x
        b, c, h, w = x.shape
        
        x = window_partition(x, self.window_size, h, w)
        
        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q.transpose(-2, -1) @ k) / self.window_size
        attn = attn.softmax(dim=-1)
        out = (v @ attn)
        out = rearrange(out, 'b c (h w) -> b c h w', h=int(self.window_size),
                        w=int(self.window_size))
        out = self.project_out(out)
        out = window_reverse(out, self.window_size, h, w)
        
        shift = torch.roll(out, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        shift_window = window_partition(shift, self.window_size, h, w)
        qkv = self.qkv_dwconv1(self.qkv_conv1(shift_window))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q.transpose(-2, -1) @ k) / self.window_size
        mask = self.create_mask(shortcut)
        attn = attn.view(b, -1, self.window_size * self.window_size,
                         self.window_size * self.window_size) + mask.unsqueeze(0)
        attn = attn.view(-1, self.window_size * self.window_size, self.window_size * self.window_size)
        attn = attn.softmax(dim=-1)
        
        out = (v @ attn)
        
        out = rearrange(out, 'b c (h w) -> b c h w', h=int(self.window_size),
                        w=int(self.window_size))
        
        out = self.project_out1(out)
        out = window_reverse(out, self.window_size, h, w)
        out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        
        return out


##########################################################################
class GlobalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(int(c / self.num_heads))
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = self.project_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, bias, adaptor=False):
        super(TransformerBlock, self).__init__()
        
        if adaptor:
            self.adap_pool = nn.AdaptiveAvgPool2d(window_size**2)
            self.adaptor = nn.Embedding(10, dim)
            self.adaptor.weight.requires_grad_(False)
            self.adap_ffn = nn.Linear(window_size**4 * dim, 10)
        else:
            self.adap_pool = None
            self.adaptor = None
            self.adap_ffn = None
        
        # global transformer
        self.glob_norm1 = LayerNorm(dim)
        self.glob_attn = GlobalTransformerBlock(dim, num_heads, bias)
        
        # local transformer
        self.local_norm1 = LayerNorm(dim)
        self.local_attn = LocalTransformerBlock(dim, window_size, shift_size, bias)
        
        # feed forward
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias)
        
        self.alpha = nn.Parameter(torch.ones((dim, 1, 1))/2.)
    
    def forward(self, x):
        
        B, C, H, W = x.shape
        shortcut = x
        
        if self.adaptor is not None:
            x_adap = self.adap_pool(x)
            x_adap = x_adap.reshape(B, -1)
            water_type = torch.argmax(F.softmax(self.adap_ffn(x_adap)), dim=1)
            type_embed = self.adaptor(water_type)
            # print(int(water_type.cpu().data))
            x = x + type_embed[..., None, None]
        
        y1 = self.glob_attn(self.glob_norm1(x))
        y1 = shortcut + y1
        
        y2 = self.local_attn(self.local_norm1(x))
        y2 = shortcut + y2
        
        alpha = self.conv(self.avg_pool(y1))
        y = alpha * y1 + (1 - alpha) * y2
        
        y = y + self.ffn(self.norm2(y))
        
        return y



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    
    def forward(self, x):
        x = self.proj(x)
        
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    
    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, n_out):
        super(Upsample, self).__init__()
        
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_out * 4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2))
    
    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    
    return x


##########################################################################
@DETECTORS.register_module()
class WaterFormerV4(BaseModule):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=36,
                 num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=2,
                 heads=[2, 2, 2, 2],
                 bias=False,
                 window_size=8,
                 shift_size=3,
                 **kwargs
                 ):
        
        super().__init__()
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], bias=bias, window_size=window_size, shift_size=shift_size) for
            i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2 ** 1, num_heads=heads[1], bias=bias, window_size=window_size,
                             shift_size=shift_size) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2 ** 2, num_heads=heads[2], bias=bias, window_size=window_size,
                             shift_size=shift_size) for i in range(num_blocks[2])])
        
        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=dim * 2 ** 3, num_heads=heads[3], bias=bias, window_size=window_size,
                             shift_size=shift_size, adaptor=(i+1)%2) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim * 2 ** 3), int(dim * 2**2))  ## From Level 4 to Level 3
        self.skip_connect3 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=1, bias=bias),
            nn.ReflectionPad2d(1),
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=3, stride=1, padding=0,
                                groups=int(dim * 2 ** 2), bias=bias),
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=1, bias=bias))
        
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[2], bias=bias, window_size=window_size,
                             shift_size=shift_size) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 3), int(dim * 2))  ## From Level 3 to Level 2
        # self.skip_connect2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.skip_connect2 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias),
            nn.ReflectionPad2d(1),
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=3, stride=1, padding=0,
                      groups=int(dim * 2 ** 1), bias=bias),
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        )
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[1], bias=bias, window_size=window_size,
                             shift_size=shift_size) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 **2), int(dim))  ## From Level 2 to Level 1
        # self.skip_connect1 = nn.Conv2d(int(dim), int(dim), kernel_size=1, bias=bias)
        self.skip_connect1 = nn.Sequential(
            nn.Conv2d(int(dim), int(dim), kernel_size=1, bias=bias),
            nn.ReflectionPad2d(1),
            nn.Conv2d(int(dim), int(dim), kernel_size=3, stride=1, padding=0,
                      groups=int(dim), bias=bias),
            nn.Conv2d(int(dim), int(dim), kernel_size=1, bias=bias)
        )
        
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], bias=bias, window_size=window_size,
                             shift_size=shift_size) for i in range(num_blocks[0])])
        
        self.output = nn.Conv2d(int(dim*2), out_channels, kernel_size=1, bias=bias)
        
        self.apply(self._init_weights)
        
        self.l1_loss = build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.ssim_loss = build_loss(dict(type='SSIMLoss', loss_weight=1.0))
        self.multi_scales = False
    
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
    
    def forward_img(self, inp_img):
        
        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        
        latent = self.latent(inp_enc_level4)
        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = cat(inp_dec_level3, self.skip_connect3(out_enc_level3))
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = cat(inp_dec_level2, self.skip_connect2(out_enc_level2))
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = cat(inp_dec_level1, self.skip_connect1(out_enc_level1))
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        ref_out = out_dec_level1
        
        out = self.output(ref_out) + inp_img
        
        return tuple([out])
    
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