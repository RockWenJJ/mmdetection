import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum

from collections import OrderedDict
from mmcv.runner import BaseModule
from ..builder import DETECTORS, build_loss, build_layer


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class ConvLayer(BaseModule):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 padding,
                 reflect_padding,
                 instance_norm):
        super(ConvLayer, self).__init__()
        self.layers = []
        conv_type = 'CIR' if instance_norm else 'CBR'
        
        layer = []
        layer_cfg = {'type': conv_type,
                     'in_ch': in_ch,
                     'out_ch': out_ch,
                     'kernel_size': kernel_size,
                     'stride': stride,
                     'padding': padding,
                     'reflect_padding': reflect_padding
                     }
        layer.append(build_layer(layer_cfg))
        layer_cfg['in_ch'] = out_ch
        layer.append(build_layer(layer_cfg))
        self.layer = nn.Sequential(*layer)
    
    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.layer(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, hw_shape


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x
    
    def flops(self, HW):
        flops = 0
        flops += HW * self.in_channels * self.kernel_size ** 2 / self.stride ** 2
        flops += HW * self.in_channels * self.out_channels
        print("SeqConv2d:{%.2f}" % (flops / 1e9))
        return flops


######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.,
                 last_stage=False, bias=True):
        super().__init__()
        
        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)
    
    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))
        
        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v
    
    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim
    
    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v
    
    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = q_L * self.dim * self.inner_dim + kv_L * self.dim * self.inner_dim * 2
        return flops
    
    #########################################


########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        
        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'
    
    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H * W, H * W)
        
        # attn = (q @ k.transpose(-2, -1))
        
        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        
        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops


########### self-attention #############
class Attention(nn.Module):
    def __init__(self, dim, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
        
        # attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
    
    def flops(self, q_num, kv_num):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))
        
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        
        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}" % (flops / 1e9))
        return flops


#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.in_features * self.hidden_features
        # fc2
        flops += H * W * self.hidden_features * self.out_features
        print("MLP:{%.2f}" % (flops / 1e9))
        return flops


## Gated-Dconv Feed-Forward Network (GDFN)
class GatedFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, bias=True):
        super(GatedFeedForward, self).__init__()
        
        self.project_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim, bias=bias)
        
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x, hw_shape):
        h, w = hw_shape
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=h, w=w)
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.project_out(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h=h, w=w)
        return x


#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=2, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
    
    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.conv(x)
        hw_shape = x.shape[-2:]
        out = x.flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out, hw_shape
    
    def flops(self, H, W):
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
    
    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        # H = int(math.sqrt(L))
        # W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.deconv(x)
        hw_shape = x.shape[-2:]
        out = x.flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out, hw_shape
    
    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel
    
    def forward(self, x):
        B, C, H, W = x.shape
        hw_shape = (H, W)
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape
    
    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3
        
        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Input_proj:{%.2f}" % (flops / 1e9))
        return flops


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel
    
    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape
    
    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3
        
        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops


#########################################
########### LeWinTransformer #############
class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='gdff',
                 modulator=False, cross_modulator=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"
        
        if modulator:
            self.modulator = nn.Embedding(win_size * win_size, dim)  # modulator
        else:
            self.modulator = None
        
        # if cross_modulator:
        #     self.cross_modulator = nn.Embedding(win_size * win_size, dim)  # cross_modulator
        #     self.cross_attn = Attention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
        #                                 proj_drop=drop,
        #                                 token_projection=token_projection, )
        #     self.norm_cross = norm_layer(dim)
        # else:
        #     self.cross_modulator = None
        
        self.cross_modulator = None
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp == 'gdff':
            self.mlp = GatedFeedForward(dim, mlp_hidden_dim)
        else:
            raise Exception("FFN error!")
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"
    
    def forward(self, x, hw_shape, mask=None):
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # pad feature maps to multiples of window size
        pad_r = (self.win_size - W % self.win_size) % self.win_size
        pad_b = (self.win_size - H % self.win_size) % self.win_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]
        
        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
            shift_mask = torch.zeros((1, H_pad, W_pad, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            # nW, win_size, 1
            mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            mask_windows = mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
        else:
            shifted_x = x
            attn_mask = None
        
        # if self.cross_modulator is not None:
        #     shortcut = x
        #     x_cross = self.norm_cross(x)
        #     x_cross = self.cross_attn(x_cross, self.cross_modulator.weight)
        #     x = shortcut + x_cross
        
        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        
        # with_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows
        
        attn_windows = self.attn(wmsa_in, mask=attn_mask)  # nW*B, win_size*win_size, C
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H_pad, W_pad)  # B H' W' C
        
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), hw_shape))
        del attn_mask
        return x, hw_shape
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        
        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)
        
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops


#########################################
########### Basic layer of Uformer ################
class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='gdff', shift_flag=True,
                 modulator=False, cross_modulator=False):
        
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, win_size=win_size,
                                      shift_size=0 if (i % 2 == 0) else win_size // 2,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                                      modulator=modulator, cross_modulator=cross_modulator)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, win_size=win_size,
                                      shift_size=0,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                                      modulator=modulator, cross_modulator=cross_modulator)
                for i in range(depth)])
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
    
    def forward(self, x, hw_shape, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, hw_shape = checkpoint.checkpoint(blk, x)
            else:
                x, hw_shape = blk(x, hw_shape, mask)
        return x, hw_shape
    
    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


@DETECTORS.register_module()
class WaTrV2(BaseModule):
    def __init__(self, img_size=128, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='gdff',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True, modulator=False,
                 cross_modulator=False, **kwargs):
        super().__init__()
        
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in
        
        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]
        
        # build layers
        
        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        
        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.skip0 = ConvLayer(in_ch=embed_dim,
                               out_ch=embed_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               reflect_padding=True,
                               instance_norm=True)
        
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.skip1 = ConvLayer(in_ch=embed_dim * 2,
                               out_ch=embed_dim * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               reflect_padding=True,
                               instance_norm=True)
        
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.skip2 = ConvLayer(in_ch=embed_dim * 4,
                               out_ch=embed_dim * 4,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               reflect_padding=True,
                               instance_norm=True)
        
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.skip3 = ConvLayer(in_ch=embed_dim * 8,
                               out_ch=embed_dim * 8,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               reflect_padding=True,
                               instance_norm=True)
        
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)
        
        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim * 16,
                                      output_dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4),
                                                        img_size // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      token_projection=token_projection, token_mlp=token_mlp, shift_flag=shift_flag)
        
        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        # self.decoderlayer_3 = ConvLayer(in_ch=embed_dim * 2,
        #                                 out_ch=embed_dim * 2,
        #                                 kernel_size=3,
        #                                 stride=1,
        #                                 padding=1,
        #                                 reflect_padding=True,
        #                                 instance_norm=True
        #                                 )
        
        self.apply(self._init_weights)
        
        self.l1_loss = build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.multi_scales = False
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"
    
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
    
    def forward_img(self, x):
        mask = None
        # Input Projection
        y, hw_shape = self.input_proj(x)
        y = self.pos_drop(y)
        # Encoder
        conv0, hw_shape = self.encoderlayer_0(y, hw_shape, mask=mask)
        pool0, hw_shape = self.dowsample_0(conv0, hw_shape)
        conv1, hw_shape = self.encoderlayer_1(pool0, hw_shape, mask=mask)
        pool1, hw_shape = self.dowsample_1(conv1, hw_shape)
        conv2, hw_shape = self.encoderlayer_2(pool1, hw_shape, mask=mask)
        pool2, hw_shape = self.dowsample_2(conv2, hw_shape)
        conv3, hw_shape = self.encoderlayer_3(pool2, hw_shape, mask=mask)
        pool3, hw_shape = self.dowsample_3(conv3, hw_shape)
        
        # Bottleneck
        conv4, hw_shape = self.conv(pool3, hw_shape, mask=mask)
        
        # Decoder
        up0, hw_shape = self.upsample_0(conv4, hw_shape)
        deconv0 = torch.cat([up0, self.skip3(conv3, hw_shape)[0]], -1)
        deconv0, hw_shape = self.decoderlayer_0(deconv0, hw_shape, mask=mask)
        
        up1, hw_shape = self.upsample_1(deconv0, hw_shape)
        deconv1 = torch.cat([up1, self.skip2(conv2, hw_shape)[0]], -1)
        deconv1, hw_shape = self.decoderlayer_1(deconv1, hw_shape, mask=mask)
        
        up2, hw_shape = self.upsample_2(deconv1, hw_shape)
        deconv2 = torch.cat([up2, self.skip1(conv1, hw_shape)[0]], -1)
        deconv2, hw_shape = self.decoderlayer_2(deconv2, hw_shape)
        
        up3, hw_shape = self.upsample_3(deconv2, hw_shape)
        deconv3 = torch.cat([up3, self.skip0(conv0, hw_shape)[0]], -1)
        deconv3, hw_shape = self.decoderlayer_3(deconv3, hw_shape)
        
        # Output Projection
        y, hw_shape = self.output_proj(deconv3, hw_shape)
        # out = x + y if self.dd_in == 3 else y
        out = y
        return tuple([out])
    
    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso, self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(self.reso, self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(self.reso // 2, self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(self.reso // 2 ** 3, self.reso // 2 ** 3)
        
        # Bottleneck
        flops += self.conv.flops()
        
        # Decoder
        flops += self.upsample_0.flops(self.reso // 2 ** 4, self.reso // 2 ** 4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2 ** 3, self.reso // 2 ** 3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso // 2, self.reso // 2) + self.decoderlayer_3.flops()
        
        # Output Projection
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops
    
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
        return loss_l1
    
    def loss(self, preds, gts, img_metas, suffix=None):
        loss_l1 = self.loss_single(preds, gts)
        
        if suffix is None:
            return dict(loss_l1=loss_l1)
        else:
            loss_dict = dict()
            loss_dict[f'loss_l1_{suffix}'] = loss_l1
            return loss_dict