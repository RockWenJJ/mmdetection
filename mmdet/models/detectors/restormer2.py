import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from collections import OrderedDict
from mmcv.runner import BaseModule
from einops import rearrange
from ..builder import DETECTORS, build_loss
from ..utils import DownSampleLayer

##########################################################################
## Layer Norm

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


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        
        hidden_features = int(dim * ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class ChannelWiseSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelWiseSelfAttention, self).__init__()
        self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.qk_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = self.qk_pool(q)
        k = self.qk_pool(k)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class PixelWiseSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(PixelWiseSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.pool = nn.AdaptiveAvgPool2d((17, 17))
        self.qkv = nn.Linear(dim, dim * 3)
        self.weight = nn.Parameter(torch.ones(64, 64))
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_ave = self.pool(x)
        # x = rearrange(x, 'b c h w -> b (h w) c')
        x_ave = rearrange(x_ave, 'b c h w -> b (h w) c')
        qkv = self.qkv(x_ave)
        q_ave, k_ave, v_ave = qkv.chunk(3, dim=2)
        
        attn_ave = (q_ave @ k_ave.transpose(-2, -1)).softmax(dim=-1)
        # attn = F.interpolate(attn_ave, mode='bilinear', size=(h*w, h*w))
        
        out_ave = attn_ave @ x_ave
        out_ave = rearrange(out_ave, 'b (h w) c -> b c h w', h=17, w=17)
        out = F.interpolate(out_ave, mode='bilinear', size=(h, w))
        weight = F.interpolate(self.weight[None, None, ...], mode='bilinear', size=(h, w))
        
        out = x + weight * out
        
        # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.c_attn = ChannelWiseSelfAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.p_attn = PixelWiseSelfAttention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    
    def forward(self, x):
        x1 = self.c_attn(self.norm1(x))
        x2 = self.p_attn(self.norm2(x))
        x = x + x1 + x2
        x = x + self.ffn(self.norm3(x))
        
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.proj = nn.Sequential(nn.ReflectionPad2d(1),
                nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=0),
                nn.InstanceNorm2d(embed_dim),
                nn.ReLU(inplace=True))
        self.downsample = DownSampleLayer(True, embed_dim)
    
    def forward(self, x):
        x = self.downsample(self.proj(x))
        
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    
    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    
    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
@DETECTORS.register_module()
class Restormer2(BaseModule):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 loss_cfg=dict(type='L1Loss', loss_weight=1.0),
                 multi_scales=False,
                 **kwargs
                 ):
        
        super(Restormer2, self).__init__()
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        
        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################
        
        self.final_up = Upsample(int(dim * 2 ** 1))
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
        self.l1_loss = build_loss(loss_cfg)
        self.multi_scales = False
        
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
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(self.final_up(out_dec_level1)) + inp_img
            # out_dec_level1 = self.output(self.final_up(out_dec_level1))
        
        return tuple([out_dec_level1])

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
    
        # # If the loss_vars has different length, GPUs will wait infinitely
        # if dist.is_available() and dist.is_initialized():
        #     log_var_length = torch.tensor(len(log_vars), device=loss.device)
        #     dist.all_reduce(log_var_length)
        #     message = (f'rank {dist.get_rank()}' +
        #                f' len(log_vars): {len(log_vars)}' + ' keys: ' +
        #                ','.join(log_vars.keys()))
        #     assert log_var_length == len(log_vars) * dist.get_world_size(), \
        #         'loss log variables are different across GPUs!\n' + message
    
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
            loss_dict=dict()
            loss_dict[f'loss_l1_{suffix}'] = loss_l1
            return loss_dict