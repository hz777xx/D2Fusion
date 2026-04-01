import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from MambaIR import VSSBlock as RSSBlock
from torch.nn import init
import numpy as np
import os
import numbers
from MSVMamba import VSSBlock as MSEBlock
from utils.module import InvertibleConv1x1


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class SpatialFeatureExtractor(nn.Module):
    def __init__(self,
                 dim,
                ):
        super(SpatialFeatureExtractor, self).__init__()
        self.block=MSEBlock(hidden_dim=dim,drop_path=0.1,convFFN=True)

    def forward(self, x):
        x_r=x.permute(0, 2, 3, 1).contiguous()
        output=self.block(x_r)
        output=output.permute(0, 3, 1, 2).contiguous()
        return output

class SpatialFeatureFusion(nn.Module):
    def __init__(self,
                 dim,
                ):
        super(SpatialFeatureFusion, self).__init__()
        self.block=MSEBlock(hidden_dim=dim,drop_path=0.1,convFFN=True)

    def forward(self, x):
        x_r=x.permute(0, 2, 3, 1).contiguous()
        output=self.block(x_r)
        output=output.permute(0, 3, 1, 2).contiguous()
        return output

class FrequencyFeatureExtractor(nn.Module):
    def __init__(self,
                 dim,
                 ):
        super(FrequencyFeatureExtractor, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.freblock = FreBlock(channels=dim)
    def forward(self,x):
        x,amp,pha = self.freblock(self.norm1(x))
        return x,amp,pha

class FrequencyFeatureFusion(nn.Module):
    def __init__(self,
                 dim,
                 ):
        super(FrequencyFeatureFusion, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.freblock=FreBlock(channels=dim)
    def forward(self, x1,x2):
        x=x1+x2
        x,_,_= self.freblock(self.norm1(x))
        return x

##########################################################################
def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)



class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        out = torch.cat((y1, y2), 1)

        return out


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            # 若不需要自定义 DenseBlockMscale 初始化，直接返回空实现（或保留最小依赖）
            # 注：若 DenseBlockMscale 是必须的，需保留其定义，否则可替换为占位模块
            class DummyBlock(nn.Module):
                def __init__(self, ch_in, ch_out):
                    super().__init__()
                    self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 1)

                def forward(self, x):
                    return self.conv(x)

            return DummyBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class MCBlock(nn.Module):

    def __init__(self, channels):
        super(MCBlock, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels

        self.spa_att_spatial = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels // 2, channels, 3, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.cha_att_spatial = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels // 2, channels * 2, 1),
            nn.Sigmoid()
        )
        self.post_spatial = nn.Conv2d(channels * 2, channels, 3, 1, 1)


        self.spa_att_freq = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels // 2, channels, 3, 1, 1),
            nn.Sigmoid()
        )
        self.cha_att_freq = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels // 2, channels * 2, 1),
            nn.Sigmoid()
        )
        self.post_freq = nn.Conv2d(channels * 2, channels, 3, 1, 1)

        self.fused_block = InvBlock(subnet('DBNet'), channels * 2, channels)
        self.post_fused = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, salient_target_feat, spatial_feat, frequency_feat):
        spa_att_spatial_map = self.spa_att_spatial(spatial_feat - salient_target_feat)
        spa_att_spatial_res = spa_att_spatial_map * salient_target_feat + salient_target_feat

        spatial_cat = torch.cat([spa_att_spatial_res, salient_target_feat], dim=1)
        spatial_cha_att = self.cha_att_spatial(
            self.contrast(spatial_cat) + self.avgpool(spatial_cat)
        ) * spatial_cat
        spatial_out = self.post_spatial(spatial_cha_att) + salient_target_feat

        spa_att_freq_map = self.spa_att_freq(frequency_feat - salient_target_feat)
        spa_att_freq_res = spa_att_freq_map * salient_target_feat + salient_target_feat

        frequency_cat = torch.cat([spa_att_freq_res, salient_target_feat], dim=1)
        frequency_cha_att = self.cha_att_freq(
            self.contrast(frequency_cat) + self.avgpool(frequency_cat)
        ) * frequency_cat
        frequency_out = self.post_freq(frequency_cha_att) + salient_target_feat

        fused_cat = torch.cat([spatial_out, frequency_out], dim=1)
        fused_result = self.post_fused(self.fused_block(fused_cat))
        return fused_result

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

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class FIBlock(nn.Module):
    def __init__(self):
        super(FIBlock, self).__init__()

    def forward(self, vis_abs,vis_pha, inf_pha,data_VIS):
        _, _, H, W = data_VIS.shape

        fused_a = vis_abs * (torch.cos(vis_pha) + torch.cos(inf_pha))
        fused_b = vis_abs * (torch.sin(vis_pha) + torch.sin(inf_pha))
        fused_y = torch.complex(fused_a, fused_b)
        fused_y = torch.abs(torch.fft.irfft2(fused_y, s=(H, W), norm='backward'))

        return fused_y


class FreBlock(nn.Module):
    #def __init__(self, channels, args):
    def __init__(self, channels):
        super(FreBlock, self).__init__()

        self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)


    def forward(self, x):
        # print("x: ", x.shape)
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        # print("msf_amp: ", msF_amp.shape)
        amp_fuse = self.amp_fuse(msF_amp)
        # print(amp_fuse.shape, msF_amp.shape)
        amp_fuse = amp_fuse + msF_amp
        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        out = self.post(out)
        out = out + x
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        # print("out: ", out.shape)
        return out,msF_amp,msF_pha


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 dim=64,
                 drop_path=0.1,
                 attn_drop_rate=0.1,
                 img_size=128
                 ):
        super(Encoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = RSSBlock(hidden_dim=dim, drop_path=drop_path, attn_drop_rate=attn_drop_rate, d_state=16, expand=2.0,
                                       is_light_sr=False)
        self.spatialFeature = SpatialFeatureExtractor(dim=dim)
        self.frequencyFeature = FrequencyFeatureExtractor(dim=dim)

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)  # (B, inp_chans, H, W) -> (B, dim, H, W)
        inp_enc_level1_3d = to_3d(inp_enc_level1)  # (B, H*W, dim)
        out_enc_level1_3d = self.encoder_level1(inp_enc_level1_3d, (H, W))
        out_enc_level1 = to_4d(out_enc_level1_3d, h=H, w=W)
        spatial_feat = self.spatialFeature(out_enc_level1)
        frequency_feat, amp, pha = self.frequencyFeature(out_enc_level1)
        return spatial_feat, frequency_feat, out_enc_level1, amp, pha


class Decoder(nn.Module):
    def __init__(self,
                 inp_channels: int = 1,
                 out_channels: int = 1,
                 dim: int = 64,
                 num_blocks: list = [4, 4],
                 heads: list = [8, 8, 8],
                 ffn_expansion_factor: float = 2,
                 bias: bool = False,
                 LayerNorm_type: str = 'WithBias'):
        super().__init__()
        self.dim = dim
        self.reduce_channel1 = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)
        self.reduce_channel2 = nn.Conv2d(3 * dim, dim, kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(
                dim=dim,
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])]
        )

        self.output_head = nn.Sequential(  #
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, *args):
        inp_img = args[0]
        out_enc_level0 = None

        if len(args) == 4:
            _, spatial_feat, frequency_feat, fused_amp = args
            out_enc_level0 = torch.cat([spatial_feat, frequency_feat, fused_amp], dim=1)
            out_enc_level0 = self.reduce_channel2(out_enc_level0)
        elif len(args) == 3:
            _, spatial_feat, frequency_feat = args
            out_enc_level0 = torch.cat([spatial_feat, frequency_feat], dim=1)
            out_enc_level0 = self.reduce_channel1(out_enc_level0)
        elif len(args) == 2:
            _, out_enc_level0 = args
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out_enc_level1 = self.output_head(out_enc_level1)

        if inp_img is not None:
            out_enc_level1 = out_enc_level1 + inp_img

        return self.sigmoid(out_enc_level1), out_enc_level0



if __name__ == '__main__':
    height = 128
    width = 128
    inp_channels = 1,
    window_size = 8
    #modelD = MambaSpaceEncoder(dim=64).cuda()
    modelD = Restormer_Encoder().cuda()
    # modelE = BaseFeatureExtraction(dim=64,patch_size=64).cuda()
    #modelE = DetailFeatureFusion(dim=64).cuda()
    #modelD=AMP();
    X = torch.rand(size=(1,1,128,128), dtype=torch.float32).cuda()
    Y = torch.rand(size=(1,1,213,325), dtype=torch.float32).cuda()
    Z = torch.rand(size=(1, 64,456,678), dtype=torch.float32).cuda()
    var1,var2,var3,var4,var5=modelD(X)
    print(modelD)
    print(var1.shape)
    print(var2.shape)
    print(var3.shape)
    # print(Y.shape)
    # print(var3.shape)
