# Code Implementation of the MSVMamba Model
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from utils.module import SelectiveScan, flops_selective_scan_fn, \
    flops_selective_scan_ref, print_jit_input_names, Mlp, CrossScan, CrossMerge, \
    selective_scan_flatten, SEModule, ConvFFN, x_selective_scan

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.d_inner = d_inner
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)
        if kwargs.get('sscore_type', 'None') != 'None':
            ms_stage, current_stage = kwargs.get('ms_stage'), kwargs.get('current_layer')
            if current_stage not in ms_stage:
                kwargs['sscore_type'] = 'None'

        if kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            forward_type = "multiscale_ssm"
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        if kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            self.K = 1 + kwargs.get('ms_split')[0]

        self.K2 = self.K if forward_type not in ["share_a"] else 1

        if kwargs.get('add_se', False):
            self.se = SEModule(d_expand, reduction=8)

        if kwargs.get('ms_fusion', None) == None:
            if kwargs.get('upsample', None) == 'interpolate':
                pass
            elif kwargs.get('upsample', None) == 'conv':
                if kwargs.get('current_layer', 0) == 3:
                    self.upsample = nn.ConvTranspose2d(
                        in_channels=d_expand,
                        out_channels=d_expand,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=d_expand,
                        bias=conv_bias,
                    )
                else:
                    self.upsample = nn.ConvTranspose2d(
                        in_channels=d_expand,
                        out_channels=d_expand,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        groups=d_expand,
                        bias=conv_bias,
                    )

        # forward_type =======================================
        self.forward_core = dict(
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            flatten_ssm=self.forward_core_flatten,
            multiscale_ssm=self.forward_core_multiscale,
        ).get(forward_type, self.forward_corev2)

        # in proj =======================================

        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            stride = 1
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                stride=stride,
                **factory_kwargs,
            )  # branch 0, convert B, C, H, W to B, C, H, W
        if kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            if kwargs.get('add_conv', True):
                b1_stride = 2
                self.conv2d_b1 = nn.Conv2d(
                    in_channels=d_expand,
                    out_channels=d_expand,
                    groups=d_expand,
                    bias=conv_bias,
                    kernel_size=7,
                    stride=b1_stride,
                    padding=3,
                    **factory_kwargs,
                )  # bracnh 1, convert B, C, H, W to B, C, H//4, W//4
            if kwargs.get('sep_norm', False):
                self.out_norm0 = self.out_norm
                self.out_norm1 = nn.LayerNorm(d_inner)

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # other kwargs =======================================
        self.kwargs = kwargs
        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        self.debug = False

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if self.debug: debug_rec = []
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = x_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
            **self.kwargs,
        )
        x, debug_rec = x[0], x[1]
        if self.ssm_low_rank:
            x = self.out_rank(x)
        if self.debug:
            return x, debug_rec
        return x

    def forward_core_flatten(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 2, 1).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = x.transpose(1, 2).contiguous()  # back to (B, L, C)
        x = selective_scan_flatten(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
            **self.kwargs,
        )  # (B, L, C)
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward_core_multiscale(self, xs: list, nrows=-1, channel_first=False):
        nrows = 1
        ys, debug_rec = [], []
        for i, x in enumerate(xs):
            if not channel_first:
                x = x.permute(0, 2, 1).contiguous()
            if self.ssm_low_rank:
                x = self.in_rank(x)
            if self.kwargs.get('sep_norm', False):
                norm_name = getattr(self, "out_norm" + str(i), nn.LayerNorm(self.d_inner))
            else:
                norm_name = getattr(self, "out_norm", None)
            if i == 0:
                proj_weight = self.x_proj_weight[[i]]
                dt_projs_weight = self.dt_projs_weight[[i]]
                dt_projs_bias = self.dt_projs_bias[[i]]
                A_logs = self.A_logs[i * self.d_inner:(i + 1) * self.d_inner]
                Ds = self.Ds[i * self.d_inner:(i + 1) * self.d_inner]
            else:
                proj_weight = self.x_proj_weight[i:]
                dt_projs_weight = self.dt_projs_weight[i:]
                dt_projs_bias = self.dt_projs_bias[i:]
                A_logs = self.A_logs[i * self.d_inner:]
                Ds = self.Ds[i * self.d_inner:]
            # if not debug  mode, remove x_rec
            x, debug = x_selective_scan(
                x, proj_weight, None, dt_projs_weight, dt_projs_bias,
                A_logs, Ds,
                norm_name,
                nrows=nrows, delta_softplus=True, force_fp32=self.training,
                **self.kwargs,
            )

            if self.ssm_low_rank:
                x = self.out_rank(x)  # (B, L, C)
            ys.append(x)
            debug_rec.append(debug)
        if self.debug:
            return ys, debug_rec
        return ys

    def forward(self, x: torch.Tensor, h_tokens=None, w_tokens=None, **kwargs):

        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            b, h, w, d = x.shape
            if not self.disable_z_act:
                z = self.act(z)  # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            x_b1 = x
            x = self.act(self.conv2d(x))  # (b, d, h, w)
            if self.kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
                if self.kwargs.get('add_conv', True):
                    x_b1 = self.act(self.conv2d_b1(x_b1))  # (b, d, h//4, w//4)
                h_b1, w_b1 = x_b1.shape[2:]
                # reverse horizontal scan
                x_hori_r = x_b1.flatten(2).flip(-1)
                # vertical scan
                x_vert = x_b1.transpose(2, 3).flatten(2).contiguous()
                # reverse vertical scan
                x_vert_r = x_b1.transpose(2, 3).flatten(2).flip(-1).contiguous()
                splits = self.kwargs.get('ms_split')[1]
                if splits == 3:
                    x_b1 = torch.cat([x_hori_r, x_vert, x_vert_r], dim=2)  # (b, d, h//2*w//2*3)
                    x = x.flatten(2)  # (b, d, h*w)
                elif splits == 2:
                    x_b1 = torch.cat([x_hori_r, x_vert_r], dim=2)
                elif splits == 1:
                    x_b1 = x_vert_r
                x = [x_b1, x]
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        if self.debug:
            y, debug_rec = self.forward_core(x, channel_first=(self.d_conv > 1))
        else:
            y = self.forward_core(x, channel_first=(self.d_conv > 1))

        if self.kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            y_b0, y_b1 = y[1], y[0]  # (b, h//4*w//4*3, d)
            if splits == 3:
                y_hori_r = y_b1[:, :h_b1 * w_b1].flip(-2).view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
                y_vert = y_b1[:, h_b1 * w_b1:h_b1 * w_b1 * 2].view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1,
                                                                                                              2)
                y_vert_r = y_b1[:, h_b1 * w_b1 * 2:].flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1,
                                                                                                              2)
                y_b1 = y_hori_r + y_vert + y_vert_r
            elif splits == 2:
                y_hori_r = y_b1[:, :h_b1 * w_b1].flip(-2).view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
                y_vert_r = y_b1[:, h_b1 * w_b1:].flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
                y_b1 = y_hori_r + y_vert_r
            elif splits == 1:
                y_vert_r = y_b1.flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
                y_b1 = y_vert_r
            y_b1 = F.interpolate(y_b1, size=(h, w), mode='bilinear', align_corners=False)
            y_b1 = y_b1.permute(0, 2, 3, 1).contiguous()
            y = y_b0.view(b, h, w, -1) + y_b1

        if getattr(self, "__DEBUG__", False):
            if self.kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
                ys_b1 = debug_rec[0]['ys'].view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b * 3, -1, h_b1, w_b1)
                ys_b1 = F.interpolate(ys_b1, size=(h, w), mode='bilinear', align_corners=False)  # (b*3, d, h, w)
                ys_b1 = ys_b1.view(b, 3, -1, h, w).contiguous()  # (b, 3, d, h, w)
                temp = ys_b1[:, 0].clone()
                ys_b1[:, 0] = ys_b1[:, 1]
                ys_b1[:, 1] = temp
                ys_b0 = debug_rec[1]['ys'].view(b, 1, -1, h, w).contiguous()  # (b, 1, d, h, w)
                ys = torch.cat([ys_b0, ys_b1], dim=1)  # (b, 4, d, h, w)

                xs_b1 = debug_rec[0]['xs'].view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b * 3, -1, h_b1, w_b1)
                xs_b1 = F.interpolate(xs_b1, size=(h, w), mode='bilinear', align_corners=False)
                xs_b1 = xs_b1.view(b, 3, -1, h, w).contiguous()
                temp = xs_b1[:, 0].clone()
                xs_b1[:, 0] = xs_b1[:, 1]
                xs_b1[:, 1] = temp
                xs_b0 = debug_rec[1]['xs'].view(b, 1, -1, h, w).contiguous()
                xs = torch.cat([xs_b0, xs_b1], dim=1).view(b, -1, h * w)  # (b, 4, d, h, w)

                A_logs_b1, dts_b1, bs_b1, cs_b1, ds_b1, delta_bias_b1 = debug_rec[0]['As'], debug_rec[0]['dts'], \
                debug_rec[0]['Bs'], debug_rec[0]['Cs'], debug_rec[0]['Ds'], debug_rec[0]['delta_bias']
                A_logs_b0, dts_b0, bs_b0, cs_b0, ds_b0, delta_bias_b0 = debug_rec[1]['As'], debug_rec[1]['dts'], \
                debug_rec[1]['Bs'], debug_rec[1]['Cs'], debug_rec[1]['Ds'], debug_rec[1]['delta_bias']
                A_logs = torch.cat([A_logs_b0, A_logs_b1.repeat(3, 1)], dim=0)
                delta_bias = torch.cat([delta_bias_b0, delta_bias_b1.repeat(3)], dim=0)
                ds = torch.cat([ds_b0, ds_b1.repeat(3)], dim=0)
                # dts, bs, cs correspond to v,k,q in transformer
                dts_b1 = dts_b1.view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b * 3, -1, h_b1, w_b1)
                dts_b1 = F.interpolate(dts_b1, size=(h, w), mode='bilinear', align_corners=False)
                dts_b1 = dts_b1.view(b, 3, -1, h * w)
                temp = dts_b1[:, 0].clone()
                dts_b1[:, 0] = dts_b1[:, 1]
                dts_b1[:, 1] = temp

                dts_b1 = dts_b1.view(b, -1, h * w).contiguous()
                dts = torch.cat([dts_b0, dts_b1], dim=1).view(b, -1, h * w)  # (b, 4, d, h*w)

                bs_b1 = bs_b1.view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b * 3, -1, h_b1, w_b1)
                bs_b1 = F.interpolate(bs_b1, size=(h, w), mode='bilinear', align_corners=False)
                bs_b1 = bs_b1.view(b, 3, -1, h * w)
                temp = bs_b1[:, 0].clone()
                bs_b1[:, 0] = bs_b1[:, 1]
                bs_b1[:, 1] = temp
                bs = torch.cat([bs_b0, bs_b1], dim=1)  # (b, 4, d, h*w)

                cs_b1 = cs_b1.view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b * 3, -1, h_b1, w_b1)
                cs_b1 = F.interpolate(cs_b1, size=(h, w), mode='bilinear', align_corners=False)
                cs_b1 = cs_b1.view(b, 3, -1, h * w)
                temp = cs_b1[:, 0].clone()
                cs_b1[:, 0] = cs_b1[:, 1]
                cs_b1[:, 1] = temp
                cs = torch.cat([cs_b0, cs_b1], dim=1)  # (b, 4, d, h*w)
            else:
                xs, ys = debug_rec['xs'], debug_rec['ys']
                A_logs, dts, bs, cs, ds, delta_bias = debug_rec['As'], debug_rec['dts'], debug_rec['Bs'], debug_rec[
                    'Cs'], debug_rec['Ds'], debug_rec['delta_bias']

            setattr(self, "__data__", dict(
                A_logs=A_logs, Bs=bs, Cs=cs, Ds=ds,
                us=xs, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))

        if self.kwargs.get('add_se', False):
            y = y.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            y = self.se(y)
            y = y.permute(0, 2, 3, 1).contiguous()
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                simple_init=ssm_simple_init,
                # ==========================
                forward_type=forward_type,
                **kwargs,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)
        if kwargs.get('convFFN', False):
            ffn_drop = kwargs.get('ffn_dropout', 0.2)
            self.convFFN = ConvFFN(hidden_dim, expansion=2, drop=ffn_drop)
            self.norm2 = norm_layer(hidden_dim)
        self.kwargs = kwargs

    def _forward(self, input: torch.Tensor, h_tokens=None, w_tokens=None):
        if self.ssm_branch:
            x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        if self.kwargs.get('convFFN', False):
            x = x + self.drop_path(
                self.convFFN(self.norm2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous())

        return x

    def forward(self, input: torch.Tensor, h_tokens=None, w_tokens=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, h_tokens=None, w_tokens=None)
        else:
            return self._forward(input, h_tokens=None, w_tokens=None)


if __name__ == "__main__":
    X = torch.rand(size=(1, 128 * 128, 64), dtype=torch.float32).cuda()
    modelD = VSSBlock(hidden_dim=64, drop_path=0.1, convFFN=True).cuda()
    X = modelD(X, (128, 128))
    print(X.shape)
    print(modelD)
