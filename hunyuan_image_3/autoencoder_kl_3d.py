"""
Reference code
[FLUX] https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
[DCAE] https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/models/efficientvit/dc_ae.py
"""
import os
from dataclasses import dataclass
from typing import Tuple, Optional
import math
import random
import numpy as np
from einops import rearrange
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from safetensors import safe_open
import os
from collections import OrderedDict
from collections.abc import Iterable
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput



class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H ,W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            reduce_dim = list(range(1, self.mean.ndim))
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=reduce_dim,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var +
                    self.var / other.var -
                    1.0 -
                    self.logvar +
                    other.logvar,
                    dim=reduce_dim,
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean

@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor
    posterior: Optional[DiagonalGaussianDistribution] = None

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(create_custom_forward(module), *inputs, use_reentrant=False)
    else:
        return module(*inputs)


class Conv3d(nn.Conv3d):
    """Perform Conv3d on patches with numerical differences from nn.Conv3d within 1e-5. Only symmetric padding is supported."""

    def forward(self, input):
        B, C, T, H, W = input.shape
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i in range(len(chunks)):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunks[i],
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
                        value=0,
                    )
                    if i > 0:
                        padded_chunk[:, :, :self.padding[0]] = chunks[i - 1][:, :, -self.padding[0]:]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0]:] = chunks[i + 1][:, :, :self.padding[0]]
                else:
                    padded_chunk = chunks[i]
                padded_chunks.append(padded_chunk)
            padding_bak = self.padding
            self.padding = (0, self.padding[1], self.padding[2])
            outputs = []
            for i in range(len(padded_chunks)):
                outputs.append(super().forward(padded_chunks[i]))
            self.padding = padding_bak
            return torch.cat(outputs, dim=-3)
        else:
            return super().forward(input)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        self.add_temporal_downsample = add_temporal_downsample
        stride = (2, 2, 2) if add_temporal_downsample else (1, 2, 2)  # THW
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=0)

    def forward(self, x: Tensor):
        spatial_pad = (0, 1, 0, 1, 0, 0)  # WHT
        x = nn.functional.pad(x, spatial_pad, mode="constant", value=0)

        temporal_pad = (0, 0, 0, 0, 0, 1) if self.add_temporal_downsample else (0, 0, 0, 0, 1, 1)
        x = nn.functional.pad(x, temporal_pad, mode="replicate")

        x = self.conv(x)
        return x


class DownsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)

        B, C, T, H, W = shortcut.shape
        shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
        return h + shortcut


class Upsample(nn.Module):
    def __init__(self, in_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        self.add_temporal_upsample = add_temporal_upsample
        self.scale_factor = (2, 2, 2) if add_temporal_upsample else (1, 2, 2)  # THW
        self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class UpsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = Conv3d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        return h + shortcut


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(i_level >= np.log2(ffactor_spatial // ffactor_temporal))
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = DownsampleDCAE(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv3d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            use_checkpointing = bool(self.training and self.gradient_checkpointing)

            # downsampling
            h = self.conv_in(x)
            for i_level in range(len(self.block_out_channels)):
                for i_block in range(self.num_res_blocks):
                    h = forward_with_checkpointing(self.down[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
                if hasattr(self.down[i_level], "downsample"):
                    h = forward_with_checkpointing(self.down[i_level].downsample, h, use_checkpointing=use_checkpointing)

            # middle
            h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

            # end
            group_size = self.block_out_channels[-1] // (2 * self.z_channels)
            shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
            h = self.norm_out(h)
            h = swish(h)
            h = self.conv_out(h)
            h += shortcut
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # z to block_in
        block_in = block_out_channels[0]
        self.conv_in = Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = UpsampleDCAE(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False


    def forward(self, z: Tensor) -> Tensor:
        with torch.no_grad():
            use_checkpointing = bool(self.training and self.gradient_checkpointing)
            # z to block_in
            repeats = self.block_out_channels[0] // (self.z_channels)
            h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)
            # middle
            h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)
            # upsampling
            for i_level in range(len(self.block_out_channels)):
                for i_block in range(self.num_res_blocks + 1):
                    h = forward_with_checkpointing(self.up[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
                if hasattr(self.up[i_level], "upsample"):
                    h = forward_with_checkpointing(self.up[i_level].upsample, h, use_checkpointing=use_checkpointing)
            # end
            h = self.norm_out(h)
            h = swish(h)
            h = self.conv_out(h)
        return h


class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        only_encoder: bool = False,
        only_decoder: bool = False,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        if not only_decoder:
            self.encoder = Encoder(
                in_channels=in_channels,
                z_channels=latent_channels,
                block_out_channels=block_out_channels,
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                downsample_match_channel=downsample_match_channel,
            )
        if not only_encoder:
            self.decoder = Decoder(
                z_channels=latent_channels,
                out_channels=out_channels,
                block_out_channels=list(reversed(block_out_channels)),
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                upsample_match_channel=upsample_match_channel,
            )

        self.use_slicing = False
        self.slicing_bsz = 1
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False
        self.use_tiling_during_training = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.125

        self.use_compile = False

        self.empty_cache = torch.empty(0, device="cuda")

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_tiling_during_training(self, use_tiling: bool = True):
        self.use_tiling_during_training = use_tiling

    def disable_tiling_during_training(self):
        self.enable_tiling_during_training(False)

    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        self.disable_spatial_tiling()

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (x / blend_extent)
        return b

    def spatial_tiled_encode(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))  # 256 * (1 - 0.25) = 192
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)  # 8 * 0.25 = 2
        row_limit = self.tile_latent_min_size - blend_extent  # 8 - 2 = 6

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=-2)
        return moments

    def temporal_tiled_encode(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))  # 64 * (1 - 0.25) = 48
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)  # 8 * 0.25 = 2
        t_limit = self.tile_latent_min_tsize - blend_extent  # 8 - 2 = 6

        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i: i + self.tile_sample_min_tsize, :, :]
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        moments = torch.cat(result_row, dim=-3)
        return moments

    def _decode_tiles_for_rank(self, z: torch.Tensor, my_linear_indices: list, num_cols: int, overlap_size: int):
        """解码当前 rank 分配到的 tiles，并返回解码结果和元信息。"""
        H_out_std = self.tile_sample_min_size
        W_out_std = self.tile_sample_min_size
        decoded_tiles = []
        decoded_metas = []

        for lin_idx in my_linear_indices:
            ri = lin_idx // num_cols
            rj = lin_idx % num_cols
            i = ri * overlap_size
            j = rj * overlap_size
            tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
            dec = self.decoder(tile)
            # 对边界 tile 的输出做右/下方向 padding 到标准尺寸
            pad_h = max(0, H_out_std - dec.shape[-2])
            pad_w = max(0, W_out_std - dec.shape[-1])
            if pad_h > 0 or pad_w > 0:
                dec = F.pad(dec, (0, pad_w, 0, pad_h, 0, 0), "constant", 0)
            decoded_tiles.append(dec)
            decoded_metas.append(torch.tensor([ri, rj, pad_w, pad_h], device=z.device, dtype=torch.int64))

        return decoded_tiles, decoded_metas

    def _pad_tiles_to_same_count(self, decoded_tiles: list, decoded_metas: list, tiles_per_rank: int,
                                  T_out: int, device, dtype):
        """将 tiles 列表填充到相同长度，以便进行 all_gather。"""
        while len(decoded_tiles) < tiles_per_rank:
            decoded_tiles.append(torch.zeros(
                [1, 3, T_out, self.tile_sample_min_size, self.tile_sample_min_size],
                device=device, dtype=dtype
            ))
            decoded_metas.append(torch.tensor(
                [-1, -1, self.tile_sample_min_size, self.tile_sample_min_size],
                device=device, dtype=torch.int64
            ))
        return decoded_tiles, decoded_metas

    def _reconstruct_tile_grid(self, tiles_gather_list: list, metas_gather_list: list,
                                num_rows: int, num_cols: int, world_size: int):
        """根据 all_gather 的结果重建 tile 网格。"""
        rows = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(world_size):
            gathered_tiles_r = tiles_gather_list[r]
            gathered_metas_r = metas_gather_list[r]
            for k in range(gathered_tiles_r.shape[0]):
                ri = int(gathered_metas_r[k][0])
                rj = int(gathered_metas_r[k][1])
                if ri < 0 or rj < 0:
                    continue
                if ri < num_rows and rj < num_cols:
                    pad_w = int(gathered_metas_r[k][2])
                    pad_h = int(gathered_metas_r[k][3])
                    h_end = None if pad_h == 0 else -pad_h
                    w_end = None if pad_w == 0 else -pad_w
                    rows[ri][rj] = gathered_tiles_r[k][:, :, :, :h_end, :w_end]
        return rows

    def _blend_and_concat_rows(self, rows: list, blend_extent: int, row_limit: int, skip_none: bool = False):
        """对 tile 网格进行融合并拼接成最终结果。"""
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if skip_none and tile is None:
                    continue
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _spatial_tiled_decode_distributed(self, z: torch.Tensor, H: int, W: int, T: int,
                                           overlap_size: int, blend_extent: int, row_limit: int):
        """分布式多卡解码逻辑。"""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        num_rows = math.ceil(H / overlap_size)
        num_cols = math.ceil(W / overlap_size)
        total_tiles = num_rows * num_cols
        tiles_per_rank = math.ceil(total_tiles / world_size)

        print(f"==={torch.distributed.get_rank()},  {total_tiles=}, {tiles_per_rank=}, {world_size=}")

        my_linear_indices = list(range(rank, total_tiles, world_size))
        if not my_linear_indices:
            my_linear_indices = [0]
        print(f"==={torch.distributed.get_rank()},  {my_linear_indices=}")

        decoded_tiles, decoded_metas = self._decode_tiles_for_rank(z, my_linear_indices, num_cols, overlap_size)

        T_out = decoded_tiles[0].shape[2] if decoded_tiles else (T - 1) * self.ffactor_temporal + 1
        dtype = decoded_tiles[0].dtype if decoded_tiles else z.dtype
        decoded_tiles, decoded_metas = self._pad_tiles_to_same_count(
            decoded_tiles, decoded_metas, tiles_per_rank, T_out, z.device, dtype
        )

        decoded_tiles = torch.stack(decoded_tiles, dim=0)
        decoded_metas = torch.stack(decoded_metas, dim=0)

        tiles_gather_list = [torch.empty_like(decoded_tiles) for _ in range(world_size)]
        metas_gather_list = [torch.empty_like(decoded_metas) for _ in range(world_size)]

        dist.all_gather(tiles_gather_list, decoded_tiles)
        dist.all_gather(metas_gather_list, decoded_metas)

        if rank != 0:
            return torch.empty(0, device=z.device)

        rows = self._reconstruct_tile_grid(tiles_gather_list, metas_gather_list, num_rows, num_cols, world_size)
        return self._blend_and_concat_rows(rows, blend_extent, row_limit, skip_none=True)

    def _spatial_tiled_decode_single(self, z: torch.Tensor, H: int, W: int,
                                      overlap_size: int, blend_extent: int, row_limit: int):
        """单卡串行解码逻辑。"""
        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        return self._blend_and_concat_rows(rows, blend_extent, row_limit, skip_none=False)

    def spatial_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))  # 24 * (1 - 0.125) = 21
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)  # 384 * 0.125 = 48
        row_limit = self.tile_sample_min_size - blend_extent  # 384 - 48 = 336

        # 分布式/多卡逻辑
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return self._spatial_tiled_decode_distributed(z, H, W, T, overlap_size, blend_extent, row_limit)

        # 单卡：原有串行逻辑
        return self._spatial_tiled_decode_single(z, H, W, overlap_size, blend_extent, row_limit)

    def temporal_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)  # 64 * 0.25 = 16
        t_limit = self.tile_sample_min_tsize - blend_extent  # 64 - 16 = 48
        assert 0 < overlap_size < self.tile_latent_min_tsize

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i: i + self.tile_latent_min_tsize, :, :]
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self.decoder(tile)
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        dec = torch.cat(result_row, dim=-3)
        return dec

    def encode(self, x: Tensor, return_dict: bool = True):

        def _encode(x):
            if self.use_temporal_tiling and x.shape[-3] > self.tile_sample_min_tsize:
                return self.temporal_tiled_encode(x)
            if self.use_spatial_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
                return self.spatial_tiled_encode(x)

            if self.use_compile:
                @torch.compile
                def encoder(x):
                    return self.encoder(x)
                return encoder(x)
            return self.encoder(x)

        if len(x.shape) != 5:  # (B, C, T, H, W)
            x = x[:, :, None]
        assert len(x.shape) == 5  # (B, C, T, H, W)
        if x.shape[2] == 1:
            x = x.expand(-1, -1, self.ffactor_temporal, -1, -1)
        else:
            assert x.shape[2] != self.ffactor_temporal and x.shape[2] % self.ffactor_temporal == 0

        if self.use_slicing and x.shape[0] > 1:
            if self.slicing_bsz == 1:
                encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            else:
                sections = [self.slicing_bsz] * (x.shape[0] // self.slicing_bsz)
                if x.shape[0] % self.slicing_bsz != 0:
                    sections.append(x.shape[0] % self.slicing_bsz)
                encoded_slices = [_encode(x_slice) for x_slice in x.split(sections)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: Tensor, return_dict: bool = True, generator=None):

        def _decode(z):
            if self.use_temporal_tiling and z.shape[-3] > self.tile_latent_min_tsize:
                return self.temporal_tiled_decode(z)
            if self.use_spatial_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
                return self.spatial_tiled_decode(z)
            return self.decoder(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return self.empty_cache

        if z.shape[-3] == 1:
            decoded = decoded[:, :, -1:]
        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def decode_dist(self, z: Tensor, return_dict: bool = True, generator=None):
        z = z.cuda()
        self.use_spatial_tiling = True
        decoded = self.decode(z)
        self.use_spatial_tiling = False
        return decoded

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_posterior: bool = True,
        return_dict: bool = True
    ):
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        return DecoderOutput(sample=dec, posterior=posterior) if return_dict else (dec, posterior)

    def random_reset_tiling(self, x: torch.Tensor):
        if x.shape[-3] == 1:
            self.disable_spatial_tiling()
            self.disable_temporal_tiling()
            return

        # tiling在input_shape和sample_size上限制很多，任意的input_shape和sample_size很可能不满足条件，因此这里使用固定值
        min_sample_size = int(1 / self.tile_overlap_factor) * self.ffactor_spatial
        min_sample_tsize = int(1 / self.tile_overlap_factor) * self.ffactor_temporal
        sample_size = random.choice([None, 1 * min_sample_size, 2 * min_sample_size, 3 * min_sample_size])
        if sample_size is None:
            self.disable_spatial_tiling()
        else:
            self.tile_sample_min_size = sample_size
            self.tile_latent_min_size = sample_size // self.ffactor_spatial
            self.enable_spatial_tiling()

        sample_tsize = random.choice([None, 1 * min_sample_tsize, 2 * min_sample_tsize, 3 * min_sample_tsize])
        if sample_tsize is None:
            self.disable_temporal_tiling()
        else:
            self.tile_sample_min_tsize = sample_tsize
            self.tile_latent_min_tsize = sample_tsize // self.ffactor_temporal
            self.enable_temporal_tiling()

def load_sharded_safetensors(model_dir):
    """
    手动加载分片的 safetensors 文件

    Args:
        model_dir: 包含分片文件的目录路径

    Returns:
        合并后的完整权重字典
    """
    # 获取所有分片文件并按编号排序
    shard_files = []
    for file in os.listdir(model_dir):
        if file.endswith(".safetensors"):
            shard_files.append(file)

    # 按分片编号排序
    shard_files.sort(key=lambda x: int(x.split("-")[1]))

    print(f"找到 {len(shard_files)} 个分片文件")

    # 合并所有权重
    merged_state_dict = dict()

    for shard_file in shard_files:
        shard_path = os.path.join(model_dir, shard_file)
        print(f"加载分片: {shard_file}")

        # 使用 safetensors 加载当前分片
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                merged_state_dict[key] = tensor

    print(f"合并完成，总键数量: {len(merged_state_dict)}")
    return merged_state_dict

def load_weights(model, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    def update_state_dict(state_dict: dict[str, torch.Tensor], name, weight):
        if name not in state_dict:
            raise ValueError(f"Unexpected weight {name}")

        model_tensor = state_dict[name]
        if model_tensor.shape != weight.shape:
            raise ValueError(
                f"Shape mismatch for weight {name}: "
                f"model tensor shape {model_tensor.shape} vs. "
                f"loaded tensor shape {weight.shape}"
            )
        if isinstance(weight, torch.Tensor):
            model_tensor.data.copy_(weight.data)
        else:
            raise ValueError(
                f"Unsupported tensor type in load_weights "
                f"for {name}: {type(weight)}"
            )

    loaded_params = set()
    for name, load_tensor in weights.items():
        updated = True
        name = name.replace('vae.', '')
        if name in model.state_dict():
            update_state_dict(model.state_dict(), name, load_tensor)
        else:
            updated = False

        if updated:
            loaded_params.add(name)

    return loaded_params

def _worker(path, config, 
    rank=None, world_size=None, port=None, req_queue=None, rsp_queue=None):
    """
    each rank's worker:
      - idle: block on req_queue.get() (CPU blocking, no GPU)
      - receive request: run runner.predict(), all ranks forward
      - only rank0 put result to rsp_queue
    """
    # _tame_cpu_threads_and_comm()
    # basic env
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # device binding should be early than all CUDA operations
    visible = torch.cuda.device_count()
    assert visible >= world_size, f"可见卡数 {visible} < world_size {world_size}"
    local_rank = int(os.environ["LOCAL_RANK"])
    
    print(f"[worker {rank}] bind to cuda:{local_rank} (visible={visible})", flush=True)
    if not torch.distributed.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    #from .. import load_vae

    #vae = load_vae(vae_type, vae_precision, device, logger, args, weights_only, only_encoder, only_decoder, sample_size, skip_create_dist=True)
    #vae = vae.cuda()
    vae = AutoencoderKLConv3D.from_config(config)
    merged_state_dict = load_sharded_safetensors(path)
    loaded_params = load_weights(vae, merged_state_dict) 
    vae = vae.cuda()
    vae.eval()  # 关闭 Dropout、BatchNorm 训练行为
    for param in vae.parameters():
        param.requires_grad = False  #
    
    while True:
        req = req_queue.get()  # blocking
        if req == "__STOP__":
            break
        out = vae.decode_dist(req, return_dict=False)
        if rank == 0:
            rsp_queue.put(out)

    #try:
    #    while True:
    #        # blocking on CPU queue
    #        req = req_queue.get()  # blocking
    #        if req == "__STOP__":
    #            break
    #        out = vae.decode_dist(req, return_dict=False)
    #        if rank == 0:
    #            rsp_queue.put(out)
    #finally:
    #    # destroy process group before exit
    #    try:
    #        dist.destroy_process_group()
    #    except Exception:
    #        pass

#def _find_free_port():
#    import socket
#    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#        s.bind(("127.0.0.1", 0))
#        return s.getsockname()[1]

# 避免端口冲突的常见做法
def _find_free_port(start_port=8100, max_attempts=900):
    import socket
    """获取一个可用的端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket() as s:
                s.bind(('localhost', port))
                return s.getsockname()[1]  # 返回实际绑定的端口
        except OSError:
            continue
    raise RuntimeError("找不到可用端口")

class AutoencoderKLConv3D_Dist(AutoencoderKLConv3D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        only_encoder: bool = False,
        only_decoder: bool = False,
    ):
        super().__init__(in_channels, out_channels, latent_channels, block_out_channels, layers_per_block, ffactor_spatial, ffactor_temporal, sample_size, sample_tsize, scaling_factor, shift_factor, downsample_match_channel, upsample_match_channel, only_encoder, only_decoder)

    def create_dist(self, path, config, 
    ):
        self.world_size = 8
        self.port = _find_free_port()
        ctx = mp.get_context("spawn")
        # 每个 rank 一个请求队列（纯 CPU），再加一个公共响应队列
        self.req_queues = [ctx.Queue() for _ in range(self.world_size)]
        self.rsp_queue = ctx.Queue()

        self.procs = []
        for rank in range(self.world_size):
            p = ctx.Process(
                target=_worker,
                args=(
                    path, config, 
                    rank, self.world_size, self.port,
                    self.req_queues[rank], self.rsp_queue,
                ),
                daemon=True,
            )
            p.start()
            self.procs.append(p)
    
    def decode(self, z: Tensor, return_dict: bool = True, generator=None):
        """
        synchronous inference: put the same request to all ranks' queues.
        return rank0's result.
        """
        # check alive
        for p in self.procs:
            if not p.is_alive():
                raise RuntimeError("One of the processes is not alive")

        # put to each rank's queue
        for q in self.req_queues:
            q.put(z)

        # wait for rank0's result
        return self.rsp_queue.get(timeout=None)
