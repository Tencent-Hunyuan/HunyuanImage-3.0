# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import random
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union, Optional, Dict, Any, Tuple, Callable

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.cuda import nvtx
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, StaticCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerationMixin, GenerationConfig, ALL_CACHE_NAMES
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
)

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

try:
    import flashinfer
except Exception as e:
    flashinfer = None

from .autoencoder_kl_3d import AutoencoderKLConv3D
from .configuration_hunyuan import HunyuanImage3Config
from .hunyuan_image_3_pipeline import HunyuanImage3Text2ImagePipeline, FlowMatchDiscreteScheduler
from .image_processor import HunyuanImage3ImageProcessor
from .siglip2 import Siglip2VisionTransformer, LightProjector
from .tokenizer_wrapper import TokenizerWrapper, ImageInfo, JointImageInfo
from .system_prompt import get_system_prompt, t2i_system_prompts


logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func

# Type aliases
BatchRaggedImages = Union[torch.Tensor, List[Union[torch.Tensor, List[torch.Tensor]]]]
BatchRaggedTensor = Union[torch.Tensor, List[torch.Tensor]]


_CONFIG_FOR_DOC = "HunyuanImage3Config"

Hunyuan_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`HunyuanImage3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# =======================================================
#     Helper Functions
# =======================================================

def default(val, d):
    return val if val is not None else d


def to_device(data, device):
    if device is None:
        return data
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    else:
        return data


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def real_batched_index_select(t, dim, idx):
    """ index_select for batched index and batched t """
    assert t.ndim >= 2 and idx.ndim >= 2, f"{t.ndim=} {idx.ndim=}"
    assert len(t) == len(idx), f"{len(t)=} != {len(idx)=}"
    return torch.stack([torch.index_select(t[i], dim - 1, idx[i]) for i in range(len(t))])


# =======================================================
#     Module Functions
# =======================================================

def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels, **kwargs):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: a nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels, **kwargs)


def topkgating(
        logits: Tensor,
        topk: int,
        group_limited_greedy: bool = False,
        n_group: int = None,
        topk_group: int = None,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        capacity_factor: float = 1.0,
        drop_tokens: bool = False,
):
    logits = logits.float()
    gates = F.softmax(logits, dim=1)

    if group_limited_greedy:
        group_shape = list(gates.shape[:-1]) + [n_group, gates.shape[-1] // n_group]
        group_scores = (
            gates.reshape(group_shape).max(dim=-1).values
        )  # [n, n_group]
        group_idx = torch.topk(
            group_scores, topk_group, dim=-1, sorted=False
        )[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                group_shape
            )
            .reshape(list(gates.shape))
        )  # [n, e]
        gates = gates.masked_fill(~score_mask.bool(), 0.0)

    num_experts = int(gates.shape[1])
    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [tokens_per_group, num_selected_experts].
    expert_gate, expert_index = torch.topk(gates, topk)
    expert_mask = F.one_hot(expert_index, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [tokens_per_group, num_experts]
    expert_mask_aux = expert_mask.max(dim=-2)[0]
    tokens_per_group_and_expert = torch.mean(expert_mask_aux.float(), dim=-2)
    router_prob_per_group_and_expert = torch.mean(gates.float(), dim=-2)
    l_aux = num_experts ** 2 * torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert)

    if drop_tokens:
        expert_capacity = int(max(topk, topk * gates.shape[0] // gates.shape[1]) * capacity_factor)
    else:
        expert_index_flat = expert_index.flatten()
        tokens_per_expert = torch.bincount(expert_index_flat, minlength=num_experts)
        expert_capacity = torch.max(tokens_per_expert).item()

    if norm_topk_prob and topk > 1:
        gates_s = torch.clamp(
            torch.matmul(expert_mask.float(), gates.unsqueeze(-1)).sum(dim=1), min=torch.finfo(gates.dtype).eps
        )
        router_probs = gates / gates_s
    else:
        router_probs = gates * routed_scaling_factor
    # Make num_selected_experts the leading axis to ensure that top-1 choices
    # have priority over top-2 choices, which have priority over top-3 choices,
    # etc.
    expert_index = torch.transpose(expert_index, 0, 1)
    # Shape: [num_selected_experts * tokens_per_group]
    expert_index = expert_index.reshape(-1)

    # Create mask out of indices.
    # Shape: [tokens_per_group * num_selected_experts, num_experts].
    expert_mask = F.one_hot(expert_index, num_experts).to(torch.int32)
    exp_counts = torch.sum(expert_mask, dim=0).detach()

    # Experts have a fixed capacity that we cannot exceed. A token's priority
    # within the expert's buffer is given by the masked, cumulative capacity of
    # its target expert.
    # Shape: [tokens_per_group * num_selected_experts, num_experts].
    token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask - 1
    # Shape: [num_selected_experts, tokens_per_group, num_experts].
    token_priority = token_priority.reshape((topk, -1, num_experts))
    # Shape: [tokens_per_group, num_selected_experts, num_experts].
    token_priority = torch.transpose(token_priority, 0, 1)
    # For each token, across all selected experts, select the only non-negative
    # (unmasked) priority. Now, for group G routing to expert E, token T has
    # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
    # is its targeted expert.
    # Shape: [tokens_per_group, num_experts].
    token_priority = torch.max(token_priority, dim=1)[0]

    # Token T can only be routed to expert E if its priority is positive and
    # less than the expert capacity. One-hot matrix will ignore indices outside
    # the range [0, expert_capacity).
    # Shape: [tokens_per_group, num_experts, expert_capacity].
    valid_mask = torch.logical_and(token_priority >= 0, token_priority < expert_capacity)
    token_priority = torch.masked_fill(token_priority, ~valid_mask, 0)
    dispatch_mask = F.one_hot(token_priority, expert_capacity).to(torch.bool)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, expert_capacity)
    dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
    # expert_capacity].
    combine_weights = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)
    exp_counts_capacity = torch.sum(dispatch_mask)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0] * topk)

    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts


# =======================================================
#     Multi-Dimensional RoPE
# =======================================================

def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
        # assert num are all integers
        num_int = [int(x) for x in num]
        assert (torch.tensor(num) == torch.tensor(num_int)).all(), f"num should be int, but got {num}"
        num = num_int
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)       # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)      # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)       # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")   # dim x [H, W]
    grid = torch.stack(grid, dim=0)     # [dim, H, W]

    return grid


def build_2d_rope(
        seq_len: int, n_elem: int, image_infos: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
        device: Optional[torch.device] = None, base: int = 10000, base_rescale_factor: float = 1.0,
        return_all_pos: bool = False,
):
    """
    Reference: https://kexue.fm/archives/10352

    Start from 1, we have
        beta_y = L + (wh - h)/2
        beta_x = L + (wh - w)/2

    Returns
    -------
    cos: torch.Tensor with shape of [seq_len, n_elem]
    sin: torch.Tensor with shape of [seq_len, n_elem]
    """
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    # theta
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)    # [1, half_d, 2]

    # position indices
    if image_infos is None:
        image_infos = []

    image_infos_list = [image_infos]
    sample_seq_lens = [seq_len]

    # Prepare position indices for each sample
    x_sections = []
    y_sections = []
    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            L = sec_slice.start   # start from 0, so image_slice.start is just L
            # previous text
            if last_pos < L:
                y_sections.append(torch.arange(last_pos, L))
                x_sections.append(torch.arange(last_pos, L))
            elif h is None:
                # Interleave data has overlapped positions for <boi> <size> <ratio> <timestep> <eoi> tokens.
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                continue
            else:
                # Interleave data has overlapped positions for noised image and the successive clean image,
                # leading to last_pos (= last text end L + noise w * h) > L (last text end L).
                pass
            # current image
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            grid = get_meshgrid_nd((beta_y, beta_x), (beta_y + h, beta_x + w))  # [2, h, w]
            grid = grid.reshape(2, -1)  # (y, x)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            # step
            last_pos = L + w * h
        # final text
        y_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id]))
        x_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id]))

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    # If there are overlap positions, we need to remove them.
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)    # [seq_len, 1, 2]

    # calc rope
    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 2)

    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)

    if return_all_pos:
        return cos, sin, all_pos

    return cos, sin


def build_batch_2d_rope(
        seq_len: int, n_elem: int, image_infos: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
        device: Optional[torch.device] = None, base: int = 10000, base_rescale_factor: float = 1.0,
        return_all_pos: bool = False,
):
    cos_list, sin_list, all_pos_list = [], [], []
    if image_infos is None:
        image_infos = [None]
    for i, image_info in enumerate(image_infos):
        res = build_2d_rope(
            seq_len, n_elem, image_infos=image_info, device=device,
            base=base, base_rescale_factor=base_rescale_factor,
            return_all_pos=return_all_pos,
        )
        if return_all_pos:
            cos, sin, all_pos = res
        else:
            cos, sin = res
            all_pos = None
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)

    stacked_cos = torch.stack(cos_list, dim=0)
    stacked_sin = torch.stack(sin_list, dim=0)

    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list

    return stacked_cos, stacked_sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass shifted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =======================================================
#     Modules for Image Generation
# =======================================================

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self,
                 hidden_size,
                 act_layer=nn.GELU,
                 frequency_embedding_size=256,
                 max_period=10000,
                 out_size=None,
                 dtype=None,
                 device=None
                 ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1, **factory_kwargs)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1, **factory_kwargs
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param in_channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
        elif down:
            self.h_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels, **factory_kwargs)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs)
            ),
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs
            )
        else:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1, **factory_kwargs)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Adaptive Group Normalization
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1. + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    """
    patch_size: one of [1, 2 ,4 ,8]
    in_channels: vae latent dim
    hidden_channels: hidden dim for reducing parameters
    out_channels: transformer model dim
    """
    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels, out_channels,
                 dropout=0.0, device=None, dtype=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList(
            [conv_nd(
                2,
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                **factory_kwargs
            )]
        )

        if self.patch_size == 1:
            self.model.append(ResBlock(
                in_channels=hidden_channels,
                emb_channels=emb_channels,
                out_channels=out_channels,
                dropout=dropout,
                **factory_kwargs
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResBlock(
                    in_channels=hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels if (i + 1) * 2 != self.patch_size else out_channels,
                    dropout=dropout,
                    down=True,
                    **factory_kwargs
                ))

    def forward(self, x, t):
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        _, _, token_h, token_w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, token_h, token_w


class UNetUp(nn.Module):
    """
    patch_size: one of [1, 2 ,4 ,8]
    in_channels: transformer model dim
    hidden_channels: hidden dim for reducing parameters
    out_channels: vae latent dim
    """
    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels, out_channels,
                 dropout=0.0, device=None, dtype=None, out_norm=False):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(ResBlock(
                in_channels=in_channels,
                emb_channels=emb_channels,
                out_channels=hidden_channels,
                dropout=dropout,
                **factory_kwargs
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResBlock(
                    in_channels=in_channels if i == 0 else hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    up=True,
                    **factory_kwargs
                ))

        if out_norm:
            self.model.append(nn.Sequential(
                normalization(hidden_channels, **factory_kwargs),
                nn.SiLU(),
                conv_nd(
                    2,
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs
                ),
            ))
        else:
            self.model.append(conv_nd(
                2,
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                **factory_kwargs
            ))

    # batch_size, seq_len, model_dim
    def forward(self, x, t, token_h, token_w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=token_h, w=token_w)
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


# =======================================================
#     Modules for Transformer Backbone
# =======================================================

@dataclass
class CausalMMOutputWithPast(CausalLMOutputWithPast):
    diffusion_prediction: Optional[torch.Tensor] = None


class HunyuanStaticCache(StaticCache):
    """
    A custom static cache for multi-modal models that supports dynamic extension of the cache
    and inplace updates of the cache.

    This cache supports batch cache_position updates.
    """
    def __init__(self, *args, **kwargs):
        self.dynamic = kwargs.pop("dynamic", False)
        super().__init__(*args, **kwargs)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
            if self.key_cache[layer_idx].device != key_states.device:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)
            k_out = self.key_cache[layer_idx]
            v_out = self.value_cache[layer_idx]
            key_states = key_states.to(k_out.dtype)
            value_states = value_states.to(v_out.dtype)
        else:
            if self.layers[layer_idx].keys is None:
                self.layers[layer_idx].lazy_initialization(key_states)
            k_out = self.layers[layer_idx].keys
            v_out = self.layers[layer_idx].values

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            if cache_position.dim() == 1:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)

                if self.dynamic:
                    end = cache_position[-1].item() + 1
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]
            else:
                assert cache_position.dim() == 2, f"multiple batch dims not yet {cache_position.shape=}"
                batch_size, idx_size = cache_position.shape
                assert batch_size == k_out.size(0)
                assert batch_size == v_out.size(0)
                assert batch_size == key_states.size(0)
                assert batch_size == value_states.size(0)
                for i in range(batch_size):
                    unbatched_dim = 1
                    k_out[i].index_copy_(unbatched_dim, cache_position[i], key_states[i])
                    v_out[i].index_copy_(unbatched_dim, cache_position[i], value_states[i])

                if self.dynamic:
                    assert len(cache_position) == 1
                    end = cache_position[0, -1].item() + 1
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]

        return k_out, v_out


class HunyuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HunyuanRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HunyuanMLP(nn.Module):
    def __init__(self, config: HunyuanImage3Config, layer_idx=None, is_shared_mlp=False, is_moe=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act

        self.intermediate_size = config.intermediate_size
        if is_shared_mlp or is_moe:
            # 如果是 moe 的话，优先用 moe_intermediate_size
            if config.moe_intermediate_size is not None:
                self.intermediate_size = config.moe_intermediate_size \
                    if isinstance(config.moe_intermediate_size, int) else config.moe_intermediate_size[layer_idx]

            if is_shared_mlp:
                num_shared_expert = config.num_shared_expert \
                    if isinstance(config.num_shared_expert, int) else config.num_shared_expert[layer_idx]
                self.intermediate_size *= num_shared_expert

        self.act_fn = ACT2FN[config.hidden_act]
        if self.hidden_act == "silu":
            self.intermediate_size *= 2  # SwiGLU
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias=config.mlp_bias)
        elif self.hidden_act == "gelu":
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        else:
            assert False, "other hidden_act are not supported"

    def forward(self, x):
        if self.hidden_act == "silu":
            gate_and_up_proj = self.gate_and_up_proj(x)
            x1, x2 = gate_and_up_proj.chunk(2, dim=2)
            down_proj = self.down_proj(x1 * self.act_fn(x2))
            return down_proj
        elif self.hidden_act == "gelu":
            intermediate = self.gate_and_up_proj(x)
            intermediate = self.act_fn(intermediate)
            output = self.down_proj(intermediate)
            return output
        else:
            assert False, "other hidden_act are not supported"


class HunyuanTopKGate(nn.Module):
    def __init__(self, config: HunyuanImage3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.moe_topk = config.moe_topk if isinstance(config.moe_topk, int) else config.moe_topk[layer_idx]
        self.drop_tokens = config.moe_drop_tokens
        self.min_capacity = 8
        self.random_routing_dropped_token = config.moe_random_routing_dropped_token
        num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
        self.wg = nn.Linear(config.hidden_size, num_experts, bias=False, dtype=torch.float32)

        # DeepSeek gating args
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.group_limited_greedy = config.group_limited_greedy

    def forward(self, hidden_states, topk_impl='default'):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        if topk_impl == 'default':
            gate_output = topkgating(logits, self.moe_topk, group_limited_greedy=self.group_limited_greedy,
                                     n_group=self.n_group, topk_group=self.topk_group,
                                     norm_topk_prob=self.norm_topk_prob,
                                     routed_scaling_factor=self.routed_scaling_factor,
                                     capacity_factor=self.config.capacity_factor,
                                     drop_tokens=self.drop_tokens)
        elif topk_impl == 'easy':
            gate_output = self.easy_topk(logits, self.moe_topk)
        else:
            raise ValueError(f"Unsupported topk_impl: {topk_impl}")

        return gate_output

    @staticmethod
    def easy_topk(logits, moe_topk):
        gates = F.softmax(logits, dim=1)
        topk_weight_1, expert_index = torch.topk(gates, moe_topk)
        weight_sums = topk_weight_1.sum(dim=1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        topk_weight = topk_weight_1 / weight_sums

        return topk_weight, expert_index


class HunyuanMoE(nn.Module):
    def __init__(self, config: HunyuanImage3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.moe_topk = config.moe_topk
        self.num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
        if config.use_mixed_mlp_moe:
            self.shared_mlp = HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=True)
        self.gate = HunyuanTopKGate(config, layer_idx=layer_idx)
        self.experts = nn.ModuleList(
            [HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=True) for _ in range(self.num_experts)]
        )

        self._moe_impl = config.moe_impl
        # For FlashInfer
        self.moe_weight = None
        self.moe_weight_2 = None
        self._weights_initialized = False

    @property
    def moe_impl(self):
        return self._moe_impl

    @moe_impl.setter
    def moe_impl(self, value):
        self._moe_impl = value
        if self._moe_impl == "flashinfer":
            assert flashinfer is not None, "When using fused_moe, flashinfer must be installed."

    def forward(self, hidden_states):
        torch.cuda.set_device(hidden_states.device.index)
        bsz, seq_len, hidden_size = hidden_states.shape

        if self.config.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states)

        reshaped_input = hidden_states.reshape(-1, hidden_size) # [bsz*seq_len, hidden_size]

        with nvtx.range("MoE"):
            if self._moe_impl == "flashinfer":
                # Get expert weights
                if not self._weights_initialized:
                    self._initialize_weights_on_device(hidden_states.device)
                topk_weight, topk_index = self.gate(hidden_states, topk_impl='easy')

                combined_output = torch.zeros_like(reshaped_input)
                _ = flashinfer.fused_moe.cutlass_fused_moe(     # noqa
                    reshaped_input.contiguous(),
                    topk_index.to(torch.int).contiguous(),
                    topk_weight.to(torch.float).contiguous(),
                    self.moe_weight,
                    self.moe_weight_2,
                    torch.bfloat16,
                    output=combined_output,
                    quant_scales=None,
                )
            else:
                # Original implementation - fallback for compatibility
                l_moe, combine_weights, dispatch_mask, exp_counts = self.gate(hidden_states, topk_impl='default')
                dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_input)
                chunks = dispatched_input.chunk(self.num_experts, dim=0)
                expert_outputs = []
                for chunk, expert in zip(chunks, self.experts):
                    expert_outputs.append(expert(chunk))

                expert_output = torch.cat(expert_outputs, dim=0)
                combined_output = torch.einsum("sec,ecm->sm", combine_weights.type_as(hidden_states), expert_output)

        combined_output = combined_output.reshape(bsz, seq_len, hidden_size)

        if self.config.use_mixed_mlp_moe:
            output = hidden_states_mlp + combined_output    # noqa
        else:
            output = combined_output

        return output

    def _initialize_weights_on_device(self, device):
        expert_weights_gate_up = []
        expert_weights_down = []

        for expert in self.experts:
            expert.to(device)
            expert_weights_gate_up.append(expert.gate_and_up_proj.weight.to(device))
            expert_weights_down.append(expert.down_proj.weight.to(device))

        self.moe_weight = torch.stack(expert_weights_gate_up).contiguous()
        self.moe_weight_2 = torch.stack(expert_weights_down).contiguous()
        # empty the expert weights
        for expert in self.experts:
            expert.gate_and_up_proj.weight.data = torch.empty(0, device=device)
            if expert.gate_and_up_proj.bias is not None:
                expert.gate_and_up_proj.bias.data = torch.empty(0, device=device)
            expert.down_proj.weight.data = torch.empty(0, device=device)
            if expert.down_proj.bias is not None:
                expert.down_proj.bias.data = torch.empty(0, device=device)

        self._weights_initialized = True


class HunyuanImage3SDPAAttention(nn.Module):
    """PyTorch SDPA attention implementation using torch.nn.functional.scaled_dot_product_attention"""

    def __init__(self, config: HunyuanImage3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = 'self'

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # self.head_dim = self.hidden_size // self.num_heads
        self.head_dim = config.attention_head_dim
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.use_qk_norm = config.use_qk_norm
        self.use_rotary_pos_emb = config.use_rotary_pos_emb
        self.hidden_size_q = self.head_dim * self.num_heads
        self.hidden_size_kv = self.head_dim * self.num_key_value_heads

        # define layers
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size_q + 2 * self.hidden_size_kv,
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.hidden_size_q, self.hidden_size, bias=config.attention_bias)

        if self.use_qk_norm:
            self.query_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if self.use_rotary_pos_emb:
            self._init_rope()

    def _init_rope(self):
        scaling_type = self.config.rope_scaling["type"]
        if scaling_type == "custom":
            # Using custom rotary embedding
            self.rotary_emb = None
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: Optional[bool] = False,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if output_attentions:
            raise NotImplementedError(
                'HunyuanImage3Model is using HunyuanImage3SDPAAttention,'
                'but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`.'
            )

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2,
                                        self.head_dim)
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            query_states = query_states.to(key_states.dtype)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
        # custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class HunyuanImage3FlashAttention2(HunyuanImage3SDPAAttention):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: Optional[bool] = False,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2,
                                        self.head_dim)
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
        # custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        target_dtype = key_states.dtype if key_states.dtype in [torch.bfloat16, torch.float16] else torch.bfloat16

        q_fa = query_states.to(target_dtype).transpose(1, 2).contiguous()
        k_fa = key_states.to(target_dtype).transpose(1, 2).contiguous()
        v_fa = value_states.to(target_dtype).transpose(1, 2).contiguous()

        mode = kwargs.get("mode", "gen_text")
        # For gen_text and gen_image, we need to handle the attention differently
        with nvtx.range("attention"):
            if mode == "gen_text":
                if attention_mask is None:
                    attn_output = flash_attn_func(q_fa, k_fa, v_fa, causal=False)   # decode attention
                else:
                    attn_output = flash_attn_func(q_fa, k_fa, v_fa, causal=True)    # prefill attention
            else:  # image attention
                gen_timestep_scatter_index: Optional[torch.Tensor] = kwargs.get("gen_timestep_scatter_index", None)
                assert gen_timestep_scatter_index is not None, \
                    "When gen_image, `gen_timestep_scatter_index` must be provided."
                # TODO: batchify
                timestep_index = gen_timestep_scatter_index[0, 0].item()
                # When image generation, different attention implementations for the first step and the following steps
                # help to improve the inference speed.
                first_step = kwargs.get("first_step", None)
                if first_step is None:
                    raise ValueError("When gen_image, `first_step` must be provided.")
                if first_step:
                    casual_len = timestep_index + 1
                    text_query_states = q_fa[:, :casual_len, :, :]
                    text_key_states = k_fa[:, :casual_len, :, :]
                    text_value_states = v_fa[:, :casual_len, :, :]
                    text_attn_output = flash_attn_func(
                        text_query_states, text_key_states, text_value_states, causal=True)
                    image_query_states = q_fa[:, casual_len:, :, :]
                    image_attn_output = flash_attn_func(image_query_states, k_fa, v_fa, causal=False)
                    attn_output = torch.cat((text_attn_output, image_attn_output), dim=1)
                else:
                    casual_len = timestep_index + 1
                    timestep_query_states = q_fa[:, 0:1, :, :]
                    timestep_key_states = k_fa[:, :casual_len, :, :]
                    timestep_value_states = v_fa[:, :casual_len, :, :]
                    timestep_attn_output = flash_attn_func(
                        timestep_query_states, timestep_key_states, timestep_value_states, causal=True)
                    image_query_states = q_fa[:, 1:, :, :]
                    image_attn_output = flash_attn_func(image_query_states, k_fa, v_fa, causal=False)
                    attn_output = torch.cat((timestep_attn_output, image_attn_output), dim=1)

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


Hunyuan_ATTENTION_CLASSES = {
    "eager": HunyuanImage3SDPAAttention,
    "sdpa": HunyuanImage3SDPAAttention,
    "flash_attention_2": HunyuanImage3FlashAttention2,
}


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(self, config: HunyuanImage3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        attn_impl = config._attn_implementation     # noqa
        if attn_impl in Hunyuan_ATTENTION_CLASSES:
            self.self_attn = Hunyuan_ATTENTION_CLASSES[attn_impl](config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unsupported attention implementation: {attn_impl}")

        if ((isinstance(config.num_experts, int) and config.num_experts > 1) or (
                isinstance(config.num_experts, list) and max(
                config.num_experts) > 1)) and layer_idx >= config.moe_layer_num_skipped:
            self.mlp = HunyuanMoE(config, layer_idx=layer_idx)
        else:
            self.mlp = HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=False)
        if config.norm_type == 'hf_rms' or config.norm_type == 'rms':
            self.input_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif config.norm_type == 'fused' or config.norm_type == 'torch_nn':
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            assert False, "other norm_type are not supported"

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor | Any]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            custom_pos_emb (`Tuple[torch.FloatTensor]`, *optional*): custom position embedding for rotary
                position embedding
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use "
                "`attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            custom_pos_emb=custom_pos_emb,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare Hunyuan Image 3 Model outputting raw hidden-states without any specific head on top.",
    Hunyuan_START_DOCSTRING,
)
class HunyuanImage3PreTrainedModel(PreTrainedModel):
    config_class = HunyuanImage3Config
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["HunyuanImage3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


Hunyuan_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Hunyuan Model outputting raw hidden-states without any specific head on top.",
    Hunyuan_START_DOCSTRING,
)
class HunyuanImage3Model(HunyuanImage3PreTrainedModel):
    def __init__(self, config: HunyuanImage3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.add_classification_head = config.add_classification_head
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HunyuanImage3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if not config.add_classification_head:
            self.ln_f = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

        self.shared_tensor = None

    @add_start_docstrings_to_model_forward(Hunyuan_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            mode: str = "gen_text",
            first_step: Optional[bool] = None,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                custom_pos_emb=custom_pos_emb,
                mode=mode,
                first_step=first_step,
                gen_timestep_scatter_index=gen_timestep_scatter_index,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if not self.add_classification_head:
            # Do ln_f outside of the model for compatibility with image generation.
            pass
            # hidden_states = self.ln_f(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class HunyuanImage3ForCausalMM(HunyuanImage3PreTrainedModel, GenerationMixin):
    def __init__(self, config: HunyuanImage3Config):
        super().__init__(config)
        self.config = config
        self._tkwrapper: Optional[TokenizerWrapper] = None

        # Initialize image preprocessor (for conditional images)
        self.image_processor = HunyuanImage3ImageProcessor(config)

        # vae and gen_image pipeline
        self.vae = AutoencoderKLConv3D.from_config(config.vae)
        self._pipeline = None

        # vit
        self.vision_model = Siglip2VisionTransformer(config.vit)
        self.vision_aligner = LightProjector(config.vit_aligner)

        # image generation related
        self.timestep_emb = TimestepEmbedder(hidden_size=config.hidden_size)
        if config.img_proj_type == "unet":
            self.patch_embed = UNetDown(
                patch_size=config.patch_size,
                emb_channels=config.hidden_size,
                in_channels=config.vae["latent_channels"],
                hidden_channels=config.patch_embed_hidden_dim,
                out_channels=config.hidden_size,
            )
            self.time_embed = TimestepEmbedder(hidden_size=config.hidden_size)

            self.final_layer = UNetUp(
                patch_size=config.patch_size,
                emb_channels=config.hidden_size,
                in_channels=config.hidden_size,
                hidden_channels=config.patch_embed_hidden_dim,
                out_channels=config.vae["latent_channels"],
                out_norm=True,
            )
            self.time_embed_2 = TimestepEmbedder(hidden_size=config.hidden_size)
        else:
            raise ValueError(f"Unknown img_proj_type {config.img_proj_type}")

        # transformer backbone
        self.model = HunyuanImage3Model(config)

        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        # linear head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def tokenizer(self):
        if self._tkwrapper is None:
            raise ValueError("Attribute `tokenizer` has not been initialized yet. Please set it first.")
        return self._tkwrapper

    def load_tokenizer(self, tokenizer):
        self._tkwrapper = TokenizerWrapper(tokenizer)

    @property
    def pipeline(self):
        if self._pipeline is None:
            self.scheduler = FlowMatchDiscreteScheduler(
                shift=self.generation_config.flow_shift, reverse=True, solver="euler",
            )
            self._pipeline = HunyuanImage3Text2ImagePipeline(
                model=self, scheduler=self.scheduler, vae=self.vae,
            )
        return self._pipeline

    @staticmethod
    def get_pos_emb(custom_pos_emb, position_ids):
        cos, sin = custom_pos_emb
        cos = real_batched_index_select(cos, dim=1, idx=position_ids)
        sin = real_batched_index_select(sin, dim=1, idx=position_ids)
        return cos, sin

    def instantiate_vae_image_tokens(
            self,
            x: torch.Tensor,
            images: BatchRaggedImages,
            ts: BatchRaggedTensor,
            image_mask: torch.Tensor,
    ):
        """
        Instantiate the VAE image embeddings into the input embedding sequence.

        Args:
            x: input sequence, (batch_size, seq_len, n_embd)
            images: BatchRaggedImages
                images can be a 4-D tensor, or a list of 4-D tensors, or a list of lists of 3-D tensors.
            ts: BatchRaggedTensor
                ts can be a 1-D tensor, or a list of 1-D tensors
            image_mask: (batch_size, seq_len)
        """
        batch_size, seq_len, n_embd = x.shape

        if isinstance(images, list):
            index = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
            t_emb = []
            for i, (image_i, t_i) in enumerate(zip(images, ts)):
                if isinstance(image_i, torch.Tensor):
                    # time_embed needs a 1-D tensor as input
                    t_i_emb = self.time_embed(t_i)
                    # n_{i} x one_image_seq_len x n_embd
                    image_i_seq, _, _ = self.patch_embed(image_i, t_i_emb)
                    # 1 x (n_{i} * one_image_seq_len)
                    image_i_scatter_index = index[i:i + 1].masked_select(image_mask[i:i + 1].bool()).reshape(1, -1)
                    x[i:i + 1].scatter_(
                        dim=1,
                        index=image_i_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                        # 1 x (n_{i} * one_image_seq_len) x n_embd
                        src=image_i_seq.reshape(1, -1, n_embd),  # 1 x (n_{i} * one_image_seq_len) x n_embd
                    )
                    t_emb.append(t_i_emb)
                elif isinstance(image_i, list):
                    # time_embed needs a 1-D tensor as input
                    t_i_emb = self.time_embed(t_i)  # n_{i} x d
                    image_i_seq_list = [], []
                    for j in range(len(image_i)):
                        image_ij = image_i[j]
                        if image_ij.dim() == 4:
                            assert image_i[j].shape[0] == 1, "image_i[j] should have a batch dimension of 1"
                        elif image_ij.dim() == 3:
                            image_ij = image_ij.unsqueeze(0)
                        else:
                            raise ValueError(f"image_i[j] should have 3 or 4 dimensions, got {image_ij.dim()}")
                        # 1 x one_image_seq_len_{j} x n_embd
                        image_i_seq_j, _, _ = self.patch_embed(image_ij, t_i_emb[j:j + 1])
                        image_i_seq_list.append(image_i_seq_j)
                    # 1 x sum_{j}(one_image_seq_len_{j}) x n_embd
                    image_i_seq = torch.cat(image_i_seq_list, dim=1)
                    # 1 x sum_{j}(one_image_seq_len_{j})
                    image_i_scatter_index = index[i:i + 1].masked_select(image_mask[i:i + 1].bool()).reshape(1, -1)
                    x[i:i + 1].scatter_(
                        dim=1,
                        index=image_i_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                        # 1 x sum_{j}(one_image_seq_len_{j}) x n_embd
                        src=image_i_seq.reshape(1, -1, n_embd),  # 1 x sum_{j}(one_image_seq_len_{j}) x n_embd
                    )
                    t_emb.append(t_i_emb)
                else:
                    raise TypeError(f"image_i should be a torch.Tensor or a list, got {type(image_i)}")
            token_h, token_w = None, None
        else:
            # images is a 4-D tensor
            batch_size, seq_len, n_embd = x.shape
            index = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
            t_emb = self.time_embed(ts)
            image_seq, token_h, token_w = self.patch_embed(images, t_emb)
            image_scatter_index = index.masked_select(image_mask.bool()).reshape(batch_size, -1)
            x.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_seq,
            )

        return x, token_h, token_w

    def instantiate_timestep_tokens(
            self,
            x: torch.Tensor,
            t: BatchRaggedTensor,
            timestep_scatter_index: BatchRaggedTensor,
    ):
        batch_size, seq_len, n_embd = x.shape
        # batch_size x n x n_embd
        timestep_scatter_src = self.timestep_emb(t.reshape(-1)).reshape(batch_size, -1, n_embd)
        x.scatter_(
            dim=1,
            index=timestep_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
            src=timestep_scatter_src,
        )

        return x

    def instantiate_vit_image_tokens(
            self,
            x: torch.Tensor,
            cond_vit_images: Union[torch.Tensor, List[torch.Tensor]],
            cond_vit_image_mask: torch.Tensor,
            vit_kwargs: Dict[str, Any],
    ):
        # 1. Forward the vit encoder and vit aligner to get the vit image embeddings and align them to the
        # transformer hidden size
        cond_vit_image_embeds = []
        for batch_idx, image in enumerate(cond_vit_images):
            cur_kwargs = {k: v[batch_idx] for k, v in vit_kwargs.items()}
            image_embed = self.vision_model(image, **cur_kwargs).last_hidden_state
            image_embed = self.vision_aligner(image_embed)
            n, seq_len, dim = image_embed.shape
            image_embed = image_embed.reshape(n * seq_len, dim)
            cond_vit_image_embeds.append(image_embed)

        # 2. Instantiate the vit image embeddings into the input sequence
        batch_size, seq_len, n_embd = x.shape
        index = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        for i, (image_embed, mask) in enumerate(zip(cond_vit_image_embeds, cond_vit_image_mask)):
            image_scatter_index = index[i:i+1].masked_select(mask.bool()).reshape(1, -1)
            x[i:i+1].scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_embed.reshape(1, -1, n_embd),
            )

        return x

    def ragged_final_layer(self, x, image_mask, timestep, token_h, token_w, first_step):
        bsz, seq_len, n_embd = x.shape
        if first_step:
            image_output = x.masked_select(image_mask.unsqueeze(-1).bool()).reshape(bsz, -1, n_embd)
        else:
            image_output = x[:, 1:, :]
        timestep_emb = self.time_embed_2(timestep)
        pred = self.final_layer(image_output, timestep_emb, token_h, token_w)
        return pred

    @staticmethod
    def _check_inputs(cond, target, check_list):
        if cond:
            for name, item in check_list:
                assert item is not None, f"`{name}` should be provided when `{target}`."

    @add_start_docstrings_to_model_forward(Hunyuan_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            mode: str = "gen_text",
            first_step: Optional[bool] = None,
            # for gen image
            images: Optional[BatchRaggedImages] = None,
            image_mask: Optional[torch.Tensor] = None,
            timestep: Optional[BatchRaggedTensor] = None,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
            # for cond image
            cond_vae_images: Optional[BatchRaggedImages] = None,
            cond_timestep: Optional[BatchRaggedTensor] = None,
            cond_vae_image_mask: Optional[torch.Tensor] = None,
            cond_vit_images: Optional[BatchRaggedImages] = None,
            cond_vit_image_mask: Optional[torch.Tensor] = None,
            vit_kwargs: Optional[Dict[str, Any]] = None,
            cond_timestep_scatter_index: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalMMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Sanity Check of Inputs
        self._check_inputs(mode == "gen_image", "in `gen_image` mode", [
            ("images", images), ("timestep", timestep), ("gen_timestep_scatter_index", gen_timestep_scatter_index),
        ])
        self._check_inputs(mode == "gen_image" and first_step, "in `gen_image` mode at the first step", [
            ("image_mask", image_mask),
        ])
        self._check_inputs(cond_vae_images is not None, "`cond_vae_images` is provided", [
            ("cond_timestep", cond_timestep), ("cond_vae_image_mask", cond_vae_image_mask),
            ("cond_timestep_scatter_index", cond_timestep_scatter_index),
        ])
        self._check_inputs(cond_vit_images is not None, "`cond_vit_images` is provided", [
            ("cond_vit_image_mask", cond_vit_image_mask), ("vit_kwargs", vit_kwargs),
        ])

        custom_pos_emb = self.get_pos_emb(custom_pos_emb, position_ids)

        inputs_embeds = self.model.wte(input_ids)
        bsz, seq_len, n_embd = inputs_embeds.shape

        # Instantiate placeholder tokens: <timestep>, <img> for the gen image
        if mode == "gen_text":
            # For gen_text, make sure gen_timestep_scatter_index is None
            gen_timestep_scatter_index = None
            token_h, token_w = None, None
        else:
            if first_step:
                inputs_embeds, token_h, token_w = self.instantiate_vae_image_tokens(
                    inputs_embeds, images, timestep, image_mask)
                inputs_embeds = self.instantiate_timestep_tokens(
                    inputs_embeds, timestep, gen_timestep_scatter_index)
            else:
                t_emb = self.time_embed(timestep)
                image_emb, token_h, token_w = self.patch_embed(images, t_emb)
                timestep_emb = self.timestep_emb(timestep).reshape(bsz, -1, n_embd)
                inputs_embeds = torch.cat([timestep_emb, image_emb], dim=1)

        # Instantiate placeholder tokens: <timestep>, <img> for cond images
        # Should only run once with kv-cache enabled.
        if cond_vae_images is not None:
            inputs_embeds, _, _ = self.instantiate_vae_image_tokens(
                inputs_embeds, cond_vae_images, cond_timestep, cond_vae_image_mask)
            inputs_embeds = self.instantiate_timestep_tokens(
                inputs_embeds, cond_timestep, cond_timestep_scatter_index)
        if cond_vit_images is not None:
            inputs_embeds = self.instantiate_vit_image_tokens(
                inputs_embeds, cond_vit_images, cond_vit_image_mask, vit_kwargs)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            custom_pos_emb=custom_pos_emb,
            mode=mode,
            first_step=first_step,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
        )
        hidden_states = outputs[0]

        if mode == "gen_text":
            hidden_states = self.model.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            diffusion_prediction = None
        else:
            logits = None
            hidden_states = hidden_states.to(input_ids.device)
            diffusion_prediction = self.ragged_final_layer(
                hidden_states, image_mask, timestep, token_h, token_w, first_step)

        if not return_dict:
            output = (logits,) + outputs[1:] + (diffusion_prediction,)
            return output

        output = CausalMMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            diffusion_prediction=diffusion_prediction,
        )

        return output

    @staticmethod
    def check_inputs(prompt=None, message_list=None):
        if prompt is None and message_list is None:
            raise ValueError("Either `prompt` or `message_list` should be provided.")
        if prompt is not None and message_list is not None:
            raise ValueError("Only one of `prompt` or `message_list` should be provided.")
        if prompt is not None:
            assert isinstance(prompt, str) or isinstance(prompt, list), \
                f"`prompt` should be a string or a list of strings, but got {type(prompt)}."
            if isinstance(prompt, list):
                assert len(prompt) > 0 and all(isinstance(p, str) for p in prompt), \
                    "`prompt` should be a non-empty list of strings."
        if message_list is not None:
            if not isinstance(message_list, list):
                raise ValueError(f"`message_list` should be a list of messages, but got {type(message_list)}.")
            assert len(message_list) > 0, "`message_list` should be a non-empty list."
            for message in message_list:
                assert isinstance(message, list) or isinstance(message, dict), \
                    f"Each message should be a list of dicts or a dict, but got {type(message)}."

    @staticmethod
    def prepare_seed(seed, batch_size):
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 10_000_000) for _ in range(batch_size)]
        elif isinstance(seed, int):
            seeds = [seed for _ in range(batch_size)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) for i in range(batch_size)]
            else:
                raise ValueError(f"Length of seed must be equal to the batch_size({batch_size}), got {seed}.")
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        return seeds

    @staticmethod
    def build_batch_rope_image_info(output, sections):
        rope_image_info = []
        for image_slices, sections_i in zip(output.all_image_slices, sections):
            image_shapes = []
            for section in sections_i:
                if 'image' in section['type']:
                    if isinstance(section['token_height'], list):
                        assert len(section['token_height']) == len(section['token_height']), \
                            (f"token_height and token_width should have the same length, "
                             f"but got {len(section['token_height'])} and {len(section['token_width'])}")
                        image_shapes.extend(list(zip(section['token_height'], section['token_width'])))
                    else:
                        image_shapes.append((section['token_height'], section['token_width']))
            assert len(image_slices) == len(image_shapes), (
                f"Size miss matching: Image slices({len(image_slices)}) != image shapes({len(image_shapes)})"
            )
            rope_image_info.append(list(zip(image_slices, image_shapes)))
        return rope_image_info

    def vae_encode(self, image, cfg_factor=1):
        config = self.vae.config

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            vae_encode_result = self.vae.encode(image)
            if isinstance(vae_encode_result, torch.Tensor):
                latents = vae_encode_result
            else:
                latents = vae_encode_result.latent_dist.sample()
            if hasattr(config, 'shift_factor') and config.shift_factor:
                latents.sub_(config.shift_factor)
            if hasattr(config, 'scaling_factor') and config.scaling_factor:
                latents.mul_(config.scaling_factor)

        if hasattr(self.vae, "ffactor_temporal"):
            assert latents.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
            latents = latents.squeeze(2)

        # Here we always use t=0 to declare it is a clean conditional image
        t = torch.zeros((latents.shape[0],))

        if cfg_factor > 1:
            t = t.repeat(cfg_factor)
            latents = latents.repeat(cfg_factor, 1, 1, 1)

        return t, latents

    def _encode_cond_image(
            self,
            batch_cond_image_info_list: List[List[JointImageInfo]],
            cfg_factor: int = 1,
    ):
        # VAE encode one by one, as we assume cond images have different sizes
        batch_cond_vae_images, batch_cond_t, batch_cond_vit_images = [], [], []
        for cond_image_info_list in batch_cond_image_info_list:
            cond_vae_image_list, cond_t_list, cond_vit_image_list = [], [], []
            for image_info in cond_image_info_list:
                cond_t_, cond_vae_image_ = self.vae_encode(
                    image_info.vae_image_info.image_tensor.to(self.device),
                )
                cond_vit_image_list.append(image_info.vision_image_info.image_tensor)
                cond_vae_image_list.append(cond_vae_image_.squeeze(0))
                cond_t_list.append(cond_t_)
            batch_cond_vae_images.append(cond_vae_image_list)
            batch_cond_t.append(cond_t_list)
            batch_cond_vit_images.append(torch.cat(cond_vit_image_list, dim=0))

        # If only one cond image for each sample and all have the same size, we can batch them together
        # In this case, cond_vae_images is a 4-D tensor.
        if all([len(items) == 1 for items in batch_cond_vae_images]) and all(
                items[0].shape == batch_cond_vae_images[0][0].shape for items in batch_cond_vae_images):
            cond_vae_images = torch.stack([items[0] for items in batch_cond_vae_images], dim=0)
            cond_t = torch.cat([items[0] for items in batch_cond_t], dim=0)
            if cfg_factor > 1:
                cond_t = cond_t.repeat(cfg_factor)
                cond_vae_images = cond_vae_images.repeat(cfg_factor, 1, 1, 1)
        else:
            # In this case, cond_vae_images is a list of 4-D tensors or a list of lists of 3-D tensors.
            cond_t = [torch.cat(item, dim=0) for item in batch_cond_t]
            cond_vae_images = []
            for items in batch_cond_vae_images:
                if all(items[0].shape == item.shape for item in items):
                    cond_vae_images.append(torch.stack(items, dim=0))
                else:
                    cond_vae_images.append(items)
            if cfg_factor > 1:
                cond_t = cond_t * cfg_factor
                cond_vae_images = cond_vae_images * cfg_factor

        if cfg_factor > 1:
            batch_cond_vit_images = batch_cond_vit_images * cfg_factor

        return cond_vae_images, cond_t, batch_cond_vit_images

    def prepare_model_inputs(
            self,
            prompt=None,
            mode="gen_text",
            system_prompt=None,
            cot_text=None,
            image_size="auto",
            message_list=None,
            device=None,
            max_new_tokens=None,
            **kwargs,
    ):
        # 1. Sanity check
        self.check_inputs(prompt, message_list)
        device = default(device, self.device)

        # 2. Format inputs
        batch_message_list = message_list
        batch_prompt = prompt
        batch_cot_text = cot_text
        batch_system_prompt = system_prompt
        batch_gen_image_info = None
        # TODO: construct with user input images
        batch_cond_image_info = None

        #   -- 2.1 message_list
        if batch_message_list is not None:
            if isinstance(batch_message_list[0], dict):
                batch_message_list = [batch_message_list]
            batch_size = len(batch_message_list)

            batch_gen_image_info = [
                [message['content'] for message in message_list_ if message['type'] == 'gen_image']
                for message_list_ in batch_message_list
            ]
            # At most one gen_image is allowed for each message_list
            batch_gen_image_info = [info[-1] if len(info) > 0 else None for info in batch_gen_image_info]
            # Multiple cond images are allowed.
            batch_cond_image_info = [
                [message['content'] for message in message_list_ if message['type'] == 'joint_image']
                for message_list_ in batch_message_list
            ]

        #   -- 2.2 Prompt, cot text, system prompt
        else:
            if isinstance(batch_prompt, str):
                batch_prompt = [batch_prompt]
            batch_size = len(batch_prompt)

            if batch_cot_text is not None:
                if isinstance(batch_cot_text, str):
                    batch_cot_text = [batch_cot_text]
                else:
                    assert isinstance(batch_cot_text, list) and len(batch_cot_text) == batch_size, \
                        "`cot_text` should be a string or a list of strings with the same length as `prompt`."

            if batch_system_prompt is not None:
                if isinstance(batch_system_prompt, str):
                    batch_system_prompt = [batch_system_prompt]
                else:
                    assert isinstance(batch_system_prompt, list) and len(batch_system_prompt) == batch_size, \
                        "`system_prompts` should be a string or a list of strings with the same length as `prompt`."

            if mode == "gen_image":
                batch_gen_image_info = [self.image_processor.build_image_info(image_size) for _ in range(batch_size)]

        #   -- 2.3 seed
        seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=batch_size)
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        # 3. apply chat template
        cfg_factor = {"gen_text": 1, "gen_image": 2}
        bot_task = kwargs.pop("bot_task", "auto")
        # If `drop_think` enabled, always drop <think> parts in the context.
        drop_think = kwargs.get('drop_think', self.generation_config.drop_think)
        # Apply batched prompt or batched message_list to build input sequence with associated info.
        out = self._tkwrapper.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_message_list=batch_message_list,
            mode=mode,
            batch_gen_image_info=batch_gen_image_info,
            batch_cond_image_info=batch_cond_image_info,
            batch_system_prompt=batch_system_prompt,
            batch_cot_text=batch_cot_text,
            max_length=kwargs.get('max_length'),
            bot_task=bot_task,
            image_base_size=self.config.image_base_size,
            sequence_template=self.generation_config.sequence_template,
            cfg_factor=cfg_factor[mode],
            drop_think=drop_think,
        )
        output, sections = out['output'], out['sections']

        # 4. Encode conditional images
        if batch_cond_image_info is not None and len(batch_cond_image_info[0]) > 0:
            cond_vae_images, cond_timestep, cond_vit_images = self._encode_cond_image(
                batch_cond_image_info, cfg_factor[mode]
            )
            # Pack vit kwargs. Siglip2-so requires spatial_shapes and attention_mask for inference.
            vit_kwargs = {"spatial_shapes": [], "attention_mask": []}
            for cond_image_info in batch_cond_image_info:
                vit_kwargs["spatial_shapes"].append(
                    torch.stack([item.vision_encoder_kwargs["spatial_shapes"] for item in cond_image_info]))
                vit_kwargs["attention_mask"].append(
                    torch.stack([item.vision_encoder_kwargs["pixel_attention_mask"] for item in cond_image_info]))
            if cfg_factor[mode] > 1:
                vit_kwargs["spatial_shapes"] = vit_kwargs["spatial_shapes"] * cfg_factor[mode]
                vit_kwargs["attention_mask"] = vit_kwargs["attention_mask"] * cfg_factor[mode]
        else:
            cond_vae_images, cond_timestep, cond_vit_images = None, None, None
            vit_kwargs = None

        # 5. Build position embeddings
        rope_image_info = self.build_batch_rope_image_info(output, sections)
        if mode == "gen_text":
            seq_len = self.generation_config.max_length
        else:
            seq_len = output.tokens.shape[1]
        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            seq_len=seq_len,
            n_elem=self.config.attention_head_dim,
            device=device,
            base=self.config.rope_theta,
        )

        # 6. Build kv cache
        if bot_task == "img_ratio":
            max_new_tokens = 1
        if mode == "gen_image":
            # Image generation will not extend sequence length, using token length as max_cache_len is enough.
            max_cache_len = output.tokens.shape[1]
        else:
            max_cache_len = output.tokens.shape[1] + default(max_new_tokens, self.generation_config.max_length)
        cache = HunyuanStaticCache(
            config=self.config,
            batch_size=batch_size * cfg_factor[mode],
            max_cache_len=max_cache_len,
            dtype=torch.bfloat16,
            dynamic=mode == "gen_text",
        )

        # 7. Build position ids
        batch_input_pos = torch.arange(
            0, output.tokens.shape[1], dtype=torch.long, device=device)[None].expand(
            batch_size * cfg_factor[mode], -1)  # use expand to share indices to save memory

        # 8. Build model input kwargs
        tkw = self._tkwrapper
        if image_size == "auto":
            extra_auto_stops = [tkw.special_token_map[f"<img_ratio_{i}>"] for i in range(33)]
        else:
            extra_auto_stops = [tkw.boi_token_id]
        stop_token_id = dict(
            auto=[tkw.eos_token_id] + extra_auto_stops,
            image=[tkw.eos_token_id],
            recaption=[tkw.end_recaption_token_id, tkw.end_answer_token_id, tkw.eos_token_id],
            think=[tkw.end_recaption_token_id, tkw.end_answer_token_id, tkw.eos_token_id],
            img_ratio=extra_auto_stops,
        )
        model_input_kwargs = dict(
            input_ids=output.tokens.to(device),
            position_ids=batch_input_pos,
            past_key_values=cache,
            custom_pos_emb=(cos, sin),
            mode=mode,
            image_mask=to_device(output.gen_image_mask, device),
            gen_timestep_scatter_index=to_device(output.gen_timestep_scatter_index, device),
            cond_vae_images=to_device(cond_vae_images, device),
            cond_timestep=to_device(cond_timestep, device),
            cond_vae_image_mask=to_device(output.cond_vae_image_mask, device),
            cond_vit_images=to_device(cond_vit_images, device),
            cond_vit_image_mask=to_device(output.cond_vit_image_mask, device),
            vit_kwargs={
                k: to_device(v, self.device) for k, v in vit_kwargs.items()
            } if vit_kwargs is not None else None,
            cond_timestep_scatter_index=to_device(output.cond_timestep_scatter_index, device),
            # for inner usage
            tokenizer_output=output,
            batch_gen_image_info=batch_gen_image_info,
            generator=generator,
            # generation config
            eos_token_id=stop_token_id[bot_task],
            max_new_tokens=max_new_tokens,
        )

        return model_input_kwargs

    def _prepare_attention_mask_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            generation_config: GenerationConfig,
            model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        # create `4d` bool attention mask (b, 1, seqlen, seqlen) using this implementation to bypass the 2d requirement
        # in the `transformers.generation_utils.GenerationMixin.generate`.
        # This implementation can handle sequences with text and image modalities, where text tokens use causal
        # attention and image tokens use full attention.
        bsz, seq_len = inputs_tensor.shape
        tokenizer_output = model_kwargs["tokenizer_output"]
        batch_image_slices = [
            tokenizer_output.joint_image_slices[i] + tokenizer_output.gen_image_slices[i]
            for i in range(bsz)
        ]
        attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril(diagonal=0).repeat(bsz, 1, 1)
        for i in range(bsz):
            for j, image_slice in enumerate(batch_image_slices[i]):
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
            tokenizer_output=None, batch_gen_image_info=None, generator=None, **kwargs
    ):
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            if input_ids.shape[1] != kwargs["position_ids"].shape[1]:    # in decode steps
                input_ids = torch.gather(input_ids, dim=1, index=kwargs["position_ids"])
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": kwargs["position_ids"],
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "custom_pos_emb": kwargs["custom_pos_emb"],
                "mode": kwargs["mode"],
                "images": kwargs.get("images"),
                "image_mask": kwargs.get("image_mask"),
                "timestep": kwargs.get("timestep"),
                "gen_timestep_scatter_index": kwargs.get("gen_timestep_scatter_index"),
                "cond_vae_images": kwargs.get("cond_vae_images"),
                "cond_timestep": kwargs.get("cond_timestep"),
                "cond_vae_image_mask": kwargs.get("cond_vae_image_mask"),
                "cond_vit_images": kwargs.get("cond_vit_images"),
                "cond_vit_image_mask": kwargs.get("cond_vit_image_mask"),
                "vit_kwargs": kwargs.get("vit_kwargs"),
                "cond_timestep_scatter_index": kwargs.get("cond_timestep_scatter_index"),
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        mode = model_kwargs["mode"]

        updated_model_kwargs = {
            "mode": mode,
            "custom_pos_emb": model_kwargs["custom_pos_emb"],
        }

        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                updated_model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        if "tokenizer_output" in model_kwargs:
            if mode == "gen_text":
                # When enable batching, we use right padding, which requires a real_pos to index the valid
                # end position of the sequence. If tokenizer_output in model_kwargs, it means we are in the
                # prefill step of generation.
                real_pos = to_device(model_kwargs["tokenizer_output"].real_pos, self.device)
                updated_model_kwargs["position_ids"] = real_pos
            else:
                # position ids
                image_mask = model_kwargs["image_mask"]
                bsz, seq_len = image_mask.shape
                index = torch.arange(seq_len, device=image_mask.device).unsqueeze(0).repeat(bsz, 1)
                position_ids = index.masked_select(image_mask.bool()).reshape(bsz, -1)
                timestep_position_ids = \
                    index[torch.arange(bsz), model_kwargs["gen_timestep_scatter_index"][:, -1]].unsqueeze(-1)
                updated_model_kwargs["position_ids"] = torch.cat([timestep_position_ids, position_ids], dim=1)

                # attention mask
                mask_list = []
                for attention_mask_i, position_ids_i in zip(
                        model_kwargs["attention_mask"], updated_model_kwargs["position_ids"]):
                    mask_list.append(torch.index_select(attention_mask_i, dim=1, index=position_ids_i.reshape(-1)))
                attention_mask = torch.stack(mask_list, dim=0)
                updated_model_kwargs["attention_mask"] = attention_mask
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs["gen_timestep_scatter_index"]

        else:
            if mode == "gen_text":
                # Now we are in the decode steps.
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"] + 1
            else:
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"]
                updated_model_kwargs["attention_mask"] = model_kwargs["attention_mask"]
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs["gen_timestep_scatter_index"]

        return updated_model_kwargs

    def _generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            use_model_defaults: Optional[bool] = None,
            generator: Optional[List[torch.Generator]] = None,
            verbose: int = 0,
            **kwargs,
    ):
        mode = kwargs.get("mode", "gen_text")

        # Log info
        if verbose >= 1:
            output = kwargs["tokenizer_output"]
            context = self._tkwrapper.tokenizer.decode(output.tokens[0], skip_special_tokens=False)
            # Replace <img><img>...<img> with [<img>]{number}
            context = re.sub(r"(<img>)+", lambda m: f"[<img>]{{{len(m.group(0)) // 5}}}", context)
            info_list = [
                ("token shape", output.tokens.shape),
                ("context[0]", context),
            ]
            gen_config = default(generation_config, self.generation_config)
            if mode == "gen_image":
                if generator is not None:
                    info_list.extend([
                        ("seed", [g.initial_seed() for g in generator]),
                    ])
                info_list.extend([
                    ("image_size", [f"{info.image_height}x{info.image_width}" for info in kwargs["batch_gen_image_info"]]),
                    ("infer_steps", kwargs.get("diff_infer_steps", gen_config.diff_infer_steps)),
                    ("guidance_scale", kwargs.get("diff_guidance_scale", gen_config.diff_guidance_scale)),
                    ("flow_shift", kwargs.get("flow_shift", gen_config.flow_shift)),
                ])
            else:
                info_list.extend([
                    ("do_sample", kwargs.get("do_sample", gen_config.do_sample)),
                    ("max_new_tokens", kwargs.get("max_new_tokens", gen_config.max_new_tokens)),
                    ("top_k", kwargs.get("top_k", gen_config.top_k)),
                    ("top_p", kwargs.get("top_p", gen_config.top_p)),
                    ("temperature", kwargs.get("temperature", gen_config.temperature)),
                    ("repetition_penalty", kwargs.get("repetition_penalty", gen_config.repetition_penalty)),
                ])
            max_key_len = max(len(k) for k, _ in info_list)
            info_str = "=" * 50 + \
                       "\nModel input info:\n" + \
                       "\n".join([f"    {k.rjust(max_key_len)}: {v}" for k, v in info_list]) + \
                       "\n--------------------------------------------------"
            print(info_str)

        if mode == "gen_text":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                return super().generate(
                    inputs,
                    generation_config,
                    logits_processor,
                    stopping_criteria,
                    prefix_allowed_tokens_fn,
                    synced_gpus,
                    assistant_model,
                    streamer,
                    negative_prompt_ids,
                    negative_prompt_attention_mask,
                    use_model_defaults,
                    **kwargs,
                )

        elif mode == "gen_image":
            batch_gen_image_info: List[ImageInfo] = kwargs.get("batch_gen_image_info")
            if batch_gen_image_info is None:
                raise ValueError("`batch_gen_image_info` should be provided when `mode` is `gen_image`.")

            results = self.pipeline(
                batch_size=len(batch_gen_image_info),
                image_size=[batch_gen_image_info[0].image_height, batch_gen_image_info[0].image_width],
                num_inference_steps=kwargs.get("diff_infer_steps", self.generation_config.diff_infer_steps),
                guidance_scale=kwargs.get("diff_guidance_scale", self.generation_config.diff_guidance_scale),
                generator=generator,
                model_kwargs=kwargs,
            )
            samples = results[0]
            return samples

        else:
            raise ValueError(f"Unknown mode {mode}, only `gen_text` and `gen_image` are supported.")

    def get_cot_text(self, output: torch.Tensor):
        if output.ndim == 2:
            return [self.get_cot_text(output_i) for output_i in output]
        elif output.ndim == 1:
            if output[-1] == self._tkwrapper.eos_token_id:
                output = output[:-1]
            cot_text = self._tkwrapper.decode(output).split("Assistant: ")[1]
            return cot_text
        else:
            raise ValueError(f"output should be 1D or 2D tensor, but got {output.ndim}D tensor.")

    def generate_image(
            self,
            prompt,
            seed=None,
            image_size="auto",
            use_system_prompt=None,
            system_prompt=None,
            bot_task=None,
            stream=False,
            **kwargs,
    ):
        max_new_tokens = kwargs.pop("max_new_tokens", 8192)
        verbose = kwargs.pop("verbose", 0)

        if stream:
            from transformers import TextStreamer
            streamer = TextStreamer(self._tkwrapper.tokenizer, skip_prompt=True, skip_special_tokens=False)
            kwargs["streamer"] = streamer

        use_system_prompt = default(use_system_prompt, self.generation_config.use_system_prompt)
        bot_task = default(bot_task, self.generation_config.bot_task)
        system_prompt = get_system_prompt(use_system_prompt, bot_task, system_prompt)

        if bot_task in ["think", "recaption"]:
            # Cot
            model_inputs = self.prepare_model_inputs(
                prompt=prompt, bot_task=bot_task, system_prompt=system_prompt, max_new_tokens=max_new_tokens)
            print(f"<{bot_task}>", end="", flush=True)
            outputs = self._generate(**model_inputs, **kwargs, verbose=verbose)
            cot_text = self.get_cot_text(outputs[0])
            # Switch system_prompt to `en_recaption` if drop_think is enabled.
            if self.generation_config.drop_think and system_prompt:
                system_prompt = t2i_system_prompts["en_recaption"][0]
        else:
            cot_text = None

        # Image ratio
        if image_size == "auto":
            model_inputs = self.prepare_model_inputs(
                prompt=prompt, cot_text=cot_text, bot_task="img_ratio", system_prompt=system_prompt, seed=seed)
            outputs = self._generate(**model_inputs, **kwargs, verbose=verbose)
            ratio_index = outputs[0, -1].item() - self._tkwrapper.ratio_token_offset
            # In some cases, the generated ratio_index is out of range. A valid ratio_index should be in [0, 32].
            # If ratio_index is out of range, we set it to 16 (i.e., 1:1).
            if ratio_index < 0 or ratio_index >= len(self.image_processor.reso_group):
                ratio_index = 16
            reso = self.image_processor.reso_group[ratio_index]
            image_size = reso.height, reso.width

        # Generate image
        model_inputs = self.prepare_model_inputs(
            prompt=prompt, cot_text=cot_text, system_prompt=system_prompt, mode="gen_image", seed=seed,
            image_size=image_size,
        )
        outputs = self._generate(**model_inputs, **kwargs, verbose=verbose)
        return outputs[0]
