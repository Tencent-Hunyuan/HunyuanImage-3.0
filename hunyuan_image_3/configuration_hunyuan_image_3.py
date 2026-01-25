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

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import List, Union, Optional


logger = logging.get_logger(__name__)


class HunyuanImage3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HunyuanImage3Model`]. It is used to instantiate
    an Hunyuan model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Hunyuan-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Hunyuan Image 3 model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`HunyuanImage3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations or shared MLP representations.
        moe_intermediate_size (`int` or `List`, *optional*, defaults to 11008):
            Dimension of the MLP representations in MoE. Use a list if you want a different size per layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether query and key in attention use norm
        use_cla (`bool`, *optional*, defaults to `False`):
            Whether to use CLA in attention
        cla_share_factor (`int`, *optional*, defaults to 1):
            The share factor of CLA
        num_experts (`int` or `List`, *optional*, defaults to 1):
            The number of experts for moe. If it is a list, it will be used as the number of experts for each layer.
        num_shared_expert (`int` or `List`, *optional*, defaults to 1):
            The number of shared experts for moe. If it is a list, it will be used as the number of shared experts
            for each layer.
        moe_topk (`int` or `List`, *optional*, defaults to 1):
            The topk value for moe. If it is a list, it will be used as the topk value for each layer.
        capacity_factor (Not used) (`float` or `List`, *optional*, defaults to 1.0):
            The capacity factor for moe. If it is a list, it will be used as the capacity factor for each layer.
        moe_layer_num_skipped (`int`, *optional*, defaults to 0):
            First moe_layer_num_skipped layers do not use MoE.
        model_version (`str`, *optional*, defaults to "HunyuanImage-3.0-Instruct"):
            The version of the model.
    """

    model_type = "Hunyuan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size: int = 290943,
            hidden_size: int = 4096,
            intermediate_size: int = 11008,
            moe_intermediate_size: Union[int, List] = None,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: Optional[int] = None,
            attention_head_dim: Optional[int] = None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            eod_token_id=3,
            im_start_id=4,
            im_end_id=5,
            text_start_id=6,
            text_end_id=7,
            image_token_id=8,
            video_start_id=9,
            video_end_id=10,
            im_newline_id=11,
            mask_init_id=12,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            use_qk_norm=False,
            use_rotary_pos_emb=True,
            use_cla=False,
            cla_share_factor=1,
            norm_type="hf_rms",
            num_experts: Union[int, List] = 1,
            use_mixed_mlp_moe=False,
            num_shared_expert: Union[int, List] = 1,
            moe_topk: Union[int, List] = 1,
            capacity_factor: int = 1.0,
            moe_drop_tokens=False,
            moe_random_routing_dropped_token=False,
            use_mla=False,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            v_head_dim=128,
            qk_nope_head_dim=128,
            moe_layer_num_skipped=0,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            group_limited_greedy=False,
            n_group=None,
            topk_group=None,
            add_classification_head=False,
            class_num=0,
            pool_type="last",
            pad_id=-1,
            # Added
            moe_impl="eager",
            vae_downsample_factor=(16, 16),     # (h, w)
            img_proj_type="unet",
            patch_size=1,
            patch_embed_hidden_dim=1024,
            image_base_size=1024,
            rope_type="2d",
            cond_token_attn_type="full",
            cond_image_type="vae_vit",
            vae_type=None,
            vae_dtype="float32",
            vae_autocast_dtype="float16",
            vae=None,
            vit_type=None,
            vit=None,
            vit_processor=None,
            vit_aligner=None,
            cfg_distilled=False,
            use_meanflow=False,
            model_version="HunyuanImage-3.0-Instruct",
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.model_version = model_version
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.moe_impl = moe_impl
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.num_shared_expert = num_shared_expert
        self.moe_topk = moe_topk
        self.capacity_factor = capacity_factor
        self.moe_drop_tokens = moe_drop_tokens
        self.moe_random_routing_dropped_token = moe_random_routing_dropped_token

        if attention_head_dim is not None:
            self.attention_head_dim = attention_head_dim
        else:
            self.attention_head_dim = self.hidden_size // num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.use_cla = use_cla
        self.cla_share_factor = cla_share_factor
        self.norm_type = norm_type
        # MLA args
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

        # DeepSeek related args
        self.moe_layer_num_skipped = moe_layer_num_skipped
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.group_limited_greedy = group_limited_greedy
        self.n_group = n_group
        self.topk_group = topk_group
        self.add_classification_head = add_classification_head
        self.class_num = class_num
        self.pool_type = pool_type
        self.pad_id = pad_id

        if self.class_num is not None:
            self.dense_list = [self.hidden_size, self.class_num]

        # Conditioning image configs
        self.cond_token_attn_type = cond_token_attn_type
        self.cond_image_type = cond_image_type

        # ViT args
        self.vit_type = vit_type
        self.vit = vit
        self.vit_processor = vit_processor
        self.vit_aligner = vit_aligner

        # Image Gen args
        self.vae_type = vae_type
        self.vae_dtype = vae_dtype
        self.vae_autocast_dtype = vae_autocast_dtype
        self.vae = vae
        self.vae_downsample_factor = vae_downsample_factor
        self.img_proj_type = img_proj_type
        self.patch_size = patch_size
        self.patch_embed_hidden_dim = patch_embed_hidden_dim
        self.image_base_size = image_base_size
        self.rope_type = rope_type

        # token id
        self.eod_token_id = eod_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.text_start_id = text_start_id
        self.text_end_id = text_end_id
        self.image_token_id = image_token_id
        self.video_start_id = video_start_id
        self.video_end_id = video_end_id
        self.im_newline_id = im_newline_id
        self.mask_init_id = mask_init_id

        # flag of cfg distilled model
        self.cfg_distilled = cfg_distilled
        # flag of meanflow distilled model
        self.use_meanflow = use_meanflow
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["HunyuanImage3Config"]
