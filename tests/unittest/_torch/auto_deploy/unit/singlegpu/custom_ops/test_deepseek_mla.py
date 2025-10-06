import logging
import math
from typing import List, Optional, Tuple

import flashinfer
import pytest
import torch
import torch.nn as nn
from transformers import DeepseekV3Config
from transformers.cache_utils import Cache

"""
References:
- https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
- https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L396-L497
- https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json
- https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_mla_decode_kernel.py
- https://sourcegraph.com/github.com/flashinfer-ai/flashinfer/-/blob/tests/test_deepseek_mla.py?L503
"""

# TODO: add cuda-graph support
# TODO: use tighter tolernaces
# TODO: commit the code
# TODO: create AD operators
# TODO: integrate the patches and the operators

logger = logging.getLogger(__name__)

GPU_DEVICE = "cuda:0"
device = GPU_DEVICE
global_workspace_buffer = None  # can.be empty initialized
global_trtllm_gen_fmha_workspace_buffer = None  # must be zero initialized
workspace_size = 128 * 1024 * 1024


def get_workspace_buffer():
    global global_trtllm_gen_fmha_workspace_buffer
    if global_trtllm_gen_fmha_workspace_buffer is None:
        global_trtllm_gen_fmha_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    return global_trtllm_gen_fmha_workspace_buffer


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
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


# This is the original DeepseekV3RotaryEmbedding from Hugging Face, with a couple of patches
# for sine and cosine caching. The cache is computed once (at initialization) and reused later.
class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

        # Patch1: removes this line because we want to use the cache later
        #
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # Patch 2: add this method for convenient access to the sine-cosine cache
    # def get_cos_sin_cache(self):
    #     return (
    #         self.cos_cached,
    #         self.sin_cached,
    #     )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]

        # Patch 3: Prevents the cache from being recomputed and asserts that
        # recomputation is not required.
        # assert self.max_seq_len_cached is not None
        # assert seq_len <= self.max_seq_len_cached
        # if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
        #     self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# This is the original DeepseekV3Attention from Hugging Face, with the original forward method
# and AD's patched forward method (ad_forward)
class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        assert self.config.rope_scaling is None

    def _init_rope(self):
        assert self.config.rope_scaling is None
        self.rotary_emb = DeepseekV3RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    # This is the Auto Deploy patched version of the forward method
    def ad_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attn_impl = kwargs.get("attn_impl", "no_absorb")
        bsz, q_len, _ = hidden_states.size()

        assert self.q_lora_rank is not None, "q_lora_rank must be set"

        # x * W^DQ (i.e. q down projection)
        q_normed_dn = self.q_a_layernorm(
            self.q_a_proj(hidden_states)
        )  # (bsz, q_len, self.q_lora_rank)

        # (x * W^DQ) * (W^UQ and W^QR) (i.e. q up projection)
        # q = self.q_b_proj(q_normed_dn)

        # q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(
        #     1, 2
        # )  # [bsz, 128, q_len, 192]

        # # Separates q into the no-positional encoding part (q_nope) and the positional encoding part (q_pe)
        # q_nope, q_pe = torch.split(
        #     q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        # )  # q_nope ~ [bsz, 128, q_len, 128], q_pe ~ [bsz, 128, q_len, 64]

        # NEW - absorbed q_pe prep
        wq_b = self.q_b_proj.weight  # (self.num_heads * self.q_head_dim, self.q_lora_rank)
        wq_b_t = wq_b.transpose(0, 1).reshape(self.q_lora_rank, self.num_heads, self.q_head_dim)
        q_b_proj_q_nope = wq_b_t[:, :, : self.qk_nope_head_dim]
        q_b_proj_q_pe = wq_b_t[:, :, self.qk_nope_head_dim :]
        # (x * W^DQ) * W^UQ (i.e. q_nope up projection)
        q_nope = torch.einsum("bsl,lhd->bhsd", q_normed_dn, q_b_proj_q_nope)
        # (x * W^DQ) * W^QR (i.e. q_pe up projection)
        q_pe = torch.einsum("bsl,lhd->bhsd", q_normed_dn, q_b_proj_q_pe)

        # c_KV = x * W^DKV (i.e. kv down projection)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [bsz, q_len, 512 + 64]
        # Separates the compressed kv into the low-rank part and the positional encoding part
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )  # compressed_kv ~ [bsz, q_len, 512 ], k_pe ~ [bsz, q_len, 64]
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        # k_pe ~ [bsz, 1, q_len, 64]
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # Patching begins here: delegate the rest to one of the AD operators
        # cos, sin = self.rotary_emb.get_cos_sin_cache()
        cos, sin = self.rotary_emb.cos_cached, self.rotary_emb.sin_cached

        wkv_b = self.kv_b_proj.weight  # [128 * 256, 512]

        if attn_impl == "no_absorb":
            use_kernel = kwargs.get("use_kernel", False)
            args = (
                # q_nope,
                # q_pe,
                q_normed_dn,
                compressed_kv,
                k_pe,
                position_ids,
                attention_mask,  # METADATA
                self.softmax_scale,
                sin,
                cos,
                wkv_b,  # CONSTANTS
                wq_b,
            )
            if not use_kernel:
                # This will be used in the final patched forward method.
                attn_output = torch_deepseek_prefill_no_absorb_attn(*args)
            else:
                # This will be used in the attention backend for prefill requests
                attn_output = flashinfer_deepseek_prefill_no_absorb_attn(*args)
        else:
            assert q_len == 1  # Use these for decode only
            use_kernel = kwargs.get("use_kernel", False)
            ckv_cache = kwargs.get("ckv_cache", None)
            k_pe_cache = kwargs.get("k_pe_cache", None)
            start_pos = kwargs.get("start_pos", 0)

            args = (
                # q_nope,
                # q_pe,
                q_normed_dn,
                compressed_kv,
                k_pe,
                position_ids,
                attention_mask,
                start_pos,  # METADATA
                ckv_cache,
                k_pe_cache,  # CACHES
                self.softmax_scale,
                sin,
                cos,
                wkv_b,  # CONSTANTS
                wq_b,
            )
            if use_kernel:
                # This will be used in the attention backend for decode requests
                attn_output = flashinfer_deepseek_decode_only_absorb_attn(*args)
            else:
                # This is a reference operator to be used for prefill requests
                attn_output = torch_deepseek_decode_only_absorb_attn(*args)
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# This is the original unmodified code from Hugging Face
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def torch_deepseek_prefill_no_absorb_attn(
    # q_nope: torch.Tensor,  # [bsz, 128, q_len, 128]
    # q_pe: torch.Tensor,  # [bsz, 128, q_len, 64]
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank=1536]
    compressed_kv: torch.Tensor,  # [bsz, q_len, 512]
    k_pe: torch.Tensor,  # [bsz, 1, q_len, 64]
    # METADATA
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    # CONSTANTS
    softmax_scale: float,
    sin: torch.Tensor,
    cos: torch.Tensor,
    wkv_b: torch.Tensor,  # [128 * 256, 512]
    wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
):
    v_head_dim = 128
    num_heads = 128
    qk_nope_head_dim = 128
    bsz, q_len, q_lora_rank = q_normed_dn.shape
    ckv_bsz, kv_seq_len, head_dim_ckv = compressed_kv.shape
    assert ckv_bsz == bsz
    assert kv_seq_len == q_len
    assert sin is not None
    assert cos is not None

    # q_normed_dn * (W^UQ and W^QR) (i.e. q up projection)
    wq_b = wq_b.reshape(num_heads, -1, q_lora_rank)
    q = torch.einsum("bsl,hdl->bhsd", q_normed_dn, wq_b)  # [bsz, 128, q_len, 192]
    q_head_dim = q.shape[-1]  # 192
    qk_rope_head_dim = q_head_dim - qk_nope_head_dim
    assert qk_rope_head_dim == 64

    # Separate q into the no-positional encoding part (q_nope) and the positional encoding part (q_pe)
    q_nope, q_pe = torch.split(
        q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1
    )  # q_nope ~ [bsz, 128, q_len, 128], q_pe ~ [bsz, 128, q_len, 64]

    # kv = c_K * W^UK (i.e. upward projection)
    kv = (
        torch.einsum(
            "bsc,xc->bsx", compressed_kv, wkv_b
        )  # [bsz, q_len, 128*512] - [[change this]] new
        .view(
            bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim
        )  # [bsz, q_len, 128, 256] - [[change this]] new
        .transpose(1, 2)  # [bsz, 128, q_len, 256] - [[change this]] new
    )

    k_nope, value_states = torch.split(
        kv, [qk_nope_head_dim, v_head_dim], dim=-1
    )  # k_nope ~ [bsz, 128, q_len, 128], value_states ~ [bsz, 128, q_len, 128] - [[change this]] new

    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos[:kv_seq_len], sin[:kv_seq_len], position_ids)

    query_states = k_pe.new_empty(
        bsz, num_heads, q_len, q_head_dim
    )  # [bsz, 128, q_len, 192] - [[change this]] new
    query_states[:, :, :, :qk_nope_head_dim] = q_nope
    query_states[:, :, :, qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(
        bsz, num_heads, q_len, q_head_dim
    )  # [bsz, 128, q_len, 192] - [[change this]] new
    key_states[:, :, :, :qk_nope_head_dim] = k_nope
    key_states[:, :, :, qk_nope_head_dim:] = k_pe

    # Batched matmul: [bsz, num_heads, q_len, 192] @ [bsz, num_heads, 192, kv_seq_len].transpose(-1, -2)
    attn_weights = (
        torch.matmul(query_states, key_states.transpose(-1, -2)) * softmax_scale
    )  # [bsz, num_heads, q_len, kv_seq_len] - [[change this]] new

    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
    # Apply attention mask (which contains proper causal masking)
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # attn_output = torch.matmul(attn_weights, v_batched_t)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, num_heads, q_len, v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, q_len, v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * v_head_dim)
    return attn_output


def flashinfer_deepseek_prefill_no_absorb_attn(
    # q_nope: torch.Tensor,  # [bsz, 128, q_len, 128]
    # q_pe: torch.Tensor,  # [bsz, 128, q_len, 64]
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
    compressed_kv: torch.Tensor,  # [bsz, q_len, 512]
    k_pe: torch.Tensor,  # [bsz, 1, q_len, 64]
    # METADATA
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    # CONSTANTS
    softmax_scale: float,
    sin: torch.Tensor,
    cos: torch.Tensor,
    wkv_b: torch.Tensor,  # [128 * 256, 512]
    wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
):
    v_head_dim = 128
    num_heads = 128
    qk_nope_head_dim = 128
    bsz, q_len, q_lora_rank = q_normed_dn.shape
    ckv_bsz, kv_seq_len, head_dim_ckv = compressed_kv.shape
    assert ckv_bsz == bsz
    assert kv_seq_len == q_len
    assert sin is not None
    assert cos is not None

    # q_normed_dn * (W^UQ and W^QR) (i.e. q up projection)
    wq_b = wq_b.reshape(num_heads, -1, q_lora_rank)
    q = torch.einsum("bsl,hdl->bhsd", q_normed_dn, wq_b)  # [bsz, 128, q_len, 192]
    q_head_dim = q.shape[-1]  # 192
    qk_rope_head_dim = q_head_dim - qk_nope_head_dim
    assert qk_rope_head_dim == 64

    # Separate q into the no-positional encoding part (q_nope) and the positional encoding part (q_pe)
    q_nope, q_pe = torch.split(
        q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1
    )  # q_nope ~ [bsz, 128, q_len, 128], q_pe ~ [bsz, 128, q_len, 64]

    # kv = c_K * W^UK (i.e. upward projection)
    kv = (
        torch.einsum("bsc,xc->bsx", compressed_kv, wkv_b)  # [bsz, q_len, 128, 512]
        .view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos[:kv_seq_len], sin[:kv_seq_len], position_ids)

    query_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
    query_states[:, :, :, :qk_nope_head_dim] = q_nope
    query_states[:, :, :, qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
    key_states[:, :, :, :qk_nope_head_dim] = k_nope
    key_states[:, :, :, qk_nope_head_dim:] = k_pe

    # https://sourcegraph.com/github.com/flashinfer-ai/flashinfer/-/blob/tests/test_hopper.py?L130-132
    backend = "auto"
    workspace_buffer = get_workspace_buffer()
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer,
        backend=backend,
        kv_layout="NHD",
        use_cuda_graph=False,
    )
    qo_indptr = torch.arange(0, bsz * q_len + 1, q_len).int().to(query_states.device)
    kv_indptr = torch.arange(0, bsz * q_len + 1, q_len).int().to(query_states.device)
    head_dim_qk = 192
    head_dim_vo = 128
    causal = True

    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim_qk,
        causal=causal,
        head_dim_vo=head_dim_vo,
        q_data_type=query_states.dtype,
        kv_data_type=value_states.dtype,
        sm_scale=softmax_scale,
    )
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()
    query_states = query_states.reshape(-1, num_heads, head_dim_qk)
    key_states = key_states.reshape(-1, num_heads, head_dim_qk)
    value_states = value_states.reshape(-1, num_heads, head_dim_vo)

    attn_output = wrapper.run(query_states, key_states, value_states, return_lse=False)
    assert not torch.isnan(attn_output).any()

    if attn_output.size() != (bsz * q_len, num_heads, v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * q_len, num_heads, v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.reshape(bsz, q_len, num_heads * v_head_dim)
    return attn_output


def flashinfer_deepseek_decode_only_absorb_attn(
    # q_nope: torch.Tensor,  # [bsz, 128, q_len, 128]
    # q_pe: torch.Tensor,  # [bsz, 128, q_len, 64]
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
    compressed_kv: torch.Tensor,  # [bsz, kv_len, 512]
    k_pe: torch.Tensor,  # [bsz, 1, kv_len, 64]
    # METADATA
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_pos: int,
    # CACHES
    ckv_cache: torch.Tensor,  # [page_num, page_size, ckv_dim]
    k_pe_cache: torch.Tensor,  # [page_num, page_size, k_pe_dim]
    # CONSTANTS
    softmax_scale: float,
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    wkv_b: torch.Tensor,  # [128 * 256, 512]
    wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
):
    """
    MLA with Matrix Absorption trick
    """
    causal = True
    use_cuda_graph = False
    device = q_normed_dn.device
    ckv_bsz, kv_seq_len, kv_lora_rank = compressed_kv.shape
    kpe_bsz, _, kpe_len, head_dim_kpe = k_pe.shape
    # bsz, num_heads, q_len, qk_nope_head_dim = q_nope.shape
    bsz, q_len, q_lora_rank = q_normed_dn.shape
    v_head_dim = 128
    q_head_dim = 192
    num_heads = 128
    qk_nope_head_dim = 128
    assert ckv_bsz == kpe_bsz == bsz
    assert kv_seq_len == kpe_len == q_len
    assert num_heads == 128
    assert qk_nope_head_dim == 128
    assert sin_cache is not None
    assert cos_cache is not None
    head_dim_ckv = kv_lora_rank
    assert q_len == 1, "q_len must be 1 for matrix absorption with flashinfer"
    assert ckv_cache is not None
    assert k_pe_cache is not None

    # NEW - absorbed q_pe prep
    wq_b_t = wq_b.transpose(0, 1).reshape(q_lora_rank, num_heads, q_head_dim)
    q_b_proj_q_nope = wq_b_t[:, :, :qk_nope_head_dim]
    q_b_proj_q_pe = wq_b_t[:, :, qk_nope_head_dim:]
    # (x * W^DQ) * W^UQ (i.e. q_nope up projection)
    q_nope = torch.einsum("bsl,lhd->bhsd", q_normed_dn, q_b_proj_q_nope)
    # (x * W^DQ) * W^QR (i.e. q_pe up projection)
    q_pe = torch.einsum("bsl,lhd->bhsd", q_normed_dn, q_b_proj_q_pe)

    wkv_b = wkv_b.view(num_heads, -1, kv_lora_rank)  # [128, 256, 512]
    w_ukv = wkv_b[:, :qk_nope_head_dim]  # W^UKV ~ [128, 128, 512]

    # Weight absorption: (c_Q * W^UQ) * W^UKV
    # q_nope is already c_Q * W^UQ
    q_nope = q_nope.transpose(1, 2).contiguous()  # [bsz, q_len, 128, 128]
    q_nope = torch.einsum("bshd,hdc->bshc", q_nope, w_ukv)  # [bsz, q_len, 128, 512]

    q_pe, k_pe = apply_rotary_pos_emb(
        q_pe, k_pe, cos_cache[:kv_seq_len], sin_cache[:kv_seq_len], position_ids
    )
    q_pe = q_pe.transpose(1, 2).contiguous()
    k_pe = k_pe.squeeze(2)  # [bsz, kv_seq_len, 64]

    page_size = 1  # TODO: move this as an argument
    if ckv_cache is not None and k_pe_cache is not None:

        def append_to_kv_cache_linear(
            ckv_cache, k_pe_cache, compressed_kv, k_pe, n_new_tokens: int
        ):
            # end_pos = start_pos + kv_seq_len
            ckv_cache[:bsz, start_pos : start_pos + n_new_tokens, :] = compressed_kv
            k_pe_cache[:bsz, start_pos : start_pos + n_new_tokens, :] = k_pe

            ckv_cache = ckv_cache[:bsz, : start_pos + n_new_tokens]
            k_pe_cache = k_pe_cache[:bsz, : start_pos + n_new_tokens]
            ckv_cache = ckv_cache.reshape(
                -1, page_size, head_dim_ckv
            )  # (num_pages, page_size, head_dim_ckv)
            k_pe_cache = k_pe_cache.reshape(
                -1, page_size, head_dim_kpe
            )  # (num_pages, page_size, head_dim_kpe)
            return ckv_cache, k_pe_cache, start_pos + n_new_tokens

        def append_to_kv_cache(ckv_cache, k_pe_cache, compressed_kv, k_pe, n_new_tokens: int):
            # end_pos = start_pos + kv_seq_len
            # ckv_cache[:bsz, start_pos:start_pos + n_new_tokens, :] = compressed_kv
            # k_pe_cache[:bsz, start_pos:start_pos + n_new_tokens, :] = k_pe

            # ckv_cache = ckv_cache[:bsz, :start_pos + n_new_tokens]
            # k_pe_cache = k_pe_cache[:bsz, :start_pos + n_new_tokens]

            ckv_cache = ckv_cache.reshape(
                -1, page_size, head_dim_ckv
            )  # (num_pages, page_size, head_dim_ckv)
            k_pe_cache = k_pe_cache.reshape(
                -1, page_size, head_dim_kpe
            )  # (num_pages, page_size, head_dim_kpe)

            ckv_cache[bsz * start_pos : bsz * (start_pos + n_new_tokens), :] = (
                compressed_kv.reshape(-1, page_size, head_dim_ckv)
            )
            k_pe_cache[bsz * start_pos : bsz * (start_pos + n_new_tokens), :] = k_pe.reshape(
                -1, page_size, head_dim_kpe
            )

            ckv_cache = ckv_cache[: bsz * (start_pos + n_new_tokens), :].clone().contiguous()
            k_pe_cache = k_pe_cache[: bsz * (start_pos + n_new_tokens), :].clone().contiguous()

            return ckv_cache, k_pe_cache, start_pos + n_new_tokens

        def calculate_last_page_len(kv_len: List[int], page_size: int):
            # Source:https://sourcegraph.com/github.com/flashinfer-ai/flashinfer/-/blob/tests/test_mla_page.py
            return [len % page_size if len % page_size != 0 else page_size for len in kv_len]

        # def append_to_kv_cache_paged(ckv_cache, k_pe_cache, compressed_kv, k_pe, n_new_tokens: int):
        #     kv_len = torch.full((bsz,), kv_seq_len, dtype=torch.int32, device=device)
        #     #kv_len = torch.full((bsz,), start_pos, dtype=torch.int32, device=device)

        #     kv_append_length = kv_len
        #     kv_append_indptr = torch.cat(
        #         [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
        #     ).int()

        #     #cache_loc = torch.full((bsz,), start_pos, dtype=torch.int32, device=device)
        #     num_pages_per_req = torch.tensor(
        #         [math.ceil(len / page_size) for len in kv_len],
        #         dtype=torch.int32,
        #         device=device,
        #     )
        #     #max_num_pages = sum(num_pages_per_req)
        #     kv_page_indptr = torch.cat(
        #         [torch.zeros(1).int().to(device), torch.cumsum(num_pages_per_req, dim=0)]
        #     ).int()
        #     kv_page_indices = torch.arange(
        #         sum(num_pages_per_req), dtype=torch.int32, device=device
        #     )
        #     kv_last_page_len = torch.tensor(
        #         calculate_last_page_len(kv_len, page_size), dtype=torch.int32, device=device
        #     )
        #     # Convert to ragged layout
        #     ckv_append = compressed_kv.reshape(-1, head_dim_ckv)
        #     kpe_append = k_pe.reshape(-1, head_dim_kpe)

        #     nnz_kv = sum(kv_len)
        #     # The batch indices of the each entry in the appended key-value pairs.
        #     batch_indices, positions = flashinfer.get_batch_indices_positions(
        #         kv_append_indptr,
        #         flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        #         nnz_kv,
        #     )

        #     flashinfer.append_paged_mla_kv_cache(
        #         ckv_append,
        #         kpe_append,
        #         batch_indices,
        #         positions,
        #         ckv_cache,
        #         k_pe_cache,
        #         kv_page_indices,
        #         kv_page_indptr,
        #         kv_last_page_len,
        #     )
        #     ckv_cache = ckv_cache.reshape(
        #         -1, page_size, head_dim_ckv
        #     )  # (num_pages, page_size, head_dim_ckv)
        #     k_pe_cache = k_pe_cache.reshape(-1, page_size, head_dim_kpe)  # (num_pages, page_size, head_dim_kpe)
        #     return ckv_cache, k_pe_cache, start_pos + n_new_tokens
        def append_to_kv_cache_paged(ckv_cache, k_pe_cache, compressed_kv, k_pe, n_new_tokens: int):
            # Convert to ragged layout
            # The compressed kv tensor to append in ragged tensor format, shape: [append_indptr[-1], ckv_dim=512]
            append_ckv = compressed_kv.reshape(-1, head_dim_ckv)
            # The K poistional-encoding tensor to append in ragged tensor format, shape: [append_indptr[-1], kpe_dim=64].
            append_kpe = k_pe.reshape(-1, head_dim_kpe)

            kv_append_length = kv_append_seq_lens = torch.full(
                (bsz,), kv_seq_len, dtype=torch.int32, device=device
            )
            # kv_len = kv_append_seq_lens
            nb_appended_tokens = nnz_kv = sum(kv_append_seq_lens)

            kv_append_indptr = torch.cat(
                [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
            ).int()

            num_pages_per_req = torch.tensor(
                [math.ceil(len / page_size) for len in kv_append_seq_lens],
                dtype=torch.int32,
                device=device,
            )
            # existing cache description
            kv_page_indptr = torch.cat(
                [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
            ).int()

            kv_append_length = kv_append_seq_lens
            kv_append_indptr = torch.cat(
                [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
            ).int()

            kv_page_indptr = torch.cat(
                [torch.zeros(1).int().to(device), torch.cumsum(num_pages_per_req, dim=0)]
            ).int()

            kv_page_indices = torch.arange(sum(num_pages_per_req), dtype=torch.int32, device=device)

            kv_last_page_len = torch.tensor(
                calculate_last_page_len(kv_append_seq_lens, page_size),
                dtype=torch.int32,
                device=device,
            )

            nnz_kv = sum(seq_len)
            # The batch indices and positions of each entry in the appended compressed kv tensor. shape: [append_indptr[-1]].
            batch_indices, positions = flashinfer.get_batch_indices_positions(
                qo_indptr,
                flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
                nnz_kv,
            )

            flashinfer.append_paged_mla_kv_cache(
                append_ckv,
                append_kpe,
                batch_indices,  # from qo_indptr - The batch indices of the each entry in the appended key-value pairs, shape: [append_indptr[-1]].
                positions,  # from qo_indptr - The positions of the each entry in the appended key-value pairs, shape: [append_indptr[-1]].
                ckv_cache,
                k_pe_cache,
                kv_page_indices,  # The page indices of the paged kv-cache, shape: [kv_indptr[-1]].
                kv_page_indptr,  # The indptr of the paged kv-cache, shape: [batch_size + 1].
                kv_last_page_len,  # The number of entries in the last page of each request in the paged kv cache, shape: [batch_size].
            )
            ckv_cache = ckv_cache.reshape(
                -1, page_size, head_dim_ckv
            )  # (num_pages, page_size, head_dim_ckv)
            k_pe_cache = k_pe_cache.reshape(
                -1, page_size, head_dim_kpe
            )  # (num_pages, page_size, head_dim_kpe)
            return ckv_cache, k_pe_cache, start_pos + n_new_tokens

        ckv_cache, k_pe_cache, kv_cache_seq_len = (
            append_to_kv_cache(  # <===========================
                ckv_cache, k_pe_cache, compressed_kv, k_pe, n_new_tokens=kv_seq_len
            )
        )
    else:
        kv_cache_seq_len = kv_seq_len
        # Reshape the inputs for flashinfer (paged)
        # num_pages = math.ceil(kv_cache_seq_len * bsz / page_size)
        ckv_cache = compressed_kv.reshape(
            -1, page_size, head_dim_ckv
        )  # (num_pages, page_size, head_dim_ckv)
        k_pe_cache = k_pe.reshape(
            -1, page_size, head_dim_kpe
        )  # (num_pages, page_size, head_dim_kpe)

    assert not (causal and q_len > kv_seq_len), "qo_len > kv_len not supported for causal attention"

    # For decoding attention, the length of each query is 1, and the content
    # of the tensor should be ``[0, 1, 2, ..., batch_size]``.
    q_indptr = torch.arange(0, bsz + 1, device=device, dtype=torch.int32) * q_len
    use_paged_kv_cache = True  # <===========================
    if not use_paged_kv_cache:
        # The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        kv_indptr = torch.arange(0, bsz + 1, device=device, dtype=torch.int32) * kv_cache_seq_len
        # The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]`` or larger
        kv_indices = torch.arange(0, bsz * kv_cache_seq_len, device=device, dtype=torch.int32)
    else:
        kv_indptr = torch.arange(0, bsz + 1, device=device, dtype=torch.int32) * kv_cache_seq_len
        kv_indices = torch.arange(0, bsz * kv_cache_seq_len, device=device, dtype=torch.int32)
        kv_indices[0:32] = torch.arange(0, 32, device=device, dtype=torch.int32)
        kv_indices[32] = 128
        kv_indices[33 : 33 + 32] = torch.arange(32, 64, device=device, dtype=torch.int32)
        kv_indices[65] = 129
        kv_indices[66 : 66 + 32] = torch.arange(64, 96, device=device, dtype=torch.int32)
        kv_indices[98] = 130
        kv_indices[99 : 99 + 32] = torch.arange(96, 128, device=device, dtype=torch.int32)
        kv_indices[131] = 131
        # for i in range(bsz):
        #     # kv_indices[(i + 1) * (kv_cache_seq_len-1)] = bsz * (kv_cache_seq_len-1) + i
        #     # kv_indptr[i* kv_cache_seq_len : (i+1) * kv_cache_seq_len] = torch.arange(
        #     #     i* (kv_cache_seq_len-1), (i+1) * (kv_cache_seq_len-1), device=device, dtype=torch.int32)
        #     kv_indices[(i+1) * kv_cache_seq_len-1] = bsz * (kv_cache_seq_len-1) + i

    # The query length of each request, shape: ``[batch_size]``.
    # kv_lens = torch.full((bsz,), q_len , dtype=torch.int32, device=device)
    kv_lens = torch.full((bsz,), kv_cache_seq_len, dtype=torch.int32, device=device)

    backend = "auto"
    workspace_buffer = get_workspace_buffer()
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer,
        backend=backend,
        use_cuda_graph=use_cuda_graph,
        # qo_indptr=torch.empty(bsz + 1, dtype=torch.int32, device=device),
        # kv_indptr=torch.empty(bsz + 1, dtype=torch.int32, device=device),
        # kv_indices=torch.empty(1048576, dtype=torch.int32, device=device),
        # kv_len_arr=torch.empty(bsz, dtype=torch.int32, device=device),
    )
    # TODO: which sm scale is correct?
    # sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale=softmax_scale,
        q_data_type=q_nope.dtype,
        kv_data_type=ckv_cache.dtype,
    )

    q_nope = q_nope.reshape(-1, num_heads, head_dim_ckv)  # ?
    q_pe = q_pe.reshape(-1, num_heads, head_dim_kpe)  # ?
    o = wrapper.run(q_nope, q_pe, ckv_cache, k_pe_cache, return_lse=False)  # [bsz, 128, 128]
    assert not torch.isnan(o).any()  # wrong shapes will yield nans

    # Weight absorption: W^UV_O
    attn_output = torch.einsum("bhc,hdc->bhd", o, wkv_b[:, -v_head_dim:])  # [bsz, q_len, 128, 128]
    return attn_output


def torch_deepseek_decode_only_absorb_attn(
    # q_nope: torch.Tensor,  # [bsz, 128, q_len, 128]
    # q_pe: torch.Tensor,  # [bsz, 128, q_len, 64]
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
    compressed_kv: torch.Tensor,  # [bsz, kv_len, 512]
    k_pe: torch.Tensor,  # [bsz, 1, kv_len, 64]
    # METADATA
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_pos: int,
    # CACHES
    ckv_cache: torch.Tensor,  # [page_num, page_size, ckv_dim]
    k_pe_cache: torch.Tensor,  # [page_num, page_size, k_pe_dim]
    # CONSTANTS
    softmax_scale: float,
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    wkv_b: torch.Tensor,  # [128 * 256, 512]
    wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
):
    """
    MLA with Matrix Absorption trick
    """

    ckv_bsz, kv_seq_len, kv_lora_rank = compressed_kv.shape
    kpe_bsz, _, kpe_len, head_dim_kpe = k_pe.shape
    # bsz, num_heads, q_len, qk_nope_head_dim = q_nope.shape
    bsz, q_len, q_lora_rank = q_normed_dn.shape
    v_head_dim = 128
    q_head_dim = 192
    num_heads = 128
    qk_nope_head_dim = 128
    assert ckv_bsz == kpe_bsz == bsz
    assert kv_seq_len == kpe_len == q_len
    assert num_heads == 128
    assert qk_nope_head_dim == 128
    assert sin_cache is not None
    assert cos_cache is not None
    head_dim_ckv = kv_lora_rank
    assert q_len == 1, "q_len must be 1 for matrix absorption with flashinfer"
    assert ckv_cache is not None
    assert k_pe_cache is not None

    # NEW - absorbed q_pe prep
    wq_b_t = wq_b.transpose(0, 1).reshape(q_lora_rank, num_heads, q_head_dim)
    q_b_proj_q_nope = wq_b_t[:, :, :qk_nope_head_dim]
    q_b_proj_q_pe = wq_b_t[:, :, qk_nope_head_dim:]
    # (x * W^DQ) * W^UQ (i.e. q_nope up projection)
    q_nope = torch.einsum("bsl,lhd->bhsd", q_normed_dn, q_b_proj_q_nope)
    # (x * W^DQ) * W^QR (i.e. q_pe up projection)
    q_pe = torch.einsum("bsl,lhd->bhsd", q_normed_dn, q_b_proj_q_pe)

    wkv_b = wkv_b.view(num_heads, -1, kv_lora_rank)  # [128, 256, 512]
    w_ukv = wkv_b[:, :qk_nope_head_dim]  #  W^UKV ~ [128, 128, 512]

    # Weight absorption: (c_Q * W^UQ) * W^UKV
    # q_nope is already c_Q * W^UQ
    q_nope = q_nope.transpose(1, 2).contiguous()  # [bsz, q_len, 128, 128]
    q_nope = torch.einsum("bshd,hdc->bshc", q_nope, w_ukv)  # [bsz, q_len, 128, 512]

    # kv = compressed_kv
    assert sin_cache is not None
    assert cos_cache is not None
    q_pe, k_pe = apply_rotary_pos_emb(
        q_pe, k_pe, cos_cache[:kv_seq_len], sin_cache[:kv_seq_len], position_ids
    )
    q_pe = q_pe.transpose(1, 2).contiguous()
    k_pe = k_pe.squeeze(2)  # [bsz, kv_seq_len, 64]

    if ckv_cache is not None and k_pe_cache is not None:

        def append_to_kv_cache(ckv_cache, k_pe_cache, compressed_kv, k_pe, n_new_tokens: int):
            # end_pos = start_pos + kv_seq_len
            ckv_cache[:bsz, start_pos : start_pos + n_new_tokens, :] = compressed_kv
            k_pe_cache[:bsz, start_pos : start_pos + n_new_tokens, :] = k_pe

            ckv_cache = ckv_cache[:bsz, : start_pos + n_new_tokens]
            k_pe_cache = k_pe_cache[:bsz, : start_pos + n_new_tokens]
            return ckv_cache, k_pe_cache, start_pos + n_new_tokens

        ckv_cache, k_pe_cache, kv_cache_seq_len = append_to_kv_cache(
            ckv_cache, k_pe_cache, compressed_kv, k_pe, n_new_tokens=kv_seq_len
        )
    else:
        kv_cache_seq_len = kv_seq_len
        ckv_cache = compressed_kv
        k_pe_cache = k_pe

    attn_weights = (
        torch.einsum("bshc,btc->bsht", q_nope, ckv_cache)  # [bsz, q_len, 128, kv_cache_seq_len]
        + torch.einsum("bshr,btr->bsht", q_pe, k_pe_cache)  # [bsz, q_len, 128, kv_cache_seq_len]
    ) * softmax_scale  # [bsz, q_len, 128, kv_cache_seq_len]

    attn_weights = attn_weights.transpose(1, 2).contiguous()  # [bsz, 128, q_len, kv_cache_seq_len]

    if attn_weights.size() != (bsz, num_heads, q_len, kv_cache_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, num_heads, q_len, kv_cache_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    assert attention_mask is not None
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_cache_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_cache_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q_pe.dtype
    )  # [bsz, 128, q_len, kv_cache_seq_len]

    x = torch.einsum("bhst,btc->bshc", attn_weights, ckv_cache)  # [bsz, q_len, 128, 512]
    attn_output = torch.einsum(
        "bshc,hdc->bshd", x, wkv_b[:, -v_head_dim:]
    )  # [bsz, q_len, 128, 128]
    return attn_output


def test_debug_flashinfer_prefill_kernel():
    torch.manual_seed(42)

    # Test configuration
    num_qo_heads = 64
    num_kv_heads = 16
    head_dim = 128
    softmax_scale = 1.0 / ((128 + 64) ** 0.5)

    bsz = 7
    q_len = 10
    cnt_qo = bsz * q_len  # Total tokens across all batches
    cnt_kv = cnt_qo

    # We're not really testing a ragged input here, so the indptr jumps by q_len for all batches.
    kv_indptr = torch.arange(0, bsz * q_len + 1, q_len).int()
    qo_indptr = torch.arange(0, bsz * q_len + 1, q_len).int()

    # Test inputs
    q = torch.randn(cnt_qo, num_qo_heads, head_dim).to(torch.bfloat16).to("cuda:0")
    k = torch.randn(cnt_kv, num_kv_heads, head_dim).to(torch.bfloat16).to("cuda:0")
    v = torch.randn(cnt_kv, num_kv_heads, head_dim).to(torch.bfloat16).to("cuda:0")

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, "NHD")

    prefill_wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        sm_scale=softmax_scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # FlashInfer output
    o = prefill_wrapper.run(q, k, v)

    # Pytorch reference implementation for ragged GQA
    # The key insight: we have 7 separate sequences of length 10, not one sequence of length 70
    # Each sequence should be processed independently with its own causal mask

    # Expand k and v to match the number of query heads
    num_groups = num_qo_heads // num_kv_heads  # 64 // 16 = 4
    k_expanded = k.repeat_interleave(num_groups, dim=1)  # [70, 16, 128] -> [70, 64, 128]
    v_expanded = v.repeat_interleave(num_groups, dim=1)  # [70, 16, 128] -> [70, 64, 128]

    # Reshape to separate batches: [70, 64, 128] -> [7, 10, 64, 128]
    q_batched = q.view(bsz, q_len, num_qo_heads, head_dim)  # [7, 10, 64, 128]
    k_batched = k_expanded.view(bsz, q_len, num_qo_heads, head_dim)  # [7, 10, 64, 128]
    v_batched = v_expanded.view(bsz, q_len, num_qo_heads, head_dim)  # [7, 10, 64, 128]

    # Process all batches simultaneously using batched operations
    # Transpose all batches at once: [7, 10, 64, 128] -> [7, 64, 10, 128]
    q_batched_t = q_batched.transpose(1, 2)  # [7, 64, 10, 128]
    k_batched_t = k_batched.transpose(1, 2)  # [7, 64, 10, 128]
    v_batched_t = v_batched.transpose(1, 2)  # [7, 64, 10, 128]

    # Batched matmul: [7, 64, 10, 128] @ [7, 64, 128, 10] = [7, 64, 10, 10]
    attn_weights = torch.matmul(q_batched_t, k_batched_t.transpose(-1, -2)) * softmax_scale

    # Create causal mask once and broadcast: [10, 10] -> [1, 1, 10, 10]
    causal_mask = torch.triu(torch.ones(q_len, q_len, device=q.device), diagonal=1).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 10, 10]

    # Apply mask to all batches and heads at once
    attn_weights.masked_fill_(causal_mask, float("-inf"))

    # Batched softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    # Batched attention: [7, 64, 10, 10] @ [7, 64, 10, 128] = [7, 64, 10, 128]
    attn_output_batched = torch.matmul(attn_weights, v_batched_t)

    # Transpose back: [7, 64, 10, 128] -> [7, 10, 64, 128]
    attn_output_batched = attn_output_batched.transpose(1, 2)

    # Reshape back to original format: [7, 10, 64, 128] -> [70, 64, 128]
    attn_output = attn_output_batched.reshape(-1, num_qo_heads, head_dim)

    print(f"attn_output.shape: {attn_output.shape}")
    print(f"o.shape: {o.shape}")

    diff = (o - attn_output).abs()
    print(f"max difference: {diff.max()}")

    assert torch.allclose(o, attn_output, atol=2e-2, rtol=1e-3)


@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seqlen_q", [1])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@torch.inference_mode()
def _test_deepseek_mla_decode_no_cache(causal, dtype, seqlen_q, batch_size, device):
    torch.manual_seed(42)
    config = DeepseekV3Config()
    hf_deepseek_attn = DeepseekV3Attention(config).to(device).to(dtype)

    # (batch_size, seq_len, dim)
    hidden_states = torch.randn(batch_size, seqlen_q, config.hidden_size, dtype=dtype).to(device)
    attention_mask = torch.zeros(batch_size, 1, seqlen_q, seqlen_q, dtype=dtype).to(device)
    # position_ids = torch.arange(seqlen_q, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    # hf_deepseek_output, _, _ = hf_deepseek_attn(hidden_states, attention_mask, position_ids)

    hf_deepseek_ref_output, _, _ = hf_deepseek_attn.forward(hidden_states, attention_mask)
    assert hf_deepseek_ref_output.shape == (batch_size, seqlen_q, config.hidden_size)

    ad_deepseek_output_no_absorb, _, _ = hf_deepseek_attn.ad_forward(
        hidden_states, attention_mask, attn_impl="no_absorb"
    )
    assert ad_deepseek_output_no_absorb.shape == (batch_size, seqlen_q, config.hidden_size)

    diff = (hf_deepseek_ref_output - ad_deepseek_output_no_absorb).abs().flatten()
    print(diff)
    print(f"decode no-absorb ref vs torch max(diff)={max(diff)}")
    assert torch.allclose(
        hf_deepseek_ref_output, ad_deepseek_output_no_absorb, atol=1e-2, rtol=1e-3
    )

    ad_deepseek_output_absorb, _, _ = hf_deepseek_attn.ad_forward(
        hidden_states, attention_mask, attn_impl="absorb"
    )
    assert ad_deepseek_output_absorb.shape == (batch_size, seqlen_q, config.hidden_size)

    diff = (hf_deepseek_ref_output - ad_deepseek_output_absorb).abs().flatten()
    print(diff)
    print(f"decode absorb ref vs torch max(diff)={max(diff)}")
    assert torch.allclose(hf_deepseek_ref_output, ad_deepseek_output_absorb, atol=1e-2, rtol=1e-3)

    ad_deepseek_output_absorb_kernel, _, _ = hf_deepseek_attn.ad_forward(
        hidden_states, attention_mask, attn_impl="absorb", use_kernel=True
    )
    assert ad_deepseek_output_absorb.shape == (batch_size, seqlen_q, config.hidden_size)

    diff = (hf_deepseek_ref_output - ad_deepseek_output_absorb_kernel).abs().flatten()
    print(diff)
    print(f"decode absorb ref vs flashinfer max(diff)={max(diff)}")
    assert torch.allclose(
        hf_deepseek_ref_output, ad_deepseek_output_absorb_kernel, atol=1e-2, rtol=1e-3
    )


def causal_attention_mask(batch_size, seqlen_q, seqlen_k, device, dtype=torch.bfloat16):
    causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=device), diagonal=1).bool()
    attention_mask = torch.zeros(batch_size, 1, seqlen_q, seqlen_k, dtype=dtype, device=device)
    attention_mask.masked_fill_(causal_mask, float("-inf"))
    return attention_mask


@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seqlen_q", [6])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@torch.inference_mode()
def test_deepseek_mla_prefill(causal, dtype, seqlen_q, batch_size, device):
    torch.manual_seed(42)
    config = DeepseekV3Config()
    hf_deepseek_attn = DeepseekV3Attention(config).to(device).to(dtype)

    hidden_states = torch.randn(batch_size, seqlen_q, config.hidden_size, dtype=dtype).to(device)

    # Create proper causal mask: 0 for past/current positions, -inf for future positions
    causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_q, device=device), diagonal=1).bool()
    attention_mask = torch.zeros(batch_size, 1, seqlen_q, seqlen_q, dtype=dtype, device=device)
    attention_mask.masked_fill_(causal_mask, float("-inf"))

    hf_deepseek_ref_output, _, _ = hf_deepseek_attn.forward(hidden_states, attention_mask)
    assert hf_deepseek_ref_output.shape == (batch_size, seqlen_q, config.hidden_size)

    ad_deepseek_output_no_absorb, _, _ = hf_deepseek_attn.ad_forward(
        hidden_states, attention_mask, attn_impl="no_absorb"
    )
    assert ad_deepseek_output_no_absorb.shape == (batch_size, seqlen_q, config.hidden_size)

    diff = (hf_deepseek_ref_output - ad_deepseek_output_no_absorb).abs().flatten()
    print(diff)
    print(f"prefill (no-absorb) ref vs torch max(diff)={max(diff)}")
    assert torch.allclose(
        hf_deepseek_ref_output, ad_deepseek_output_no_absorb, atol=1e-2, rtol=1e-3
    )

    ad_deepseek_output_no_absorb_kernel, _, _ = hf_deepseek_attn.ad_forward(
        hidden_states, attention_mask, attn_impl="no_absorb", use_kernel=True
    )
    assert ad_deepseek_output_no_absorb_kernel.shape == (batch_size, seqlen_q, config.hidden_size)

    diff = (hf_deepseek_ref_output - ad_deepseek_output_no_absorb_kernel).abs()
    print(diff)
    print(f"prefill (no-absorb) ref vs flashinfer max(diff)={diff.max()}")
    assert torch.allclose(
        hf_deepseek_ref_output, ad_deepseek_output_no_absorb_kernel, atol=1e-2, rtol=1e-3
    )


def create_linear_caches(
    max_batch_size: int, max_seq_len: int, device: str, dtype: torch.dtype, init_randn: bool = False
):
    """
    Create ompressed KV and K positional encoding caches (CKV and KPE) with a [B,S,D] layout.
    Note that there is only one (shared) head.
    """

    def _create_cache(head_dim: int):
        tensor_init = torch.randn if init_randn else torch.zeros
        return tensor_init(
            max_batch_size,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

    head_dim_ckv = 512  # AKA kv_lora_rank
    qk_rope_head_dim = 64
    ckv_cache = _create_cache(head_dim_ckv)
    kpe_cache = _create_cache(qk_rope_head_dim)
    return ckv_cache, kpe_cache


def create_paged_caches(
    num_pages: int, page_size: int, device: str, dtype: torch.dtype, init_randn: bool = False
):
    """
    The compressed KV and K positional encoding caches (CKV and KPE) with paged layout.
    Note that there is only one (shared) head.
    """

    def _create_cache(head_dim: int):
        tensor_init = torch.randn if init_randn else torch.zeros
        return tensor_init(
            num_pages,
            page_size,
            head_dim,
            device=device,
            dtype=dtype,
        )

    head_dim_ckv = 512  # AKA kv_lora_rank
    qk_rope_head_dim = 64
    ckv_cache = _create_cache(head_dim_ckv)
    kpe_cache = _create_cache(qk_rope_head_dim)
    return ckv_cache, kpe_cache


@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seqlen_q", [1])
@pytest.mark.parametrize("seqlen_kv", [32])  # length of pre-cached kv
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("max_batch_size", [32])
@pytest.mark.parametrize("max_seq_len", [64])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@torch.inference_mode()
def test_deepseek_mla_decode_cache(
    causal, dtype, seqlen_q, seqlen_kv, batch_size, max_batch_size, max_seq_len, page_size, device
):
    torch.manual_seed(42)
    assert max_batch_size >= batch_size
    assert max_seq_len >= seqlen_q + seqlen_kv
    assert seqlen_q == 1  # decode only
    assert page_size == 1  # decode only

    config = DeepseekV3Config()
    hf_deepseek_attn = DeepseekV3Attention(config).to(device).to(dtype)

    # (batch_size, seq_len, dim)
    hidden_states = torch.randn(batch_size, seqlen_q, config.hidden_size, dtype=dtype).to(device)
    # the model is decoding the next token (L+1), the attention mask has shape [1, L+1].
    # The mask will allow the new token to attend to all past tokens—including itself
    attention_mask = torch.zeros(batch_size, 1, seqlen_q, seqlen_q + seqlen_kv, dtype=dtype).to(
        device
    )
    # attention_mask = causal_attention_mask(batch_size, seqlen_q, seqlen_q + seqlen_kv, device, dtype)

    # attention_mask = torch.tril(torch.ones(seqlen_q, seqlen_kv+seqlen_q)).bool()
    compressed_kv_normed_cache, k_pe_cache = create_linear_caches(
        max_batch_size, max_seq_len, device, dtype, init_randn=True
    )

    ad_deepseek_output_absorb, _, _ = hf_deepseek_attn.ad_forward(
        hidden_states,
        attention_mask,
        attn_impl="absorb",
        ckv_cache=compressed_kv_normed_cache,
        k_pe_cache=k_pe_cache,
        start_pos=seqlen_kv,
    )
    assert ad_deepseek_output_absorb.shape == (batch_size, seqlen_q, config.hidden_size)

    num_pages = math.ceil(max_batch_size / page_size) * max_seq_len
    compressed_kv_normed_cache_paged, k_pe_cache_paged = create_paged_caches(
        num_pages, page_size, device, dtype, init_randn=False
    )

    # copy from the linear cache to the paged cache
    ckv_to_copy = compressed_kv_normed_cache[:batch_size, :seqlen_kv, :]
    k_pe_to_copy = k_pe_cache[:batch_size, :seqlen_kv, :]
    ckv_to_copy = ckv_to_copy.clone().contiguous().reshape(-1, page_size, 512)
    k_pe_to_copy = k_pe_to_copy.clone().contiguous().reshape(-1, page_size, 64)
    compressed_kv_normed_cache_paged[: batch_size * seqlen_kv, :] = ckv_to_copy
    k_pe_cache_paged[: batch_size * seqlen_kv, :] = k_pe_to_copy

    ad_deepseek_output_absorb_kernel, _, _ = hf_deepseek_attn.ad_forward(
        hidden_states,
        attention_mask,
        attn_impl="absorb",
        use_kernel=True,
        ckv_cache=compressed_kv_normed_cache_paged,
        k_pe_cache=k_pe_cache_paged,
        start_pos=seqlen_kv,
    )
    assert ad_deepseek_output_absorb_kernel.shape == (batch_size, seqlen_q, config.hidden_size)

    diff = (ad_deepseek_output_absorb - ad_deepseek_output_absorb_kernel).abs()
    print(diff)
    print(f"decode torch vs flashinfermax: (diff)={diff.max()}")
    assert torch.allclose(
        ad_deepseek_output_absorb, ad_deepseek_output_absorb_kernel, atol=1e-2, rtol=1e-3
    )
