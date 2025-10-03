import types
import warnings
from typing import Dict, Optional

import torch
import torch.utils.checkpoint
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache


@torch.inference_mode()
def deepseek_v3_attention(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.IntTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):

    """DeepSeekV3Attention forward function rewritten to wrap MultiheadLatentAttention as a custom op."""
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    assert self.q_lora_rank is not None, "q_lora_rank must be set"
    
    # x * W^DQ (i.e. q down projection)
    q_normed_dn = self.q_a_layernorm(self.q_a_proj(hidden_states)) # (bsz, q_len, self.q_lora_rank)

    # (x * W^DQ) * (W^UQ and W^QR) (i.e. q up projection)
    #q = self.q_b_proj(q_normed_dn)

    # q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(
    #     1, 2
    # )  # [bsz, 128, q_len, 192]

    # # Separates q into the no-positional encoding part (q_nope) and the positional encoding part (q_pe)
    # q_nope, q_pe = torch.split(
    #     q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    # )  # q_nope ~ [bsz, 128, q_len, 128], q_pe ~ [bsz, 128, q_len, 64]


    # NEW - absrobed q_pe prep
    wq_b = self.q_b_proj.weight # (self.num_heads * self.q_head_dim, self.q_lora_rank)
    #wq_b_t = wq_b.transpose(0, 1).reshape(self.q_lora_rank, self.num_heads, self.q_head_dim)
    #q_b_proj_q_nope = wq_b_t[:, :, :self.qk_nope_head_dim]
    #q_b_proj_q_pe = wq_b_t[:, :, self.qk_nope_head_dim:]
    # (x * W^DQ) * W^UQ (i.e. q_nope up projection)
    #q_nope = torch.einsum("bsl,lhd->bhsd", q_normed_dn, q_b_proj_q_nope)
    # (x * W^DQ) * W^QR (i.e. q_pe up projection)
    #q_pe = torch.einsum("bsl,lhd->bhsd", q_normed_dn, q_b_proj_q_pe)

    # c_KV = x * W^DKV (i.e. kv down projection)
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [bsz, q_len, 512 + 64]
    # Separates the compressed kv into the low-rank part and the positional encoding part
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )  # compressed_kv ~ [bsz, q_len, 512 ], k_pe ~ [bsz, q_len, 64]
    compressed_kv = self.kv_a_layernorm(compressed_kv)

    # return torch.randn_like(hidden_states), None, None

    # k_pe ~ [bsz, 1, q_len, 64]
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

    # Patching begins here: delegate the rest to one of the AD operators
    # cos, sin = self.rotary_emb.get_cos_sin_cache()
    cos, sin = self.rotary_emb.cos_cached, self.rotary_emb.sin_cached

    wkv_b = self.kv_b_proj.weight  # [128 * 256, 512]

    # Use custom op to capture mla. This does not handle KV cache
    # as passing transformers Cache into a custom op is throwing an error.
    # Is not an issue, because we intend to replace mla op with our implementation further along the pipeline
    # attn_output = torch.ops.auto_deploy.torch_attention_deepseek_fused_mla(
    #     q_nope,
    #     q_pe,
    #     kv,
    #     k_pe,
    #     cos,
    #     sin,
    #     position_ids,
    #     attention_mask,
    #     self.softmax_scale,
    # )
    args = (
                #q_nope,
                #q_pe,
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
    attn_output = torch.ops.auto_deploy.torch_deepseek_prefill_no_absorb_attn(*args)
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

    ##################################################################
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

        print(f"Debug: In apply_rotary_pos_emb - cos.shape={cos.shape}, sin.shape={sin.shape}")
        print(f"Debug: position_ids.shape={position_ids.shape}, position_ids={position_ids}")
        print(f"Debug: unsqueeze_dim={unsqueeze_dim}")
        
        # Add bounds checking for the indexing operation that could be causing the device assertion
        # if position_ids.max() >= cos.shape[0]:
        #     raise RuntimeError(f"In apply_rotary_pos_emb: position_ids.max() ({position_ids.max()}) >= cos.shape[0] ({cos.shape[0]})")
        # if position_ids.min() < 0:
        #     raise RuntimeError(f"In apply_rotary_pos_emb: position_ids.min() ({position_ids.min()}) < 0")
            
        try:
            _cos = cos[position_ids].unsqueeze(unsqueeze_dim)
            _sin = sin[position_ids].unsqueeze(unsqueeze_dim)
            print(f"Debug: Successfully indexed cos and sin, new shapes: cos={cos.shape}, sin={sin.shape}")
        except Exception as e:
            print(f"Debug: Error indexing cos/sin: {e}")
            raise

        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        q_embed = (q * _cos) + (rotate_half(q) * _sin)
        k_embed = (k * _cos) + (rotate_half(k) * _sin)
        return q_embed, k_embed

    v_head_dim = 128
    num_heads = 128
    qk_nope_head_dim = 128
    bsz, q_len, q_lora_rank = q_normed_dn.shape
    ckv_bsz, kv_seq_len, head_dim_ckv = compressed_kv.shape
    assert ckv_bsz == bsz
    assert kv_seq_len == q_len
    assert sin is not None
    assert cos is not None
    
    # Validate dimensions match expectations before reshape
    expected_wq_b_shape = (num_heads * (qk_nope_head_dim + 64), q_lora_rank)  # 64 = qk_rope_head_dim
    if wq_b.shape != expected_wq_b_shape:
        raise RuntimeError(f"wq_b shape mismatch: got {wq_b.shape}, expected {expected_wq_b_shape}")
    
    wq_b = wq_b.reshape(num_heads, -1, q_lora_rank)
    print(f"Debug: About to perform einsum with q_normed_dn.shape={q_normed_dn.shape}, wq_b.shape={wq_b.shape}")
    
    try:
        q = torch.einsum("bsl,hdl->bhsd", q_normed_dn, wq_b) # [bsz, 128, q_len, 192]
        print(f"Debug: einsum successful, q.shape={q.shape}")
    except Exception as e:
        print(f"Debug: Error in einsum: {e}")
        raise
    q_head_dim = q.shape[-1] # 192
    qk_rope_head_dim = q_head_dim - qk_nope_head_dim
    assert qk_rope_head_dim == 64

    # return torch.randn_like(hidden_states), None, None

    # Separate q into the no-positional encoding part (q_nope) and the positional encoding part (q_pe)
    q_nope, q_pe = torch.split(
        q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1
    )  # q_nope ~ [bsz, 128, q_len, 128], q_pe ~ [bsz, 128, q_len, 64]
    
    # Ensure contiguous memory layout for CUDA operations
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()

    # kv = c_K * W^UK (i.e. upward projection)
    print(f"Debug: About to perform kv einsum with compressed_kv.shape={compressed_kv.shape}, wkv_b.shape={wkv_b.shape}")
    
    try:
        kv_result = torch.einsum(
            "bsc,xc->bsx", compressed_kv, wkv_b
        )  # [bsz, q_len, 128*512] - [[change this]] new
        print(f"Debug: kv einsum successful, result.shape={kv_result.shape}")
        
        kv = (
            kv_result
            .view(
                bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim
            )  # [bsz, q_len, 128, 256] - [[change this]] new
            .transpose(1, 2)  # [bsz, 128, q_len, 256] - [[change this]] new
        )
        print(f"Debug: kv reshape and transpose successful, kv.shape={kv.shape}")
    except Exception as e:
        print(f"Debug: Error in kv operations: {e}")
        raise

    k_nope, value_states = torch.split(
        kv, [qk_nope_head_dim, v_head_dim], dim=-1
    )  # k_nope ~ [bsz, 128, q_len, 128], value_states ~ [bsz, 128, q_len, 128] - [[change this]] new
    
    # Ensure contiguous memory layout for CUDA operations
    k_nope = k_nope.contiguous()
    value_states = value_states.contiguous()

    # Add debugging for rotary embedding inputs
    print(f"Debug: Before rotary_pos_emb - cos.shape={cos.shape}, sin.shape={sin.shape}")
    print(f"Debug: kv_seq_len={kv_seq_len}, position_ids.shape={position_ids.shape}")
    print(f"Debug: position_ids min={position_ids.min()}, max={position_ids.max()}")
    
    # Check bounds before slicing cos/sin
    if kv_seq_len > cos.shape[0]:
        raise RuntimeError(f"kv_seq_len ({kv_seq_len}) > cos.shape[0] ({cos.shape[0]})")
    if kv_seq_len > sin.shape[0]:  
        raise RuntimeError(f"kv_seq_len ({kv_seq_len}) > sin.shape[0] ({sin.shape[0]})")
    
    # Check position_ids bounds
    # if position_ids.max() >= cos.shape[0]:
    #     raise RuntimeError(f"position_ids.max() ({position_ids.max()}) >= cos.shape[0] ({cos.shape[0]})")
    # if position_ids.min() < 0:
    #     raise RuntimeError(f"position_ids.min() ({position_ids.min()}) < 0")
    
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    print(f"Debug: After rotary_pos_emb - q_pe.shape={q_pe.shape}, k_pe.shape={k_pe.shape}")
    # return torch.randn_like(hidden_states), None, None
    query_states = k_pe.new_empty(
        bsz, num_heads, q_len, q_head_dim
    )  # [bsz, 128, q_len, 192] - [[change this]] new
    
    # Validate tensor dimensions before indexing to prevent device-side assertions
    if q_nope.shape != (bsz, num_heads, q_len, qk_nope_head_dim):
        raise RuntimeError(f"q_nope shape mismatch: got {q_nope.shape}, expected {(bsz, num_heads, q_len, qk_nope_head_dim)}")
    if q_pe.shape[-1] != q_head_dim - qk_nope_head_dim:
        raise RuntimeError(f"q_pe last dimension mismatch: got {q_pe.shape[-1]}, expected {q_head_dim - qk_nope_head_dim}")
    if qk_nope_head_dim > q_head_dim:
        raise RuntimeError(f"qk_nope_head_dim ({qk_nope_head_dim}) > q_head_dim ({q_head_dim})")
        
    # Add step-by-step debugging for tensor assignments
    print(f"Debug: About to assign q_nope to query_states[:, :, :, :{qk_nope_head_dim}]")
    print(f"Debug: query_states slice shape: {query_states[:, :, :, :qk_nope_head_dim].shape}")
    print(f"Debug: q_nope shape: {q_nope.shape}")
    
    try:
        query_states[:, :, :, :qk_nope_head_dim] = q_nope
        print("Debug: Successfully assigned q_nope")
    except Exception as e:
        print(f"Debug: Error assigning q_nope: {e}")
        raise
    
    print(f"Debug: About to assign q_pe to query_states[:, :, :, {qk_nope_head_dim}:]")
    print(f"Debug: query_states slice shape: {query_states[:, :, :, qk_nope_head_dim:].shape}")
    print(f"Debug: q_pe shape: {q_pe.shape}")
    
    try:
        query_states[:, :, :, qk_nope_head_dim:] = q_pe
        print("Debug: Successfully assigned q_pe")
    except Exception as e:
        print(f"Debug: Error assigning q_pe: {e}")
        raise

    # return torch.randn_like(hidden_states), None, None

    key_states = k_pe.new_empty(
        bsz, num_heads, q_len, q_head_dim
    )  # [bsz, 128, q_len, 192] - [[change this]] new
    key_states[:, :, :, :qk_nope_head_dim] = k_nope
    key_states[:, :, :, qk_nope_head_dim:] = k_pe

    # Batched matmul: [bsz, num_heads, q_len, 192] @ [bsz, num_heads, 192, kv_seq_len].transpose(-1, -2)
    attn_weights = (
        torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale
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

    # return torch.randn_like(hidden_states), None, None
    # upcast attention to fp32
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
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


    ##################################################################
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# This patched module matches exactly with HF generate
@torch.inference_mode()
def deepseek_v3_moe_exact(self, hidden_states):
    """DeepSeekV3MoE forward function rewritten to enable torch export.

    This custom implementation matches exactly with the deepseek implementation. There are
    some errors in the output tensors when the index_add based implementation is used, leading
    to some mismatch in the outputs for some prompts. This ensures exact match between HF output
    without custom patch and with custom patch.
    """
    identity = hidden_states
    batch_size, sequence_length, hidden_dim = hidden_states.shape

    selected_experts, routing_weights, *_ = self.gate(hidden_states)

    hidden_states = hidden_states.view(-1, hidden_dim)
    idxs = torch.argsort(selected_experts.view(-1), stable=True)

    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=self.experts_per_rank
    ).permute(2, 1, 0)
    outputs = []
    for expert_idx in range(len(self.experts)):
        expert_layer = self.experts[expert_idx]
        _, top_x = torch.where(expert_mask[expert_idx])
        # Sort the top_xs and idx
        sorted, _ = torch.sort(top_x)
        tokens_for_this_expert = hidden_states[None, sorted].reshape(-1, hidden_dim)
        expert_out = expert_layer(tokens_for_this_expert)
        outputs.append(expert_out)

    outs = torch.cat(outputs, dim=0)
    # Wrap torch.zeros() in a custom op to fix meta device issue during inference.
    new_x = torch.zeros(
        (*selected_experts.view(-1).shape, hidden_dim),
        device=selected_experts.device,
        dtype=outs.dtype,
    )
    new_x[idxs] = outs
    final_hidden_states = (
        new_x.view(*selected_experts.shape, -1)
        .type(routing_weights.dtype)
        .mul_(routing_weights.unsqueeze(-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    if self.config.n_shared_experts is not None:
        final_hidden_states = final_hidden_states + self.shared_experts(identity)

    return final_hidden_states.to(hidden_states.dtype)


@torch.inference_mode()
def deepseek_v3_moe(self, hidden_states):
    """DeepSeekV3MoE forward function rewritten in Mixtral style to enable torch export."""

    selected_experts, routing_weights, *_ = self.gate(hidden_states)
    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        w1_weight=[expert.gate_proj.weight for expert in self.experts],
        w2_weight=[expert.down_proj.weight for expert in self.experts],
        w3_weight=[expert.up_proj.weight for expert in self.experts],
    )

    if self.config.n_shared_experts is not None:
        final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)

    return final_hidden_states.to(hidden_states.dtype)


def deepseek_v3_rope(self, x, seq_len=None):
    """DeepSeekV3 Rotary Embedding forward function rewritten to enable torch export.
    We return the full cached cos and sin values, instead of slicing them based on seq_len as this
    would cause an issue during the generate phase (when seq_len=1 from input_ids). We also move the cos
    sin buffers to appropriate device to enable export.
    """

    return (
        self.cos_cached.to(dtype=x.dtype).to(device=x.device),
        self.sin_cached.to(dtype=x.dtype).to(device=x.device),
    )


_from_config_original = AutoModelForCausalLM.from_config

CUSTOM_MODULE_PATCHES: Dict[str, callable] = {
    "DeepseekV3MoE": deepseek_v3_moe,
    "DeepseekV2MoE": deepseek_v3_moe,
    "DeepseekV3RotaryEmbedding": deepseek_v3_rope,
    "DeepseekV3YarnRotaryEmbedding": deepseek_v3_rope,
    "DeepseekV2RotaryEmbedding": deepseek_v3_rope,
    "DeepseekV2YarnRotaryEmbedding": deepseek_v3_rope,
    "DeepseekV3Attention": deepseek_v3_attention,
}


def get_model_from_config_patched(config, **kwargs):
    model = _from_config_original(config, **kwargs)
    # Patch modules
    for _, module in model.named_modules():
        if type(module).__name__ in CUSTOM_MODULE_PATCHES.keys():
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[type(module).__name__], module)

    return model


# TODO: figure out how this can be incorporated into the export patch system
AutoModelForCausalLM.from_config = get_model_from_config_patched
