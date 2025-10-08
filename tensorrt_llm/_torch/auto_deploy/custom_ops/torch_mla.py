import torch
import torch.nn as nn


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
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d).contiguous()

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d).contiguous()

    q_embed = (q * cos) + (rotate_half(q) * sin).contiguous()
    k_embed = (k * cos) + (rotate_half(k) * sin).contiguous()
    return q_embed, k_embed


# @torch.library.custom_op("auto_deploy::torch_deepseek_prefill_no_absorb_attn", mutates_args=())
# def torch_deepseek_prefill_no_absorb_attn(
# MHA mode for MLA
# Invoked from the deepseek.py patch
@torch.library.custom_op("auto_deploy::torch_deepseek_mla_no_cache", mutates_args=())
def torch_deepseek_mla_no_cache(
    # q_nope: torch.Tensor,  # [bsz, 128, q_len, 128]
    # q_pe: torch.Tensor,  # [bsz, 128, q_len, 64]
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank=1536]
    compressed_kv: torch.Tensor,  # [bsz, q_len, 512]
    k_pe: torch.Tensor,  # [bsz, 1, q_len, 64]
    sin: torch.Tensor,
    cos: torch.Tensor,
    wkv_b: torch.Tensor,  # [128 * 256, 512]
    wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
    # METADATA
    position_ids: torch.Tensor,
    # CONSTANTS
    softmax_scale: float,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
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

    # Ensure contiguous memory layout for CUDA operations
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()

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

    # Ensure contiguous memory layout for CUDA operations
    k_nope = k_nope.contiguous()
    value_states = value_states.contiguous()

    if position_ids is None:
        q_pe, k_pe = apply_rotary_pos_emb(
            q_pe, k_pe, cos[:kv_seq_len], sin[:kv_seq_len], position_ids
        )
    else:
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

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
    else:
        causal_mask = (
            torch.triu(
                torch.ones(q_len, kv_seq_len, device=q_normed_dn.device, dtype=torch.bool),
                diagonal=1,  # Use diagonal=1 for standard causal masking
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        attn_weights.masked_fill_(causal_mask, float("-inf"))

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


@torch_deepseek_mla_no_cache.register_fake
def torch_deepseek_mla_no_cache_fake(
    q_normed_dn: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    wkv_b: torch.Tensor,
    wq_b: torch.Tensor,
    # METADATA
    position_ids: torch.Tensor,
    # CONSTANTS
    # attention_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    v_head_dim = 128
    num_heads = 128
    bsz, q_len, q_lora_rank = q_normed_dn.shape

    attn_output = torch.empty(bsz, q_len, num_heads * v_head_dim).to(
        device=q_normed_dn.device, dtype=q_normed_dn.dtype
    )
    return attn_output


# @torch.library.custom_op("auto_deploy::torch_deepseek_prefill_no_absorb_attn", mutates_args=())
@torch.library.custom_op("auto_deploy::torch_deepseek_mla_with_kv_cache", mutates_args=())
def torch_deepseek_mla_with_kv_cache(
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
    compressed_kv: torch.Tensor,  # [bsz, kv_len, 512]
    k_pe: torch.Tensor,  # [bsz, 1, kv_len, 64]
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    wkv_b: torch.Tensor,  # [128 * 256, 512]
    wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
    # METADATA
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    position_ids: torch.Tensor,
    # CACHES
    ckv_cache: torch.Tensor,  # [page_num, page_size, ckv_dim]
    k_pe_cache: torch.Tensor,  # [page_num, page_size, k_pe_dim]
    # CONSTANTS
    # attention_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """MHA mode for MLA (i.e. no weight absorption)

    * Cached
    * Serves both prefill and decode
    * Requests are either all decode or mixed prefill+decode.
    * When all requests are decode then input shape is [bsz=nb_seqs, q_seq_len=1, dim]. I.e. request is its own sequence of len 1.
    * When requests are mixed prefill+decode then input shape is [bsz=1, q_seq_len=sum(seq_len), dim]. I.e. all requests are flattened into a single sequence of length q_len.
      This is like a ragged tensor. Use seq_lens to determine the actual number of sequences and tokens per sequence. Use seq_start to determine the actual start position of each sequence.

    Expect one of two cases here:
    1. b > 0, s==1: this indicates a generate-only batch of tokens.
    2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
       and number of tokens per sequence are encoded in seq_len and seq_start.

    If q_seq_len == 1 ==> this is decode-only, otherwise it's mixed prefill+decode
    """

    original_bsz, q_seq_len = q_normed_dn.shape[0:2]
    # v_head_dim = 128
    # num_heads = 128
    # candidate = torch.randn(q_normed_dn.shape[0:2] + (num_heads * v_head_dim,)).to(
    #     device=q_normed_dn.device, dtype=q_normed_dn.dtype
    # )
    # return candidate
    if original_bsz != 1:
        # This is a decode-only batch
        assert q_seq_len == 1, "q_seq_len of each request must be 1 for decode-only batch"
        # (bsz, 1, dim) -> (1, bsz, dim) like in the case of mixed prefill+decode
        q_normed_dn = q_normed_dn.transpose(0, 1)
        compressed_kv = compressed_kv.transpose(0, 1)
        k_pe = k_pe.transpose(0, 2)
        position_ids = position_ids.transpose(0, 1)

    #
    # From this point on, we treat decode-only and mixed prefill+decode the same
    #

    v_head_dim = 128
    num_heads = 128
    qk_nope_head_dim = 128
    # Todo: is q_len even a meaningful variable here?
    bsz, q_len, q_lora_rank = q_normed_dn.shape
    ckv_bsz, kv_seq_len, head_dim_ckv = compressed_kv.shape
    assert ckv_bsz == bsz
    assert kv_seq_len == q_len
    assert sin_cache is not None
    assert cos_cache is not None
    assert bsz == 1

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

    # Ensure contiguous memory layout for CUDA operations
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()

    # Apply RoPE only on the new (uncached) tokens positional encodings
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_cache, sin_cache, position_ids)

    # Store the tokens in the cache, after adding position information
    update_ckv_kpe_cache(
        compressed_kv, k_pe, ckv_cache, k_pe_cache, seq_len, input_pos, cache_loc, seq_start
    )

    # kv = c_K * W^UK (i.e. upward projection)
    # Use entire cache
    kv = (
        torch.einsum(
            # "bsc,xc->bsx", compressed_kv, wkv_b
            "bsc,xc->bsx",
            ckv_cache,
            wkv_b,
        )  # [bsz, q_len, 128*512] - [[change this]] new
        .view(
            ckv_cache.shape[0], ckv_cache.shape[1], num_heads, qk_nope_head_dim + v_head_dim
        )  # [bsz, q_len, 128, 256] - [[change this]] new
        .transpose(1, 2)  # [bsz, 128, ckv_cache.shape[1], 256] - [[change this]] new
    )

    # Over entire cache
    k_nope, value_states = torch.split(
        kv, [qk_nope_head_dim, v_head_dim], dim=-1
    )  # k_nope ~ [bsz, 128, q_len, 128], value_states ~ [bsz, 128, q_len, 128] - [[change this]] new

    # Ensure contiguous memory layout for CUDA operations
    k_nope = k_nope.contiguous()
    value_states = value_states.contiguous()

    # # Apply RoPE only on the new (uncached) tokens positional encodings
    # q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_cache, sin_cache, position_ids)

    # # Store the tokens in the cache, after adding position information
    # update_ckv_kpe_cache(
    #     compressed_kv, k_pe, ckv_cache, k_pe_cache, seq_len, input_pos, cache_loc, seq_start
    # )

    # Concatenate q_nope and q_pe
    query_states = k_pe.new_empty(
        bsz, num_heads, q_len, q_head_dim
    )  # [bsz, 128, q_len, 192] - [[change this]] new
    query_states[..., :qk_nope_head_dim] = q_nope
    query_states[..., qk_nope_head_dim:] = q_pe

    # Concatenate k_nope and k_pe
    key_states = k_pe.new_empty(
        # bsz, num_heads, q_len, q_head_dim
        kv.shape[:-1] + (q_head_dim,)
    )  # [bsz, 128, q_len, 192] - [[change this]] new
    key_states[..., :qk_nope_head_dim] = k_nope
    key_states[..., qk_nope_head_dim:] = k_pe_cache.unsqueeze(1)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Use fused_mla_ref for reference

    # Process one sequence at a time
    attn_outputs = []
    nb_seqs = seq_len.shape[0]
    for i in range(nb_seqs):
        seq_len_i = seq_len[i]
        seq_start_i = seq_start[i]
        input_pos_i = input_pos[i]
        cache_loc_i = cache_loc[i]

        # Batched matmul: [bsz, num_heads, q_len, 192] @ [bsz, num_heads, 192, kv_seq_len].transpose(-1, -2)
        attn_weights = (
            torch.matmul(
                query_states[:, :, seq_start_i : seq_start_i + seq_len_i, :],
                key_states[cache_loc_i, :, seq_start_i : seq_start_i + seq_len_i, :].transpose(
                    -1, -2
                ),
            )
            * softmax_scale
        )  # [bsz, num_heads, q_len, kv_seq_len] - [[change this]] new

        if attn_weights.size() != (bsz, num_heads, seq_len_i, seq_len_i):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, seq_len_i, seq_len_i)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply attention mask for prefill requests only
        if seq_len_i > 1:
            kv_seq_len = input_pos_i + seq_len_i
            causal_mask = (
                torch.triu(
                    torch.ones(seq_len_i, kv_seq_len, device=q_normed_dn.device, dtype=torch.bool),
                    diagonal=1,  # Use diagonal=1 for standard causal masking
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        # upcast attention to fp32 ???
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # attn_output = torch.matmul(attn_weights, v_batched_t)
        attn_output = torch.matmul(
            attn_weights, value_states[cache_loc_i, :, seq_start_i : seq_start_i + seq_len_i, :]
        )

        if attn_output.size() != (bsz, num_heads, seq_len_i, v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, seq_len_i, v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len_i, num_heads * v_head_dim)
        attn_outputs.append(attn_output)

    attn_output = torch.cat(attn_outputs, dim=-2)  # .unsqueeze(0)
    if original_bsz != 1:
        attn_output = attn_output.transpose(0, 1)
    return attn_output


@torch_deepseek_mla_with_kv_cache.register_fake
def torch_deepseek_mla_with_kv_cache_fake(
    q_normed_dn: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    wkv_b: torch.Tensor,
    wq_b: torch.Tensor,
    # METADATA
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    position_ids: torch.Tensor,
    # CACHES
    ckv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    # CONSTANTS
    # attention_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    v_head_dim = 128
    num_heads = 128
    bsz, q_len, _ = q_normed_dn.shape
    attn_output = torch.empty(bsz, q_len, num_heads * v_head_dim).to(
        device=q_normed_dn.device, dtype=q_normed_dn.dtype
    )

    return attn_output


def update_ckv_kpe_cache(
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    seq_len: torch.Tensor,  # metadata
    input_pos: torch.Tensor,  # metadata
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation for update kv cache function. Assumes KV cache layout to be [B,S,D].
    This function can be used to build reference attention implementations that use KV cache.
    """

    for idx in range(seq_len.shape[0]):
        ckv_cache[cache_loc[idx], input_pos[idx] : input_pos[idx] + seq_len[idx], :] = ckv[
            :, seq_start[idx] : seq_start[idx] + seq_len[idx], ...
        ]
        kpe_cache[cache_loc[idx], input_pos[idx] : input_pos[idx] + seq_len[idx], :] = kpe[
            :, :, seq_start[idx] : seq_start[idx] + seq_len[idx], ...
        ]


import math


def update_ckv_kpe_cache_paged(
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    seq_len: torch.Tensor,  # metadata
    input_pos: torch.Tensor,  # metadata
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """
    Reference implementation for update kv cache function. Assumes KV cache layout to be [max_nb_pages,page_size,D].
    This function can be used to build reference attention implementations that use KV cache.
    """

    seq_start_loc = 0
    for seq_idx in range(seq_len.shape[0]):
        kv_page_indptr = math.cumsum(seq_len[:seq_idx])  # location of the first page in the block
        nb_pages_in_seq = math.ceil(seq_len[seq_idx] / page_size)
        tokens_to_copy = (
            seq_len[seq_idx] % page_size if seq_len[seq_idx] % page_size != 0 else page_size
        )

        # Copy the sequence tokens one page at a time
        for page_idx_in_block in range(nb_pages_in_seq):
            if page_idx_in_block == 0:
                tokens_to_copy -= input_pos[seq_idx]
                copy_start_pos = input_pos[seq_idx]
                src_start_pos = seq_start[seq_idx]
            else:
                copy_start_pos = 0
                tokens_to_copy = page_size
                src_start_pos += page_size
            if page_idx_in_block == nb_pages_in_seq - 1:
                tokens_to_copy = (
                    seq_len[seq_idx] % page_size if seq_len[seq_idx] % page_size != 0 else page_size
                )
            virtual_page_loc = kv_page_indptr + page_idx_in_block

            ckv_cache[
                cache_loc[virtual_page_loc], copy_start_pos : copy_start_pos + tokens_to_copy, :
            ] = ckv[:, src_start_pos : src_start_pos + tokens_to_copy, ...]
            kpe_cache[
                cache_loc[virtual_page_loc],
                input_pos[seq_idx] : input_pos[seq_idx] + seq_len[seq_idx],
                :,
            ] = kpe[:, :, src_start_pos : src_start_pos + seq_len[seq_idx], ...]
            seq_start_loc += seq_len[seq_idx]


#####################################################################################################

# import flashinfer

# global_workspace_buffer = None  # can.be empty initialized
# global_trtllm_gen_fmha_workspace_buffer = None  # must be zero initialized
# workspace_size = 128 * 1024 * 1024


# def get_workspace_buffer(device):
#     global global_trtllm_gen_fmha_workspace_buffer
#     if global_trtllm_gen_fmha_workspace_buffer is None:
#         global_trtllm_gen_fmha_workspace_buffer = torch.zeros(
#             workspace_size, dtype=torch.int8, device=device
#         )
#     return global_trtllm_gen_fmha_workspace_buffer


# ###################################################################################

# # # @torch.library.custom_op("auto_deploy::torch_deepseek_prefill_no_absorb_attn", mutates_args=())
# @torch.library.custom_op("auto_deploy::flashinfer_deepseek_mla_with_kv_cache", mutates_args=())
# def flashinfer_deepseek_mla_with_kv_cache(
#     q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
#     compressed_kv: torch.Tensor,  # [bsz, kv_len, 512]
#     k_pe: torch.Tensor,  # [bsz, 1, kv_len, 64]
#     sin_cache: torch.Tensor,
#     cos_cache: torch.Tensor,
#     wkv_b: torch.Tensor,  # [128 * 256, 512]
#     wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
#     # METADATA
#     q_indptr: torch.Tensor,
#     kv_page_indptr: torch.Tensor,
#     kv_page_indices: torch.Tensor,
#     kv_last_page_len: torch.Tensor,
#     batch_indices: torch.Tensor,
#     positions: torch.Tensor,
#     kv_lens: torch.Tensor,
#     page_size: torch.Tensor,
#     position_ids: torch.Tensor,
#     # CACHES
#     ckv_cache: torch.Tensor,  # [page_num, page_size, ckv_dim]
#     k_pe_cache: torch.Tensor,  # [page_num, page_size, k_pe_dim]
#     # CONSTANTS
#     # attention_mask: torch.Tensor,
#     softmax_scale: float,
# ) -> torch.Tensor:
#     """MQA mode for MLA (i.e. with weight absorption)

#     * Cached
#     * Serves both prefill and decode
#     * Requests are either all decode or mixed prefill+decode.
#     * When all requests are decode then input shape is [bsz=nb_seqs, q_seq_len=1, dim]. I.e. request is its own sequence of len 1.
#     * When requests are mixed prefill+decode then input shape is [bsz=1, q_seq_len=sum(seq_len), dim]. I.e. all requests are flattened into a single sequence of length q_len.
#       This is like a ragged tensor. Use seq_lens to determine the actual number of sequences and tokens per sequence. Use seq_start to determine the actual start position of each sequence.

#     Expect one of two cases here:
#     1. b > 0, s==1: this indicates a generate-only batch of tokens.
#     2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
#        and number of tokens per sequence are encoded in seq_len and seq_start.

#     If q_seq_len == 1 ==> this is decode-only, otherwise it's mixed prefill+decode
#     """

#     # v_head_dim = 128
#     # num_heads = 128
#     # candidate = torch.randn(q_normed_dn.shape[0:2] + (num_heads * v_head_dim,)).to(
#     #     device=q_normed_dn.device, dtype=q_normed_dn.dtype
#     # )
#     #return candidate

#     original_bsz, q_seq_len = q_normed_dn.shape[0:2]
#     if original_bsz != 1:
#         # This is a decode-only batch
#         assert q_seq_len == 1, "q_seq_len of each request must be 1 for decode-only batch"
#         # (bsz, 1, dim) -> (1, bsz, dim) like in the case of mixed prefill+decode
#         q_normed_dn = q_normed_dn.transpose(0, 1).contiguous()
#         compressed_kv = compressed_kv.transpose(0, 1).contiguous()
#         k_pe = k_pe.transpose(0, 2).contiguous()
#         position_ids = position_ids.transpose(0, 1).contiguous()

#     q_normed_dn = q_normed_dn.contiguous()
#     #
#     # From this point on, we treat decode-only and mixed prefill+decode the same
#     #

#     v_head_dim = 128
#     num_heads = 128
#     qk_nope_head_dim = 128
#     qk_rope_head_dim = 64
#     q_head_dim = 192
#     # Todo: is q_len even a meaningful variable here?
#     bsz, q_len, q_lora_rank = q_normed_dn.shape
#     ckv_bsz, kv_seq_len, head_dim_ckv = compressed_kv.shape
#     kv_lora_rank = head_dim_ckv
#     assert ckv_bsz == bsz
#     assert kv_seq_len == q_len
#     assert sin_cache is not None
#     assert cos_cache is not None
#     assert bsz == 1


#     # NEW - absorbed q_pe prep
#     wq_b_t = wq_b.transpose(0, 1).reshape(q_lora_rank, num_heads, q_head_dim).contiguous() # [q_lora_rank, 128, 192]
#     q_b_proj_q_nope = wq_b_t[:, :, :qk_nope_head_dim].contiguous() # [q_lora_rank, 128, 128]
#     q_b_proj_q_pe = wq_b_t[:, :, qk_nope_head_dim:].contiguous() # [q_lora_rank, 128, 64]
#     # q_normed_dn is x * W^DQ
#     # q_nope = (x * W^DQ) * W^UQ (i.e. q_nope up projection)
#     q_nope = torch.einsum("bsl,lhd->bhsd", q_normed_dn, q_b_proj_q_nope) # [bsz, 128, q_len, 128]
#     # (x * W^DQ) * W^QR (i.e. q_pe up projection)
#     q_pe = torch.einsum("bsl,lhd->bhsd", q_normed_dn, q_b_proj_q_pe)

#     wkv_b = wkv_b.view(num_heads, -1, kv_lora_rank)  # [128, 256, 512]
#     w_ukv = wkv_b[:, :qk_nope_head_dim].contiguous()  # W^UKV ~ [128, 128, 512]

#     # Weight absorption: (c_Q * W^UQ) * W^UKV
#     # q_nope is already c_Q * W^UQ
#     q_nope = q_nope.transpose(1, 2).contiguous()  # [bsz, q_len, 128, 128]
#     q_nope_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, w_ukv); q_nope = None  # [bsz, q_len, 128, 512]

#     assert position_ids is not None
#     q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_cache, sin_cache, position_ids)
#     q_pe = q_pe.transpose(1, 2).contiguous()
#     k_pe = k_pe.squeeze(1)  # [bsz, kv_seq_len, 64]

#     #############################
#      # Store the tokens in the cache, after adding position information

#     flashinfer.append_paged_mla_kv_cache(
#         compressed_kv.reshape(-1, head_dim_ckv),
#         k_pe.reshape(-1, qk_rope_head_dim),
#         batch_indices,  # from qo_indptr - The batch indices of the each entry in the appended key-value pairs, shape: [append_indptr[-1]].
#         positions,  # from qo_indptr - The positions of the each entry in the appended key-value pairs, shape: [append_indptr[-1]].
#         ckv_cache,
#         k_pe_cache,
#         kv_page_indices,  # The page indices of the paged kv-cache, shape: [kv_indptr[-1]].
#         kv_page_indptr,  # The indptr of the paged kv-cache, shape: [batch_size + 1].
#         kv_last_page_len,  # The number of entries in the last page of each request in the paged kv cache, shape: [batch_size].
#     )

#     #############################

#     # https://sourcegraph.com/github.com/flashinfer-ai/flashinfer/-/blob/tests/test_hopper.py?L130-132
#     backend = "auto"
#     use_cuda_graph = False
#     causal = True
#     workspace_buffer = get_workspace_buffer(device=k_pe.device)

#     wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
#         workspace_buffer,
#         backend=backend,
#         use_cuda_graph=use_cuda_graph,
#         # qo_indptr=torch.empty(bsz + 1, dtype=torch.int32, device=device),
#         # kv_indptr=torch.empty(bsz + 1, dtype=torch.int32, device=device),
#         # kv_indices=torch.empty(1048576, dtype=torch.int32, device=device),
#         # kv_len_arr=torch.empty(bsz, dtype=torch.int32, device=device),
#     )

#     head_dim_kpe = k_pe_cache.shape[-1]
#     wrapper.plan(
#         q_indptr,
#         kv_page_indptr,
#         kv_page_indices,
#         kv_lens,
#         num_heads,
#         head_dim_ckv,
#         head_dim_kpe,
#         page_size.item(),
#         causal,
#         sm_scale=softmax_scale,
#         q_data_type=q_nope_absorbed.dtype,
#         kv_data_type=ckv_cache.dtype,
#     )

#     q_nope_absorbed = q_nope_absorbed.reshape(-1, num_heads, head_dim_ckv)
#     q_pe_absorbed = q_pe.reshape(-1, num_heads, head_dim_kpe); q_pe = None
#     o = wrapper.run(q_nope_absorbed, q_pe_absorbed, ckv_cache, k_pe_cache, return_lse=False)  # [bsz, 128, 128]
#     assert not torch.isnan(o).any()  # wrong shapes will yield nans

#     # Weight absorption: W^UV_O
#     attn_output = torch.einsum("bhc,hdc->bhd", o, wkv_b[:, -v_head_dim:])  # [bsz * q_len, 128, 128]
#     attn_output = attn_output.reshape(bsz, q_len, num_heads * v_head_dim)

#     if original_bsz != 1:
#         attn_output = attn_output.transpose(0, 1)
#     return attn_output


# @flashinfer_deepseek_mla_with_kv_cache.register_fake
# def flashinfer_deepseek_mla_with_kv_cache_fake(
#     q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
#     compressed_kv: torch.Tensor,  # [bsz, kv_len, 512]
#     k_pe: torch.Tensor,  # [bsz, 1, kv_len, 64]
#     sin_cache: torch.Tensor,
#     cos_cache: torch.Tensor,
#     wkv_b: torch.Tensor,  # [128 * 256, 512]
#     wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
#     # METADATA
#     q_indptr: torch.Tensor,
#     paged_kv_indptr: torch.Tensor,
#     paged_kv_indices: torch.Tensor,
#     paged_kv_last_page_len: torch.Tensor,
#     batch_indices: torch.Tensor,
#     positions: torch.Tensor,
#     kv_indices: torch.Tensor,
#     kv_lens: torch.Tensor,
#     page_size: torch.Tensor,
#     position_ids: torch.Tensor,
#     # CACHES
#     ckv_cache: torch.Tensor,  # [page_num, page_size, ckv_dim]
#     k_pe_cache: torch.Tensor,  # [page_num, page_size, k_pe_dim]
#     # CONSTANTS
#     # attention_mask: torch.Tensor,
#     softmax_scale: float,

# ) -> torch.Tensor:
#     v_head_dim = 128
#     num_heads = 128
#     bsz, q_len, q_lora_rank = q_normed_dn.shape

#     attn_output = torch.empty(bsz, q_len, num_heads * v_head_dim).to(
#         device=q_normed_dn.device, dtype=q_normed_dn.dtype
#     )
#     return attn_output


# #############################################################################################
# # Usused reference

# @torch.library.custom_op("auto_deploy::flashinfer_deepseek_mla_no_cache", mutates_args=())
# def flashinfer_deepseek_mla_no_cache(
#     q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
#     compressed_kv: torch.Tensor,  # [bsz, kv_len, 512]
#     k_pe: torch.Tensor,  # [bsz, 1, kv_len, 64]
#     sin_cache: torch.Tensor,
#     cos_cache: torch.Tensor,
#     wkv_b: torch.Tensor,  # [128 * 256, 512]
#     wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
#     position_ids: torch.Tensor,
#     # CONSTANTS
#     softmax_scale: float,
#     attention_mask: torch.Tensor = None
# ) -> torch.Tensor:
#     """MHA mode for MLA (i.e. no weight absorption)

#     * No cache
#     * Serves both prefill and decode
#     * Requests are either all decode or mixed prefill+decode.
#     """

#     v_head_dim = 128
#     num_heads = 128
#     qk_nope_head_dim = 128
#     bsz, q_len, q_lora_rank = q_normed_dn.shape
#     ckv_bsz, kv_seq_len, head_dim_ckv = compressed_kv.shape
#     assert ckv_bsz == bsz
#     assert kv_seq_len == q_len
#     assert sin_cache is not None
#     assert cos_cache is not None

#     # q_normed_dn * (W^UQ and W^QR) (i.e. q up projection)
#     wq_b = wq_b.reshape(num_heads, -1, q_lora_rank)
#     q = torch.einsum("bsl,hdl->bhsd", q_normed_dn, wq_b)  # [bsz, 128, q_len, 192]
#     q_head_dim = q.shape[-1]  # 192
#     qk_rope_head_dim = q_head_dim - qk_nope_head_dim
#     assert qk_rope_head_dim == 64

#     # Separate q into the no-positional encoding part (q_nope) and the positional encoding part (q_pe)
#     q_nope, q_pe = torch.split(
#         q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1
#     )  # q_nope ~ [bsz, 128, q_len, 128], q_pe ~ [bsz, 128, q_len, 64]

#     # kv = c_K * W^UK (i.e. upward projection)
#     kv = (
#         torch.einsum("bsc,xc->bsx", compressed_kv, wkv_b)  # [bsz, q_len, 128, 512]
#         .view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)
#         .transpose(1, 2)
#     )

#     k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

#     q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_cache[:kv_seq_len], sin_cache[:kv_seq_len], position_ids)

#     query_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
#     query_states[:, :, :, :qk_nope_head_dim] = q_nope
#     query_states[:, :, :, qk_nope_head_dim:] = q_pe

#     key_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
#     key_states[:, :, :, :qk_nope_head_dim] = k_nope
#     key_states[:, :, :, qk_nope_head_dim:] = k_pe

#     # https://sourcegraph.com/github.com/flashinfer-ai/flashinfer/-/blob/tests/test_hopper.py?L130-132
#     backend = "auto"
#     workspace_buffer = get_workspace_buffer(query_states.device)
#     wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
#         workspace_buffer,
#         backend=backend,
#         kv_layout="NHD",
#         use_cuda_graph=False,
#     )
#     qo_indptr = torch.arange(0, bsz * q_len + 1, q_len).int().to(query_states.device)
#     kv_indptr = torch.arange(0, bsz * q_len + 1, q_len).int().to(query_states.device)
#     head_dim_qk = 192
#     head_dim_vo = 128
#     causal = True

#     wrapper.plan(
#         qo_indptr,
#         kv_indptr,
#         num_heads,
#         num_heads,
#         head_dim_qk,
#         causal=causal,
#         head_dim_vo=head_dim_vo,
#         q_data_type=query_states.dtype,
#         kv_data_type=value_states.dtype,
#         sm_scale=softmax_scale,
#     )
#     query_states = query_states.transpose(1, 2).contiguous()
#     key_states = key_states.transpose(1, 2).contiguous()
#     value_states = value_states.transpose(1, 2).contiguous()
#     query_states = query_states.reshape(-1, num_heads, head_dim_qk)
#     key_states = key_states.reshape(-1, num_heads, head_dim_qk)
#     value_states = value_states.reshape(-1, num_heads, head_dim_vo)

#     attn_output = wrapper.run(query_states, key_states, value_states, return_lse=False)
#     assert not torch.isnan(attn_output).any()

#     if attn_output.size() != (bsz * q_len, num_heads, v_head_dim):
#         raise ValueError(
#             f"`attn_output` should be of size {(bsz * q_len, num_heads, v_head_dim)}, but is"
#             f" {attn_output.size()}"
#         )

#     attn_output = attn_output.reshape(bsz, q_len, num_heads * v_head_dim)
#     return attn_output


# @flashinfer_deepseek_mla_with_kv_cache.register_fake
# def flashinfer_deepseek_mla_no_cache_fake(
#     q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
#     compressed_kv: torch.Tensor,  # [bsz, kv_len, 512]
#     k_pe: torch.Tensor,  # [bsz, 1, kv_len, 64]
#     sin_cache: torch.Tensor,
#     cos_cache: torch.Tensor,
#     wkv_b: torch.Tensor,  # [128 * 256, 512]
#     wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
#     # METADATA
#     position_ids: torch.Tensor,
#     # CONSTANTS
#     # attention_mask: torch.Tensor,
#     softmax_scale: float,

# ) -> torch.Tensor:
#     v_head_dim = 128
#     num_heads = 128
#     bsz, q_len, q_lora_rank = q_normed_dn.shape

#     attn_output = torch.empty(bsz, q_len, num_heads * v_head_dim).to(
#         device=q_normed_dn.device, dtype=q_normed_dn.dtype
#     )
#     return attn_output
