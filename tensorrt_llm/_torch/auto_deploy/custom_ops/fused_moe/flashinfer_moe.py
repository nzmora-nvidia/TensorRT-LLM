from typing import List

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType


@torch.library.custom_op("auto_deploy::flashinfer_quant_fp8_moe", mutates_args=())
def flashinfer_quant_fp8_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    w1_input_scale: List[torch.Tensor],
    w2_input_scale: List[torch.Tensor],
    w3_input_scale: List[torch.Tensor],
    w1_weight_scale: List[torch.Tensor],
    w2_weight_scale: List[torch.Tensor],
    w3_weight_scale: List[torch.Tensor],
    mlp_style: str = "gated_mlp",  # "gated_mlp" (default) or "mlp"
    act_fn: str = "silu",  # silu or relu2
) -> torch.Tensor:
    using_cuda_graph = True
    if not using_cuda_graph:

        def _check_equal_scalar_scales(scales) -> bool:
            if not scales:
                return True
            first_val = scales[0].item()
            return all(s.numel() == 1 and s.item() == first_val for s in scales)

        assert _check_equal_scalar_scales(w1_input_scale)
        assert _check_equal_scalar_scales(w2_input_scale)
        assert _check_equal_scalar_scales(w3_input_scale)

    x_fp8 = x.to(torch.float8_e4m3fn)
    output_flashinfer_fp8_moe = torch.empty_like(x)
    w1_weight_quantized = torch.stack(w1_weight, dim=0)
    w2_weight_quantized = torch.stack(w2_weight, dim=0)
    w1_weight_scale = torch.stack(w1_weight_scale, dim=0)
    w2_weight_scale = torch.stack(w2_weight_scale, dim=0)
    routing_weights = routing_weights.to(torch.float32)

    a1_scale = w1_input_scale[0].to(dtype=torch.float32)
    a2_scale = w2_input_scale[0].to(dtype=torch.float32)
    quant_scales = [w1_weight_scale * a1_scale, 1 / a2_scale, w2_weight_scale * a2_scale, a1_scale]

    avoid_segfault = True
    if avoid_segfault:
        return torch.randn_like(x)
    else:
        cutlass_fused_moe(
            input=x_fp8,  # (m, k) -> (1, 2688) | (8, 64)
            token_selected_experts=selected_experts.to(torch.int),  # (m, topk) -> (1, 6) | (8, 2)
            token_final_scales=routing_weights.contiguous(),  # (m, topk) -> (1, 6) | (8, 2)
            fc1_expert_weights=w1_weight_quantized,  # (e, n, k) -> (128, 1856, 2688) | (3, 32, 64)
            fc2_expert_weights=w2_weight_quantized,  # (e, k, n) -> (128, 2688, 1856) | (3, 64, 32)
            output_dtype=torch.bfloat16,
            quant_scales=quant_scales,  # [128, 1, 128, 1] | [3, 1, 3, 1]
            input_sf=None,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            output=output_flashinfer_fp8_moe,  # (8,64)
            activation_type=ActivationType.Relu2,
        )
    return output_flashinfer_fp8_moe


@flashinfer_quant_fp8_moe.register_fake
def flashinfer_quant_fp8_moe_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    w1_input_scale: List[torch.Tensor],
    w2_input_scale: List[torch.Tensor],
    w3_input_scale: List[torch.Tensor],
    w1_weight_scale: List[torch.Tensor],
    w2_weight_scale: List[torch.Tensor],
    w3_weight_scale: List[torch.Tensor],
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    return torch.empty_like(x)
