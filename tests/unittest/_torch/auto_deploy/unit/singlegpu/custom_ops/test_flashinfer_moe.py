# from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.flashinfer_moe import flashinfer_quant_fp8_moe
# Each one of these includes causes a segmentation fault i nthe culass fp8 moe kernel
# import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
# from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
# from tensorrt_llm._torch.modules.fused_moe import MoE  # noqa: F401
from typing import List

import pytest
import torch
from _torch.helpers import reference_moe_torch
from _torch_test_utils import fp8_compatible
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

avoid_segfault = False
if not avoid_segfault:
    from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.flashinfer_moe import (
        flashinfer_quant_fp8_moe,
    )
else:

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
        def _check_equal_scalar_scales(scales):
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
        quant_scales = [
            w1_weight_scale * a1_scale,
            1 / a2_scale,
            w2_weight_scale * a2_scale,
            a1_scale,
        ]

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


from test_ad_moe_op import setup_moe_test


def input_to_float8(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values "
    "with tensor-wise quantization."""
    dtype = torch.float8_e4m3fn
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def quant_fp8_per_tensor_batches(a):
    num_batches = a.size(0)
    a_quant = []
    a_scales = []

    for i in range(num_batches):
        a_fp8, a_global_sf = input_to_float8(a[i])
        a_global_sf = 1.0 / a_global_sf
        a_quant.append(a_fp8)
        a_scales.append(a_global_sf)

    result_a_quant = torch.stack(a_quant)
    result_a_scales = torch.stack(a_scales)

    return result_a_quant, result_a_scales


# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
# def test_flashinfer_fp8_moe_op_run(dtype):
#     num_experts = 128 # 2
#     (
#         x,
#         selected_experts,
#         final_scales,
#         w1_weight,
#         w2_weight,
#         w3_weight,
#         weights,
#         fused_w3_w1_stacked_weight,
#         fused_w2_weight,
#     ) = setup_moe_test(dtype, num_experts)

#     w1_input_scale, w2_input_scale, w3_input_scale = [], [], []
#     w1_weight_scale, w2_weight_scale, w3_weight_scale = [], [], []
#     for i in range(num_experts):
#         inp_scale_val = torch.tensor(1.0).float().cuda()
#         wt_scale_factor = 448 if dtype == torch.bfloat16 else 432  # float16 overflow with 448
#         wt_scale_val = (torch.max(torch.abs(w1_weight[i])) / wt_scale_factor).float().to("cuda")
#         w1_input_scale.append(inp_scale_val)
#         w2_input_scale.append(inp_scale_val)
#         w3_input_scale.append(inp_scale_val)
#         w1_weight_scale.append(wt_scale_val)
#         w2_weight_scale.append(wt_scale_val)
#         w3_weight_scale.append(wt_scale_val)
#         # Cast the expert weight tensors and fused weights to FP8.
#         w1_weight[i] = (w1_weight[i] / w1_weight_scale[i]).to(torch.float8_e4m3fn)
#         w2_weight[i] = (w2_weight[i] / w2_weight_scale[i]).to(torch.float8_e4m3fn)
#         w3_weight[i] = (w3_weight[i] / w3_weight_scale[i]).to(torch.float8_e4m3fn)
#         fused_w3_w1_stacked_weight[i] = (fused_w3_w1_stacked_weight[i] / w1_weight_scale[i]).to(
#             torch.float8_e4m3fn
#         )
#         fused_w2_weight[i] = (fused_w2_weight[i] / w2_weight_scale[i]).to(torch.float8_e4m3fn)

#     with torch.inference_mode():
#         ref_output = reference_moe_torch(
#             x, selected_experts, final_scales, num_experts, weights, mlp_style="mlp", act_fn="relu2")

#     output_flashinfer_fp8_moe = torch.empty_like(x)
#     w1_weight_quantized = torch.stack(w1_weight, dim=0)
#     w2_weight_quantized = torch.stack(w2_weight, dim=0)
#     w1_weight_scale = torch.stack(w1_weight_scale, dim=0)
#     w2_weight_scale = torch.stack(w2_weight_scale, dim=0)
#     #scaled_hidden_states, a1_scale = input_to_float8(x)
#     #assert a1_scale == w1_input_scale
#     #a1_scale = torch.stack(w1_input_scale, dim=0).contiguous()    #assert a1_scale == w1_input_scale
#     a1_scale = torch.tensor(1.0).to(device="cuda").to(dtype=torch.float32)
#     a2_scale = torch.scalar_tensor(1.0).to(device="cuda").to(dtype=torch.float32)
#     scaled_hidden_states = x.to(torch.float8_e4m3fn)

#     print("a1_scale")
#     # w1_weight_quantized, w1_weight_scale = quant_fp8_per_tensor_batches(w1_weight)
#     # w2_weight_quantized, w2_weight_scale = quant_fp8_per_tensor_batches(w2_weight)
#     quant_scales = [
#         w1_weight_scale * a1_scale,
#         1 / a2_scale,
#         w2_weight_scale * a2_scale,
#         a1_scale]
#     final_scales = final_scales.to(torch.float32)
#     with torch.inference_mode():
#         cutlass_fused_moe(
#             input=scaled_hidden_states.contiguous(), # (m, k) -> (1, 2688) | (8, 64)
#             token_selected_experts=selected_experts.to(torch.int), # (m, topk) -> (1, 6) | (8, 2)
#             token_final_scales=final_scales.contiguous(), # (m, topk) -> (1, 6) | (8, 2)
#             fc1_expert_weights=w1_weight_quantized, # (e, n, k) -> (128, 1856, 2688) | (3, 32, 64)
#             fc2_expert_weights=w2_weight_quantized, # (e, k, n) -> (128, 2688, 1856) | (3, 64, 32)
#             output_dtype=torch.bfloat16,
#             quant_scales=quant_scales, # [128, 1, 128, 1] | [3, 1, 3, 1]
#             input_sf=None,
#             tp_size=1,
#             tp_rank=0,
#             ep_size=1,
#             ep_rank=0,
#             output=output_flashinfer_fp8_moe, # (8,64)
#             activation_type=ActivationType.Relu2
#         )

#     torch.cuda.synchronize()
#     rtol = 0.5 if dtype == torch.bfloat16 else 1.5
#     atol = 0.8 if dtype == torch.bfloat16 else 1
#     #torch.testing.assert_close(output_torch_fp8_moe, output_torch_moe, rtol=rtol, atol=atol)
#     #torch.testing.assert_close(output_torch_fp8_moe, ref_output, rtol=rtol, atol=atol)
#     print("output_flashinfer_fp8_moe")
#     print(output_flashinfer_fp8_moe)
#     print("ref_output")
#     print(ref_output)
#     torch.testing.assert_close(output_flashinfer_fp8_moe, ref_output, rtol=0.1, atol=0.1)


# Todo: check if scales are reciprocal
# Todo: check if the scales are per tensor


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_flashinfer_fp8_moe_op_run(dtype):
    num_experts = 128  # 2
    (
        x,
        selected_experts,
        final_scales,
        w1_weight,
        w2_weight,
        w3_weight,
        weights,
        fused_w3_w1_stacked_weight,
        fused_w2_weight,
    ) = setup_moe_test(dtype, num_experts)

    w1_input_scale, w2_input_scale, w3_input_scale = [], [], []
    w1_weight_scale, w2_weight_scale, w3_weight_scale = [], [], []
    for i in range(num_experts):
        inp_scale_val = torch.tensor(1.0).float().cuda()
        wt_scale_factor = 448 if dtype == torch.bfloat16 else 432  # float16 overflow with 448
        wt_scale_val = (torch.max(torch.abs(w1_weight[i])) / wt_scale_factor).float().to("cuda")
        w1_input_scale.append(inp_scale_val)
        w2_input_scale.append(inp_scale_val)
        w3_input_scale.append(inp_scale_val)
        w1_weight_scale.append(wt_scale_val)
        w2_weight_scale.append(wt_scale_val)
        w3_weight_scale.append(wt_scale_val)
        # Cast the expert weight tensors and fused weights to FP8.
        w1_weight[i] = (w1_weight[i] / w1_weight_scale[i]).to(torch.float8_e4m3fn)
        w2_weight[i] = (w2_weight[i] / w2_weight_scale[i]).to(torch.float8_e4m3fn)
        w3_weight[i] = (w3_weight[i] / w3_weight_scale[i]).to(torch.float8_e4m3fn)
        fused_w3_w1_stacked_weight[i] = (fused_w3_w1_stacked_weight[i] / w1_weight_scale[i]).to(
            torch.float8_e4m3fn
        )
        fused_w2_weight[i] = (fused_w2_weight[i] / w2_weight_scale[i]).to(torch.float8_e4m3fn)

    with torch.inference_mode():
        ref_output = reference_moe_torch(
            x, selected_experts, final_scales, num_experts, weights, mlp_style="mlp", act_fn="relu2"
        )

    # final_scales = final_scales.to(torch.float32)
    with torch.inference_mode():
        output_flashinfer_fp8_moe = torch.ops.auto_deploy.flashinfer_quant_fp8_moe(
            x,
            selected_experts,
            final_scales,
            w1_weight,
            w2_weight,
            w3_weight,
            w1_input_scale,
            w2_input_scale,
            w3_input_scale,
            w1_weight_scale,
            w2_weight_scale,
            w3_weight_scale,
        )

    torch.cuda.synchronize()
    rtol = 0.5 if dtype == torch.bfloat16 else 1.5
    atol = 0.8 if dtype == torch.bfloat16 else 1
    # torch.testing.assert_close(output_torch_fp8_moe, output_torch_moe, rtol=rtol, atol=atol)
    # torch.testing.assert_close(output_torch_fp8_moe, ref_output, rtol=rtol, atol=atol)
    print("output_flashinfer_fp8_moe")
    print(output_flashinfer_fp8_moe)
    print("ref_output")
    print(ref_output)
    torch.testing.assert_close(output_flashinfer_fp8_moe, ref_output, rtol=0.1, atol=0.1)
