import pytest
import torch
from _graph_test_helpers import FakeFactory, run_test_transformed_gm
from _model_test_utils import MoEOpModel

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

TEST_CASES = [
    ("NVFP4", torch.ops.auto_deploy.trtllm_quant_nvfp4_moe_fused, "trtllm"),
    ("NVFP4", torch.ops.auto_deploy.torch_quant_nvfp4_moe, "torch"),
    ("FP8", torch.ops.auto_deploy.torch_quant_fp8_moe, "torch"),
    # ("FP8", torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused, "trtllm"),
]

MLP_STYLES = [("gated_mlp", "silu"), ("mlp", "relu2")]


@pytest.mark.parametrize("quant_algo, expected_op, backend", TEST_CASES)
@pytest.mark.parametrize("mlp_style, act_fn", MLP_STYLES)
def test_quantize_moe_transformation(
    quant_algo: str,
    expected_op: torch.ops.OpOverloadPacket,
    backend: str,
    mlp_style: str,
    act_fn: str,
):
    device = "cuda"
    hidden_size = 128
    intermediate_size = 256
    num_experts = 4
    top_k = 2

    torch.manual_seed(42)

    model = MoEOpModel(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        mlp_style=mlp_style,
        act_fn=act_fn,
    ).to(device=device, dtype=torch.bfloat16)

    x = model.get_input(device=device, dtype=torch.bfloat16) * 0.01

    def _check_transformed_graph(gm):
        return any(is_op(n, expected_op) for n in gm.graph.nodes)

    def _expected_num_params(n):
        """
        Return expected parameter count after quantization.
        For FP4, weights are quantized to half-size (simulate 4-bit).
        """
        # gate: Linear(hidden_size, num_experts)
        gate_params = (hidden_size + 1) * num_experts  # with bias

        if quant_algo == "NVFP4":
            num_weights = 3 if mlp_style == "gated_mlp" else 2
            expert_params = num_experts * num_weights * hidden_size * intermediate_size // 2
            # 3 weights per expert, of shape [hidden_size, intermediate_size] or
            # [intermediate_size, hidden_size], shape will be halved to store quantized uint8 weight
            return gate_params + expert_params
        else:
            return n

    quant_config = {"quant_algo": quant_algo, "backend": backend}

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        FakeFactory(quant_config=quant_config),
        {
            "quantize_fp8_moe": {
                "stage": "pattern_matcher",
            },
            "quantize_nvfp4_moe": {
                "stage": "pattern_matcher",
                "backend": backend,
            },
        },
    )(None, gm)

    run_test_transformed_gm(
        model=model,
        x=x,
        gm_transformed=gm_transformed,
        check_transformed_graph=_check_transformed_graph,
        _get_expected_num_params=_expected_num_params,
        atol=1e-5,
        rtol=1e-5,
        test_load_hook=False,
        strict_loading=False,
    )
