# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 routed-expert W2A16 quantization.

The checkpoint keeps every non-routed-expert tensor in its original dtype.  Expert
weights use four signed two-bit values per byte and one symmetric scale per input
group.  The CUDA implementation dequantizes inside the MoE GEMM; the torch path is
kept as a correctness oracle and for CPU unit tests.
"""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import set_weight_attrs

W2_LEVELS = (-2, -1, 0, 1)


def pack_w2(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed codes in ``[-2, 1]`` along the last dimension."""
    if codes.shape[-1] % 4:
        raise ValueError("W2 input dimension must be divisible by four")
    if torch.any((codes < -2) | (codes > 1)):
        raise ValueError("W2 codes must be in [-2, 1]")
    u = (codes.to(torch.int16) + 2).to(torch.uint8).reshape(*codes.shape[:-1], -1, 4)
    return u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)


def unpack_w2(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.tensor((0, 2, 4, 6), dtype=torch.uint8, device=packed.device)
    values = ((packed.unsqueeze(-1) >> shifts) & 3).to(torch.int8) - 2
    return values.flatten(-2)


def pack_w4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed codes in ``[-8, 7]`` along the last dimension."""
    if codes.shape[-1] % 2:
        raise ValueError("W4 input dimension must be divisible by two")
    if torch.any((codes < -8) | (codes > 7)):
        raise ValueError("W4 codes must be in [-8, 7]")
    u = (codes.to(torch.int16) + 8).to(torch.uint8).reshape(*codes.shape[:-1], -1, 2)
    return u[..., 0] | (u[..., 1] << 4)


def unpack_w4(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.tensor((0, 4), dtype=torch.uint8, device=packed.device)
    values = ((packed.unsqueeze(-1) >> shifts) & 15).to(torch.int8) - 8
    return values.flatten(-2)


def dequantize_w2(
    packed: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    codes = unpack_w2(packed).to(scales.dtype)
    if codes.shape[-1] != scales.shape[-1] * group_size:
        raise ValueError("scale shape does not match packed W2 input dimension")
    return codes * scales.repeat_interleave(group_size, dim=-1)


class KimiK3W2A16Config(QuantizationConfig):
    def __init__(
        self, group_size: int = 128, w4_channel_fraction: float = 0.05
    ) -> None:
        super().__init__()
        if group_size < 32 or group_size % 32:
            raise ValueError("group_size must be a multiple of 32")
        self.group_size = group_size
        if not 0.0 <= w4_channel_fraction < 1.0:
            raise ValueError("w4_channel_fraction must be in [0, 1)")
        self.w4_channel_fraction = w4_channel_fraction

    @classmethod
    def get_name(cls) -> str:
        return "kimi_k3_w2a16"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def get_scaled_act_names(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KimiK3W2A16Config:
        bits = int(config.get("bits", 2))
        if bits != 2:
            raise ValueError(f"kimi_k3_w2a16 requires bits=2, got {bits}")
        return cls(
            group_size=int(config.get("group_size", 128)),
            w4_channel_fraction=float(config.get("w4_channel_fraction", 0.05)),
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        if hf_quant_cfg and hf_quant_cfg.get("quant_method") == cls.get_name():
            return cls.get_name()
        return cls.get_name() if user_quant == cls.get_name() else None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if isinstance(layer, FusedMoE) and ".experts" in prefix:
            return KimiK3W2A16MoEMethod(self)
        # K3 checkpoint quantizes routed experts only.
        return UnquantizedLinearMethod()


class KimiK3W2A16MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: KimiK3W2A16Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        group = self.quant_config.group_size
        if hidden_size % group or intermediate_size_per_partition % group:
            raise ValueError("K3 W2 dimensions must be divisible by group_size")
        layer.group_size = group
        layer.w4_channel_fraction = self.quant_config.w4_channel_fraction
        promoted_per_projection = round(
            intermediate_size_per_partition * layer.w4_channel_fraction
        )
        promoted_w13 = 2 * promoted_per_projection
        promoted_w2 = round(hidden_size * layer.w4_channel_fraction)
        attrs = dict(extra_weight_attrs)
        attrs["weight_loader"] = self._get_weight_loader(layer)
        attrs.update({"quant_method": "group", "is_transposed": False})
        for name, shape, dtype in (
            (
                "w13_qweight",
                (num_experts, 2 * intermediate_size_per_partition, hidden_size // 4),
                torch.uint8,
            ),
            (
                "w2_qweight",
                (num_experts, hidden_size, intermediate_size_per_partition // 4),
                torch.uint8,
            ),
            (
                "w13_scales",
                (
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // group,
                ),
                params_dtype,
            ),
            (
                "w2_scales",
                (num_experts, hidden_size, intermediate_size_per_partition // group),
                params_dtype,
            ),
            (
                "w13_w4_qweight",
                (num_experts, promoted_w13, hidden_size // 2),
                torch.uint8,
            ),
            (
                "w2_w4_qweight",
                (num_experts, promoted_w2, intermediate_size_per_partition // 2),
                torch.uint8,
            ),
            (
                "w13_w4_scales",
                (num_experts, promoted_w13, hidden_size // group),
                params_dtype,
            ),
            (
                "w2_w4_scales",
                (num_experts, promoted_w2, intermediate_size_per_partition // group),
                params_dtype,
            ),
            ("w13_w4_indices", (num_experts, promoted_w13), torch.int32),
            ("w2_w4_indices", (num_experts, promoted_w2), torch.int32),
        ):
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, attrs)

    @staticmethod
    def _get_weight_loader(layer):
        def load(param, loaded_weight, weight_name, shard_id, expert_id):
            tp_rank = get_parallel().moe_tp_rank
            tp_size = layer.moe_tp_size
            target = param.data[expert_id]
            if shard_id in {"w1", "w3"}:
                if loaded_weight.shape[0] % tp_size:
                    raise ValueError(f"{weight_name} rows do not divide across TP")
                rows = loaded_weight.shape[0] // tp_size
                local = loaded_weight.narrow(0, tp_rank * rows, rows)
                if "w4_indices" in weight_name:
                    local = local - layer.intermediate_size_per_partition * tp_rank
                start = 0 if shard_id == "w1" else target.shape[0] // 2
                target[start : start + rows].copy_(
                    local.to(param.device, dtype=param.dtype)
                )
            else:
                if "w4_indices" in weight_name:
                    local = loaded_weight
                else:
                    width = target.shape[-1]
                    if loaded_weight.shape[-1] != width * tp_size:
                        raise ValueError(
                            f"{weight_name} width does not match TP layout"
                        )
                    local = loaded_weight.narrow(-1, tp_rank * width, width)
                target.copy_(local.to(param.device, dtype=param.dtype))

        return load

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config

    def get_triton_quant_info(self, layer):
        raise NotImplementedError("W2 uses the dedicated K3 runner")

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.kimi_k3_w2 import kimi_k3_w2_moe
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        x, _x_scale, topk_output = dispatch_output
        topk_weights, topk_ids, _ = topk_output
        out = kimi_k3_w2_moe(
            x,
            topk_ids,
            topk_weights,
            layer.w13_qweight,
            layer.w13_scales,
            layer.w2_qweight,
            layer.w2_scales,
            layer.w13_w4_qweight,
            layer.w13_w4_scales,
            layer.w13_w4_indices,
            layer.w2_w4_qweight,
            layer.w2_w4_scales,
            layer.w2_w4_indices,
            layer.group_size,
            beta=self.moe_runner_config.gemm1_alpha or 4.0,
            linear_beta=self.moe_runner_config.gemm1_clamp_limit,
            layer_idx=self.moe_runner_config.layer_id,
        )
        return StandardCombineInput(hidden_states=out)
