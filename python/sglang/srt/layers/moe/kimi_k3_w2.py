# SPDX-License-Identifier: Apache-2.0
"""Reference and CUDA entry point for Kimi-K3 W2 routed experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.kimi_k3_w2a16 import (
    dequantize_w2,
    unpack_w4,
)


def _apply_promoted_row_correction(base, packed, scales, indices, group_size):
    if indices.shape[-1] == 0:
        return base
    promoted = unpack_w4(packed).to(scales.dtype)
    promoted *= scales.repeat_interleave(group_size, dim=-1)
    base[indices.long()] += promoted
    return base


def kimi_k3_w2_moe(
    x,
    topk_ids,
    topk_weights,
    w13_qweight,
    w13_scales,
    w2_qweight,
    w2_scales,
    w13_w4_qweight,
    w13_w4_scales,
    w13_w4_indices,
    w2_w4_qweight,
    w2_w4_scales,
    w2_w4_indices,
    group_size,
    beta=4.0,
    linear_beta=25.0,
    layer_idx=None,
):
    # CUDA is isolated behind this import so checkpoint/unit tooling has no Triton dependency.
    if x.is_cuda:
        from sglang.srt.layers.moe.kimi_k3_w2_triton import kimi_k3_w2_moe_cuda

        return kimi_k3_w2_moe_cuda(
            x,
            topk_ids,
            topk_weights,
            w13_qweight,
            w13_scales,
            w2_qweight,
            w2_scales,
            w13_w4_qweight,
            w13_w4_scales,
            w13_w4_indices,
            w2_w4_qweight,
            w2_w4_scales,
            w2_w4_indices,
            group_size,
            beta,
            linear_beta,
            layer_idx,
        )

    out = torch.zeros_like(x)
    for expert in torch.unique(topk_ids).tolist():
        token, slot = torch.where(topk_ids == expert)
        if token.numel() == 0:
            continue
        w13 = dequantize_w2(w13_qweight[expert], w13_scales[expert], group_size)
        w13 = _apply_promoted_row_correction(
            w13,
            w13_w4_qweight[expert],
            w13_w4_scales[expert],
            w13_w4_indices[expert],
            group_size,
        )
        gate_up = F.linear(x[token], w13)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = beta * torch.tanh(gate.float() / beta) * torch.sigmoid(gate.float())
        if linear_beta is not None:
            up = linear_beta * torch.tanh(up.float() / linear_beta)
        hidden = (gate * up).to(x.dtype)
        w2 = dequantize_w2(w2_qweight[expert], w2_scales[expert], group_size)
        w2 = _apply_promoted_row_correction(
            w2,
            w2_w4_qweight[expert],
            w2_w4_scales[expert],
            w2_w4_indices[expert],
            group_size,
        )
        contribution = F.linear(hidden, w2)
        out.index_add_(
            0, token, contribution * topk_weights[token, slot, None].to(x.dtype)
        )
    return out
