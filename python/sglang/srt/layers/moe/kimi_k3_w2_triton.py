# SPDX-License-Identifier: Apache-2.0
"""Triton W2-dequantize-and-MoE-GEMM kernels for NVIDIA SM100/B300."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _w2_gate_up_kernel(
    x,
    expert_ids,
    qweight,
    scales,
    output,
    x_stride_m: tl.constexpr,
    q_stride_e: tl.constexpr,
    q_stride_n: tl.constexpr,
    s_stride_e: tl.constexpr,
    s_stride_n: tl.constexpr,
    o_stride_m: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    block_n = tl.program_id(1)
    token = route // TOPK
    expert = tl.load(expert_ids + route).to(tl.int64)
    n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(x + token * x_stride_m + k, mask=k < K, other=0.0)
        packed_k = k // 4
        byte = tl.load(
            qweight + expert * q_stride_e + n[:, None] * q_stride_n + packed_k[None, :],
            mask=(n[:, None] < N) & (k[None, :] < K),
            other=0,
        )
        shift = (k[None, :] & 3) * 2
        code = ((byte >> shift) & 3).to(tl.float32) - 2.0
        scale = tl.load(
            scales
            + expert * s_stride_e
            + n[:, None] * s_stride_n
            + (k[None, :] // GROUP),
            mask=(n[:, None] < N) & (k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(code * scale * a[None, :], axis=1)
    tl.store(output + route * o_stride_m + n, acc, mask=n < N)


@triton.jit
def _situ_w2_down_kernel(
    gate_up,
    expert_ids,
    qweight,
    scales,
    output,
    gu_stride_m: tl.constexpr,
    q_stride_e: tl.constexpr,
    q_stride_n: tl.constexpr,
    s_stride_e: tl.constexpr,
    s_stride_n: tl.constexpr,
    o_stride_m: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    block_n = tl.program_id(1)
    expert = tl.load(expert_ids + route).to(tl.int64)
    n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        gate = tl.load(gate_up + route * gu_stride_m + k, mask=k < K, other=0.0).to(
            tl.float32
        )
        up = tl.load(gate_up + route * gu_stride_m + K + k, mask=k < K, other=0.0).to(
            tl.float32
        )
        gate = BETA * libdevice.tanh(gate / BETA) * tl.sigmoid(gate)
        if LINEAR_BETA > 0.0:
            up = LINEAR_BETA * libdevice.tanh(up / LINEAR_BETA)
        a = gate * up
        packed_k = k // 4
        byte = tl.load(
            qweight + expert * q_stride_e + n[:, None] * q_stride_n + packed_k[None, :],
            mask=(n[:, None] < N) & (k[None, :] < K),
            other=0,
        )
        shift = (k[None, :] & 3) * 2
        code = ((byte >> shift) & 3).to(tl.float32) - 2.0
        scale = tl.load(
            scales
            + expert * s_stride_e
            + n[:, None] * s_stride_n
            + (k[None, :] // GROUP),
            mask=(n[:, None] < N) & (k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(code * scale * a[None, :], axis=1)
    tl.store(output + route * o_stride_m + n, acc, mask=n < N)


@triton.jit
def _w4_row_correction_kernel(
    x,
    expert_ids,
    qweight,
    scales,
    row_indices,
    output,
    x_stride_m: tl.constexpr,
    q_stride_e: tl.constexpr,
    q_stride_n: tl.constexpr,
    s_stride_e: tl.constexpr,
    s_stride_n: tl.constexpr,
    i_stride_e: tl.constexpr,
    o_stride_m: tl.constexpr,
    K: tl.constexpr,
    P: tl.constexpr,
    GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    block_n = tl.program_id(1)
    token = route // TOPK
    expert = tl.load(expert_ids + route).to(tl.int64)
    p = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_row = tl.load(row_indices + expert * i_stride_e + p, mask=p < P, other=0).to(
        tl.int64
    )
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(x + token * x_stride_m + k, mask=k < K, other=0.0)
        byte = tl.load(
            qweight + expert * q_stride_e + p[:, None] * q_stride_n + (k[None, :] // 2),
            mask=(p[:, None] < P) & (k[None, :] < K),
            other=0,
        )
        code = ((byte >> ((k[None, :] & 1) * 4)) & 15).to(tl.float32) - 8.0
        scale = tl.load(
            scales
            + expert * s_stride_e
            + p[:, None] * s_stride_n
            + (k[None, :] // GROUP),
            mask=(p[:, None] < P) & (k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(code * scale * a[None, :], axis=1)
    ptr = output + route * o_stride_m + output_row
    previous = tl.load(ptr, mask=p < P, other=0.0)
    tl.store(ptr, previous + acc, mask=p < P)


@triton.jit
def _situ_w4_row_correction_kernel(
    gate_up,
    expert_ids,
    qweight,
    scales,
    row_indices,
    output,
    gu_stride_m: tl.constexpr,
    q_stride_e: tl.constexpr,
    q_stride_n: tl.constexpr,
    s_stride_e: tl.constexpr,
    s_stride_n: tl.constexpr,
    i_stride_e: tl.constexpr,
    o_stride_m: tl.constexpr,
    K: tl.constexpr,
    P: tl.constexpr,
    GROUP: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    block_n = tl.program_id(1)
    expert = tl.load(expert_ids + route).to(tl.int64)
    p = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_row = tl.load(row_indices + expert * i_stride_e + p, mask=p < P, other=0).to(
        tl.int64
    )
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        gate = tl.load(gate_up + route * gu_stride_m + k, mask=k < K, other=0.0).to(
            tl.float32
        )
        up = tl.load(gate_up + route * gu_stride_m + K + k, mask=k < K, other=0.0).to(
            tl.float32
        )
        gate = BETA * libdevice.tanh(gate / BETA) * tl.sigmoid(gate)
        if LINEAR_BETA > 0.0:
            up = LINEAR_BETA * libdevice.tanh(up / LINEAR_BETA)
        a = gate * up
        byte = tl.load(
            qweight + expert * q_stride_e + p[:, None] * q_stride_n + (k[None, :] // 2),
            mask=(p[:, None] < P) & (k[None, :] < K),
            other=0,
        )
        code = ((byte >> ((k[None, :] & 1) * 4)) & 15).to(tl.float32) - 8.0
        scale = tl.load(
            scales
            + expert * s_stride_e
            + p[:, None] * s_stride_n
            + (k[None, :] // GROUP),
            mask=(p[:, None] < P) & (k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(code * scale * a[None, :], axis=1)
    ptr = output + route * o_stride_m + output_row
    previous = tl.load(ptr, mask=p < P, other=0.0)
    tl.store(ptr, previous + acc, mask=p < P)


def kimi_k3_w2_moe_cuda(
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
):
    major, _minor = torch.cuda.get_device_capability(x.device)
    if major < 10:
        raise RuntimeError("kimi_k3_w2a16 requires SM100 or newer (B300/B200)")
    if not (x.is_contiguous() and topk_ids.is_contiguous()):
        x, topk_ids = x.contiguous(), topk_ids.contiguous()
    tokens, hidden = x.shape
    topk = topk_ids.shape[1]
    routes = tokens * topk
    intermediate = w2_qweight.shape[-1] * 4
    gate_up = torch.empty((routes, 2 * intermediate), dtype=x.dtype, device=x.device)
    block_n, block_k = 32, 32
    _w2_gate_up_kernel[(routes, triton.cdiv(2 * intermediate, block_n))](
        x,
        topk_ids,
        w13_qweight,
        w13_scales,
        gate_up,
        x.stride(0),
        w13_qweight.stride(0),
        w13_qweight.stride(1),
        w13_scales.stride(0),
        w13_scales.stride(1),
        gate_up.stride(0),
        hidden,
        2 * intermediate,
        group_size,
        topk,
        block_n,
        block_k,
        num_warps=4,
        num_stages=3,
    )
    promoted_w13 = w13_w4_indices.shape[1]
    if promoted_w13:
        _w4_row_correction_kernel[(routes, triton.cdiv(promoted_w13, block_n))](
            x,
            topk_ids,
            w13_w4_qweight,
            w13_w4_scales,
            w13_w4_indices,
            gate_up,
            x.stride(0),
            w13_w4_qweight.stride(0),
            w13_w4_qweight.stride(1),
            w13_w4_scales.stride(0),
            w13_w4_scales.stride(1),
            w13_w4_indices.stride(0),
            gate_up.stride(0),
            hidden,
            promoted_w13,
            group_size,
            topk,
            block_n,
            block_k,
            num_warps=4,
            num_stages=3,
        )
    if layer_idx is not None:
        from sglang.srt.debug_utils.k3_tensor_capture import k3_capture

        k3_capture("moe_w13_input", layer_idx, {"hidden_states": x})
        k3_capture(
            "moe_w2_input",
            layer_idx,
            {"gate_up": gate_up},
            metadata={"beta": beta, "linear_beta": linear_beta},
        )
    route_out = torch.empty((routes, hidden), dtype=x.dtype, device=x.device)
    _situ_w2_down_kernel[(routes, triton.cdiv(hidden, block_n))](
        gate_up,
        topk_ids,
        w2_qweight,
        w2_scales,
        route_out,
        gate_up.stride(0),
        w2_qweight.stride(0),
        w2_qweight.stride(1),
        w2_scales.stride(0),
        w2_scales.stride(1),
        route_out.stride(0),
        intermediate,
        hidden,
        group_size,
        float(beta),
        -1.0 if linear_beta is None else float(linear_beta),
        block_n,
        block_k,
        num_warps=4,
        num_stages=3,
    )
    promoted_w2 = w2_w4_indices.shape[1]
    if promoted_w2:
        _situ_w4_row_correction_kernel[(routes, triton.cdiv(promoted_w2, block_n))](
            gate_up,
            topk_ids,
            w2_w4_qweight,
            w2_w4_scales,
            w2_w4_indices,
            route_out,
            gate_up.stride(0),
            w2_w4_qweight.stride(0),
            w2_w4_qweight.stride(1),
            w2_w4_scales.stride(0),
            w2_w4_scales.stride(1),
            w2_w4_indices.stride(0),
            route_out.stride(0),
            intermediate,
            promoted_w2,
            group_size,
            float(beta),
            -1.0 if linear_beta is None else float(linear_beta),
            block_n,
            block_k,
            num_warps=4,
            num_stages=3,
        )
    weighted = route_out.view(tokens, topk, hidden) * topk_weights[..., None].to(
        x.dtype
    )
    return weighted.sum(dim=1)
