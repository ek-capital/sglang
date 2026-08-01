#!/usr/bin/env python3
"""Stream Kimi-K3 MXFP4 routed experts into importance-calibrated W2 safetensors.

Run this on a machine with enough storage for the source and destination snapshots.
Every non-expert tensor is copied unchanged.  Source shards are processed and released
one at a time, and the output index is written only after all shards validate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from sglang.srt.layers.quantization.kimi_k3_w2a16 import (
    dequantize_w2,
    pack_w2,
    pack_w4,
)

EXPERT_RE = re.compile(r"^(.*\.experts\.\d+\.w[123])\.weight_packed$")
_IMPORTANCE_CACHE: dict[Path, dict[str, torch.Tensor]] = {}


def decode_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Decode two OCP E2M1 values per byte, preserving checkpoint order."""
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    return lut[nibbles.long()]


def decode_e8m0(scales: torch.Tensor) -> torch.Tensor:
    # OCP E8M0 encodes exact powers of two with bias 127; 255 is NaN.
    if torch.any(scales == 255):
        raise ValueError("MXFP4 scale contains reserved E8M0 NaN encoding")
    return torch.pow(2.0, scales.to(torch.float32) - 127.0)


def dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = decode_e2m1(packed)
    if values.shape[-1] != scales.shape[-1] * 32:
        raise ValueError(
            f"bad MXFP4 shapes: {tuple(packed.shape)}, {tuple(scales.shape)}"
        )
    return values * decode_e8m0(scales).repeat_interleave(32, dim=-1)


def quantize_importance_w2(
    weight: torch.Tensor, importance: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted least-squares W2 with diagonal input Hessian importance."""
    if weight.shape[-1] % group_size:
        raise ValueError("weight input dimension is not divisible by group_size")
    if importance.numel() != weight.shape[-1]:
        raise ValueError("importance vector has the wrong width")
    w = weight.float().reshape(weight.shape[0], -1, group_size)
    h = importance.float().clamp_min(1e-8).reshape(1, -1, group_size)
    # Alternating nearest-level assignment and weighted LS scale update.
    scale = (w.abs().amax(-1, keepdim=True) / 2).clamp_min(1e-8)
    for _ in range(3):
        q = torch.round(w / scale).clamp(-2, 1)
        scale = (
            (
                (h * w * q).sum(-1, keepdim=True)
                / (h * q.square()).sum(-1, keepdim=True).clamp_min(1e-12)
            )
            .abs()
            .clamp_min(1e-8)
        )
    q = torch.round(w / scale).clamp(-2, 1).to(torch.int8).reshape_as(weight)
    return pack_w2(q), scale.squeeze(-1).to(torch.bfloat16)


def quantize_residual_w4(
    residual: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = residual.float().reshape(residual.shape[0], -1, group_size)
    scale = (grouped.abs().amax(-1, keepdim=True) / 8).clamp_min(1e-8)
    for _ in range(2):
        q = torch.round(grouped / scale).clamp(-8, 7)
        scale = (
            (
                (grouped * q).sum(-1, keepdim=True)
                / q.square().sum(-1, keepdim=True).clamp_min(1e-12)
            )
            .abs()
            .clamp_min(1e-8)
        )
    q = torch.round(grouped / scale).clamp(-8, 7).to(torch.int8)
    return pack_w4(q.reshape_as(residual)), scale.squeeze(-1).to(torch.bfloat16)


def quantize_mixed_w2_w4(
    weight: torch.Tensor,
    importance: torch.Tensor,
    group_size: int,
    w4_fraction: float,
    tp_stripes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qweight, scales = quantize_importance_w2(weight, importance, group_size)
    reconstructed = dequantize_w2(qweight, scales.float(), group_size)
    residual = weight.float() - reconstructed
    score = (residual.square() * importance.float().reshape(1, -1)).sum(-1)
    if tp_stripes > 1:
        if weight.shape[0] % tp_stripes:
            raise ValueError("output rows must divide evenly across TP stripes")
        stripe_rows = weight.shape[0] // tp_stripes
        quota = round(stripe_rows * w4_fraction)
        selected = []
        for stripe in range(tp_stripes):
            start = stripe * stripe_rows
            local = torch.topk(score[start : start + stripe_rows], quota).indices
            selected.append(local.sort().values + start)
        indices = torch.cat(selected)
    else:
        quota = round(weight.shape[0] * w4_fraction)
        indices = torch.topk(score, quota).indices.sort().values
    w4_qweight, w4_scales = quantize_residual_w4(residual[indices], group_size)
    return qweight, scales, w4_qweight, w4_scales, indices.to(torch.int32)


def load_importance(
    path: Path | None, layer: int, projection: str, width: int
) -> torch.Tensor:
    if path is None:
        return torch.ones(width)
    payload = _IMPORTANCE_CACHE.get(path)
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _IMPORTANCE_CACHE[path] = payload
    key = f"layers.{layer}.{projection}.input_second_moment"
    value = payload.get(key)
    if value is None:
        raise KeyError(f"importance artifact lacks {key}")
    return value


def quantize_shard(
    src: Path,
    dst: Path,
    importance_path: Path | None,
    group_size: int,
    w4_fraction: float,
    output_name: str | None = None,
) -> dict[str, str]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(src, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        consumed: set[str] = set()
        for name in keys:
            if name in consumed:
                continue
            match = EXPERT_RE.match(name)
            if not match:
                tensors[name] = handle.get_tensor(name)
                continue
            stem = match.group(1)
            scale_name = f"{stem}.weight_scale"
            if scale_name not in keys:
                raise KeyError(f"missing {scale_name} beside {name}")
            packed = handle.get_tensor(name)
            mx_scale = handle.get_tensor(scale_name)
            weight = dequantize_mxfp4(packed, mx_scale)
            layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
            proj = stem.rsplit(".", 1)[-1]
            importance = load_importance(importance_path, layer, proj, weight.shape[-1])
            tp_stripes = 4 if proj in {"w1", "w3"} else 1
            qweight, scales, w4_qweight, w4_scales, w4_indices = quantize_mixed_w2_w4(
                weight, importance, group_size, w4_fraction, tp_stripes
            )
            tensors[f"{stem}.qweight"] = qweight
            tensors[f"{stem}.scales"] = scales
            tensors[f"{stem}.w4_qweight"] = w4_qweight
            tensors[f"{stem}.w4_scales"] = w4_scales
            tensors[f"{stem}.w4_indices"] = w4_indices
            consumed.add(scale_name)
    save_file(tensors, dst, metadata={"format": "pt", "quant_method": "kimi_k3_w2a16"})
    return {name: output_name or dst.name for name in tensors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--importance", type=Path)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--w4-channel-fraction", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--wait-for-shards",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Poll for source shards at this interval, enabling download/quantize pipelining.",
    )
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    for name, shard in weight_map.items():
        match = EXPERT_RE.match(name)
        if match is None:
            continue
        scale_name = f"{match.group(1)}.weight_scale"
        if weight_map.get(scale_name) != shard:
            raise ValueError(
                f"{name} and {scale_name} must be in the same source shard; "
                f"got {shard!r} and {weight_map.get(scale_name)!r}"
            )
    output_map: dict[str, str] = {}
    for shard in sorted(set(index["weight_map"].values())):
        src, dst = args.source / shard, args.destination / shard
        while not src.is_file() and args.wait_for_shards > 0:
            print(f"waiting for {src.name}", flush=True)
            time.sleep(args.wait_for_shards)
        if not src.is_file():
            raise FileNotFoundError(src)
        sidecar = dst.with_suffix(dst.suffix + ".index.json")
        if args.resume and dst.exists() and sidecar.exists():
            shard_map = json.loads(sidecar.read_text())
        else:
            partial = dst.with_suffix(dst.suffix + ".partial")
            shard_map = quantize_shard(
                src,
                partial,
                args.importance,
                args.group_size,
                args.w4_channel_fraction,
                output_name=dst.name,
            )
            os.replace(partial, dst)
            sidecar.write_text(json.dumps(shard_map, sort_keys=True) + "\n")
        output_map.update(shard_map)
    for path in args.source.iterdir():
        if (
            path.name.endswith(".safetensors")
            or path.name == "model.safetensors.index.json"
        ):
            continue
        target = args.destination / path.name
        if path.is_file() and not target.exists():
            shutil.copy2(path, target)
    config_path = args.destination / "config.json"
    config = json.loads(config_path.read_text())
    qcfg = {
        "quant_method": "kimi_k3_w2a16",
        "bits": 2,
        "group_size": args.group_size,
        "symmetric": True,
        "scope": "routed_experts_only",
        "importance_calibrated": args.importance is not None,
        "w4_channel_fraction": args.w4_channel_fraction,
        "checkpoint_tp_size": 4,
    }
    config["quantization_config"] = qcfg
    if isinstance(config.get("text_config"), dict):
        config["text_config"]["quantization_config"] = qcfg
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.destination / "quantize_config.json").write_text(
        json.dumps(qcfg, indent=2, sort_keys=True) + "\n"
    )
    total = sum((args.destination / s).stat().st_size for s in set(output_map.values()))
    final_index = {
        "metadata": {"total_size": total, "quant_method": "kimi_k3_w2a16"},
        "weight_map": output_map,
    }
    (args.destination / "model.safetensors.index.json").write_text(
        json.dumps(final_index, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
