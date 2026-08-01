#!/usr/bin/env python3
"""Reduce K3 W2 capture shards into per-layer input second moments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def _load_tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-tp", type=int, default=4)
    args = parser.parse_args()

    sums: dict[tuple[str, int, int], torch.Tensor] = {}
    counts: dict[tuple[str, int, int], int] = defaultdict(int)
    for manifest in sorted(args.capture_dir.glob("rank-*/manifest.jsonl")):
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        run = next(record for record in records if record["record_type"] == "run")
        tp_rank = int(run["tp_rank"])
        for record in records:
            if record.get("record_type") != "capture":
                continue
            point = record["point"]
            if point not in {"moe_w13_input", "moe_w2_input"}:
                continue
            layer = int(record["layer"])
            path = manifest.parent / record["file"]
            if point == "moe_w13_input":
                value = _load_tensor(path, "hidden_states")
            else:
                gate_up = _load_tensor(path, "gate_up")
                gate, up = gate_up.chunk(2, dim=-1)
                metadata = record.get("metadata") or {}
                beta = float(metadata.get("beta") or 4.0)
                linear_beta = metadata.get("linear_beta")
                gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
                if linear_beta is not None:
                    linear_beta = float(linear_beta)
                    up = linear_beta * torch.tanh(up / linear_beta)
                value = gate * up
            key = point, layer, tp_rank
            contribution = value.square().sum(dim=0, dtype=torch.float64)
            sums[key] = sums.get(key, torch.zeros_like(contribution)) + contribution
            counts[key] += value.shape[0]

    layers = sorted({layer for point, layer, _rank in sums if point == "moe_w13_input"})
    if not layers:
        raise ValueError("no moe_w13_input captures found")
    output: dict[str, torch.Tensor] = {}
    for layer in layers:
        w13_ranks = sorted(
            rank
            for point, lid, rank in sums
            if point == "moe_w13_input" and lid == layer
        )
        rank = w13_ranks[0]
        w13 = (
            sums[("moe_w13_input", layer, rank)]
            / counts[("moe_w13_input", layer, rank)]
        ).float()
        output[f"layers.{layer}.w1.input_second_moment"] = w13
        output[f"layers.{layer}.w3.input_second_moment"] = w13.clone()

        w2_parts = []
        for tp_rank in range(args.expected_tp):
            key = "moe_w2_input", layer, tp_rank
            if key not in sums:
                raise ValueError(
                    f"missing layer {layer} w2 capture for TP rank {tp_rank}"
                )
            w2_parts.append((sums[key] / counts[key]).float())
        output[f"layers.{layer}.w2.input_second_moment"] = torch.cat(w2_parts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"wrote {len(output)} importance vectors for {len(layers)} layers to {args.output}"
    )


if __name__ == "__main__":
    main()
