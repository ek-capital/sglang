# SPDX-License-Identifier: Apache-2.0
"""Model-neutral, utility-first replay capture contracts.

This module deliberately separates *what makes an operation replayable* from
the transport used to persist tensors.  Model adapters register semantic
operation specifications; hot-path call sites submit one atomic boundary
bundle containing every required input/output/state tensor.  The existing
asynchronous safetensors writer remains the transport for now.

Capture runs are not profiling runs.  Enabling replay capture can change graph
selection, add copies, and perturb streams, so timings collected while this
module is armed must never be used as production latency measurements.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ReplaySpec:
    name: str
    version: int
    required: frozenset[str]
    optional: frozenset[str] = frozenset()
    policy: str = "single"
    description: str = ""
    model_families: frozenset[str] = frozenset()

    def validate(self, tensors: Mapping[str, torch.Tensor | None]) -> None:
        missing = sorted(name for name in self.required if tensors.get(name) is None)
        if missing:
            raise ValueError(
                f"replay bundle {self.name}@{self.version} is incomplete; "
                f"missing {', '.join(missing)}"
            )
        unknown = sorted(set(tensors) - self.required - self.optional)
        if unknown:
            raise ValueError(
                f"replay bundle {self.name}@{self.version} has undeclared "
                f"tensors: {', '.join(unknown)}"
            )


class ReplayRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ReplaySpec] = {}

    def register(self, spec: ReplaySpec) -> None:
        previous = self._specs.get(spec.name)
        if previous is not None and previous != spec:
            raise ValueError(f"replay spec already registered: {spec.name}")
        if spec.policy not in {"single", "paired", "sequence", "collective"}:
            raise ValueError(f"invalid replay policy: {spec.policy}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ReplaySpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown replay operation: {name}") from exc

    def capabilities(self, model_family: str | None = None) -> dict[str, Any]:
        selected = [
            spec
            for spec in self._specs.values()
            if model_family is None
            or not spec.model_families
            or model_family in spec.model_families
        ]
        return {
            spec.name: {
                "version": spec.version,
                "required": sorted(spec.required),
                "optional": sorted(spec.optional),
                "policy": spec.policy,
                "description": spec.description,
            }
            for spec in sorted(selected, key=lambda item: item.name)
        }


REGISTRY = ReplayRegistry()


def register_replay_spec(spec: ReplaySpec) -> ReplaySpec:
    REGISTRY.register(spec)
    return spec


def _register_builtin_specs() -> None:
    kimi = frozenset({"kimi_k3"})
    specs = (
        ReplaySpec(
            "speculative.draft_round",
            1,
            frozenset(
                {
                    "anchor_token_ids",
                    "positions",
                    "proposed_token_ids",
                    "confidence",
                    "verify_width",
                    "accepted_length",
                }
            ),
            frozenset(
                {
                    "parent_indices",
                    "proposal_scores",
                    "draft_state_before",
                    "draft_state_after",
                    "rng_state_before",
                    "rng_state_after",
                    "target_logits",
                    "replay_inputs",
                }
            ),
            policy="sequence",
            description="One complete speculative proposal/verify/commit round.",
        ),
        ReplaySpec(
            "moe.shared_mlp",
            1,
            frozenset({"hidden_states", "shared_output"}),
            frozenset({"expert_0_output", "expert_1_output"}),
            description="Source-available shared expert boundary.",
            model_families=kimi,
        ),
        ReplaySpec(
            "moe.tail",
            1,
            frozenset(
                {
                    "routed_output",
                    "shared_output",
                    "pending_residual",
                    "tail_output",
                }
            ),
            frozenset({"latent_before_up_proj", "routed_partial"}),
            description="Routed/shared MoE finalization boundary.",
            model_families=kimi,
        ),
        ReplaySpec(
            "residual.attnres",
            1,
            frozenset(
                {
                    "hidden_states",
                    "residual_bank_before",
                    "valid_block_count",
                    "aggregation_scores",
                    "softmax_weights",
                    "normalized_output",
                    "residual_bank_after",
                }
            ),
            frozenset({"pending_prefix", "aggregated_prefix", "write_index"}),
            policy="paired",
            description="AttnRes aggregation plus optional bank transition.",
            model_families=kimi,
        ),
        ReplaySpec(
            "collective.tp_residual",
            1,
            frozenset(
                {
                    "rank_partial",
                    "pending_residual",
                    "collective_output",
                }
            ),
            frozenset({"normalized_output"}),
            policy="collective",
            description="Same logical fused residual collective on every rank.",
        ),
    )
    for spec in specs:
        register_replay_spec(spec)


_register_builtin_specs()


def _stable_bundle_id(
    operation: str,
    layer_idx: int,
    metadata: Mapping[str, Any] | None,
) -> str:
    identity = {
        "operation": operation,
        "layer": layer_idx,
        "run_id": os.environ.get("SGLANG_REPLAY_CAPTURE_RUN_ID")
        or os.environ.get("SGLANG_K3_CAPTURE_RUN_ID"),
        **{
            key: (metadata or {}).get(key)
            for key in (
                "request_id",
                "batch_id",
                "forward_id",
                "spec_round_id",
                "collective_sequence",
            )
            if (metadata or {}).get(key) is not None
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def replay_capture_wants(operation: str, layer_idx: int) -> bool:
    spec = REGISTRY.get(operation)
    from sglang.srt.debug_utils.k3_tensor_capture import get_replay_tensor_capture

    capture = get_replay_tensor_capture()
    if not capture.wants(f"bundle.{operation}", layer_idx):
        return False
    # Model-local activations are TP-replicated or reconstructable from one
    # selected rank. Collective boundaries are the only policy that requires
    # every rank, avoiding an otherwise silent world-size multiplier in bytes.
    return spec.policy == "collective" or capture.rank == 0


def replay_capture_path_wants(operation: str, layer_idx: int) -> bool:
    """Whether every TP rank must take a capture-compatible execution path."""
    REGISTRY.get(operation)
    from sglang.srt.debug_utils.k3_tensor_capture import get_replay_tensor_capture

    return get_replay_tensor_capture().path_wants(f"bundle.{operation}", layer_idx)


def replay_capture_set_forward_context(forward_batch: Any) -> None:
    from sglang.srt.debug_utils.k3_tensor_capture import get_replay_tensor_capture

    get_replay_tensor_capture().set_forward_context(forward_batch)


def replay_capture_bundle(
    operation: str,
    layer_idx: int,
    tensors: Mapping[str, torch.Tensor | None],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Validate and persist one atomic replay boundary bundle.

    Tensors with a shared leading dimension are sampled with identical indices
    by the transport.  A bundle is accepted only when every declared required
    tensor is present, preventing expensive but unreplayable partial datasets.
    """
    spec = REGISTRY.get(operation)
    if not replay_capture_wants(operation, layer_idx):
        return None
    spec.validate(tensors)
    merged = {
        **dict(metadata or {}),
        "semantic_operation": operation,
        "schema_version": spec.version,
        "capture_policy": spec.policy,
        "bundle_id": _stable_bundle_id(operation, layer_idx, metadata),
        "timing_valid": False,
    }
    from sglang.srt.debug_utils.k3_tensor_capture import k3_capture

    return k3_capture(f"bundle.{operation}", layer_idx, tensors, metadata=merged)


def write_capability_manifest(path: str, model_family: str | None = None) -> None:
    payload = {
        "format_version": 1,
        "model_family": model_family,
        "operations": REGISTRY.capabilities(model_family),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
