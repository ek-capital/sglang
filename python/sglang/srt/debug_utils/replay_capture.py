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


def build_dspark_training_window(
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    target_hidden: torch.Tensor,
    target_layer_ids: list[int],
    sequence_lengths: list[int],
    request_pool_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build shifted labels without ever crossing a packed-sequence boundary.

    The last token in each captured chunk has no in-chunk successor, so it is
    masked rather than accidentally learning the first token of the next
    request.  Keeping this transformation on device also avoids transferring
    uncaptured rows through host memory.
    """
    token_count = int(input_ids.numel())
    if input_ids.ndim != 1 or positions.ndim != 1:
        raise ValueError("DSpark training capture requires flat token/position tensors")
    if positions.numel() != token_count or target_hidden.shape[0] != token_count:
        raise ValueError("DSpark training tensors disagree on token count")
    if len(sequence_lengths) != int(request_pool_indices.numel()):
        raise ValueError("one sequence length is required per request")
    if any(length < 0 for length in sequence_lengths):
        raise ValueError("sequence lengths must be non-negative")
    if sum(sequence_lengths) != token_count:
        raise ValueError(
            f"packed sequence lengths sum to {sum(sequence_lengths)}, "
            f"but the forward contains {token_count} tokens"
        )

    offsets_cpu = [0]
    for length in sequence_lengths:
        offsets_cpu.append(offsets_cpu[-1] + length)
    offsets = torch.tensor(offsets_cpu, dtype=torch.int64, device=input_ids.device)
    target_ids = torch.full_like(input_ids, -100)
    loss_mask = torch.zeros(token_count, dtype=torch.bool, device=input_ids.device)
    for start, end in zip(offsets_cpu[:-1], offsets_cpu[1:]):
        if end - start > 1:
            target_ids[start : end - 1] = input_ids[start + 1 : end]
            loss_mask[start : end - 1] = True

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "positions": positions,
        "sequence_offsets": offsets,
        "request_pool_indices": request_pool_indices,
        "target_hidden": target_hidden,
        "target_layer_ids": torch.tensor(
            target_layer_ids, dtype=torch.int32, device=input_ids.device
        ),
    }


@dataclass(frozen=True)
class ReplaySpec:
    name: str
    version: int
    required: frozenset[str]
    optional: frozenset[str] = frozenset()
    policy: str = "single"
    description: str = ""
    model_families: frozenset[str] = frozenset()

    @property
    def preserve_rows(self) -> bool:
        """Whether the leading dimension is semantic and must not be sampled."""
        return self.policy == "sequence"

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
        if spec.policy not in {
            "single",
            "paired",
            "sequence",
            "collective",
            "rank_local",
        }:
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
            "speculative.draft_generation",
            1,
            frozenset(
                {
                    "anchor_token_ids",
                    "positions",
                    "verify_token_ids",
                    "proposed_token_ids",
                    "draft_hidden_after",
                    "greedy_mask",
                    "temperatures",
                }
            ),
            frozenset(
                {
                    "draft_hidden_before",
                    "proposal_scores",
                    "confidence",
                }
            ),
            policy="sequence",
            description=(
                "DSpark draft-model and Markov proposal boundary. Model weights are "
                "referenced by the run's pinned draft checkpoint revision."
            ),
            model_families=kimi,
        ),
        ReplaySpec(
            "speculative.verify_plan",
            1,
            frozenset(
                {
                    "prefix_lengths",
                    "request_pool_indices",
                    "verify_width",
                }
            ),
            frozenset({"confidence", "cap_trim_lengths"}),
            policy="sequence",
            description="Confidence-to-budget and compact verify-layout boundary.",
            model_families=kimi,
        ),
        ReplaySpec(
            "speculative.acceptance",
            1,
            frozenset(
                {
                    "proposed_token_ids",
                    "verify_token_ids",
                    "target_logits",
                    "correct_length",
                    "accepted_length",
                    "bonus_token_ids",
                    "committed_token_ids",
                    "greedy_mask",
                    "temperatures",
                }
            ),
            frozenset(
                {
                    "proposal_scores",
                    "accept_uniform_samples",
                    "accept_uniform_samples_final",
                    "cap_trim_lengths",
                }
            ),
            policy="sequence",
            description=(
                "DSpark target verification acceptance with the exact random "
                "variates consumed by rejection sampling."
            ),
            model_families=kimi,
        ),
        ReplaySpec(
            "speculative.draft_round",
            2,
            frozenset(
                {
                    "anchor_token_ids",
                    "positions",
                    "proposed_token_ids",
                    "verify_token_ids",
                    "verify_width",
                    "correct_length",
                    "accepted_length",
                    "bonus_token_ids",
                    "committed_token_ids",
                    "new_sequence_lengths",
                    "request_pool_indices",
                    "draft_hidden_after",
                    "greedy_mask",
                    "temperatures",
                }
            ),
            frozenset(
                {
                    "confidence",
                    "parent_indices",
                    "proposal_scores",
                    "draft_hidden_before",
                    "cap_trim_lengths",
                    "rng_state_before",
                    "rng_state_after",
                    "accept_uniform_samples",
                    "accept_uniform_samples_final",
                    "target_logits",
                    "replay_inputs",
                }
            ),
            policy="sequence",
            description=(
                "One complete DSpark proposal/verify/accept/commit round; confidence "
                "is optional because the public static checkpoint is headless."
            ),
            model_families=kimi,
        ),
        ReplaySpec(
            "attention.kda_target_verify",
            1,
            frozenset(
                {
                    "mixed_qkv",
                    "forget_gate",
                    "beta",
                    "cache_indices",
                    "query_lengths",
                    "conv_states_before",
                    "ssm_states_before",
                    "conv_weights",
                    "a_log",
                    "dt_bias",
                    "lower_bound",
                    "intermediate_conv_windows_after",
                    "core_attn_out",
                }
            ),
            frozenset(
                {
                    "conv_bias",
                    "intermediate_ssm_after",
                    "replayssm_rawv_before",
                    "replayssm_rawk_before",
                    "replayssm_g_before",
                    "replayssm_beta_before",
                    "replayssm_rawv_after",
                    "replayssm_rawk_after",
                    "replayssm_g_after",
                    "replayssm_beta_after",
                    "retrieve_next_token",
                    "retrieve_next_sibling",
                    "retrieve_parent_token",
                    "output_norm_gate",
                    "output_norm_weight",
                    "output_norm_eps",
                }
            ),
            policy="rank_local",
            description=(
                "Per-rank KDA TARGET_VERIFY inputs, recurrent/conv state before, "
                "candidate state after every verify token, and kernel output."
            ),
            model_families=kimi,
        ),
        ReplaySpec(
            "speculative.dspark_training_window",
            1,
            frozenset(
                {
                    "input_ids",
                    "target_ids",
                    "loss_mask",
                    "positions",
                    "sequence_offsets",
                    "request_pool_indices",
                    "target_hidden",
                    "target_layer_ids",
                }
            ),
            frozenset({"topk_token_ids", "topk_logits", "logsumexp"}),
            policy="sequence",
            description=(
                "A contiguous teacher-forced K3 window at the exact hidden-state "
                "boundary consumed by DSpark."
            ),
            model_families=kimi,
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
    return spec.policy in {"collective", "rank_local"} or capture.rank == 0


def replay_capture_path_wants(operation: str, layer_idx: int) -> bool:
    """Whether every TP rank must take a capture-compatible execution path."""
    REGISTRY.get(operation)
    from sglang.srt.debug_utils.k3_tensor_capture import get_replay_tensor_capture

    return get_replay_tensor_capture().path_wants(f"bundle.{operation}", layer_idx)


def replay_capture_configured(operation: str, layer_idx: int) -> bool:
    """Whether an operation is selected even if runtime capture is unarmed."""
    REGISTRY.get(operation)
    from sglang.srt.debug_utils.k3_tensor_capture import get_replay_tensor_capture

    return get_replay_tensor_capture().configured(f"bundle.{operation}", layer_idx)


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

    return k3_capture(
        f"bundle.{operation}",
        layer_idx,
        tensors,
        metadata=merged,
        preserve_rows=spec.preserve_rows,
    )


def write_capability_manifest(path: str, model_family: str | None = None) -> None:
    payload = {
        "format_version": 1,
        "model_family": model_family,
        "operations": REGISTRY.capabilities(model_family),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
