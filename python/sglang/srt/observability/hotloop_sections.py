# SPDX-License-Identifier: Apache-2.0
"""Model-neutral decode-hotloop section and kernel ownership registry.

CUDA graph replay does not reliably retain Python/NVTX nesting for every graph
node.  Section ownership therefore has two independent signals: explicit
``sglang.hotloop/*`` spans and ordered kernel-name rules.  Every kernel is
assigned to at most one leaf section so reported shares cannot double count.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class HotloopSection:
    name: str
    description: str
    kernel_patterns: tuple[str, ...] = ()
    capture_operations: tuple[str, ...] = ()
    actionable: bool = True
    priority: int = 0
    model_families: tuple[str, ...] = ()

    def supports(self, model_family: str | None) -> bool:
        return (
            model_family is None
            or not self.model_families
            or model_family in self.model_families
        )

    @property
    def capture_ready(self) -> bool:
        return bool(self.capture_operations)

    def to_dict(self) -> dict:
        return {**asdict(self), "capture_ready": self.capture_ready}


class HotloopSectionRegistry:
    def __init__(self, sections: Iterable[HotloopSection] = ()) -> None:
        self._sections: dict[str, HotloopSection] = {}
        self._compiled: dict[str, tuple[re.Pattern[str], ...]] = {}
        for section in sections:
            self.register(section)

    def register(self, section: HotloopSection) -> None:
        if section.name in self._sections:
            raise ValueError(f"duplicate hotloop section: {section.name}")
        self._sections[section.name] = section
        self._compiled[section.name] = tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in section.kernel_patterns
        )

    def get(self, name: str) -> HotloopSection:
        try:
            return self._sections[name]
        except KeyError as exc:
            raise KeyError(f"unknown hotloop section: {name}") from exc

    def sections(self, model_family: str | None = None) -> list[HotloopSection]:
        return sorted(
            (
                section
                for section in self._sections.values()
                if section.supports(model_family)
            ),
            key=lambda section: (-section.priority, section.name),
        )

    def classify_kernel(self, kernel_name: str, model_family: str | None = None) -> str:
        for section in self.sections(model_family):
            if any(
                pattern.search(kernel_name) for pattern in self._compiled[section.name]
            ):
                return section.name
        return "other.kernels"

    def capabilities(self, model_family: str | None = None) -> dict[str, dict]:
        return {
            section.name: section.to_dict() for section in self.sections(model_family)
        }


BUILTIN_SECTIONS = (
    # Speculative decoding is intentionally split more finely than the rest of
    # the stack: proposal model, planning, target verification, acceptance and
    # state commits have different optimization surfaces and replay contracts.
    HotloopSection(
        "speculative.draft",
        "Draft-model forward, Markov head and proposal sampling.",
        (
            r"dspark.*(?:draft|markov|sample)",
            r"(?:draft|markov).*(?:gemm|kernel|sample)",
            r"sample_step_tokens",
        ),
        ("speculative.draft_generation",),
        priority=100,
    ),
    HotloopSection(
        "speculative.plan",
        "Confidence, token-budget and verify-layout planning.",
        (r"dspark.*(?:confidence|budget|layout|verify_window)",),
        ("speculative.verify_plan",),
        priority=99,
    ),
    HotloopSection(
        "speculative.accept",
        "Greedy/rejection acceptance, bonus sampling and output construction.",
        (
            r"accept_(?:greedy|sampling)",
            r"reject_sampling",
            r"finalize_accept",
            r"build_out_tokens",
            r"select_mixed_accept",
        ),
        ("speculative.acceptance",),
        priority=98,
    ),
    HotloopSection(
        "speculative.state_commit",
        "Target recurrent/KV state and draft-hidden-state commit.",
        (r"(?:mamba|kda|kv|hidden).*(?:commit|inject|rollback)",),
        ("speculative.draft_round", "attention.kda_target_verify"),
        priority=97,
    ),
    HotloopSection(
        "attention.kda",
        "Kimi Delta Attention projection, convolution and recurrent update.",
        (r"(?:kda|delta_rule|causal_conv1d|fused_recurrent)",),
        ("attention.kda_target_verify",),
        priority=90,
        model_families=("kimi_k3", "qwen"),
    ),
    HotloopSection(
        "attention.mla",
        "MLA projection, paged attention and output projection.",
        (r"(?:flashmla|mla_|multi_latent|trtllm_mla|cutedsl_mla)",),
        priority=89,
        model_families=("kimi_k3", "deepseek_v4"),
    ),
    HotloopSection(
        "moe.routed_experts",
        "Router, dispatch, expert GEMMs and combine.",
        (
            r"(?:moe|expert).*(?:gemm|dispatch|combine|permute|topk)",
            r"(?:flashinfer|trtllm).*moe",
            r"marlin.*(?:moe|gemm)",
        ),
        ("moe.tail",),
        priority=80,
    ),
    HotloopSection(
        "moe.shared_experts",
        "Shared-expert MLP branch.",
        (r"shared.*expert",),
        ("moe.shared_mlp",),
        priority=81,
    ),
    HotloopSection(
        "collective.tp_ep",
        "Tensor/expert-parallel collectives on the decode critical path.",
        (r"(?:nccl|allreduce|all_reduce|reduce_scatter|alltoall|all_to_all)",),
        ("collective.tp_residual",),
        priority=75,
    ),
    HotloopSection(
        "residual.attnres",
        "Attention Residual aggregation and residual-bank update.",
        (r"attnres|attn_res",),
        ("residual.attnres",),
        priority=70,
        model_families=("kimi_k3",),
    ),
    HotloopSection(
        "dense.projections",
        "Unattributed dense GEMMs and projections.",
        (r"(?:gemm|matmul|cublas|cutlass)",),
        actionable=False,
        priority=10,
    ),
    HotloopSection(
        "other.kernels",
        "Kernels not yet resolved to an actionable owner.",
        actionable=False,
        priority=-100,
    ),
)


REGISTRY = HotloopSectionRegistry(BUILTIN_SECTIONS)
