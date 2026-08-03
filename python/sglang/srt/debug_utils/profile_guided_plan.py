# SPDX-License-Identifier: Apache-2.0
"""Compile measured hotloop winners into a bounded replay-capture plan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from sglang.srt.observability.hotloop_sections import REGISTRY

_DEFAULT_POLICIES = {
    "attention.kda_target_verify": "rank_local",
    "collective.tp_residual": "collective",
    "speculative.draft_round": "sequence",
    "speculative.draft_generation": "sequence",
    "speculative.verify_plan": "sequence",
    "speculative.acceptance": "sequence",
    "speculative.dspark_training_window": "sequence",
}


def _capability_policies(payload: dict[str, Any] | None) -> dict[str, str]:
    policies = dict(_DEFAULT_POLICIES)
    if payload:
        operations = payload.get("operations", payload)
        for name, capability in operations.items():
            if isinstance(capability, dict) and capability.get("policy"):
                policies[str(name)] = str(capability["policy"])
    return policies


def compile_capture_plan(
    selection: dict[str, Any],
    *,
    case_bytes: dict[str, int],
    cases_per_section: int = 16,
    world_size: int,
    total_byte_limit: int = 5_000_000_000_000,
    safety_factor: float = 1.25,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if cases_per_section <= 0:
        raise ValueError("cases_per_section must be positive")
    selected = selection.get("selected") or []
    if len(selected) not in (2, 3):
        raise ValueError("selection must contain exactly two or three sections")
    policies = _capability_policies(capabilities)
    operations: dict[str, int] = {}
    estimates: dict[str, dict[str, int]] = {}
    requires_all_ranks = False
    estimated_total = 0
    for row in selected:
        section_name = str(row["section"])
        section = REGISTRY.get(section_name)
        if not section.actionable or not section.capture_ready:
            raise ValueError(f"section is not capture-ready: {section_name}")
        if section_name not in case_bytes or int(case_bytes[section_name]) <= 0:
            raise ValueError(
                f"missing positive preflight byte estimate for {section_name}"
            )
        bytes_one = int(case_bytes[section_name])
        section_rank_multiplier = 1
        for operation in section.capture_operations:
            policy = policies.get(operation, "single")
            if policy in {"collective", "rank_local"}:
                requires_all_ranks = True
                section_rank_multiplier = world_size
            operations[operation] = max(operations.get(operation, 0), cases_per_section)
        section_total = bytes_one * cases_per_section * section_rank_multiplier
        estimated_total += section_total
        estimates[section_name] = {
            "bytes_per_case": bytes_one,
            "cases": cases_per_section,
            "rank_multiplier": section_rank_multiplier,
            "estimated_bytes": section_total,
        }
    budgeted_total = math.ceil(estimated_total * safety_factor)
    if budgeted_total > total_byte_limit:
        raise ValueError(
            f"capture estimate {budgeted_total} exceeds limit {total_byte_limit}"
        )
    # The transport ceiling is per rank. A conservative equal split is safe for
    # rank-local captures and intentionally leaves headroom on rank-zero-only ops.
    per_rank = max(
        1, math.ceil(budgeted_total / (world_size if requires_all_ranks else 1))
    )
    return {
        "format_version": 1,
        "methodology": "profile_guided_top_k",
        "model_family": selection.get("model_family"),
        "profile_timing_basis": selection.get("timing_basis"),
        "selected_sections": [str(row["section"]) for row in selected],
        "selected_share": selection.get("selected_share"),
        "ranks": "all" if requires_all_ranks else "0",
        "require_all_ranks": requires_all_ranks,
        "phases": ["decode"],
        "operations": sorted(operations),
        "max_rows_per_operation": dict(sorted(operations.items())),
        "max_gib_per_rank": math.ceil(per_rank / 2**30 * 1000) / 1000,
        "budget": {
            "estimated_bytes": estimated_total,
            "safety_factor": safety_factor,
            "budgeted_bytes": budgeted_total,
            "hard_limit_bytes": total_byte_limit,
            "per_section": estimates,
        },
    }


def _parse_case_bytes(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        try:
            section, raw_bytes = value.split("=", 1)
            result[section] = int(raw_bytes)
        except ValueError as exc:
            raise ValueError(
                f"invalid --case-bytes {value!r}; expected section=bytes"
            ) from exc
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--case-bytes", action="append", default=[])
    parser.add_argument("--cases-per-section", type=int, default=16)
    parser.add_argument("--total-byte-limit", type=int, default=5_000_000_000_000)
    parser.add_argument("--capabilities", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    capabilities = (
        None
        if args.capabilities is None
        else json.loads(args.capabilities.read_text(encoding="utf-8"))
    )
    plan = compile_capture_plan(
        json.loads(args.selection.read_text(encoding="utf-8")),
        case_bytes=_parse_case_bytes(args.case_bytes),
        cases_per_section=args.cases_per_section,
        world_size=args.world_size,
        total_byte_limit=args.total_byte_limit,
        capabilities=capabilities,
    )
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
