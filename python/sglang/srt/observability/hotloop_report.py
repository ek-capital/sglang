# SPDX-License-Identifier: Apache-2.0
"""Build an exclusive decode-hotloop timing table and selection manifest."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from sglang.srt.observability.hotloop_sections import REGISTRY

_RANK_RE = re.compile(r"(?:TP-|rank[-_]?)(\d+)", re.IGNORECASE)
_KERNEL_CATEGORIES = {
    "kernel",
    "cuda_kernel",
    "gpu_kernel",
    "concurrent_kernel",
}


@dataclass(frozen=True)
class SectionTiming:
    section: str
    time_ms: float
    share: float
    calls: int
    actionable: bool
    capture_ready: bool


def _load_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _rank(path: Path, payload: dict[str, Any]) -> int:
    explicit = payload.get("rank")
    if explicit is not None:
        return int(explicit)
    match = _RANK_RE.search(path.name)
    return int(match.group(1)) if match else 0


def _duration_us(event: dict[str, Any]) -> float:
    if event.get("dur") is not None:
        return float(event["dur"])
    args = event.get("args") or {}
    for key in ("gpu_duration_us", "duration_us", "Duration (us)"):
        if args.get(key) is not None:
            return float(args[key])
    return 0.0


def _is_kernel(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", "")).lower()
    if category in _KERNEL_CATEGORIES or "kernel" in category:
        return True
    args = event.get("args") or {}
    return any(key in args for key in ("grid", "blocks per SM", "registers per thread"))


def _semantic_section(name: str) -> str | None:
    prefix = "sglang.hotloop/"
    if not name.startswith(prefix):
        return None
    semantic = name[len(prefix) :].split("/", 1)[0]
    ownership = (
        (("speculative.draft",), "speculative.draft"),
        (
            ("speculative.confidence", "speculative.schedule", "speculative.plan"),
            "speculative.plan",
        ),
        (("speculative.accept",), "speculative.accept"),
        (
            ("speculative.kda_commit", "speculative.hidden_commit"),
            "speculative.state_commit",
        ),
        (("attention.kda", "kda."), "attention.kda"),
        (("attention.mla", "mla."), "attention.mla"),
        (("moe.shared",), "moe.shared_experts"),
        (("moe.",), "moe.routed_experts"),
        (("collective.", "tp."), "collective.tp_ep"),
        (("residual.attnres", "attnres."), "residual.attnres"),
    )
    for prefixes, section in ownership:
        if semantic.startswith(prefixes):
            return section
    return semantic if semantic in REGISTRY.capabilities() else None


def _step_count(payload: dict[str, Any]) -> int:
    explicit = payload.get("decode_steps")
    if explicit is not None:
        return max(1, int(explicit))
    steps = {
        str(event.get("name"))
        for event in payload.get("traceEvents", [])
        if str(event.get("name", "")).startswith("ProfilerStep#")
    }
    return max(1, len(steps))


def _events(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    # A compact normalized format is useful for Nsight exporters and tests.
    if "kernel_events" in payload:
        for event in payload["kernel_events"]:
            yield {
                "name": event["name"],
                "dur": event["duration_us"],
                "cat": "kernel",
                "args": {"section": event.get("section")},
            }
        return
    yield from payload.get("traceEvents", [])


def analyze_traces(
    paths: Iterable[Path], *, model_family: str | None = None
) -> dict[str, Any]:
    per_rank_us: defaultdict[int, Counter[str]] = defaultdict(Counter)
    per_rank_calls: defaultdict[int, Counter[str]] = defaultdict(Counter)
    per_rank_steps: Counter[int] = Counter()
    evidence: Counter[str] = Counter()

    for path in paths:
        payload = _load_json(path)
        rank = _rank(path, payload)
        per_rank_steps[rank] += _step_count(payload)
        kernels_seen = 0
        projected: list[tuple[str, float]] = []
        for event in _events(payload):
            name = str(event.get("name", ""))
            duration = _duration_us(event)
            if duration <= 0:
                continue
            if _is_kernel(event):
                kernels_seen += 1
                args = event.get("args") or {}
                explicit = args.get("section")
                section = (
                    str(explicit)
                    if explicit
                    else REGISTRY.classify_kernel(name, model_family)
                )
                per_rank_us[rank][section] += duration
                per_rank_calls[rank][section] += 1
                evidence["exclusive_kernel"] += 1
                continue
            semantic = _semantic_section(name)
            args = event.get("args") or {}
            if semantic and args.get("gpu_projected") is True:
                projected.append((semantic, duration))
        if kernels_seen == 0:
            # Fallback for an Nsight exporter that provides projected NVTX GPU
            # durations but no individual kernels. Such ranges may overlap and
            # are marked explicitly in the report.
            for section, duration in projected:
                per_rank_us[rank][section] += duration
                per_rank_calls[rank][section] += 1
                evidence["projected_semantic_fallback"] += 1

    if not per_rank_us:
        raise ValueError("no CUDA kernel or projected semantic events found")
    rank_totals = {rank: sum(values.values()) for rank, values in per_rank_us.items()}
    critical_rank = max(rank_totals, key=rank_totals.get)
    total_us = rank_totals[critical_rank]
    steps = max(1, per_rank_steps[critical_rank])
    rows: list[SectionTiming] = []
    for section, duration in per_rank_us[critical_rank].most_common():
        try:
            capability = REGISTRY.get(section)
            actionable = capability.actionable
            capture_ready = capability.capture_ready
        except KeyError:
            actionable = False
            capture_ready = False
        rows.append(
            SectionTiming(
                section=section,
                time_ms=duration / 1000.0 / steps,
                share=duration / total_us if total_us else 0.0,
                calls=per_rank_calls[critical_rank][section],
                actionable=actionable,
                capture_ready=capture_ready,
            )
        )
    return {
        "schema_version": 1,
        "model_family": model_family,
        "critical_rank": critical_rank,
        "decode_steps": steps,
        "total_attributed_ms_per_step": total_us / 1000.0 / steps,
        "timing_basis": (
            "exclusive_kernel"
            if evidence["exclusive_kernel"]
            else "projected_semantic_fallback_may_overlap"
        ),
        "sections": [asdict(row) for row in rows],
        "rank_totals_ms": {
            str(rank): duration / 1000.0 / max(1, per_rank_steps[rank])
            for rank, duration in sorted(rank_totals.items())
        },
    }


def select_sections(report: dict[str, Any], top_k: int) -> dict[str, Any]:
    if top_k not in (2, 3):
        raise ValueError("top_k must be 2 or 3")
    selected = [
        row for row in report["sections"] if row["actionable"] and row["capture_ready"]
    ][:top_k]
    if len(selected) != top_k:
        raise ValueError(
            f"only {len(selected)} actionable capture-ready sections; need {top_k}"
        )
    return {
        "schema_version": 1,
        "model_family": report.get("model_family"),
        "critical_rank": report["critical_rank"],
        "timing_basis": report["timing_basis"],
        "selected": selected,
        "selected_share": sum(float(row["share"]) for row in selected),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "| Hotloop section | Time/step | Share | Calls | Capture-ready |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in report["sections"]:
        lines.append(
            f"| {row['section']} | {row['time_ms']:.3f} ms | "
            f"{100 * row['share']:.1f}% | {row['calls']} | "
            f"{'yes' if row['capture_ready'] else 'no'} |"
        )
    lines.append(
        f"| **Total attributed kernel time** | "
        f"**{report['total_attributed_ms_per_step']:.3f} ms** | **100%** | | |"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--model-family")
    parser.add_argument("--top-k", type=int, choices=(2, 3), default=3)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--selection-out", type=Path)
    args = parser.parse_args()
    report = analyze_traces(args.traces, model_family=args.model_family)
    selection = select_sections(report, args.top_k)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.markdown_out:
        args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    if args.selection_out:
        args.selection_out.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
