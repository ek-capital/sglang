# SPDX-License-Identifier: Apache-2.0
"""Committed-token latency report for speculative decode rounds.

The DSpark info dumper records one row per complete speculative round.  This
module normalizes phase timings by tokens actually committed to each request,
not by draft candidates, target-verify rows, profiler steps, or batch-level
throughput.  It intentionally keeps wall-latency phase timing separate from
exclusive CUDA-kernel work reported by :mod:`hotloop_report`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    label: str
    field: str


PHASES = (
    PhaseSpec("prepare_window", "Prepare verify window", "prepare_window_gpu_ms"),
    PhaseSpec("draft", "Draft generation", "draft_gpu_ms"),
    PhaseSpec("confidence_budget", "Confidence and budget", "confidence_budget_gpu_ms"),
    PhaseSpec("schedule_layout", "Verify layout", "schedule_layout_gpu_ms"),
    PhaseSpec("target_verify", "Target verification", "target_verify_gpu_ms"),
    PhaseSpec("accept_finalize", "Acceptance and finalize", "accept_finalize_gpu_ms"),
    PhaseSpec("state_commit", "State commit", "state_commit_gpu_ms"),
)


def _committed_tokens(record: dict[str, Any]) -> int:
    reqs = record.get("reqs")
    if not isinstance(reqs, list) or not reqs:
        raise ValueError(
            "decode record has no per-request commit lengths; enable "
            "SGLANG_DSPARK_DEBUG_DUMP=core,reqs,phase_gpu_times"
        )
    committed = sum(int(req["acc_len"]) for req in reqs)
    if committed <= 0:
        raise ValueError("decode record committed no tokens")
    return committed


def analyze_decode_latency(payload: dict[str, Any]) -> dict[str, Any]:
    """Return request-latency-normalized speculative phase timings.

    A round taking ``L`` milliseconds for a batch of ``B`` requests contributes
    ``B * L`` request-milliseconds.  Dividing by all committed request tokens
    answers "how long did generating the next token take for an average
    request?".  Dividing ``L`` directly by committed tokens is also returned as
    service cost, a throughput-oriented metric with different semantics.
    """

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("no speculative decode records found")

    complete: list[tuple[dict[str, Any], int, int, float]] = []
    skipped = 0
    for record in records:
        bs = int(record.get("bs", -1))
        step_ms = record.get("step_gpu_ms")
        if bs <= 0 or step_ms is None:
            skipped += 1
            continue
        committed = _committed_tokens(record)
        complete.append((record, bs, committed, float(step_ms)))
    if not complete:
        raise ValueError("no complete timed speculative decode rounds found")

    committed_tokens = sum(row[2] for row in complete)
    request_rounds = sum(row[1] for row in complete)
    step_request_ms = sum(bs * step_ms for _, bs, _, step_ms in complete)
    step_service_ms = sum(step_ms for _, _, _, step_ms in complete)

    phase_rows: list[dict[str, Any]] = []
    known_request_ms = 0.0
    missing_fields: dict[str, int] = {}
    for phase in PHASES:
        values: list[tuple[int, float]] = []
        missing = 0
        for record, bs, _, _ in complete:
            value = record.get(phase.field)
            if value is None:
                missing += 1
            else:
                values.append((bs, float(value)))
        weighted_ms = sum(bs * value for bs, value in values)
        known_request_ms += weighted_ms
        if missing:
            missing_fields[phase.name] = missing
        phase_rows.append(
            {
                "phase": phase.name,
                "label": phase.label,
                "avg_ms_per_next_token": weighted_ms / committed_tokens,
                "share_of_next_token_latency": (
                    weighted_ms / step_request_ms if step_request_ms else 0.0
                ),
                "timed_rounds": len(values),
            }
        )

    unattributed_request_ms = max(0.0, step_request_ms - known_request_ms)
    phase_rows.append(
        {
            "phase": "runtime_unattributed",
            "label": "Runtime, gaps, and unattributed",
            "avg_ms_per_next_token": unattributed_request_ms / committed_tokens,
            "share_of_next_token_latency": (
                unattributed_request_ms / step_request_ms if step_request_ms else 0.0
            ),
            "timed_rounds": len(complete),
        }
    )

    return {
        "schema_version": 1,
        "normalization": "request_weighted_committed_token",
        "mode": payload.get("mode"),
        "rounds": len(complete),
        "skipped_records": skipped,
        "request_rounds": request_rounds,
        "committed_tokens": committed_tokens,
        "avg_committed_tokens_per_request_round": committed_tokens / request_rounds,
        "avg_gpu_ms_per_next_token": step_request_ms / committed_tokens,
        "gpu_service_ms_per_token": step_service_ms / committed_tokens,
        "phase_timing_coverage": known_request_ms / step_request_ms,
        "phase_overlap_request_ms": max(0.0, known_request_ms - step_request_ms),
        "missing_phase_records": missing_fields,
        "phases": phase_rows,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "| Speculative decode phase | Avg ms/next token | Share | Timed rounds |",
        "|---|---:|---:|---:|",
    ]
    for row in report["phases"]:
        lines.append(
            f"| {row['label']} | {row['avg_ms_per_next_token']:.4f} ms | "
            f"{100 * row['share_of_next_token_latency']:.2f}% | "
            f"{row['timed_rounds']} |"
        )
    lines.append(
        f"| **Total speculative decode** | "
        f"**{report['avg_gpu_ms_per_next_token']:.4f} ms** | **100.00%** | "
        f"**{report['rounds']}** |"
    )
    lines.extend(
        (
            "",
            f"- Average committed tokens/request/round: "
            f"{report['avg_committed_tokens_per_request_round']:.3f}",
            f"- GPU service cost: {report['gpu_service_ms_per_token']:.4f} ms/token",
            f"- Phase timing coverage: {100 * report['phase_timing_coverage']:.2f}%",
        )
    )
    return "\n".join(lines) + "\n"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "dspark_info_record" in payload:
        payload = payload["dspark_info_record"]
    return payload


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)

    report = analyze_decode_latency(_load(args.records))
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.markdown_out:
        args.markdown_out.write_text(markdown_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
