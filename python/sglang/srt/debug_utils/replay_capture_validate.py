# SPDX-License-Identifier: Apache-2.0
"""Validate replay-capture utility without downloading tensor payloads."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sglang.srt.debug_utils.replay_capture import REGISTRY


def _records(path: Path) -> Iterable[dict[str, Any]]:
    for manifest in sorted(path.glob("rank-*/manifest.jsonl")):
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {manifest}:{line_number}"
                    ) from exc


def _entries(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if record.get("record_type") == "shard":
        yield from record.get("entries", [])
    elif record.get("record_type") == "capture":
        yield record


def validate_capture_root(
    root: Path,
    *,
    required_operations: Iterable[str] = (),
    minimum_bundles: int = 1,
) -> dict[str, Any]:
    runs: dict[int, dict[str, Any]] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    invalid: list[str] = []
    collective_ranks: defaultdict[tuple[str, Any], set[int]] = defaultdict(set)

    for record in _records(root):
        if record.get("record_type") == "run":
            runs[int(record["rank"])] = record
            continue
        for entry in _entries(record):
            point = str(entry.get("point", ""))
            if not point.startswith("bundle."):
                continue
            operation = point.removeprefix("bundle.")
            try:
                spec = REGISTRY.get(operation)
                names = set(entry.get("tensors", {}))
                spec.validate({name: object() for name in names})  # type: ignore[arg-type]
            except (KeyError, ValueError) as exc:
                invalid.append(str(exc))
                continue
            counts[operation] += 1
            if spec.policy == "collective":
                metadata = entry.get("metadata", {})
                sequence = metadata.get("collective_sequence")
                collective_ranks[(operation, sequence)].add(int(entry["rank"]))

    world_sizes = {int(run.get("world_size", 1)) for run in runs.values()}
    if len(world_sizes) > 1:
        invalid.append(f"inconsistent world sizes: {sorted(world_sizes)}")
    world_size = next(iter(world_sizes), 1)
    expected_ranks = set(range(world_size))
    for (operation, sequence), ranks in sorted(
        collective_ranks.items(), key=lambda item: str(item[0])
    ):
        if ranks != expected_ranks:
            invalid.append(
                f"{operation} sequence {sequence} has ranks {sorted(ranks)}, "
                f"expected {sorted(expected_ranks)}"
            )

    missing = {
        operation: counts[operation]
        for operation in required_operations
        if counts[operation] < minimum_bundles
    }
    if missing:
        invalid.append(
            "insufficient complete bundles: "
            + ", ".join(
                f"{name}={count}<{minimum_bundles}"
                for name, count in sorted(missing.items())
            )
        )
    return {
        "valid": not invalid,
        "world_size": world_size,
        "ranks_with_run_record": sorted(runs),
        "complete_bundles": dict(sorted(counts.items())),
        "errors": invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--minimum-bundles", type=int, default=1)
    args = parser.parse_args()
    report = validate_capture_root(
        args.root,
        required_operations=args.require,
        minimum_bundles=args.minimum_bundles,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
