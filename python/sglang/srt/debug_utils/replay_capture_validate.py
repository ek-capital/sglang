# SPDX-License-Identifier: Apache-2.0
"""Validate replay-capture utility without downloading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sglang.srt.debug_utils.replay_capture import REGISTRY


def _records(path: Path) -> Iterable[dict[str, Any]]:
    for manifest in sorted(path.glob("rank-*/manifest.jsonl")):
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                    record["_manifest_dir"] = str(manifest.parent)
                    yield record
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
    verify_sha256: bool = False,
    require_closed: bool = False,
    hash_workers: int = 8,
    coverage_requirements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runs: dict[int, dict[str, Any]] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    invalid: list[str] = []
    collective_ranks: defaultdict[tuple[str, Any], set[int]] = defaultdict(set)
    operation_entries: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

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
            operation_entries[operation].append(entry)
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

    coverage_report: dict[str, Any] = {}
    for operation, requirement in sorted((coverage_requirements or {}).items()):
        entries = operation_entries[operation]
        operation_errors: list[str] = []
        minimum = int(requirement.get("minimum_bundles", 1))
        if len(entries) < minimum:
            operation_errors.append(f"bundles={len(entries)}<{minimum}")

        ranks = {int(entry.get("rank", -1)) for entry in entries}
        expected = requirement.get("required_ranks")
        expected_ranks = (
            expected_ranks
            if expected == "all"
            else {int(rank) for rank in (expected or [])}
        )
        if expected_ranks and not expected_ranks.issubset(ranks):
            operation_errors.append(
                f"ranks={sorted(ranks)}, missing={sorted(expected_ranks - ranks)}"
            )
        per_rank_counts = Counter(int(entry.get("rank", -1)) for entry in entries)
        minimum_per_rank = int(requirement.get("minimum_bundles_per_rank", 0))
        if minimum_per_rank:
            ranks_to_check = expected_ranks or ranks
            short_ranks = {
                rank: per_rank_counts[rank]
                for rank in ranks_to_check
                if per_rank_counts[rank] < minimum_per_rank
            }
            if short_ranks:
                operation_errors.append(
                    "bundles_per_rank="
                    + ",".join(
                        f"{rank}:{count}<{minimum_per_rank}"
                        for rank, count in sorted(short_ranks.items())
                    )
                )

        metadata = [entry.get("metadata", {}) for entry in entries]
        modes = {str(item.get("forward_mode", "")).lower() for item in metadata}
        required_modes = {
            str(mode).lower() for mode in requirement.get("forward_modes", [])
        }
        if not required_modes.issubset(modes):
            operation_errors.append(
                f"forward_modes={sorted(modes)}, "
                f"missing={sorted(required_modes - modes)}"
            )

        layers = {int(entry.get("layer", -1)) for entry in entries}
        required_layers = {int(layer) for layer in requirement.get("layers", [])}
        if not required_layers.issubset(layers):
            operation_errors.append(
                f"layers={sorted(layers)}, missing={sorted(required_layers - layers)}"
            )
        minimum_layers = int(requirement.get("minimum_layers", 0))
        semantic_layers = {layer for layer in layers if layer >= 0}
        if len(semantic_layers) < minimum_layers:
            operation_errors.append(
                f"distinct_layers={len(semantic_layers)}<{minimum_layers}"
            )

        forward_ids = {
            int(item["forward_id"])
            for item in metadata
            if item.get("forward_id") is not None and int(item["forward_id"]) >= 0
        }
        minimum_forward_ids = int(requirement.get("minimum_forward_ids", 0))
        if len(forward_ids) < minimum_forward_ids:
            operation_errors.append(
                f"distinct_forward_ids={len(forward_ids)}<{minimum_forward_ids}"
            )

        minimum_tokens = int(requirement.get("minimum_tokens", 0))
        token_count = sum(int(item.get("token_count", 0)) for item in metadata)
        if token_count < minimum_tokens:
            operation_errors.append(f"tokens={token_count}<{minimum_tokens}")

        batch_sizes = {
            int(item["batch_size"])
            for item in metadata
            if item.get("batch_size") is not None
        }
        minimum_observed_bs = requirement.get("minimum_observed_batch_size")
        if minimum_observed_bs is not None and (
            not batch_sizes or min(batch_sizes) > int(minimum_observed_bs)
        ):
            operation_errors.append(
                f"min_batch_size={min(batch_sizes) if batch_sizes else None}"
                f">{int(minimum_observed_bs)}"
            )
        maximum_observed_bs = requirement.get("maximum_observed_batch_size")
        if maximum_observed_bs is not None and (
            not batch_sizes or max(batch_sizes) < int(maximum_observed_bs)
        ):
            operation_errors.append(
                f"max_batch_size={max(batch_sizes) if batch_sizes else None}"
                f"<{int(maximum_observed_bs)}"
            )

        if operation_errors:
            invalid.extend(f"{operation}: {error}" for error in operation_errors)
        coverage_report[operation] = {
            "bundles": len(entries),
            "ranks": sorted(ranks),
            "forward_modes": sorted(modes),
            "layers": sorted(layers),
            "distinct_forward_ids": len(forward_ids),
            "tokens": token_count,
            "batch_sizes": sorted(batch_sizes),
            "valid": not operation_errors,
        }

    if require_closed:
        for rank in sorted(runs):
            close_path = root / f"rank-{rank:05d}" / "capture-close.json"
            try:
                close = json.loads(close_path.read_text(encoding="utf-8"))
                if close.get("rank") != rank or close.get("complete") is not True:
                    raise ValueError("invalid close record")
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                invalid.append(
                    f"rank {rank} has no valid transactional close marker: {exc}"
                )

    checked_shards = 0
    if verify_sha256:

        def shard_jobs() -> Iterable[tuple[Path, str]]:
            for record in _records(root):
                if record.get("record_type") != "shard":
                    continue
                yield (
                    Path(record["_manifest_dir"]) / str(record["file"]),
                    str(record.get("sha256", "")),
                )

        def check(job: tuple[Path, str]) -> tuple[Path, str | None]:
            path, expected = job
            try:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                actual = digest.hexdigest()
            except OSError as exc:
                return path, str(exc)
            if not expected or actual != expected:
                return path, f"sha256 {actual} != {expected or '<missing>'}"
            return path, None

        def consume(result: tuple[Path, str | None]) -> None:
            nonlocal checked_shards
            path, error = result
            checked_shards += 1
            if error:
                invalid.append(f"invalid shard {path}: {error}")

        workers = max(1, hash_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            for job in shard_jobs():
                pending.append(pool.submit(check, job))
                if len(pending) >= workers * 2:
                    consume(pending.popleft().result())
            while pending:
                consume(pending.popleft().result())
    return {
        "valid": not invalid,
        "world_size": world_size,
        "ranks_with_run_record": sorted(runs),
        "complete_bundles": dict(sorted(counts.items())),
        "hashed_shards": checked_shards,
        "coverage": coverage_report,
        "errors": invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--minimum-bundles", type=int, default=1)
    parser.add_argument("--hash-workers", type=int, default=8)
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--allow-open", action="store_true")
    parser.add_argument(
        "--coverage",
        type=Path,
        help="JSON map of operation names to semantic coverage requirements",
    )
    args = parser.parse_args()
    coverage_requirements = None
    if args.coverage is not None:
        coverage_requirements = json.loads(args.coverage.read_text(encoding="utf-8"))
    report = validate_capture_root(
        args.root,
        required_operations=args.require,
        minimum_bundles=args.minimum_bundles,
        verify_sha256=not args.skip_hashes,
        require_closed=not args.allow_open,
        hash_workers=args.hash_workers,
        coverage_requirements=coverage_requirements,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
