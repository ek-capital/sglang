# SPDX-License-Identifier: Apache-2.0
"""Validate replay-capture utility without downloading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
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
    args = parser.parse_args()
    report = validate_capture_root(
        args.root,
        required_operations=args.require,
        minimum_bundles=args.minimum_bundles,
        verify_sha256=not args.skip_hashes,
        require_closed=not args.allow_open,
        hash_workers=args.hash_workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
