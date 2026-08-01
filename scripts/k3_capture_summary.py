#!/usr/bin/env python3
"""Validate and summarize a Kimi-K3 tensor capture directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--expect-layers", type=int, default=93)
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument("--write-complete", action="store_true")
    args = parser.parse_args()
    if args.write_complete and args.skip_hash:
        raise SystemExit("--write-complete requires hash validation")

    rows = defaultdict(int)
    sizes = defaultdict(int)
    layers = defaultdict(set)
    files = 0
    validated_shards = []
    validated_manifests = []
    expert_quotas = []
    for manifest in sorted(args.capture_dir.glob("rank-*/manifest.jsonl")):
        validated_manifests.append(
            {
                "file": str(manifest.relative_to(args.capture_dir)),
                "sha256": sha256_file(manifest),
            }
        )
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("record_type") == "expert_quota":
                expert_quotas.append(record)
                continue
            if record.get("record_type") == "shard":
                shard = manifest.parent / record["file"]
                if not shard.is_file():
                    raise SystemExit(f"missing shard: {shard}")
                if not args.skip_hash:
                    digest = sha256_file(shard)
                    if digest != record["sha256"]:
                        raise SystemExit(f"hash mismatch: {shard}")
                validated_shards.append(
                    {
                        "file": str(shard.relative_to(args.capture_dir)),
                        "sha256": record["sha256"],
                        "size": shard.stat().st_size,
                    }
                )
                for entry in record["entries"]:
                    point = entry["point"]
                    rows[point] += int(entry["rows"])
                    sizes[point] += sum(
                        int(meta.get("saved_nbytes", 0))
                        for meta in entry.get("tensors", {}).values()
                    )
                    layers[point].add(int(entry["layer"]))
                    files += 1
                continue
            if record.get("record_type") != "capture":
                continue
            shard = manifest.parent / record["file"]
            if not shard.is_file():
                raise SystemExit(f"missing shard: {shard}")
            if not args.skip_hash:
                digest = sha256_file(shard)
                if digest != record["sha256"]:
                    raise SystemExit(f"hash mismatch: {shard}")
            validated_shards.append(
                {
                    "file": str(shard.relative_to(args.capture_dir)),
                    "sha256": record["sha256"],
                    "size": shard.stat().st_size,
                }
            )
            point = record["point"]
            rows[point] += int(record["rows"])
            sizes[point] += shard.stat().st_size
            layers[point].add(int(record["layer"]))
            files += 1

    if files == 0:
        raise SystemExit("no capture records found")
    print(f"validated {files} shards")
    for point in sorted(rows):
        missing = sorted(set(range(args.expect_layers)) - layers[point])
        coverage = f"{len(layers[point])}/{args.expect_layers}"
        print(
            f"{point:24s} rows={rows[point]:9d} "
            f"size_gib={sizes[point] / 2**30:8.3f} layers={coverage}"
        )
        if missing and point not in {"lm_head"}:
            print(f"  missing layers: {missing}")
    if expert_quotas:
        complete = sum(
            record["experts_at_quota"] == len(record["assignments"])
            for record in expert_quotas
        )
        print(
            f"expert quotas complete={complete}/{len(expert_quotas)} "
            f"minimum={min(record['min_assignments'] for record in expert_quotas)}"
        )

    if args.write_complete:
        entries = {"manifests": validated_manifests, "shards": validated_shards}
        canonical = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
        completion = {
            "format_version": 1,
            "root_sha256": hashlib.sha256(canonical).hexdigest(),
            **entries,
        }
        final_path = args.capture_dir / "COMPLETE.json"
        partial_path = args.capture_dir / ".COMPLETE.json.partial"
        partial_path.write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial_path, final_path)
        print(f"wrote {final_path} root_sha256={completion['root_sha256']}")


if __name__ == "__main__":
    main()
