# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load(name: str, relative: str):
    path = Path(__file__).parents[4] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replay_capture = _load(
    "sglang.srt.debug_utils.replay_capture",
    "python/sglang/srt/debug_utils/replay_capture.py",
)
validator = _load(
    "replay_capture_validate",
    "python/sglang/srt/debug_utils/replay_capture_validate.py",
)


def _write_manifest(root: Path, rank: int, world_size: int, entries: list[dict]):
    directory = root / f"rank-{rank:05d}"
    directory.mkdir(parents=True)
    records = [
        {"record_type": "run", "rank": rank, "world_size": world_size},
        {"record_type": "shard", "entries": entries},
    ]
    (directory / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def _collective_entry(rank: int, sequence: int) -> dict:
    return {
        "point": "bundle.collective.tp_residual",
        "rank": rank,
        "metadata": {"collective_sequence": sequence},
        "tensors": {
            "rank_partial": {},
            "pending_residual": {},
            "collective_output": {},
        },
    }


def test_validator_requires_every_collective_rank(tmp_path):
    _write_manifest(tmp_path, 0, 2, [_collective_entry(0, 7)])
    report = validator.validate_capture_root(tmp_path)
    assert not report["valid"]
    assert "expected [0, 1]" in report["errors"][0]


def test_validator_accepts_complete_collective_and_minimum(tmp_path):
    _write_manifest(tmp_path, 0, 2, [_collective_entry(0, 7)])
    _write_manifest(tmp_path, 1, 2, [_collective_entry(1, 7)])
    report = validator.validate_capture_root(
        tmp_path,
        required_operations=["collective.tp_residual"],
        minimum_bundles=2,
    )
    assert report["valid"]
    assert report["complete_bundles"]["collective.tp_residual"] == 2
