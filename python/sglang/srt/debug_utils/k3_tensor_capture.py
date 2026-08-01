# SPDX-License-Identifier: Apache-2.0
"""Bounded, opt-in tensor capture for Kimi-K3 replay datasets.

This module is intentionally inert unless ``SGLANG_K3_CAPTURE_DIR`` is set.
Capture is synchronous: tensors are cloned to CPU before the hotloop can reuse
their storage, then written as atomic safetensors shards plus a JSONL manifest.
That is slower than serving, but it makes a one-off corpus collection reliable.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_DEFAULT_POINTS = {
    "attention_input",
    "attention_output",
    "attn_res",
    "attn_res_state",
    "attn_res_static",
    "kda",
    "kda_state_decode",
    "kda_state_extend",
    "layer_input",
    "layer_output",
    "lm_head",
    "mla_latent",
    "mla_gate",
    "moe_input",
    "moe_w13_input",
    "moe_w2_input",
    "moe_output",
    "routing",
    "routing_static",
}


def _parse_layers(value: str) -> set[int] | None:
    value = value.strip().lower()
    if not value or value == "all":
        return None
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            result.update(range(lo, hi + 1))
        else:
            result.add(int(part))
    return result


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class K3TensorCapture:
    def __init__(self) -> None:
        capture_dir = os.environ.get("SGLANG_K3_CAPTURE_DIR", "").strip()
        self.enabled = bool(capture_dir)
        self.root = Path(capture_dir) if capture_dir else None
        arm_file = os.environ.get("SGLANG_K3_CAPTURE_ARM_FILE", "").strip()
        self.arm_file = Path(arm_file) if arm_file else None
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        self.tp_rank = int(os.environ.get("TP_RANK", str(self.rank)))
        self.default_max_rows = max(
            1, int(os.environ.get("SGLANG_K3_CAPTURE_MAX_ROWS", "512"))
        )
        self.layers = _parse_layers(os.environ.get("SGLANG_K3_CAPTURE_LAYERS", "all"))
        raw_points = os.environ.get("SGLANG_K3_CAPTURE_POINTS", "all").strip()
        self.points = (
            set(_DEFAULT_POINTS)
            if not raw_points or raw_points.lower() == "all"
            else {point.strip() for point in raw_points.split(",") if point.strip()}
        )
        rank_filter = os.environ.get("SGLANG_K3_CAPTURE_RANKS", "0").strip().lower()
        self.rank_allowed = rank_filter == "all" or self.rank in {
            int(item) for item in rank_filter.split(",") if item.strip()
        }
        self._rows: dict[tuple[str, int], int] = {}
        self._events: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()
        self._process_dir: Path | None = None
        self._manifest: Path | None = None

        if self.enabled and self.rank_allowed:
            assert self.root is not None
            self._process_dir = self.root / f"rank-{self.rank:05d}"
            self._process_dir.mkdir(parents=True, exist_ok=True)
            self._manifest = self._process_dir / "manifest.jsonl"
            run_record = {
                "record_type": "run",
                "format_version": 1,
                "created_unix_ns": time.time_ns(),
                "rank": self.rank,
                "local_rank": self.local_rank,
                "tp_rank": self.tp_rank,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "run_id": os.environ.get("SGLANG_K3_CAPTURE_RUN_ID"),
                "model_revision": os.environ.get("SGLANG_K3_CAPTURE_MODEL_REVISION"),
                "sglang_revision": os.environ.get("SGLANG_K3_CAPTURE_SGLANG_REVISION"),
                "corpus_sha256": os.environ.get("SGLANG_K3_CAPTURE_CORPUS_SHA256"),
                "sampling_seed": os.environ.get("SGLANG_K3_CAPTURE_SAMPLING_SEED"),
                "points": sorted(self.points),
                "layers": "all" if self.layers is None else sorted(self.layers),
                "default_max_rows": self.default_max_rows,
            }
            with self._manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(run_record, sort_keys=True) + "\n")

    def wants(self, point: str, layer_idx: int) -> bool:
        return (
            self.enabled
            and self.rank_allowed
            and (self.arm_file is None or self.arm_file.exists())
            and point in self.points
            and (self.layers is None or layer_idx in self.layers)
        )

    def _row_limit(self, point: str) -> int:
        env_name = "SGLANG_K3_CAPTURE_MAX_ROWS_" + point.upper().replace("-", "_")
        return max(1, int(os.environ.get(env_name, str(self.default_max_rows))))

    def remaining_rows(self, point: str, layer_idx: int) -> int:
        if not self.wants(point, layer_idx):
            return 0
        return max(0, self._row_limit(point) - self._rows.get((point, layer_idx), 0))

    def capture(
        self,
        point: str,
        layer_idx: int,
        tensors: Mapping[str, torch.Tensor | None],
        *,
        metadata: Mapping[str, Any] | None = None,
        once: bool = False,
    ) -> Path | None:
        """Write one bounded tensor bundle and return its path.

        The first non-scalar tensor defines the row count. Tensors with the
        same leading dimension are sliced consistently; static tensors (for
        example correction bias) are retained in full.
        """
        if not self.wants(point, layer_idx):
            return None
        key = (point, layer_idx)
        with self._lock:
            if once and self._events.get(key, 0) > 0:
                return None
            remaining = self.remaining_rows(point, layer_idx)
            if remaining <= 0:
                return None

            present = {
                name: value for name, value in tensors.items() if value is not None
            }
            row_count = (
                1
                if once
                else next(
                    (
                        int(value.shape[0])
                        for value in present.values()
                        if value.ndim > 0
                    ),
                    1,
                )
            )
            take = min(row_count, remaining)
            if take <= 0:
                return None

            cpu_tensors: dict[str, torch.Tensor] = {}
            tensor_meta: dict[str, Any] = {}
            for name, value in present.items():
                original_shape = list(value.shape)
                original_stride = list(value.stride())
                selected = (
                    value[:take]
                    if not once and value.ndim > 0 and value.shape[0] == row_count
                    else value
                )
                cpu_tensors[name] = selected.detach().contiguous().to("cpu").clone()
                tensor_meta[name] = {
                    "shape": original_shape,
                    "stride": original_stride,
                    "saved_shape": list(cpu_tensors[name].shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "device": str(value.device),
                }

            event = self._events.get(key, 0)
            self._events[key] = event + 1
            self._rows[key] = self._rows.get(key, 0) + take
            assert self._process_dir is not None and self._manifest is not None
            stem = f"l{layer_idx:03d}-{point}-{event:06d}"
            final_path = self._process_dir / f"{stem}.safetensors"
            tmp_path = self._process_dir / f".{stem}.{os.getpid()}.tmp"

            from safetensors.torch import save_file

            save_file(cpu_tensors, str(tmp_path))
            os.replace(tmp_path, final_path)
            sha256 = _sha256_file(final_path)
            record = {
                "record_type": "capture",
                "format_version": 1,
                "created_unix_ns": time.time_ns(),
                "point": point,
                "layer": layer_idx,
                "event": event,
                "rows": take,
                "rows_total": self._rows[key],
                "rank": self.rank,
                "local_rank": self.local_rank,
                "tp_rank": self.tp_rank,
                "file": final_path.name,
                "sha256": sha256,
                "tensors": tensor_meta,
                "metadata": _jsonable(dict(metadata or {})),
            }
            with self._manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            return final_path


_CAPTURE: K3TensorCapture | None = None


def get_k3_tensor_capture() -> K3TensorCapture:
    global _CAPTURE
    if _CAPTURE is None:
        _CAPTURE = K3TensorCapture()
    return _CAPTURE


def k3_capture_wants(point: str, layer_idx: int) -> bool:
    return get_k3_tensor_capture().wants(point, layer_idx)


def k3_capture_remaining_rows(point: str, layer_idx: int) -> int:
    return get_k3_tensor_capture().remaining_rows(point, layer_idx)


def k3_capture(
    point: str,
    layer_idx: int,
    tensors: Mapping[str, torch.Tensor | None],
    *,
    metadata: Mapping[str, Any] | None = None,
    once: bool = False,
) -> Path | None:
    return get_k3_tensor_capture().capture(
        point, layer_idx, tensors, metadata=metadata, once=once
    )
