# SPDX-License-Identifier: Apache-2.0
"""Bounded tensor transport for replay datasets.

The scalable path samples token rows on GPU, stages them into pinned host
memory on a dedicated CUDA stream, packs many capture entries into large local
NVMe safetensors shards, and uploads completed shards in the background.  The
legacy synchronous path remains available for small smoke tests.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

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
    "expert_output",
    "routing",
    "routing_static",
}


class _PendingCapture(NamedTuple):
    event: Any
    tensors: dict[str, torch.Tensor]
    record: dict[str, Any]
    nbytes: int


class _CompletedShard(NamedTuple):
    local_path: Path
    record: dict[str, Any]


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a small audit record without exposing a torn final file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _claim_new_directory(path: Path, label: str) -> None:
    """Atomically claim a fresh directory instead of appending to an old run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"{label} already exists: {path}; choose a fresh capture run directory"
        ) from exc


def _parallel_identity() -> tuple[int, int, int, int]:
    """Resolve ranks after SGLang has initialized torch.distributed.

    SGLang's multiprocessing launcher passes ranks as function arguments rather
    than exporting RANK/LOCAL_RANK in every scheduler worker.  Environment-only
    discovery therefore makes every TP worker look like rank zero and causes
    capture-file collisions.  Explicit capture overrides remain useful for
    tests and non-standard launchers; otherwise prefer live process-group state.
    """
    explicit_rank = os.environ.get("SGLANG_K3_CAPTURE_RANK")
    explicit_local_rank = os.environ.get("SGLANG_K3_CAPTURE_LOCAL_RANK")
    explicit_tp_rank = os.environ.get("SGLANG_K3_CAPTURE_TP_RANK")
    explicit_world_size = os.environ.get("SGLANG_K3_CAPTURE_WORLD_SIZE")

    dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = (
        int(explicit_rank)
        if explicit_rank is not None
        else (
            torch.distributed.get_rank()
            if dist_ready
            else int(os.environ.get("RANK", "0"))
        )
    )
    world_size = (
        int(explicit_world_size)
        if explicit_world_size is not None
        else (
            torch.distributed.get_world_size()
            if dist_ready
            else int(os.environ.get("WORLD_SIZE", "1"))
        )
    )
    local_rank = (
        int(explicit_local_rank)
        if explicit_local_rank is not None
        else (
            int(os.environ["LOCAL_RANK"])
            if "LOCAL_RANK" in os.environ
            else torch.cuda.current_device() if torch.cuda.is_available() else rank
        )
    )

    if explicit_tp_rank is not None:
        tp_rank = int(explicit_tp_rank)
    else:
        try:
            from sglang.srt.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
            )

            tp_rank = get_tensor_model_parallel_rank()
        except (AssertionError, ImportError, RuntimeError):
            tp_rank = int(os.environ.get("TP_RANK", str(rank)))

    return rank, local_rank, tp_rank, world_size


def _tp_world_size(default: int) -> int:
    explicit = os.environ.get("SGLANG_K3_CAPTURE_TP_WORLD_SIZE")
    if explicit is not None:
        return max(1, int(explicit))
    try:
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )

        return max(1, int(get_tensor_model_parallel_world_size()))
    except (AssertionError, ImportError, RuntimeError):
        return max(1, default)


class ReplayTensorCapture:
    def __init__(self) -> None:
        plan_path = os.environ.get("SGLANG_REPLAY_CAPTURE_PLAN", "").strip()
        plan: dict[str, Any] = {}
        if plan_path:
            with Path(plan_path).open(encoding="utf-8") as handle:
                plan = json.load(handle)
            if int(plan.get("format_version", 0)) != 1:
                raise ValueError("unsupported replay capture plan format_version")
        capture_dir = (
            os.environ.get("SGLANG_REPLAY_CAPTURE_DIR")
            or os.environ.get("SGLANG_K3_CAPTURE_DIR", "")
        ).strip()
        self.enabled = bool(capture_dir)
        if self.enabled and os.environ.get("SGLANG_HOTLOOP_PROFILE", "0").lower() in {
            "1",
            "true",
            "yes",
        }:
            raise ValueError(
                "replay capture and production hotloop profiling are mutually exclusive"
            )
        self.root = Path(capture_dir) if capture_dir else None
        arm_file = os.environ.get("SGLANG_K3_CAPTURE_ARM_FILE", "").strip()
        self.arm_file = Path(arm_file) if arm_file else None
        if self.enabled and plan_path and self.arm_file is None:
            raise ValueError(
                "replay capture starts unarmed; set SGLANG_K3_CAPTURE_ARM_FILE "
                "to a local sentinel path"
            )
        if self.root is not None and self.arm_file is not None:
            try:
                self.arm_file.resolve().relative_to(self.root.resolve())
            except ValueError:
                pass
            else:
                raise ValueError(
                    "SGLANG_K3_CAPTURE_ARM_FILE must not live under the capture "
                    "destination; use a local sentinel such as /tmp/sglang-capture-arm"
                )
        self.rank, self.local_rank, self.tp_rank, self.world_size = _parallel_identity()
        self.tp_world_size = _tp_world_size(self.world_size)
        self.default_max_rows = max(
            1, int(os.environ.get("SGLANG_K3_CAPTURE_MAX_ROWS", "512"))
        )
        # Per-rank hard ceiling.  Eight GiB keeps a TP8 supplement below 64 GiB
        # before compression; operators can lower it for narrow puzzle plans.
        max_gib = float(
            os.environ.get(
                "SGLANG_REPLAY_CAPTURE_MAX_GIB",
                str(plan.get("max_gib_per_rank", 8)),
            )
        )
        self.max_bytes = max(1, int(max_gib * 2**30))
        self._bytes_reserved = 0
        self.layers = _parse_layers(os.environ.get("SGLANG_K3_CAPTURE_LAYERS", "all"))
        plan_phases = plan.get("phases", "all")
        if isinstance(plan_phases, list):
            plan_phases = ",".join(str(phase) for phase in plan_phases)
        raw_phases = (
            os.environ.get("SGLANG_K3_CAPTURE_PHASES", str(plan_phases)).strip().lower()
        )
        self.phases = (
            None
            if not raw_phases or raw_phases == "all"
            else {phase.strip() for phase in raw_phases.split(",") if phase.strip()}
        )
        plan_operations = plan.get("operations", [])
        plan_points = plan.get("points", [])
        configured_points = [f"bundle.{name}" for name in plan_operations]
        configured_points.extend(str(name) for name in plan_points)
        extra_points = (
            os.environ.get("SGLANG_REPLAY_CAPTURE_POINTS")
            or os.environ.get("SGLANG_K3_CAPTURE_POINTS", "")
        ).strip()
        if extra_points:
            configured_points.extend(
                point.strip() for point in extra_points.split(",") if point.strip()
            )
        raw_points = ",".join(configured_points) if configured_points else "all"
        self.all_points = not raw_points or raw_points.lower() == "all"
        self.points = (
            set(_DEFAULT_POINTS)
            if self.all_points
            else {point.strip() for point in raw_points.split(",") if point.strip()}
        )
        rank_filter = (
            os.environ.get("SGLANG_K3_CAPTURE_RANKS", str(plan.get("ranks", "0")))
            .strip()
            .lower()
        )
        if (
            "collective.tp_residual" in plan_operations
            or bool(plan.get("require_all_ranks", False))
        ) and rank_filter != "all":
            raise ValueError("this replay plan requires SGLANG_K3_CAPTURE_RANKS=all")
        self.rank_allowed = rank_filter == "all" or self.rank in {
            int(item) for item in rank_filter.split(",") if item.strip()
        }
        self._rows: dict[tuple[str, int], int] = {}
        self._plan_row_limits = {
            f"bundle.{name}": int(limit)
            for name, limit in plan.get("max_rows_per_operation", {}).items()
        }
        self._plan_row_limits.update(
            {
                str(name): int(limit)
                for name, limit in plan.get("max_rows_per_point", {}).items()
            }
        )
        self._rows_by_phase: dict[tuple[str, int, str], int] = {}
        self._events: dict[tuple[str, int], int] = {}
        self._calls: dict[tuple[str, int], int] = {}
        self._expert_counts: dict[tuple[int, str], torch.Tensor] = {}
        self._lock = threading.Lock()
        self._process_dir: Path | None = None
        self._manifest: Path | None = None
        self._context: dict[str, Any] = {}
        self.async_enabled = self.enabled and os.environ.get(
            "SGLANG_K3_CAPTURE_ASYNC", "0"
        ).lower() in {"1", "true", "yes"}
        local_dir = os.environ.get("SGLANG_K3_CAPTURE_LOCAL_DIR", "").strip()
        if self.async_enabled and plan_path and not local_dir:
            raise ValueError(
                "async replay capture requires SGLANG_K3_CAPTURE_LOCAL_DIR "
                "on local NVMe"
            )
        self._configured_local_root = Path(local_dir) if local_dir else self.root
        self.shard_target_bytes = int(
            float(os.environ.get("SGLANG_K3_CAPTURE_SHARD_MB", "256")) * 2**20
        )
        self.expert_quota = max(
            0, int(os.environ.get("SGLANG_K3_CAPTURE_EXPERT_QUOTA", "0"))
        )
        self.num_experts = max(
            1, int(os.environ.get("SGLANG_K3_CAPTURE_NUM_EXPERTS", "896"))
        )
        self._copy_stream: Any = None
        self._pending_queue: queue.Queue[Any] | None = None
        self._upload_queue: queue.Queue[Any] | None = None
        self._writer_thread: threading.Thread | None = None
        self._uploader_thread: threading.Thread | None = None
        self._async_error: BaseException | None = None
        self._closed = False
        self._local_process_dir: Path | None = None
        self._local_manifest: Path | None = None
        self._shard_seq = 0
        self.delete_local_after_upload = os.environ.get(
            "SGLANG_K3_CAPTURE_DELETE_LOCAL_AFTER_UPLOAD", "1"
        ).lower() in {"1", "true", "yes"}
        self.split_rows_across_ranks = (
            os.environ.get("SGLANG_K3_CAPTURE_SPLIT_ROWS_ACROSS_RANKS", "0").lower()
            in {
                "1",
                "true",
                "yes",
            }
            or "collective.tp_residual" in plan_operations
        )

        if self.enabled and self.rank_allowed:
            assert self.root is not None
            self._process_dir = self.root / f"rank-{self.rank:05d}"
            _claim_new_directory(self._process_dir, "capture rank directory")
            self._manifest = self._process_dir / "manifest.jsonl"
            run_record = {
                "record_type": "run",
                "format_version": 1,
                "created_unix_ns": time.time_ns(),
                "rank": self.rank,
                "local_rank": self.local_rank,
                "tp_rank": self.tp_rank,
                "world_size": self.world_size,
                "run_id": os.environ.get("SGLANG_K3_CAPTURE_RUN_ID"),
                "model_revision": os.environ.get("SGLANG_K3_CAPTURE_MODEL_REVISION"),
                "draft_model_revision": os.environ.get(
                    "SGLANG_DSPARK_CAPTURE_DRAFT_REVISION"
                ),
                "tokenizer_revision": os.environ.get(
                    "SGLANG_DSPARK_CAPTURE_TOKENIZER_REVISION"
                ),
                "chat_template_sha256": os.environ.get(
                    "SGLANG_DSPARK_CAPTURE_CHAT_TEMPLATE_SHA256"
                ),
                "sglang_revision": os.environ.get("SGLANG_K3_CAPTURE_SGLANG_REVISION"),
                "corpus_sha256": os.environ.get("SGLANG_K3_CAPTURE_CORPUS_SHA256"),
                "sampling_seed": os.environ.get("SGLANG_K3_CAPTURE_SAMPLING_SEED"),
                "points": sorted(self.points),
                "layers": "all" if self.layers is None else sorted(self.layers),
                "phases": "all" if self.phases is None else sorted(self.phases),
                "default_max_rows": self.default_max_rows,
                "async_capture": self.async_enabled,
                "shard_target_bytes": self.shard_target_bytes,
                "expert_quota": self.expert_quota,
                "max_bytes": self.max_bytes,
            }
            with self._manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(run_record, sort_keys=True) + "\n")
            if self.async_enabled:
                assert self._configured_local_root is not None
                local_root = self._configured_local_root
                self._local_process_dir = local_root / f"rank-{self.rank:05d}"
                if self._local_process_dir != self._process_dir:
                    _claim_new_directory(
                        self._local_process_dir, "local staging rank directory"
                    )
                self._local_manifest = self._local_process_dir / "manifest.jsonl"
                if self._local_manifest != self._manifest:
                    with self._local_manifest.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(run_record, sort_keys=True) + "\n")
                max_pending = max(
                    1, int(os.environ.get("SGLANG_K3_CAPTURE_MAX_PENDING", "8"))
                )
                self._pending_queue = queue.Queue(maxsize=max_pending)
                self._upload_queue = queue.Queue(maxsize=2)
                if torch.cuda.is_available():
                    self._copy_stream = torch.cuda.Stream(device=self.local_rank)
                self._writer_thread = threading.Thread(
                    target=self._writer_loop,
                    name=f"k3-capture-writer-r{self.rank}",
                    daemon=True,
                )
                self._uploader_thread = threading.Thread(
                    target=self._uploader_loop,
                    name=f"k3-capture-uploader-r{self.rank}",
                    daemon=True,
                )
                self._writer_thread.start()
                self._uploader_thread.start()
                atexit.register(self.close)

    def set_forward_context(self, forward_batch: Any) -> None:
        """Attach phase/cache strata to all capture points in this forward."""
        forward_mode = getattr(forward_batch, "forward_mode", "unknown")
        mode_name = getattr(forward_mode, "name", None)
        mode = str(mode_name if mode_name is not None else forward_mode).lower()
        decode_checks = (
            getattr(forward_mode, "is_decode", None),
            getattr(forward_mode, "is_target_verify", None),
            getattr(forward_mode, "is_draft_extend_v2", None),
        )
        decode_like = any(bool(check()) for check in decode_checks if callable(check))
        phase = (
            "decode"
            if decode_like
            or any(tag in mode for tag in ("decode", "target_verify", "draft_extend"))
            else "prefill"
        )
        prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        seq_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        prefix_lens = [] if prefix_lens is None else prefix_lens
        seq_lens = [] if seq_lens is None else seq_lens
        max_prefix = max((int(x) for x in prefix_lens), default=0)
        max_seq = max((int(x) for x in seq_lens), default=0)
        forward_iter = getattr(forward_batch, "forward_iter", -1)
        self._context = {
            "phase": phase,
            "forward_mode": mode,
            # Runtime state is deliberately distinct from corpus grouping.  A
            # request tagged for prefix reuse can still be the group's first
            # (cache-miss) member.
            "runtime_prefix_cache": "hit" if max_prefix > 0 else "miss",
            "prefix_cache": "hit" if max_prefix > 0 else "miss",
            "corpus_metadata": _jsonable(
                getattr(forward_batch, "capture_corpus_metadata", None)
            ),
            "max_prefix_len": max_prefix,
            "max_sequence_len": max_seq,
            "batch_size": int(getattr(forward_batch, "batch_size", 0) or 0),
            "forward_id": -1 if forward_iter is None else int(forward_iter),
            "tp_rank": self.tp_rank,
        }

    def _phase(self, metadata: Mapping[str, Any] | None) -> str:
        value = str((metadata or {}).get("phase", self._context.get("phase", "all")))
        return value.lower()

    def _check_async_error(self) -> None:
        if self._async_error is not None:
            raise RuntimeError(
                "K3 asynchronous capture worker failed"
            ) from self._async_error

    def _append_manifest(self, path: Path, record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _writer_loop(self) -> None:
        from safetensors.torch import save_file

        pending: list[_PendingCapture] = []
        pending_bytes = 0

        def flush() -> None:
            nonlocal pending, pending_bytes
            if not pending:
                return
            assert self._local_process_dir is not None
            assert self._local_manifest is not None
            tensors: dict[str, torch.Tensor] = {}
            entries: list[dict[str, Any]] = []
            for entry_idx, item in enumerate(pending):
                if item.event is not None:
                    item.event.synchronize()
                tensor_keys = {}
                for name, value in item.tensors.items():
                    key = f"e{entry_idx:05d}__{name}"
                    tensors[key] = value
                    tensor_keys[name] = key
                record = dict(item.record)
                record["tensor_keys"] = tensor_keys
                entries.append(record)
            name = f"shard-{self._shard_seq:06d}.safetensors"
            self._shard_seq += 1
            final_path = self._local_process_dir / name
            tmp_path = self._local_process_dir / f".{name}.{os.getpid()}.tmp"
            save_file(tensors, str(tmp_path))
            os.replace(tmp_path, final_path)
            shard_record = {
                "record_type": "shard",
                "format_version": 2,
                "created_unix_ns": time.time_ns(),
                "rank": self.rank,
                "file": name,
                "size": final_path.stat().st_size,
                "sha256": _sha256_file(final_path),
                "entries": entries,
            }
            self._append_manifest(self._local_manifest, shard_record)
            assert self._upload_queue is not None
            self._upload_queue.put(_CompletedShard(final_path, shard_record))
            pending = []
            pending_bytes = 0

        try:
            assert self._pending_queue is not None
            while True:
                try:
                    item = self._pending_queue.get(timeout=2.0)
                except queue.Empty:
                    flush()
                    continue
                if item is None:
                    flush()
                    self._pending_queue.task_done()
                    break
                pending.append(item)
                pending_bytes += item.nbytes
                self._pending_queue.task_done()
                if pending_bytes >= self.shard_target_bytes:
                    flush()
        except BaseException as exc:
            self._async_error = exc
        finally:
            if self._upload_queue is not None:
                self._upload_queue.put(None)

    def _uploader_loop(self) -> None:
        try:
            assert self._upload_queue is not None
            assert self._process_dir is not None and self._manifest is not None
            while True:
                item = self._upload_queue.get()
                if item is None:
                    self._upload_queue.task_done()
                    break
                assert isinstance(item, _CompletedShard)
                destination = self._process_dir / item.local_path.name
                if destination != item.local_path:
                    partial = self._process_dir / (
                        f".{item.local_path.name}.{os.getpid()}.uploading"
                    )
                    shutil.copyfile(item.local_path, partial)
                    os.replace(partial, destination)
                    self._append_manifest(self._manifest, item.record)
                    if self.delete_local_after_upload:
                        item.local_path.unlink()
                self._upload_queue.task_done()
        except BaseException as exc:
            self._async_error = exc

    def close(self) -> None:
        if self._closed or not self.enabled or not self.rank_allowed:
            return
        self._closed = True
        if self.async_enabled:
            assert self._pending_queue is not None
            self._pending_queue.put(None)
            if self._writer_thread is not None:
                self._writer_thread.join()
            if self._uploader_thread is not None:
                self._uploader_thread.join()
            self._check_async_error()

        quotas = []
        if self.expert_quota > 0:
            for (layer_idx, phase), counts in sorted(self._expert_counts.items()):
                values = counts.to("cpu").tolist()
                quotas.append(
                    {
                        "layer": layer_idx,
                        "phase": phase,
                        "quota": self.expert_quota,
                        "experts_at_quota": sum(
                            value >= self.expert_quota for value in values
                        ),
                        "min_assignments": min(values),
                        "max_assignments": max(values),
                        "assignments": values,
                    }
                )
        assert self._process_dir is not None
        _atomic_json(
            self._process_dir / "capture-close.json",
            {
                "record_type": "capture_close",
                "format_version": 2,
                "created_unix_ns": time.time_ns(),
                "rank": self.rank,
                "world_size": self.world_size,
                "bytes_reserved": self._bytes_reserved,
                "expert_quotas": quotas,
                "complete": True,
            },
        )

    def path_wants(self, point: str, layer_idx: int) -> bool:
        """Whether every TP rank must take the capture-compatible code path."""
        return self.configured(point, layer_idx) and (
            self.arm_file is None or self.arm_file.exists()
        )

    def configured(self, point: str, layer_idx: int) -> bool:
        """Whether a point is selected, independent of the runtime arm gate.

        Model construction uses this for capture paths that require persistent
        structural configuration. Serialization must continue to use
        :meth:`wants`/``path_wants`` so an unarmed process writes nothing.
        """
        return (
            self.enabled
            and point in self.points
            and (layer_idx < 0 or self.layers is None or layer_idx in self.layers)
        )

    def wants(self, point: str, layer_idx: int) -> bool:
        """Whether this rank should serialize the requested capture point."""
        return self.rank_allowed and self.path_wants(point, layer_idx)

    def _row_limit(self, point: str, phase: str = "all") -> int:
        env_name = "SGLANG_K3_CAPTURE_MAX_ROWS_" + point.upper().replace(
            "-", "_"
        ).replace(".", "_")
        phase_name = f"{env_name}_{phase.upper()}"
        default = self._plan_row_limits.get(point, self.default_max_rows)
        value = os.environ.get(phase_name, os.environ.get(env_name, str(default)))
        global_limit = max(1, int(value))
        if self.split_rows_across_ranks and self.rank_allowed:
            # The configured target is aggregate across TP, not per rank.
            return max(
                0,
                (global_limit + self.tp_world_size - 1 - self.tp_rank)
                // self.tp_world_size,
            )
        return global_limit

    def _fit_byte_budget(
        self,
        present: Mapping[str, torch.Tensor],
        row_count: int,
        take: int,
        once: bool,
    ) -> int:
        """Shrink a row-consistent bundle to the remaining hard byte budget."""
        fixed = 0
        per_row = 0
        for value in present.values():
            nbytes = value.numel() * value.element_size()
            if not once and value.ndim > 0 and int(value.shape[0]) == row_count:
                per_row += nbytes // max(1, row_count)
            else:
                fixed += nbytes
        remaining = self.max_bytes - self._bytes_reserved
        if fixed > remaining:
            return 0
        if per_row == 0:
            fitted = take
        else:
            fitted = min(take, (remaining - fixed) // per_row)
        if fitted <= 0:
            return 0
        self._bytes_reserved += fixed + fitted * per_row
        return int(fitted)

    def remaining_rows(
        self, point: str, layer_idx: int, phase: str | None = None
    ) -> int:
        if not self.wants(point, layer_idx):
            return 0
        resolved_phase = phase or str(self._context.get("phase", "all"))
        if self.phases is not None and resolved_phase.lower() not in self.phases:
            return 0
        key = (point, layer_idx, resolved_phase)
        return max(
            0,
            self._row_limit(point, resolved_phase) - self._rows_by_phase.get(key, 0),
        )

    def _gpu_sample_indices(
        self, row_count: int, take: int, device: torch.device
    ) -> torch.Tensor:
        """Deterministic GPU sampling, interleaved across TP-rank strata."""
        if take >= row_count:
            return torch.arange(row_count, device=device, dtype=torch.long)
        denominator = take * self.tp_world_size
        strata = (
            torch.arange(take, device=device, dtype=torch.long) * self.tp_world_size
            + self.tp_rank
        )
        return torch.div(
            strata * row_count + denominator // 2,
            denominator,
            rounding_mode="floor",
        ).clamp_max_(row_count - 1)

    def _expert_sample_indices(
        self,
        layer_idx: int,
        phase: str,
        topk_ids: torch.Tensor,
        take: int,
    ) -> torch.Tensor:
        key = (layer_idx, phase)
        counts = self._expert_counts.get(key)
        if counts is None:
            counts = torch.zeros(
                self.num_experts, dtype=torch.int32, device=topk_ids.device
            )
            self._expert_counts[key] = counts
        valid_ids = topk_ids.clamp(0, self.num_experts - 1).to(torch.long)
        deficits = (self.expert_quota - counts).clamp_min_(0)
        scores = deficits[valid_ids].amax(dim=1)
        candidates = torch.nonzero(scores > 0, as_tuple=False).flatten()
        if candidates.numel() > take:
            chosen = self._gpu_sample_indices(
                int(candidates.numel()), take, candidates.device
            )
            candidates = candidates.index_select(0, chosen)
        selected_ids = valid_ids.index_select(0, candidates).flatten()
        counts.add_(
            torch.bincount(selected_ids, minlength=self.num_experts).to(counts.dtype)
        ).clamp_max_(self.expert_quota)
        return candidates

    def _capture_async(
        self,
        point: str,
        layer_idx: int,
        present: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any] | None,
        once: bool,
        row_count: int,
        take: int,
        event: int,
        phase: str,
        indices: torch.Tensor | None = None,
        preserve_rows: bool = False,
    ) -> Path:
        self._check_async_error()
        first = next((value for value in present.values() if value.ndim > 0), None)
        device = first.device if first is not None else torch.device("cpu")
        if indices is None:
            indices = self._gpu_sample_indices(row_count, take, device)

        cpu_tensors: dict[str, torch.Tensor] = {}
        tensor_meta: dict[str, Any] = {}
        current_stream = (
            torch.cuda.current_stream(device) if device.type == "cuda" else None
        )
        copy_event = None
        # Materialize sampled tensors on the model stream.  Selecting on the
        # copy stream would let the model stream immediately reuse or mutate
        # the source buffers while index_select was still reading them.
        prepared: dict[str, torch.Tensor] = {}
        for name, value in present.items():
            selected = (
                value.index_select(0, indices)
                if not once
                and not preserve_rows
                and value.ndim > 0
                and value.shape[0] == row_count
                # contiguous() may alias an already-contiguous workspace.  A
                # real producer-stream clone is required before the model can
                # reuse it while the copy stream is still transferring.
                else value.detach().contiguous().clone()
            )
            prepared[name] = selected.contiguous()
        if self._copy_stream is not None and current_stream is not None:
            self._copy_stream.wait_stream(current_stream)
        stream_context = (
            torch.cuda.stream(self._copy_stream)
            if self._copy_stream is not None and current_stream is not None
            else torch.no_grad()
        )
        with stream_context:
            for name, value in present.items():
                original_shape = list(value.shape)
                original_stride = list(value.stride())
                selected = prepared[name]
                if selected.device.type == "cuda":
                    host = torch.empty_like(selected, device="cpu", pin_memory=True)
                    host.copy_(selected, non_blocking=True)
                    selected.record_stream(self._copy_stream)
                else:
                    host = selected.detach().contiguous().to("cpu").clone()
                cpu_tensors[name] = host
                tensor_meta[name] = {
                    "shape": original_shape,
                    "stride": original_stride,
                    "saved_shape": list(host.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "device": str(value.device),
                    "saved_nbytes": host.numel() * host.element_size(),
                }
            if self._copy_stream is not None and current_stream is not None:
                copy_event = torch.cuda.Event()
                copy_event.record(self._copy_stream)

        merged_metadata = {**self._context, **dict(metadata or {})}
        record = {
            "point": point,
            "layer": layer_idx,
            "event": event,
            "rows": take,
            "rows_total": self._rows[(point, layer_idx)],
            "rows_phase_total": self._rows_by_phase[(point, layer_idx, phase)],
            "phase": phase,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "tp_rank": self.tp_rank,
            "tensors": tensor_meta,
            "metadata": _jsonable(merged_metadata),
        }
        nbytes = sum(
            value.numel() * value.element_size() for value in cpu_tensors.values()
        )
        assert self._pending_queue is not None
        self._pending_queue.put(
            _PendingCapture(copy_event, cpu_tensors, record, nbytes)
        )
        return Path(f"queued://rank-{self.rank}/{point}/l{layer_idx}/e{event}")

    def capture(
        self,
        point: str,
        layer_idx: int,
        tensors: Mapping[str, torch.Tensor | None],
        *,
        metadata: Mapping[str, Any] | None = None,
        once: bool = False,
        preserve_rows: bool = False,
    ) -> Path | None:
        """Write one bounded tensor bundle and return its path.

        The first non-scalar tensor defines the row count. Tensors with the
        same leading dimension are sliced consistently; static tensors (for
        example correction bias) are retained in full. ``preserve_rows`` makes
        the complete bundle one budgeted event, which is required for token
        sequences whose continuity would be destroyed by row sampling.
        """
        if not self.wants(point, layer_idx):
            return None
        key = (point, layer_idx)
        with self._lock:
            self._check_async_error()
            call_index = self._calls.get(key, 0)
            self._calls[key] = call_index + 1
            if once and self._events.get(key, 0) > 0:
                return None
            phase = self._phase(metadata)
            phase_key = (point, layer_idx, phase)
            remaining = self.remaining_rows(point, layer_idx, phase)
            if remaining <= 0:
                return None

            present = {
                name: value for name, value in tensors.items() if value is not None
            }
            row_count = (
                1
                if once or preserve_rows
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
            if (
                self.async_enabled
                and self.split_rows_across_ranks
                and not once
                and not preserve_rows
            ):
                if row_count < self.tp_world_size:
                    if call_index % self.tp_world_size != self.tp_rank:
                        return None
                else:
                    take = min(
                        take,
                        (row_count + self.tp_world_size - 1) // self.tp_world_size,
                    )
            if take <= 0:
                return None

            take = self._fit_byte_budget(present, row_count, take, once)
            if take <= 0:
                return None

            indices = None
            if (
                self.async_enabled
                and point == "expert_output"
                and self.expert_quota > 0
                and "topk_ids" in present
            ):
                indices = self._expert_sample_indices(
                    layer_idx, phase, present["topk_ids"], take
                )
                take = int(indices.numel())
                if take <= 0:
                    return None

            event = self._events.get(key, 0)
            self._events[key] = event + 1
            self._rows[key] = self._rows.get(key, 0) + take
            self._rows_by_phase[phase_key] = (
                self._rows_by_phase.get(phase_key, 0) + take
            )

            if self.async_enabled:
                return self._capture_async(
                    point,
                    layer_idx,
                    present,
                    metadata,
                    once,
                    row_count,
                    take,
                    event,
                    phase,
                    indices,
                    preserve_rows,
                )

            cpu_tensors: dict[str, torch.Tensor] = {}
            tensor_meta: dict[str, Any] = {}
            for name, value in present.items():
                original_shape = list(value.shape)
                original_stride = list(value.stride())
                selected = (
                    value[:take]
                    if not once
                    and not preserve_rows
                    and value.ndim > 0
                    and value.shape[0] == row_count
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


# Backward-compatible name for the existing K3 call sites and launch scripts.
K3TensorCapture = ReplayTensorCapture


_CAPTURE: ReplayTensorCapture | None = None


def get_replay_tensor_capture() -> ReplayTensorCapture:
    global _CAPTURE
    if _CAPTURE is None:
        _CAPTURE = ReplayTensorCapture()
    return _CAPTURE


get_k3_tensor_capture = get_replay_tensor_capture


def k3_capture_wants(point: str, layer_idx: int) -> bool:
    # This predicate controls graph structure in K3's router.  Every TP rank
    # must choose the same fused/unfused path even when only one rank writes.
    return get_replay_tensor_capture().path_wants(point, layer_idx)


def k3_capture_configured(point: str, layer_idx: int) -> bool:
    return get_replay_tensor_capture().configured(point, layer_idx)


def k3_capture_remaining_rows(point: str, layer_idx: int) -> int:
    return get_replay_tensor_capture().remaining_rows(point, layer_idx)


def k3_capture_set_forward_context(forward_batch: Any) -> None:
    get_replay_tensor_capture().set_forward_context(forward_batch)


def k3_capture_close() -> None:
    get_replay_tensor_capture().close()


def k3_capture(
    point: str,
    layer_idx: int,
    tensors: Mapping[str, torch.Tensor | None],
    *,
    metadata: Mapping[str, Any] | None = None,
    once: bool = False,
    preserve_rows: bool = False,
) -> Path | None:
    return get_replay_tensor_capture().capture(
        point,
        layer_idx,
        tensors,
        metadata=metadata,
        once=once,
        preserve_rows=preserve_rows,
    )
