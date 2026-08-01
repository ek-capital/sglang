import hashlib
import importlib.util
import json
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/debug_utils/k3_tensor_capture.py"
)
_SPEC = importlib.util.spec_from_file_location("k3_tensor_capture", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
K3TensorCapture = _MODULE.K3TensorCapture


def test_k3_tensor_capture_is_bounded_and_hashed(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_K3_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_POINTS", "routing,routing_static")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_LAYERS", "1-2")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_MAX_ROWS", "3")

    capture = K3TensorCapture()
    assert capture.wants("routing", 1)
    assert not capture.wants("routing", 0)

    logits = torch.arange(20, dtype=torch.float32).view(5, 4)
    hidden = torch.arange(30, dtype=torch.bfloat16).view(5, 6)
    capture.capture("routing", 1, {"logits": logits, "hidden": hidden})
    assert capture.capture("routing", 1, {"logits": logits}) is None

    capture.capture(
        "routing_static",
        1,
        {"bias": torch.arange(4, dtype=torch.float32)},
        once=True,
    )
    assert (
        capture.capture(
            "routing_static",
            1,
            {"bias": torch.arange(4, dtype=torch.float32)},
            once=True,
        )
        is None
    )

    rank_dir = tmp_path / "rank-00000"
    records = [
        json.loads(line)
        for line in (rank_dir / "manifest.jsonl").read_text().splitlines()
    ]
    captures = [record for record in records if record["record_type"] == "capture"]
    assert [record["point"] for record in captures] == ["routing", "routing_static"]
    assert captures[0]["rows"] == 3

    shard = rank_dir / captures[0]["file"]
    assert hashlib.sha256(shard.read_bytes()).hexdigest() == captures[0]["sha256"]
    saved = load_file(shard)
    torch.testing.assert_close(saved["logits"], logits[:3])
    torch.testing.assert_close(saved["hidden"], hidden[:3])


def test_k3_tensor_capture_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_K3_CAPTURE_DIR", raising=False)
    capture = K3TensorCapture()
    assert not capture.enabled
    assert capture.capture("routing", 0, {"x": torch.ones(1)}) is None


def test_k3_tensor_capture_waits_for_arm_file(tmp_path, monkeypatch):
    arm_file = tmp_path / "ARMED"
    monkeypatch.setenv("SGLANG_K3_CAPTURE_DIR", str(tmp_path / "capture"))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_ARM_FILE", str(arm_file))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_POINTS", "layer_input")

    capture = K3TensorCapture()
    assert not capture.wants("layer_input", 0)
    assert capture.capture("layer_input", 0, {"x": torch.ones(1)}) is None

    arm_file.touch()
    assert capture.wants("layer_input", 0)
    assert capture.capture("layer_input", 0, {"x": torch.ones(1)}) is not None


def test_k3_tensor_capture_uses_live_distributed_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_K3_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_RANKS", "3")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("TP_RANK", raising=False)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    capture = K3TensorCapture()

    assert capture.rank == 3
    assert capture.local_rank == 3
    assert capture.tp_rank == 3
    assert capture.world_size == 8
    assert capture.rank_allowed
    assert (tmp_path / "rank-00003" / "manifest.jsonl").is_file()


def test_capture_path_is_tp_uniform_but_writes_remain_rank_filtered(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SGLANG_K3_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_POINTS", "routing")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_RANKS", "0")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_RANK", "5")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_WORLD_SIZE", "8")

    capture = K3TensorCapture()

    assert capture.path_wants("routing", 1)
    assert not capture.wants("routing", 1)
    assert capture.capture("routing", 1, {"x": torch.ones(1)}) is None
    assert not (tmp_path / "rank-00005").exists()


def test_phase_quotas_are_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_K3_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_MAX_ROWS_LAYER_INPUT_PREFILL", "3")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_MAX_ROWS_LAYER_INPUT_DECODE", "2")
    capture = K3TensorCapture()

    capture.set_forward_context(
        SimpleNamespace(
            forward_mode="EXTEND",
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[10],
            batch_size=1,
        )
    )
    capture.capture("layer_input", 0, {"x": torch.arange(5).view(5, 1)})
    assert capture.remaining_rows("layer_input", 0) == 0

    capture.set_forward_context(
        SimpleNamespace(
            forward_mode="DECODE",
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=None,
            batch_size=1,
        )
    )
    assert capture.remaining_rows("layer_input", 0) == 2
    capture.capture("layer_input", 0, {"x": torch.arange(5).view(5, 1)})
    assert capture.remaining_rows("layer_input", 0) == 0


def test_numeric_forward_mode_uses_decode_predicate(tmp_path, monkeypatch):
    class NumericForwardMode(IntEnum):
        EXTEND = 1
        DECODE = 2

        def is_decode(self):
            return self is NumericForwardMode.DECODE

    monkeypatch.setenv("SGLANG_K3_CAPTURE_DIR", str(tmp_path))
    capture = K3TensorCapture()

    capture.set_forward_context(
        SimpleNamespace(
            forward_mode=NumericForwardMode.DECODE,
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=None,
            batch_size=1,
        )
    )

    assert capture._context["phase"] == "decode"
    assert capture._context["forward_mode"] == "decode"


def test_async_capture_packs_and_uploads_composite_shard(tmp_path, monkeypatch):
    persistent = tmp_path / "persistent"
    local = tmp_path / "local"
    monkeypatch.setenv("SGLANG_K3_CAPTURE_DIR", str(persistent))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_LOCAL_DIR", str(local))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_ASYNC", "1")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_SHARD_MB", "0.001")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_DELETE_LOCAL_AFTER_UPLOAD", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    capture = K3TensorCapture()

    tensor = torch.arange(4096, dtype=torch.float32).view(1024, 4)
    capture.capture("layer_input", 0, {"hidden_states": tensor})
    capture.capture("layer_output", 0, {"hidden_states": tensor + 1})
    capture.close()

    manifest = persistent / "rank-00000" / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    shards = [record for record in records if record["record_type"] == "shard"]
    assert shards
    assert sum(len(record["entries"]) for record in shards) == 2
    for record in shards:
        path = manifest.parent / record["file"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert not list((local / "rank-00000").glob("*.safetensors"))


def test_expert_quota_sampling_records_assignment_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_K3_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_ASYNC", "1")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_POINTS", "expert_output")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_EXPERT_QUOTA", "2")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_NUM_EXPERTS", "4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    capture = K3TensorCapture()
    topk_ids = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
    values = torch.arange(16, dtype=torch.float32).view(4, 4)

    capture.capture(
        "expert_output",
        1,
        {"topk_ids": topk_ids, "expert_output": values},
        metadata={"phase": "prefill"},
    )
    capture.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-00000" / "manifest.jsonl").read_text().splitlines()
    ]
    quota = next(record for record in records if record["record_type"] == "expert_quota")
    assert quota["experts_at_quota"] == 4
    assert quota["min_assignments"] == 2


def test_gpu_sampling_interleaves_tp_rank_strata(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_K3_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_RANK", "3")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_WORLD_SIZE", "8")
    capture = K3TensorCapture()
    indices = capture._gpu_sample_indices(800, 10, torch.device("cpu"))
    assert indices.tolist() == [30, 110, 190, 270, 350, 430, 510, 590, 670, 750]


def test_single_row_decode_calls_are_round_robined_across_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_K3_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_K3_CAPTURE_ASYNC", "1")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_SPLIT_ROWS_ACROSS_RANKS", "1")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_RANK", "1")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_WORLD_SIZE", "2")
    monkeypatch.setenv("SGLANG_K3_CAPTURE_RANKS", "all")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    capture = K3TensorCapture()
    value = torch.ones(1, 4)

    assert capture.capture("layer_input", 0, {"x": value}) is None
    assert capture.capture("layer_input", 0, {"x": value}) is not None
    capture.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-00001" / "manifest.jsonl").read_text().splitlines()
    ]
    entries = [
        entry
        for record in records
        if record["record_type"] == "shard"
        for entry in record["entries"]
    ]
    assert len(entries) == 1
    assert entries[0]["rows"] == 1
