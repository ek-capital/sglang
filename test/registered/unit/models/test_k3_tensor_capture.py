import hashlib
import importlib.util
import json
from pathlib import Path

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
