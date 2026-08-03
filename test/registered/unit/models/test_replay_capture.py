# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).parents[4] / "python/sglang/srt/debug_utils/replay_capture.py"
)
_SPEC = importlib.util.spec_from_file_location("replay_capture", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
replay_capture = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = replay_capture
_SPEC.loader.exec_module(replay_capture)


def test_builtin_specs_describe_replayable_hotloop_boundaries():
    capabilities = replay_capture.REGISTRY.capabilities("kimi_k3")
    assert capabilities["speculative.draft_round"]["policy"] == "sequence"
    assert capabilities["residual.attnres"]["policy"] == "paired"
    assert capabilities["collective.tp_residual"]["policy"] == "collective"
    assert "tail_output" in capabilities["moe.tail"]["required"]


def test_bundle_validation_rejects_expensive_partial_capture():
    spec = replay_capture.REGISTRY.get("moe.tail")
    with pytest.raises(ValueError, match="shared_output, tail_output"):
        spec.validate(
            {
                "routed_output": torch.zeros(1, 2),
                "pending_residual": torch.zeros(1, 2),
            }
        )


def test_bundle_validation_rejects_undeclared_tensor():
    spec = replay_capture.REGISTRY.get("moe.shared_mlp")
    with pytest.raises(ValueError, match="undeclared"):
        spec.validate(
            {
                "hidden_states": torch.zeros(1, 2),
                "shared_output": torch.zeros(1, 2),
                "accidental_gigantic_state": torch.zeros(1, 2),
            }
        )


def test_capability_manifest_is_model_filtered(tmp_path):
    path = tmp_path / "capabilities.json"
    replay_capture.write_capability_manifest(str(path), "qwen")
    payload = path.read_text()
    assert "collective.tp_residual" in payload
    assert "moe.tail" not in payload
