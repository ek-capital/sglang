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
    training = capabilities["speculative.dspark_training_window"]
    assert training["policy"] == "sequence"
    assert "target_hidden" in training["required"]
    draft = capabilities["speculative.draft_round"]
    assert draft["version"] == 2
    assert "confidence" in draft["optional"]
    assert "target_logits" in draft["optional"]
    verify = capabilities["attention.kda_target_verify"]
    assert verify["policy"] == "rank_local"
    assert "conv_states_before" in verify["required"]
    assert "ssm_states_before" in verify["required"]
    assert "conv_weights" in verify["required"]
    assert "intermediate_ssm_after" in verify["optional"]
    assert "replayssm_rawv_before" in verify["optional"]


def test_dspark_training_window_shifts_labels_within_each_sequence():
    window = replay_capture.build_dspark_training_window(
        input_ids=torch.tensor([10, 11, 12, 20, 21]),
        positions=torch.tensor([0, 1, 2, 0, 1]),
        target_hidden=torch.zeros(5, 8),
        target_layer_ids=[7, 23],
        sequence_lengths=[3, 2],
        request_pool_indices=torch.tensor([4, 9]),
    )
    assert window["target_ids"].tolist() == [11, 12, -100, 21, -100]
    assert window["loss_mask"].tolist() == [True, True, False, True, False]
    assert window["sequence_offsets"].tolist() == [0, 3, 5]
    assert window["target_layer_ids"].tolist() == [7, 23]


def test_dspark_training_window_rejects_cross_sequence_misalignment():
    with pytest.raises(ValueError, match="packed sequence lengths"):
        replay_capture.build_dspark_training_window(
            input_ids=torch.tensor([10, 11, 12]),
            positions=torch.tensor([0, 1, 2]),
            target_hidden=torch.zeros(3, 8),
            target_layer_ids=[7],
            sequence_lengths=[2],
            request_pool_indices=torch.tensor([4]),
        )


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
