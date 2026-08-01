import importlib.util
from pathlib import Path

import sglang.srt.layers.quantization.kimi_k3_w2a16 as k3_quant
import torch
from sglang.srt.layers.quantization.kimi_k3_w2a16 import (
    KimiK3W2A16Config,
    dequantize_w2,
    pack_w2,
    pack_w4,
    unpack_w2,
    unpack_w4,
)

SCRIPT = Path(__file__).parents[5] / "scripts" / "kimi_k3_quantize_w2.py"
SPEC = importlib.util.spec_from_file_location("kimi_k3_quantize_w2", SCRIPT)
QUANTIZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(QUANTIZER)


def test_w2_pack_roundtrip():
    codes = torch.tensor([[-2, -1, 0, 1, 1, 0, -1, -2]], dtype=torch.int8)
    assert torch.equal(unpack_w2(pack_w2(codes)), codes)


def test_w2_dequant_groups():
    codes = torch.tensor([[-2, -1, 0, 1] * 2], dtype=torch.int8)
    scales = torch.tensor([[0.5, 2.0]])
    expected = codes.float() * scales.repeat_interleave(4, dim=-1)
    assert torch.equal(dequantize_w2(pack_w2(codes), scales, 4), expected)


def test_w4_pack_roundtrip():
    codes = torch.tensor([[-8, -7, -1, 0, 1, 6, 7, 3]], dtype=torch.int8)
    assert torch.equal(unpack_w4(pack_w4(codes)), codes)


def test_mxfp4_decode_known_levels():
    # low nibble +0.5, high nibble -6.0, scale 2**1
    packed = torch.tensor([[0xF1]], dtype=torch.uint8)
    scales = torch.tensor([[128]], dtype=torch.uint8)
    # Expand to one legal 32-value source block.
    packed = packed.repeat(1, 16)
    decoded = QUANTIZER.dequantize_mxfp4(packed, scales)
    assert decoded[0, 0].item() == 1.0
    assert decoded[0, 1].item() == -12.0


def test_importance_changes_weighted_error_choice():
    weight = torch.tensor([[1.9, 1.0, -1.0, -1.9]], dtype=torch.float32)
    uniform = torch.ones(4)
    important_edge = torch.tensor([1000.0, 1.0, 1.0, 1000.0])
    q0, s0 = QUANTIZER.quantize_importance_w2(weight, uniform, 4)
    q1, s1 = QUANTIZER.quantize_importance_w2(weight, important_edge, 4)
    e0 = ((weight - dequantize_w2(q0, s0, 4)) ** 2 * important_edge).sum()
    e1 = ((weight - dequantize_w2(q1, s1, 4)) ** 2 * important_edge).sum()
    assert e1 <= e0


def test_config_contract():
    cfg = KimiK3W2A16Config.from_config(
        {
            "quant_method": "kimi_k3_w2a16",
            "bits": 2,
            "group_size": 128,
            "w4_channel_fraction": 0.05,
        }
    )
    assert cfg.get_name() == "kimi_k3_w2a16"
    assert cfg.get_min_capability() == 100
    assert cfg.w4_channel_fraction == 0.05


def test_mixed_quant_promotes_equal_tp4_quota():
    torch.manual_seed(7)
    weight = torch.randn(16, 32)
    result = QUANTIZER.quantize_mixed_w2_w4(weight, torch.ones(32), 32, 0.25, 4)
    _q2, _s2, q4, s4, indices = result
    assert indices.tolist() == sorted(indices.tolist())
    assert torch.equal(
        torch.bincount(indices // 4, minlength=4), torch.ones(4, dtype=torch.long)
    )
    correction = unpack_w4(q4).float() * s4.float().repeat_interleave(32, -1)
    assert correction.shape == (4, 32)


def test_tp4_loader_uses_output_major_packed_layout(monkeypatch):
    layer = type(
        "Layer",
        (),
        {"moe_tp_size": 4, "intermediate_size_per_partition": 4},
    )()
    monkeypatch.setattr(
        k3_quant,
        "get_parallel",
        lambda: type("Parallel", (), {"moe_tp_rank": 2})(),
    )
    loader = k3_quant.KimiK3W2A16MoEMethod._get_weight_loader(layer)

    w13 = torch.nn.Parameter(torch.zeros(1, 8, 2), requires_grad=False)
    source_w1 = torch.arange(32).reshape(16, 2)
    source_w3 = source_w1 + 100
    loader(w13, source_w1, "experts.w13_qweight", "w1", 0)
    loader(w13, source_w3, "experts.w13_qweight", "w3", 0)
    assert torch.equal(w13[0, :4], source_w1[8:12])
    assert torch.equal(w13[0, 4:], source_w3[8:12])

    indices = torch.nn.Parameter(
        torch.zeros(1, 4, dtype=torch.int32), requires_grad=False
    )
    source_indices = torch.tensor([0, 1, 4, 5, 8, 9, 12, 13], dtype=torch.int32)
    loader(indices, source_indices, "experts.w13_w4_indices", "w1", 0)
    assert indices[0, :2].tolist() == [0, 1]

    w2 = torch.nn.Parameter(torch.zeros(1, 3, 2), requires_grad=False)
    source_w2 = torch.arange(24).reshape(3, 8)
    loader(w2, source_w2, "experts.w2_qweight", "w2", 0)
    assert torch.equal(w2[0], source_w2[:, 4:6])
