import torch

from sglang.srt.layers.attention.linear.kda_backend import _kda_chunk_transitions


def test_kda_chunk_transitions_do_not_cross_sequence_boundaries():
    # Two packed sequences with 3 and 2 chunks.  h[c] is the state before
    # chunk c; the committed cache rows supply each sequence's final after-state.
    h = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1, 1)
    final_states = torch.tensor([10.0, 20.0]).view(2, 1, 1, 1)

    captured = _kda_chunk_transitions(h, final_states, [130, 65], chunk_size=64)

    assert captured["ssm_state_before"].flatten().tolist() == [0, 1, 2, 3, 4]
    assert captured["ssm_state_after"].flatten().tolist() == [1, 2, 10, 4, 20]
    assert captured["sequence_index"].tolist() == [0, 0, 0, 1, 1]
    assert captured["chunk_index"].tolist() == [0, 1, 2, 0, 1]
    assert captured["token_start"].tolist() == [0, 64, 128, 0, 64]
    assert captured["token_end"].tolist() == [64, 128, 130, 64, 65]
