import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_attention_source_distance_redteam import sequence_head_writes


class FakeProjection:
    def __init__(self, weight): self.weight = weight


class FakeAttention:
    def __init__(self, weight, heads, head_dimension):
        self.c_proj = FakeProjection(weight); self.n_head = heads; self.head_dim = head_dimension


class FakeBlock:
    def __init__(self, weight, heads, head_dimension): self.attn = FakeAttention(weight, heads, head_dimension)


def test_sequence_head_writes_sum_to_full_projection_delta():
    generator = torch.Generator().manual_seed(23); heads, head_dimension, dimension = 3, 4, 12
    weight = torch.randn(dimension, dimension, generator=generator); base = torch.randn(2, heads, 5, head_dimension, generator=generator); changed = torch.randn(2, heads, 5, head_dimension, generator=generator)
    writes = sequence_head_writes(FakeBlock(weight, heads, head_dimension), base, changed)
    flattened = (changed - base).permute(0, 2, 1, 3).reshape(2, 5, dimension)
    expected = flattened @ weight.T
    assert torch.allclose(writes.sum(0), expected, atol=2e-6)
