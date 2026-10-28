import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_attention_qkov_depth_audit import FACTORS, vector_shapley


def test_qkov_shapley_recomposes_vector_game():
    generator = torch.Generator().manual_seed(19)
    worlds = torch.randn(16, 2, 17, generator=generator)
    result = vector_shapley(worlds)
    assert set(result) == set(FACTORS)
    assert torch.allclose(sum(result.values()), worlds[15] - worlds[0], atol=2e-6)


def test_qkov_shapley_recovers_additive_factors():
    base = torch.randn(2, 8); pieces = torch.randn(4, 2, 8); worlds = []
    for mask in range(16): worlds.append(base + sum((pieces[i] for i in range(4) if mask & (1 << i)), torch.zeros_like(base)))
    result = vector_shapley(torch.stack(worlds))
    for index, factor in enumerate(FACTORS): assert torch.allclose(result[factor], pieces[index], atol=2e-6)
