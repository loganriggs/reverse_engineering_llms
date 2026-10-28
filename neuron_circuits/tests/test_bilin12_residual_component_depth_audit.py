import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_residual_component_depth_audit import COMPONENTS, vector_shapley


def test_four_factor_vector_shapley_recomposes():
    generator = torch.Generator().manual_seed(11)
    worlds = torch.randn(16, 2, 13, generator=generator)
    result = vector_shapley(worlds)
    assert set(result) == set(COMPONENTS)
    assert torch.allclose(sum(result.values()), worlds[15] - worlds[0], atol=2e-6)


def test_four_factor_additive_game_is_exact():
    base = torch.randn(3, 7)
    pieces = torch.randn(4, 3, 7)
    worlds = []
    for mask in range(16):
        worlds.append(base + sum((pieces[i] for i in range(4) if mask & (1 << i)), torch.zeros_like(base)))
    result = vector_shapley(torch.stack(worlds))
    for index, component in enumerate(COMPONENTS):
        assert torch.allclose(result[component], pieces[index], atol=2e-6)
