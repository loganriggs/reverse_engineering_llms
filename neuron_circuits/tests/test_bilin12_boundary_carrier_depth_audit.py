import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_boundary_carrier_depth_audit import FACTORS, fisher_terms, shapley_effects


def test_shapley_effects_recompose_total_vector_effect():
    generator = torch.Generator().manual_seed(7)
    worlds = torch.randn(8, 3, 11, generator=generator)
    effects = shapley_effects(worlds)
    assert set(effects) == set(FACTORS)
    assert torch.allclose(sum(effects.values()), worlds[7] - worlds[0], atol=1e-6)


def test_shapley_assigns_additive_factor_effects_exactly():
    base = torch.randn(2, 5)
    pieces = torch.randn(3, 2, 5)
    worlds = []
    for mask in range(8):
        worlds.append(base + sum((pieces[i] for i in range(3) if mask & (1 << i)), torch.zeros_like(base)))
    result = shapley_effects(torch.stack(worlds))
    for index, factor in enumerate(FACTORS):
        assert torch.allclose(result[factor], pieces[index], atol=1e-6)


def test_fisher_terms_are_centered_and_exact():
    reference = torch.tensor([[1.0, -0.5, 0.2]])
    target = torch.tensor([[0.3, -0.2, 0.7]])
    energy, prediction_energy, cross, error = fisher_terms(reference, target, target)
    assert torch.allclose(energy, prediction_energy)
    assert torch.allclose(energy, cross)
    assert torch.allclose(error, torch.zeros_like(error), atol=1e-7)


def test_fisher_terms_ignore_constant_logit_shift():
    reference = torch.randn(4, 9)
    target = torch.randn(4, 9)
    shifted = target + 12.0
    first = fisher_terms(reference, target, target)
    second = fisher_terms(reference, target, shifted)
    for left, right in zip(first[:3], second[:3]):
        assert torch.allclose(left, right, atol=2e-6)
    assert torch.allclose(second[3], torch.zeros_like(second[3]), atol=2e-6)
