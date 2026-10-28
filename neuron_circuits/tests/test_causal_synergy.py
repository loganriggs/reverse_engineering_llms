import numpy as np
import torch

from neuron_circuits.causal_synergy import (
    factorial_statistics, pair_complementary_information, pair_information_synergy,
)


def test_factorial_interaction_is_zero_for_additive_effects():
    margins = torch.tensor([[7.0, 4.0, 5.0, 2.0]])
    result = factorial_statistics(margins)
    assert torch.allclose(result["interaction"], torch.zeros(1))
    assert not result["sign_flip"].item()


def test_factorial_interaction_and_sign_flip_detect_context_dependence():
    # A contributes -1 alone but +2 with B.
    margins = torch.tensor([[3.0, -1.0, 1.0, 0.0]])
    result = factorial_statistics(margins)
    assert torch.allclose(result["interaction"], torch.tensor([3.0]))
    assert result["sign_flip"].item()


def test_xor_has_joint_only_information_synergy():
    first = np.tile([0, 0, 1, 1], 32)
    second = np.tile([0, 1, 0, 1], 32)
    target = first ^ second
    assert abs(pair_information_synergy(target, first, second) - 1.0) < 1e-9
    assert abs(pair_complementary_information(target, first, second) - 1.0) < 1e-9
