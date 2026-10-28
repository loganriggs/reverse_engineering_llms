import torch

from neuron_circuits.bilinear_memory import (
    BilinearMemory,
    MAXIMUM,
    MINIMUM,
    held_out_pair,
    task_targets,
)


def test_pair_activation_is_symmetric_and_matches_quadratic_matrix():
    torch.manual_seed(4)
    model = BilinearMemory(8, 4, 5, 3)
    first = torch.randn(6, 5)
    second = torch.randn(6, 5)
    forward = model.bilinear_activations(first, second)
    reverse = model.bilinear_activations(second, first)
    analytic = torch.einsum("bi,hij,bj->bh", first, model.neuron_matrices(), second)
    assert torch.allclose(forward, reverse, atol=1e-6)
    assert torch.allclose(forward, analytic, atol=1e-6)


def test_pair_split_and_targets_are_deterministic():
    first = torch.tensor([0, 1, 2, 3])
    second = torch.tensor([2, 1, 0, 3])
    assert torch.equal(held_out_pair(first, second), held_out_pair(first, second))
    values = torch.tensor([3, 0, 2, 1])
    maximum = task_targets(values, first, second, torch.full_like(first, MAXIMUM))
    minimum = task_targets(values, first, second, torch.full_like(first, MINIMUM))
    assert maximum.tolist() == [3, 0, 3, 1]
    assert minimum.tolist() == [2, 0, 2, 1]
