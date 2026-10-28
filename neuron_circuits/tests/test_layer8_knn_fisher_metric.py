import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_knn_fisher_metric import (  # noqa: E402
    nearest_indices,
    neighbor_responses,
)


def test_nearest_indices_uses_normalized_pca_coordinates():
    mean = torch.zeros(2)
    basis = torch.eye(2)
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert nearest_indices(torch.tensor([2.0, 0.1]), mean, basis, bank, 1).item() == 0


def test_neighbor_responses_averages_metric_not_amplitude():
    gradients = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    weights = torch.ones(2, 1)
    tangents = torch.eye(2)
    response = neighbor_responses(gradients, weights, torch.tensor([0, 1]), tangents)
    assert torch.allclose(response.square().sum(0), torch.tensor([1.0, 0.0]))
