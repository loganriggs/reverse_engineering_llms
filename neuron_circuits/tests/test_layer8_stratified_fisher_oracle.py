import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_stratified_fisher_oracle import weighted_fisher_impact  # noqa: E402


def test_weighted_fisher_impact_is_probability_weighted_second_moment():
    gradients = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    weights = torch.tensor([0.25, 0.75])
    modes = torch.eye(2)
    actual = weighted_fisher_impact(gradients, weights, modes)
    assert torch.allclose(actual, torch.tensor([0.25, 3.0]))
