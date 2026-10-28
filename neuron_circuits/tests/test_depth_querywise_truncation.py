import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_depth_querywise_truncation import ordered_eigensystem  # noqa: E402


def test_ordered_eigensystem_reconstructs_and_sorts_by_absolute_value():
    matrix = torch.tensor([[2.0, 0.0, 0.0], [0.0, -5.0, 0.0], [0.0, 0.0, 1.0]])[None]
    values, vectors = ordered_eigensystem(matrix)
    assert torch.allclose(values[0], torch.tensor([-5.0, 2.0, 1.0]))
    reconstructed = vectors @ torch.diag_embed(values) @ vectors.transpose(1, 2)
    assert torch.allclose(reconstructed, matrix)
