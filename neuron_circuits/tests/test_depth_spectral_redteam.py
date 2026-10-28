import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_depth_spectral_redteam import output_map_eigensystem  # noqa: E402


def test_operator_gram_matches_materialized_symmetric_tensor():
    generator = torch.Generator().manual_seed(17)
    output, channels, dimension = 7, 11, 5
    down = torch.randn(output, channels, generator=generator, dtype=torch.float64)
    left = torch.randn(channels, dimension, generator=generator, dtype=torch.float64)
    right = torch.randn(channels, dimension, generator=generator, dtype=torch.float64)
    tensor = torch.einsum("oh,hi,hj->oij", down, left, right)
    tensor = 0.5 * (tensor + tensor.transpose(1, 2))
    expected = torch.linalg.svdvals(tensor.flatten(1))
    actual, vectors = output_map_eigensystem(down, left, right)
    assert torch.allclose(actual, expected, rtol=1e-8, atol=1e-8)
    reconstructed_gram = vectors @ torch.diag(actual.square()) @ vectors.T
    assert torch.allclose(reconstructed_gram, tensor.flatten(1) @ tensor.flatten(1).T, rtol=1e-8, atol=1e-8)
