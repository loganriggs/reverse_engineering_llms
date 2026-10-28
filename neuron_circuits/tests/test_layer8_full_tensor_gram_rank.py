import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_full_tensor_gram_rank import (  # noqa: E402
    gram_eigensystem,
    low_rank_response,
)


def test_low_rank_response_reconstructs_gram_at_full_rank():
    torch.manual_seed(5)
    response = torch.randn(7, 4)
    values, vectors = gram_eigensystem(response)
    reconstructed = low_rank_response(values, vectors, 4)
    assert torch.allclose(
        reconstructed.T @ reconstructed,
        response.T @ response,
        atol=1e-5,
        rtol=1e-5,
    )
