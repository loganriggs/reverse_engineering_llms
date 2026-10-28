import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin18_layer01_prospective_tangent_router import cosine, jaccard, rms_jvp


def test_rms_jvp_matches_finite_difference():
    torch.manual_seed(0)
    state = torch.randn(17, dtype=torch.float64)
    direction = torch.randn(17, dtype=torch.float64)
    epsilon = 1e-6
    predicted = rms_jvp(state, direction)
    finite = (
        torch.nn.functional.rms_norm(state + epsilon * direction, (17,))
        - torch.nn.functional.rms_norm(state - epsilon * direction, (17,))
    ) / (2 * epsilon)
    assert torch.allclose(predicted, finite, atol=1e-8, rtol=1e-7)


def test_support_and_cosine_helpers():
    assert jaccard(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])) == 0.5
    assert abs(cosine(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) - 1.0) < 1e-7
