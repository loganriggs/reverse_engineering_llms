import sys
from pathlib import Path

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_direct_tensor_jacobian import (  # noqa: E402
    head_rms_norm_jvp,
    rms_norm_jvp,
)


def test_rms_norm_jvp_matches_autograd():
    torch.manual_seed(2)
    x = torch.randn(7, dtype=torch.float64)
    tangent = torch.randn(7, dtype=torch.float64)
    _, expected = torch.autograd.functional.jvp(
        lambda value: F.rms_norm(value, (7,)), x, tangent,
    )
    actual = rms_norm_jvp(x, tangent[None])[0]
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_head_rms_norm_jvp_matches_per_head_autograd():
    torch.manual_seed(3)
    x = torch.randn(2, 5, dtype=torch.float64)
    tangents = torch.randn(4, 2, 5, dtype=torch.float64)
    actual = head_rms_norm_jvp(x, tangents)
    expected = []
    for tangent in tangents:
        _, value = torch.autograd.functional.jvp(
            lambda item: F.rms_norm(item, (5,)), x, tangent,
        )
        expected.append(value)
    assert torch.allclose(actual, torch.stack(expected), atol=1e-10, rtol=1e-10)
