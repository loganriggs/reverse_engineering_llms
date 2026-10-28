import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_direct_tensor_jacobian import rotate_heads  # noqa: E402


def test_rotate_heads_zero_angle_is_identity():
    values = torch.randn(3, 2, 8)
    cosine = torch.ones(1, 1, 4)
    sine = torch.zeros(1, 1, 4)
    assert torch.equal(rotate_heads(values, cosine, sine), values)
