import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_frozen_fisher_quotient import capture_fraction  # noqa: E402


def test_capture_fraction_for_coordinate_subspace():
    gradients = torch.eye(3)
    basis = torch.eye(3)[:2]
    assert abs(capture_fraction(gradients, basis) - 2 / 3) < 1e-6
