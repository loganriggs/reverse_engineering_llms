import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_adaptive_chart_router import native_density, write_fidelity


def test_write_fidelity_and_density_have_expected_limits():
    decoder = torch.eye(4)
    target = torch.tensor([1.0, 2.0, 0.0, 0.0])
    assert write_fidelity(decoder, target, target) == 1.0
    assert write_fidelity(decoder, target, torch.zeros_like(target)) == 0.0
    assert native_density(torch.tensor([1.0, 1e-5, 0.5, 0.0])) == 0.5
