import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin18_dynamic_mlp_jacobian_spectrum import analytic_mlp_jacobian


class _MLP:
    def __init__(self, dimension, channels):
        self.Left = torch.nn.Linear(dimension, channels, bias=False)
        self.Right = torch.nn.Linear(dimension, channels, bias=False)
        self.Down = torch.nn.Linear(channels, dimension, bias=False)


class _Block:
    def __init__(self, dimension, channels):
        self.mlp = _MLP(dimension, channels)


def test_analytic_mlp_jacobian_matches_autograd():
    torch.manual_seed(7)
    dimension, channels = 11, 17
    block = _Block(dimension, channels)
    vector = torch.randn(dimension, dtype=torch.float32)

    def forward(x):
        state = x / x.square().mean().sqrt()
        return block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state))

    expected = torch.autograd.functional.jacobian(forward, vector)
    actual = analytic_mlp_jacobian(block, vector)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
