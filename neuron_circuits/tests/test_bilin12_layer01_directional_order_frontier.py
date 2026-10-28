import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_directional_order_frontier import derivative_frontier, taylor_sum


def test_fifth_order_frontier_is_exact_for_quintic_map():
    base = torch.tensor([.7, -1.1], dtype=torch.float64)
    direction = torch.tensor([.2, .3], dtype=torch.float64)
    function = lambda value: value ** 5
    derivatives, functions = derivative_frontier(function, base, direction, 5)
    midpoint = taylor_sum(derivatives, .5, integral=False)
    endpoint = taylor_sum(derivatives, 1.0, integral=True)
    assert torch.allclose(midpoint, functions[1](base + .5 * direction), atol=1e-11)
    assert torch.allclose(endpoint, function(base + direction) - function(base), atol=1e-11)
