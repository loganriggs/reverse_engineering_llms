import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_directional_taylor_selector import directional_derivatives


def test_cubic_directional_taylor_is_exact_for_cubic_map():
    base = torch.tensor([1.2, -0.7], dtype=torch.float64)
    direction = torch.tensor([0.4, -0.2], dtype=torch.float64)
    function = lambda value: value ** 3
    first, second, third, first_function, _ = directional_derivatives(function, base, direction)
    midpoint_prediction = first + .5 * second + .125 * third
    midpoint_exact = first_function(base + .5 * direction)
    endpoint_prediction = first + .5 * second + third / 6
    endpoint_exact = function(base + direction) - function(base)
    assert torch.allclose(midpoint_prediction, midpoint_exact, atol=1e-12)
    assert torch.allclose(endpoint_prediction, endpoint_exact, atol=1e-12)
