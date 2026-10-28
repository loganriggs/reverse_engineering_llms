import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_chart_empirical_direction_control import sphere_endpoint_from_direction


def test_transported_direction_matches_radius_and_chord():
    torch.manual_seed(3)
    source = torch.randn(64, dtype=torch.float64)
    source = source / source.norm() * 8
    direction = torch.randn_like(source)
    distance = torch.tensor(5.5, dtype=torch.float64)
    endpoint = sphere_endpoint_from_direction(source, distance, direction)
    assert torch.allclose(endpoint.norm(), source.norm(), atol=1e-10)
    assert torch.allclose((endpoint - source).norm(), distance, atol=1e-10)
