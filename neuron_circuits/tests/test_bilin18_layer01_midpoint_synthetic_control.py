import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin18_layer01_midpoint_synthetic_control import sphere_endpoint


def test_sphere_endpoint_matches_norm_and_distance():
    torch.manual_seed(0)
    source=torch.randn(32,dtype=torch.float64); source=source/source.norm()*7
    endpoint=sphere_endpoint(source,torch.tensor(5.0,dtype=torch.float64),3)
    assert torch.allclose(endpoint.norm(),source.norm(),atol=1e-10)
    assert torch.allclose((endpoint-source).norm(),torch.tensor(5.0,dtype=torch.float64),atol=1e-10)
