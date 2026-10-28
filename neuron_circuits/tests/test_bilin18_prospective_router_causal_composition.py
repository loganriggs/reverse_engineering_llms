import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_prospective_router_causal_composition import reconstruct


def test_reconstruct_projects_onto_discovery_basis():
    torch.manual_seed(8)
    basis, _ = torch.linalg.qr(torch.randn(7, 7))
    mean = torch.randn(7)
    vector = torch.randn(7)
    actual = reconstruct(vector, mean, basis, 3)
    expected = mean + basis[:, :3] @ (basis[:, :3].T @ (vector - mean))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_full_rank_reconstruct_is_identity():
    torch.manual_seed(9)
    basis, _ = torch.linalg.qr(torch.randn(7, 7))
    mean = torch.randn(7)
    vector = torch.randn(7)
    assert torch.allclose(reconstruct(vector, mean, basis, 7), vector, atol=1e-5)
