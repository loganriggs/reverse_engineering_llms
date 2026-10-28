import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_prospective_operator_router import (
    operator_coordinates,
    pca_coordinates,
    pca_projection,
    pca_reconstruct,
)


def test_pca_coordinates_use_only_discovery_mean_and_basis():
    torch.manual_seed(3)
    train = torch.randn(100, 8)
    test = torch.randn(20, 8) + 2
    train_coordinates, test_coordinates = pca_coordinates(train, test, 4)
    assert train_coordinates.shape == (100, 4)
    assert test_coordinates.shape == (20, 4)
    assert torch.allclose(train_coordinates.mean(0), torch.zeros(4), atol=1e-5)
    assert test_coordinates.mean().abs() > 0.05


def test_operator_coordinates_matches_declared_linear_map():
    torch.manual_seed(4)
    state = torch.randn(7, 5)
    q, _ = torch.linalg.qr(torch.randn(5, 5))
    m, _ = torch.linalg.qr(torch.randn(5, 5))
    basis = {
        "mean": torch.randn(5),
        "state_vectors": q,
        "state_rms": torch.rand(5) + 0.5,
        "operator_modes_first": m,
    }
    expected = ((state - basis["mean"]) @ q / basis["state_rms"]) @ m[:, :3]
    assert torch.allclose(operator_coordinates(state.cuda(), basis, 3).cpu(), expected, atol=1e-5)


def test_full_rank_pca_reconstruction_is_exact_on_low_rank_data():
    torch.manual_seed(5)
    map_ = torch.randn(3, 8)
    offset = torch.randn(8)
    train = torch.randn(100, 3) @ map_ + offset
    test = torch.randn(20, 3) @ map_ + offset
    mean, vectors, _, test_coordinates = pca_projection(train, test, 3)
    reconstructed = pca_reconstruct(mean, vectors, test_coordinates, 3)
    assert torch.allclose(reconstructed, test, atol=2e-5)
