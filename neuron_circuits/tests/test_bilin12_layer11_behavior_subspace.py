import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin12_layer11_behavior_subspace import canonical_overlap, make_basis  # noqa: E402


def test_global_basis_is_orthonormal():
    parameter = torch.randn(17, 5)
    basis = make_basis(parameter, None)
    assert torch.allclose(basis.T @ basis, torch.eye(5), atol=1e-5)


def test_anchored_basis_preserves_anchor_and_orthogonality():
    anchor = torch.linalg.qr(torch.randn(17, 3), mode="reduced").Q
    parameter = torch.randn(17, 5)
    basis = make_basis(parameter, anchor)
    assert basis.shape == (17, 8)
    assert torch.allclose(basis[:, :3], anchor)
    assert torch.allclose(basis.T @ basis, torch.eye(8), atol=1e-5)


def test_canonical_overlap_is_basis_invariant():
    first = torch.linalg.qr(torch.randn(19, 6), mode="reduced").Q
    rotation = torch.linalg.qr(torch.randn(6, 6), mode="reduced").Q
    assert canonical_overlap(first, first @ rotation)["mean"] > 0.999


def test_config_locks_exact_behavior_and_ood_redteam():
    config = json.loads(Path("configs/bilin12_layer11_behavior_subspace.json").read_text())
    assert "locked" in config["status"]
    assert config["ranks"] == [20, 68]
    assert config["optimization_steps"] >= 200
    assert set(config["evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    assert len(config["hypotheses"]) == 7


def test_unmodified_initialization_is_a_selectable_checkpoint():
    source = Path("scripts/run_bilin12_layer11_behavior_subspace.py").read_text()
    assert '"step": -1' in source
    assert "initial_basis = make_basis(initialization, anchor)" in source
