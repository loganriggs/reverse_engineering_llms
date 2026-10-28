import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_taskblind_mlp_subspace_finite.py"
spec = importlib.util.spec_from_file_location("taskblind_mlp_subspace_finite", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_affine_pca_reconstructs_rank_two_data():
    rng = np.random.default_rng(5); values = rng.normal(size=(20, 2)) @ rng.normal(size=(2, 7)) + 3
    mean, basis = module.affine_pca(values, 2)
    tensor = torch.as_tensor(values, dtype=torch.float32)
    recovered = module.project(tensor, torch.as_tensor(mean), torch.as_tensor(basis))
    assert torch.allclose(tensor, recovered, atol=1e-4)


def test_affine_pca_drops_centering_null_direction():
    values = np.random.default_rng(7).normal(size=(8, 20))
    _, basis = module.affine_pca(values, 20)
    assert len(basis) <= 7
    assert np.isfinite(basis).all()


def test_weight_basis_is_orthonormal():
    basis = module.output_weight_basis(np.random.default_rng(2).normal(size=(6, 9)), 4)
    assert np.allclose(basis @ basis.T, np.eye(4), atol=1e-6)


def test_random_basis_is_orthonormal():
    basis = module.random_basis(9, 3, np.random.default_rng(3))
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-6)
