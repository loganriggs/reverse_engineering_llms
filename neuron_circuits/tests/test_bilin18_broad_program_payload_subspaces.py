import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_broad_program_payload_subspaces.py"
spec = importlib.util.spec_from_file_location("broad_program_payload_subspaces", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_fit_basis_recovers_known_plane():
    values = torch.tensor([[1.0, 0, 0], [0, 2.0, 0], [1.0, 1.0, 0]])
    basis, _ = module.fit_basis(values, 2)
    projected = (values @ basis.T) @ basis
    torch.testing.assert_close(projected, values, atol=1e-6, rtol=1e-6)


def test_gaussian_basis_is_orthonormal():
    basis = module.gaussian_basis(20, 3, 4)
    torch.testing.assert_close(basis @ basis.T, torch.eye(3), atol=1e-6, rtol=1e-6)


def test_vector_metric_distinguishes_projection_from_zero():
    target = torch.tensor([[1.0, 2.0]])
    exact = module.vector_metric(target, target)
    zero = module.vector_metric(target, torch.zeros_like(target))
    assert exact["energy_fidelity"] == 1.0
    assert zero["energy_fidelity"] == 0.0
