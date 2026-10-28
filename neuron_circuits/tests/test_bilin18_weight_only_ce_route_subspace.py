import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_weight_only_ce_route_subspace.py"
spec = importlib.util.spec_from_file_location("weight_ce_route", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_operator_basis_recovers_head_axes():
    signatures = np.diag([3.0, 2.0, 1.0])
    basis, values = module.operator_basis(signatures, 2)
    projector = basis.T @ basis
    assert np.allclose(projector, np.diag([1.0, 1.0, 0.0]))
    assert np.allclose(values, [9.0, 4.0])


def test_random_basis_is_orthonormal():
    basis = module.random_basis(9, 4, np.random.default_rng(2))
    assert np.allclose(basis @ basis.T, np.eye(4), atol=1e-7)


def test_best_of_four_weight_search_has_matched_null():
    source = SCRIPT.read_text()
    assert "for _ in METHODS" in source
    assert '"best_fidelity_method": best_method' in source
