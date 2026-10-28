import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_stratified_natural_route_census.py"
spec = importlib.util.spec_from_file_location("stratified_natural_route_census", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_distance_classes_are_exhaustive():
    values = module.distance_class(np.asarray([0, 1, 4, 5, 16, 17, 64, 65]))
    assert values.tolist() == ["self", "local1-4", "local1-4", "near5-16", "near5-16", "mid17-64", "mid17-64", "far65+"]


def test_effect_fidelity_is_exact_for_full_basis():
    values = np.asarray([[1.0, 2.0], [2.0, -1.0]])
    gradients = np.asarray([[0.5, 1.0], [-1.0, 0.25]])
    assert np.isclose(module.effect_fidelity(values, gradients, np.eye(2)), 1.0)


def test_weighted_basis_is_orthonormal():
    rng = np.random.default_rng(1)
    basis = module.weighted_basis(rng.standard_normal((100, 9)), rng.standard_normal(100), 4)
    assert np.allclose(basis @ basis.T, np.eye(4), atol=1e-8)


def test_token_classes_use_fixed_precedence():
    flags = np.zeros((2, len(module.FEATURES)), dtype=np.uint8)
    flags[0, module.FEATURES.index("capitalized")] = 1
    flags[1, module.FEATURES.index("capitalized")] = 1
    flags[1, module.FEATURES.index("punctuation")] = 1
    assert module.token_class_from_flags(flags).tolist() == ["capitalized", "punctuation"]
