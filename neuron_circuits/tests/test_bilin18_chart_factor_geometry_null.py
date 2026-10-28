import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_chart_factor_geometry_null.py"
spec = importlib.util.spec_from_file_location("chart_factor_geometry_null", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_matched_alternative_preserves_angles():
    rng = np.random.default_rng(1)
    reference = np.eye(2, 9)
    cosines = np.asarray([.8, .3])
    alternative = module.matched_alternative(reference, cosines, rng)
    assert np.allclose(np.linalg.svd(reference @ alternative.T, compute_uv=False), cosines)
    assert np.allclose(alternative @ alternative.T, np.eye(2))


def test_principal_coordinates_diagonalize_overlap():
    first = np.eye(2, 3)
    second = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    a, b, singular = module.principal_coordinates(first, second)
    assert np.allclose(a @ b.T, np.diag(singular))
