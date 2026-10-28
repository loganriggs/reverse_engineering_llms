import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_ce_adjoint_residual_novelty.py"
spec = importlib.util.spec_from_file_location("ce_residual_novelty", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_complement_removes_basis():
    values = np.asarray([[1.0, 2.0, 3.0]])
    basis = np.asarray([[1.0, 0.0, 0.0]])
    result = module.normalized_complement(values, basis)
    assert np.isclose(result[0, 0], 0.0)
    assert np.isclose(np.linalg.norm(result), 1.0)
