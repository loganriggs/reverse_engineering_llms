import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_weight_context_route_hybrid.py"
spec = importlib.util.spec_from_file_location("weight_context_hybrid", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_hybrid_has_declared_rank_and_is_orthonormal():
    rng = np.random.default_rng(4)
    weight = np.linalg.qr(rng.normal(size=(9, 4)))[0].T
    discovery = rng.normal(size=(100, 9))
    for k in range(5):
        basis = module.hybrid_basis(weight, discovery, k, 4)
        assert basis.shape == (4, 9)
        assert np.allclose(basis @ basis.T, np.eye(4), atol=1e-7)
