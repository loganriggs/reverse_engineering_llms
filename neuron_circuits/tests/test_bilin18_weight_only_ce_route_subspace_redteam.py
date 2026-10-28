import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_weight_only_ce_route_subspace.py"
spec = importlib.util.spec_from_file_location("weight_ce_redteam", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_fidelity_exact_basis():
    values = np.eye(3)
    assert np.isclose(module.fidelity(values, np.eye(3)), 1.0)
