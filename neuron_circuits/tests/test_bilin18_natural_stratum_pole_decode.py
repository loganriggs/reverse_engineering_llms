import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_natural_stratum_pole_decode.py"
spec = importlib.util.spec_from_file_location("natural_stratum_pole_decode", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_effective_support():
    assert np.isclose(module.effective_support([1, 0, 0]), 1)
    assert np.isclose(module.effective_support([1, 1, 0]), 2)


def test_profile_has_fixed_dimension():
    row = {"source_class": "other", "destination_class": "function", "distance_class": "self", "residual_effect": 1.0, "source_id": 1, "next_id": 2}
    assert len(module.profile([row])) == 2 * len(module.TOKEN_CLASSES) + len(module.DISTANCES) + 2
