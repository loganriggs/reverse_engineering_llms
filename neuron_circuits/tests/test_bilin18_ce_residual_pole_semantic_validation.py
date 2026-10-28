import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_bilin18_ce_residual_pole_semantics.py"
spec = importlib.util.spec_from_file_location("ce_pole_validation", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_profile_cosine_exact():
    row = {"median_distance": 3.0, "self_rate": .2, "distinct_source_tokens": 12}
    assert np.isclose(module.cosine(module.vector(row), module.vector(row)), 1.0)
