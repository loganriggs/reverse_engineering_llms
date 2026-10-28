import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_chart_head_binding_hard_null.py"
spec = importlib.util.spec_from_file_location("chart_head_binding_hard_null", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_local_fidelity_exact_for_identity():
    rng = np.random.default_rng(1)
    values = rng.standard_normal((20, 3)); gradients = rng.standard_normal((20, 3)); basis = np.eye(3)
    assert np.isclose(module.local_fidelity(values, gradients, basis, basis), 1.0)


def test_sampled_permutations_are_unique():
    values = module.sample_permutations(5, 20, np.random.default_rng(2))
    assert len({tuple(value) for value in values}) == 20
