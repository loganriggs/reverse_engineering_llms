import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_chart_head_binding.py"
spec = importlib.util.spec_from_file_location("chart_head_binding", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_unique_permutations_exclude_identity():
    values = module.unique_permutations(4, 10, np.random.default_rng(1))
    assert len(values) == 10
    assert len({tuple(value) for value in values}) == 10
    assert all(not np.array_equal(value, np.arange(4)) for value in values)
