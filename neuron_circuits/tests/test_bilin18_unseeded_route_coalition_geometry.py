import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_unseeded_route_coalition_geometry.py"
spec = importlib.util.spec_from_file_location("unseeded_route_coalition_geometry", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_best_center_matching_recovers_permutation():
    first = np.eye(3)
    second = first[[2, 0, 1]]
    result = module.best_center_matching(first, second)
    assert result["mean_cosine"] == 1
    assert result["permutation"] == [1, 2, 0]


def test_effective_support():
    assert module.effective_support([1, 0, 0]) == 1
    assert abs(module.effective_support([1, 1, 1]) - 3) < 1e-12
