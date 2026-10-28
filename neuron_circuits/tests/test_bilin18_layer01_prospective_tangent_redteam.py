import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin18_layer01_prospective_tangent_redteam import pair_bootstrap, spearman


def test_spearman_ordering():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) > 0.999
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) < -0.999


def test_pair_bootstrap_uses_pair_medians():
    rows = [
        {"pair": "a", "left": 3.0, "right": 1.0},
        {"pair": "a", "left": 5.0, "right": 1.0},
        {"pair": "b", "left": 4.0, "right": 1.0},
    ]
    center, interval = pair_bootstrap(rows, "left", "right", 1000, 0)
    assert np.isclose(center, 3.0)
    assert interval[0] <= center <= interval[1]
