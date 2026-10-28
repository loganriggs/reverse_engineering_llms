import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_prospective_read_relation_test import matched_auc  # noqa: E402


def test_matched_auc_respects_strata():
    candidate = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    values = np.array([3, 1, 4, 1, 5, 0], dtype=float)
    strata = np.array([0, 0, 1, 1, 2, 2])
    result = matched_auc(candidate, values, strata, seed=1, repetitions=4, minimum=3)
    assert result["matched_edges_per_class"] == 3
    assert result["auc_mean"] == 1.0
