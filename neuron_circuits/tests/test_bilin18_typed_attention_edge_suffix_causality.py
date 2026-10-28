import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_typed_attention_edge_suffix_causality import spearman


def test_spearman_detects_monotonic_order_not_scale():
    assert np.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1)
    assert np.isclose(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1)
