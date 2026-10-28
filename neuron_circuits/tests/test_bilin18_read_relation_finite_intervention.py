import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_read_relation_finite_intervention import compare, edge_records  # noqa: E402


def test_edge_records_roundtrip():
    positions = [1, 3]
    rows, sources, destinations = edge_records([0, 1, 2, 5], positions)
    assert rows.tolist() == [0, 0, 0, 0]
    assert sources.tolist() == [0, 1, 0, 3]
    assert destinations.tolist() == [1, 1, 3, 3]


def test_compare_orders_effects():
    candidate = {"local": np.array([3., 4.]), "kl": np.array([2., 3.]), "ce_delta": np.array([1., 1.]), "margin_delta": np.array([-2., -2.])}
    control = {"local": np.array([1., 2.]), "kl": np.array([0., 1.]), "ce_delta": np.array([0., 0.]), "margin_delta": np.array([0., 0.])}
    result = compare(candidate, control)
    assert result["local_auc"] > .8
    assert result["kl_auc"] > .8
    assert result["local_sum_ratio"] > 2
