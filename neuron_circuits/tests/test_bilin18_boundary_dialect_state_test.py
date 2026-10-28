import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_boundary_dialect_state_test import matched_rows  # noqa: E402


def test_matching_balances_labels_and_address_strata():
    # Token 9 is boundary; US=1, UK=2. Both labels have identical source/distance.
    raw = np.array([[1, 0, 9, 0], [2, 0, 9, 0], [1, 0, 9, 0], [2, 0, 9, 0]])
    lookup = np.zeros(10, dtype=bool); lookup[9] = True
    rows = matched_rows(raw, lookup, {1}, {2}, destination=3, maximum=8, seed=1)
    assert sum(row[2] for row in rows) == len(rows) // 2
    assert {row[4] for row in rows} == {0}
