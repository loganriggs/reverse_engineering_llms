import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records  # noqa: E402


def test_disjoint_contrasts_are_balanced():
    matrix = make_contrasts(20, [(1, 2), (3, 4)], 5, 7, "cpu")
    torch.testing.assert_close(matrix[:, 0].sum(), torch.tensor(0.0))
    assert torch.all((matrix != 0).sum(0) == 4)


def test_counterfactual_records_are_balanced():
    raw = np.array([[1, 0, 0, 0, 9, 0, 0, 0], [2, 0, 0, 0, 9, 0, 0, 0], [1, 0, 0, 0, 9, 0, 0, 0], [2, 0, 0, 0, 9, 0, 0, 0]])
    boundary = np.zeros(10, dtype=bool); boundary[9] = True
    records = select_records(raw, boundary, [(1, 2)], np.ones(10), 7, 8, 1)
    assert len(records) == 4
    assert sum(row[4] for row in records) == 2
    assert {row[5] for row in records if row[4]} == {2}
