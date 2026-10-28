import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_conditional_coalition_selector import (
    fit_dual_ridge,
    ridge_predict,
    shuffled_pair_indices,
)


def test_dual_ridge_predicts_training_map():
    torch.manual_seed(1)
    x = torch.randn(12, 5)
    y = x @ torch.randn(5, 7)
    fitted = fit_dual_ridge(x, y, 1e-8)
    prediction = ridge_predict(fitted, x)
    assert float((prediction - y).norm() / y.norm()) < 1e-4


def test_pair_shuffle_has_no_fixed_pair_and_preserves_row_count():
    rows = [{"pair": pair} for pair in ("a", "a", "b", "b", "c", "c")]
    indices, mapping = shuffled_pair_indices(rows, 3)
    assert len(indices) == len(rows)
    assert all(left != right for left, right in mapping.items())
    assert sorted(indices) == list(range(len(rows)))
