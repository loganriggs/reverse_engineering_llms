import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_contextual_read_function_discovery import (  # noqa: E402
    fit_tree, predict_tree, predicate_jaccard,
)


def test_readable_tree_learns_conjunction_and_predicts_every_row():
    x = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)] * 20, dtype=bool)
    y = x[:, 0] & x[:, 1]
    tree = fit_tree(x, y, ["a", "b", "c"], leaves=4, maximum_depth=3, minimum_leaf=10)
    score = predict_tree(tree, x)
    assert np.all((score > .5) == y)


def test_predicate_jaccard():
    assert predicate_jaccard(["a", "b"], ["b", "c"]) == 1 / 3
    assert predicate_jaccard([], []) == 0
