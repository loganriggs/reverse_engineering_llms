import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_readable_source_grammar import fit_readable_tree


def test_readable_tree_splits_two_profile_groups():
    matrix = torch.tensor([[1., 0.], [1., .1], [0., 1.], [.1, 1.]])
    weights = torch.ones(4)
    predicates = torch.tensor([[0], [0], [1], [1]], dtype=torch.bool)
    assignment, splits, rules = fit_readable_tree(matrix, weights, predicates, ["group"], 2, .1)
    assert assignment[0] == assignment[1]
    assert assignment[2] == assignment[3]
    assert assignment[0] != assignment[2]
    assert splits[0]["predicate"] == "group"
    assert len(rules) == 2
