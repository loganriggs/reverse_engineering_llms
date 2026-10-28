import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_routing_primitive_factorization import greedy_alignment, weighted_nmf


def test_nmf_recovers_nonnegative_low_rank_matrix():
    left = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., .2]])
    right = torch.tensor([[1., 2., 0.], [0., 1., 3.]])
    matrix = left @ right
    _, _, error = weighted_nmf(matrix, torch.ones(4), 2, 200, 3, 11)
    assert error < .01


def test_greedy_alignment_handles_permutation():
    first = torch.eye(3)
    second = first[[2, 0, 1]]
    pairs = greedy_alignment(first, second)
    assert [row[2] for row in pairs] == [1, 2, 0]
    assert all(row[0] == 1 for row in pairs)
