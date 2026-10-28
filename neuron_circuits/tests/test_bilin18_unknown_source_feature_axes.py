import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_unknown_source_feature_axes import category_centroids, greedy_axis_alignment, heldout_recovery


def test_category_centroids_and_residual_recovery():
    matrix = torch.tensor([[1., 0.], [1., 2.], [3., 0.], [3., 2.]])
    category = torch.tensor([0, 0, 1, 1]); weights = torch.ones(4)
    centroid = category_centroids(matrix, category, weights, 2)
    assert torch.allclose(centroid, torch.tensor([[1., 1.], [3., 1.]]))
    residual = matrix - centroid[category]
    axes = torch.tensor([[0., 1.]])
    assert heldout_recovery(matrix, centroid[category], residual, axes, weights) > .999


def test_axis_alignment_accepts_sign_flips():
    first = torch.eye(2); second = -first[[1, 0]]
    pairs = greedy_axis_alignment(first, second)
    assert [row[2] for row in pairs] == [1, 0]
    assert all(row[0] == 1 and row[3] == -1 for row in pairs)
