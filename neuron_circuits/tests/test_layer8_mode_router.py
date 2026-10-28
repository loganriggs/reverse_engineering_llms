import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_mode_router import assign_clusters, kmeans, make_routes  # noqa: E402


def test_kmeans_and_routes_separate_two_contexts():
    points = torch.tensor([[-2.0], [-1.8], [2.0], [2.2]])
    centroids, assignments = kmeans(points, clusters=2, iterations=20, seed=3)
    predicted = assign_clusters(points, centroids)
    assert torch.equal(predicted, assignments)
    assert assignments[:2].unique().numel() == 1
    assert assignments[2:].unique().numel() == 1
    assert assignments[0] != assignments[2]
    contributions = torch.tensor([
        [[10.0, 1.0]], [[9.0, 1.0]], [[1.0, 8.0]], [[1.0, 9.0]],
    ])
    routes = make_routes(contributions.abs(), assignments, clusters=2, budget=1)
    assert routes[assignments[0], 0, 0] == 0
    assert routes[assignments[2], 0, 0] == 1
