import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_knn_router import route_grid  # noqa: E402


def test_nearest_context_routes_recover_matching_support():
    discovery_features = torch.tensor([[-1.0], [1.0]])
    discovery_absolute = torch.tensor([[[10.0, 0.0]], [[0.0, 10.0]]])
    evaluation_features = torch.tensor([[-0.9], [0.9]])
    evaluation_contributions = torch.tensor([[[8.0, 0.0]], [[0.0, 9.0]]])
    result = route_grid(
        discovery_features,
        discovery_absolute,
        evaluation_features,
        evaluation_contributions,
        neighbors=[1],
        budget=1,
        batch_size=2,
    )
    assert result["neighbors"]["1"]["median_oracle_support_recall"] == 1.0
    assert result["neighbors"]["1"]["median_correlation"] > 0.999
