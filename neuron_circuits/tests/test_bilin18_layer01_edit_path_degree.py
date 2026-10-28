import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin18_layer01_edit_path_degree import polynomial_metrics


def test_polynomial_metrics_identify_quadratic_path():
    train_nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
    evaluation_nodes = [0.125, 0.375, 0.625, 0.875]

    def values(nodes):
        t = torch.tensor(nodes, dtype=torch.float64)[:, None]
        return torch.cat([1 + 2 * t + 3 * t.square(), -2 + t - t.square()], dim=1)

    result = polynomial_metrics(
        values(train_nodes), values(evaluation_nodes), train_nodes, evaluation_nodes, [1, 2, 3, 4],
    )
    assert result["2"] > 1 - 1e-12
    assert result["3"] > 1 - 1e-12
    assert result["rank2_energy"] > 1 - 1e-12
