import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_fisher_chart import adjoint_covariance


def test_fisher_chart_is_candidate_independent_and_locked():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_fisher_chart.json").read_text())
    assert config["status"] == "locked-after-oracle-dynamic-chart-result-before-prospective-fisher-chart"
    assert config["exact_top_tokens"] >= 8
    assert config["tail_samples_per_split"] >= 8


def test_adjoint_covariance_sums_positions_and_probability_weights():
    adjoint = torch.tensor([[[[1., 0.], [0., 2.]]], [[[0., 1.], [1., 0.]]]])
    weights = torch.tensor([4., 1.], dtype=torch.float64)
    covariance = adjoint_covariance(adjoint, weights)
    expected = torch.tensor([[5., 0.], [0., 17.]], dtype=torch.float64)
    torch.testing.assert_close(covariance, expected)
