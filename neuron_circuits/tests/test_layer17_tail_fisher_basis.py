import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer17_tail_fisher_basis import (  # noqa: E402
    fisher_quadratic,
    greedy_axis_selection,
)


def test_fisher_quadratic_matches_manual_batch_form():
    generator = torch.Generator().manual_seed(31)
    error = torch.randn(11, 4, generator=generator, dtype=torch.float64)
    factor = torch.randn(11, 4, 4, generator=generator, dtype=torch.float64)
    metric = factor @ factor.transpose(1, 2)
    expected = torch.stack([error[i] @ metric[i] @ error[i] for i in range(len(error))])
    assert torch.allclose(fisher_quadratic(error, metric), expected)


def test_greedy_selection_uses_metric_relevant_nonleading_axis():
    # Candidate 2 is the only one that reconstructs the Fisher-important first score.
    z = torch.tensor([[1.0, 0.0, 2.0], [2.0, 0.0, 1.0]], dtype=torch.float64)
    forms = torch.zeros(2, 3, 3, dtype=torch.float64)
    forms[0, 2, 2] = 1.0
    target = torch.stack((z[:, 2].square(), z[:, 0].square()), 1)
    metric = torch.zeros(2, 2, 2, dtype=torch.float64)
    metric[:, 0, 0] = 1.0
    selected, _ = greedy_axis_selection(z, forms, target, metric, rank=1)
    assert selected == [2]


def test_fisher_basis_experiment_is_locked_and_matched_width():
    config = json.loads(Path("configs/bilin18_layer17_tail_fisher_basis.json").read_text())
    source = Path("scripts/run_bilin18_layer17_tail_fisher_basis.py").read_text()
    assert "locked" in config["status"]
    assert config["candidate_pool_rank"] >= 128
    assert config["random_subsets"] >= 64
    for required in (
        "final_logit_fisher", "greedy_axis_selection", "frobenius",
        "score_mse", "exact_kl", "random_subsets",
    ):
        assert required in source
