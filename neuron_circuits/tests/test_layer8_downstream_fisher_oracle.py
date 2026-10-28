import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_downstream_fisher_oracle import (  # noqa: E402
    centered_fisher_impact,
    matched_local_controls,
)


def test_centered_fisher_impact_removes_constant_gradient():
    gradients = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    modes = torch.eye(2)
    assert torch.equal(centered_fisher_impact(gradients, modes), torch.zeros(2))


def test_matched_controls_exclude_fisher_top_pool():
    local = torch.arange(1, 65, dtype=torch.float32)
    fisher = torch.arange(64, dtype=torch.float32)
    selected = fisher.topk(4).indices
    tops, controls = matched_local_controls(local, fisher, selected, 4)
    excluded = set(fisher.topk(32).indices.tolist())
    assert tops == selected.tolist()
    assert not (set(controls) & excluded)
    assert len(set(controls)) == 4
