import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin12_layer5_shared_channel_backbone import aggregate_backbone_order  # noqa: E402


def test_aggregate_backbone_order_finds_shared_dominant_channel():
    profiles = torch.tensor([
        [10.0, 1.0, 0.0, 0.0],
        [8.0, 0.0, 2.0, 0.0],
        [9.0, 0.0, 0.0, 1.0],
    ])
    order, normalized = aggregate_backbone_order(profiles)
    assert int(order[0]) == 0
    assert torch.allclose(normalized.sum(1), torch.ones(3))


def test_config_locks_shared_sufficiency_specificity_and_residual_tests():
    config = json.loads(Path("configs/bilin12_layer5_shared_channel_backbone.json").read_text())
    assert "locked" in config["status"]
    assert config["primary_backbone_size"] == 16
    assert 128 in config["backbone_sizes"]
    assert len(config["hypotheses"]) == 8
