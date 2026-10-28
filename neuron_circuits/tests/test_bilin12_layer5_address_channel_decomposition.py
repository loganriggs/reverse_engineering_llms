import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin12_layer5_address_channel_decomposition import channel_decomposition  # noqa: E402


def test_channel_decomposition_reconstructs_operator_and_allocates_energy():
    torch.manual_seed(8)
    dimension, channels = 9, 17
    down = torch.randn(dimension, channels)
    context_rows = torch.randn(channels, dimension)
    row = channel_decomposition(down, context_rows, [1, 4, 17])
    assert row["operator_identity_relative_error"] < 1e-6
    assert row["shapley_sum_relative_error"] < 1e-6
    assert abs(row["prefix_rows"][-1]["operator_r2"] - 1.0) < 1e-6


def test_config_locks_native_channel_sufficiency_and_cooperation_tests():
    config = json.loads(Path("configs/bilin12_layer5_address_channel_decomposition.json").read_text())
    assert "locked" in config["status"]
    assert config["channel_prefixes"][-1] == 3072
    assert 128 in config["channel_prefixes"]
    assert len(config["hypotheses"]) == 8
