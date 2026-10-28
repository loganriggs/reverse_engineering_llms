import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin12_layer5_weight_conditioned_addresses import (  # noqa: E402
    output_conditioned_form,
    restricted_operator,
)


def test_output_conditioned_form_matches_scalar_bilinear_write():
    torch.manual_seed(3)
    dimension, channels = 7, 11
    down = torch.randn(dimension, channels)
    left = torch.randn(channels, dimension)
    right = torch.randn(channels, dimension)
    output = torch.randn(dimension)
    x = torch.randn(dimension)
    form = output_conditioned_form(down, left, right, output)
    native = output @ (down @ ((left @ x) * (right @ x)))
    assert torch.allclose(x @ form @ x, native, atol=1e-4)


def test_restricted_operator_matches_native_jvp():
    torch.manual_seed(5)
    dimension, channels = 9, 13
    down = torch.randn(dimension, channels)
    left = torch.randn(channels, dimension)
    right = torch.randn(channels, dimension)
    edit = torch.randn(dimension)
    context = torch.randn(17, dimension)
    operator = restricted_operator(down, left, right, edit)
    formula = context @ operator.T
    native = torch.stack([
        down @ ((left @ edit) * (right @ row) + (right @ edit) * (left @ row))
        for row in context
    ])
    assert torch.allclose(formula, native, atol=2e-4)


def test_config_locks_weight_only_discovery_and_nulls():
    config = json.loads(Path("configs/bilin12_layer5_weight_conditioned_addresses.json").read_text())
    assert "locked" in config["status"]
    assert config["layer"] == 5
    assert config["leading_addresses"] == 4
    assert config["sign_nulls"] >= 8
    assert config["random_addresses"] >= 8
    assert config["discovery"]["rows"] * len(config["positions"]) >= 8192
    assert set(config["evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    assert len(config["hypotheses"]) == 9
