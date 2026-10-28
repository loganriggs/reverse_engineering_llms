import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path("scripts").resolve()))
from run_bilin18_layer8_native_channel_program import channel_target, coefficients  # noqa: E402


def test_native_channel_coefficients_reconstruct_bilinear_derivative():
    generator = torch.Generator().manual_seed(11)
    left = torch.randn(9, 6, generator=generator)
    right = torch.randn(9, 6, generator=generator)
    down = torch.randn(7, 9, generator=generator)
    edit = torch.randn(6, 3, generator=generator)
    states = torch.randn(4, 6, generator=generator)
    actual = channel_target(coefficients(left, right, edit, states), down)
    expected = torch.einsum("oh,ha,nh->nao", down, left @ edit, states @ right.T)
    expected += torch.einsum("oh,ha,nh->nao", down, right @ edit, states @ left.T)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_native_channel_config_has_static_oracle_and_random_controls():
    config = json.loads(Path("configs/bilin18_layer8_native_channel_program.json").read_text())
    assert config["static_ranks"][-1] == 4608
    assert 256 in config["oracle_ranks"]
    assert config["random_replicates"] >= 8
