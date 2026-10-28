import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path("scripts").resolve()))
from run_bilin18_layer8_channel_shapley_redteam import shapley_scores  # noqa: E402
from run_bilin18_layer8_native_channel_program import channel_target  # noqa: E402


def test_channel_shapley_allocations_sum_to_squared_total():
    generator = torch.Generator().manual_seed(13)
    coefficients = torch.randn(5, 3, 11, generator=generator)
    down = torch.randn(7, 11, generator=generator)
    target = channel_target(coefficients, down)
    scores = shapley_scores(coefficients, down, target)
    assert torch.allclose(scores.sum(), target.square().sum(), atol=1e-4, rtol=1e-5)


def test_shapley_redteam_is_locked_after_energy_result():
    source = Path("configs/bilin18_layer8_channel_shapley_redteam.json").read_text()
    assert "cooperative-attribution" in source
    assert "S7_sufficiency_remains_broad" in source
