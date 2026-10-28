import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin18_layers0_3_two_source_selector import (
    masks,
    selected_vector,
    split_vector,
)


def test_near_windows_are_source_relative_and_clipped():
    result = masks(source_layer=3, block_count=18, widths=[4, 12, 20])
    assert result["near_4"] == {4, 5, 6, 7}
    assert result["near_12"] == set(range(4, 16))
    assert result["near_20"] == set(range(4, 18))
    assert result["late_8"] == set(range(10, 18))
    assert result["all_attention"] == set(range(4, 18))


def test_flat_multisource_vector_round_trips_by_position():
    vector = torch.arange(12)
    pieces = split_vector(vector, positions=(3, 9), width=6)
    assert torch.equal(pieces[3], torch.arange(6))
    assert torch.equal(pieces[9], torch.arange(6, 12))


def test_selected_vector_uses_one_global_budget_across_positions():
    delta = torch.arange(1, 9, dtype=torch.float32)
    score = torch.tensor([1.0, 9.0, 2.0, 8.0, 7.0, 3.0, 6.0, 4.0])
    selected, indices = selected_vector(delta, score, rank=3)
    assert set(indices.tolist()) == {1, 3, 4}
    assert torch.equal(selected, torch.tensor([0.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0]))
