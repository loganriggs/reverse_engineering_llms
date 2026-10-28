import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_attention_head_operator_roles import (
    coordinates,
    distance_masks,
    normalized_state,
    operator_map,
)


def test_head_subset_state_is_literal_sum_then_rmsnorm():
    torch.manual_seed(2)
    mixed = torch.randn(5, 7)
    heads = torch.randn(5, 3, 7)
    actual = normalized_state(mixed, heads, (0, 2))
    expected = torch.nn.functional.rms_norm(mixed + heads[:, 0] + heads[:, 2], (7,))
    assert torch.allclose(actual, expected)


def test_operator_map_matches_whitened_coordinates():
    torch.manual_seed(3)
    q, _ = torch.linalg.qr(torch.randn(6, 6)); modes, _ = torch.linalg.qr(torch.randn(6, 6))
    basis = {"mean": torch.randn(6), "state_vectors": q, "state_rms": torch.rand(6) + .2,
             "operator_modes_first": modes, "operator_singular_first": torch.rand(6)}
    state = torch.randn(9, 6)
    mean, mapping, _ = operator_map(basis, 4, "cpu")
    expected = ((state - basis["mean"]) @ q / basis["state_rms"]) @ modes[:, :4]
    assert torch.allclose(coordinates(state, mean, mapping), expected, atol=1e-5)


def test_distance_masks_are_disjoint_and_cover_causal_sources():
    masks = distance_masks([3, 7], 8, [0, 1, 2, 4, 8], "cpu")
    assert torch.all(masks.sum(0) <= 1)
    assert int(masks[:, 0].sum()) == 4
    assert int(masks[:, 1].sum()) == 8
