import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_attention_source_edge_causality import distance_bin, masks_for_site, value_norm_matched_masks


def test_distance_bin_boundaries():
    boundaries = [0, 1, 2, 4, 8]
    assert [distance_bin(value, boundaries) for value in (0, 1, 2, 3, 7)] == [0, 1, 2, 2, 3]


def test_random_masks_match_selected_counts_by_distance_bin():
    tokens = np.asarray([0, 1, 0, 0, 1, 0, 0, 0])
    flags = np.asarray([0, 1], dtype=np.uint8)
    boundaries = [0, 1, 2, 4, 8]
    selected, controls = masks_for_site(tokens, 7, flags, boundaries, 5, np.random.default_rng(2))
    bins = np.asarray([distance_bin(7 - source, boundaries) for source in range(8)])
    for control in controls:
        for bin_index in np.unique(bins[selected]):
            assert int(control[bins == bin_index].sum()) == int(selected[bins == bin_index].sum())


def test_value_norm_matching_preserves_counts_and_prefers_nearby_norms():
    selected = np.zeros((1, 8), dtype=bool); selected[0, 4] = True
    import torch
    value_norm = torch.tensor([[1., 2., 3., 4., 10., 9., 8., 7.]])
    controls = value_norm_matched_masks(value_norm, torch.as_tensor(selected), 7, [0, 1, 2, 4, 8], 2)
    assert controls.sum(-1).tolist() == [[1, 1]]
    # Source 5 has the nearest non-selected norm in the same distance bin.
    assert bool(controls[0, 0, 5])
