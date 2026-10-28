import importlib.util
import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_bilin18_layer8_suffix_attention_subset.py"
SPEC = importlib.util.spec_from_file_location("suffix_attention_subset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

from run_bilin18_layer8_joint_fisher_program import greedy_normalized_joint


def test_gram_greedy_matches_response_greedy():
    generator = torch.Generator().manual_seed(7)
    response = torch.randn(19, 11, generator=generator)
    amplitudes = torch.rand(11, generator=generator) + 0.2
    expected = greedy_normalized_joint(response, amplitudes, 5)
    actual = MODULE.greedy_from_gram(response.T @ response, amplitudes, 5)
    assert torch.equal(actual, expected)
    expected_objective = response[:, expected].sum(-1).square().sum() / amplitudes[expected].square().sum()
    assert torch.allclose(torch.tensor(MODULE.gram_objective(response.T @ response, amplitudes, expected)), expected_objective)


def test_random_masks_are_unique_and_exclude_selected():
    selected = (9, 10, 11)
    masks = MODULE.random_size_three_masks(list(range(9, 18)), selected, 8, 3)
    assert len(masks) == len(set(masks)) == 8
    assert selected not in masks
    assert all(len(mask) == 3 for mask in masks)


def test_bootstrap_interval_contains_constant():
    assert MODULE.bootstrap_median_interval([0.4] * 8, 9) == [0.4, 0.4]
