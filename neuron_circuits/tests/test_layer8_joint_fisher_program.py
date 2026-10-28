import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_joint_fisher_program import (  # noqa: E402
    cosine,
    coherence_ratio,
    greedy_normalized_joint,
)


def test_greedy_prefers_constructively_aligned_columns():
    responses = torch.tensor([[1.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
    amplitudes = torch.ones(3)
    selected = greedy_normalized_joint(responses, amplitudes, 2)
    assert set(selected.tolist()) == {0, 1}
    assert coherence_ratio(responses, selected) == 2.0


def test_cosine_is_scale_invariant():
    first = torch.tensor([1.0, 2.0])
    assert abs(cosine(first, 3 * first) - 1) < 1e-6
