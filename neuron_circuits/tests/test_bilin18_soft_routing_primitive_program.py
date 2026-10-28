import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_soft_routing_primitive_program import normalize_memberships, support_keep_weights


def test_membership_normalization_is_partition_of_unity_including_empty_rows():
    result = normalize_memberships(torch.tensor([[1., 3.], [0., 0.]]))
    assert torch.allclose(result.sum(1), torch.ones(2))
    assert torch.allclose(result[1], torch.tensor([.5, .5]))


def test_all_soft_types_keep_every_causal_edge():
    membership = normalize_memberships(torch.tensor([[[1., 2.], [3., 1.], [0., 0.]]]))
    distance = torch.tensor([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    support = list(range(2 * 2 * 2))
    keep = support_keep_weights(membership, distance, support, heads=2, distances=2, variables=2)
    valid = distance >= 0
    assert torch.allclose(keep[0, :, 0][valid], torch.ones(int(valid.sum())))
    assert torch.allclose(keep[0, :, 1][valid], torch.ones(int(valid.sum())))
