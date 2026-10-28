import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_read_relation_payload_ontology import (  # noqa: E402
    cross_capture, normalized_svd, subspace_overlap,
)


def test_subspace_metrics_identify_same_and_orthogonal_spaces():
    first = torch.eye(8)[:4]
    same = first * 3
    orthogonal = torch.eye(8)[4:]
    assert cross_capture(first, same, 4) > .999
    assert subspace_overlap(first, same, 4) > .999
    assert subspace_overlap(first, orthogonal, 4) < 1e-6


def test_effective_rank():
    _, _, rank = normalized_svd(torch.eye(5))
    assert abs(rank - 5) < 1e-5
