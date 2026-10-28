import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_bilin18_layer17_tail_routed_seed_stability import best_alignment  # noqa: E402


def test_best_alignment_recovers_permuted_identical_experts():
    generator = torch.Generator().manual_seed(12)
    reference = torch.linalg.qr(torch.randn(4, 16, 3, generator=generator)).Q
    permutation = torch.tensor([2, 0, 3, 1])
    candidate = reference[permutation]
    result = best_alignment(reference, candidate)
    assert result["mean_overlap"] > 0.999
    inverse = result["candidate_to_reference"]
    assert sorted(inverse) == [0, 1, 2, 3]


def test_seed_stability_source_has_alignment_and_route_controls():
    source = Path("scripts/run_bilin18_layer17_tail_routed_seed_stability.py").read_text()
    for required in ("best_alignment", "candidate_to_reference", "route_agreement", "global32", "geometric_mean"):
        assert required in source
