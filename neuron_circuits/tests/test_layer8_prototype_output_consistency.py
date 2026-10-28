import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_bilin18_layer8_prototype_output_consistency import orthonormal_output_basis, overlap_batch  # noqa: E402


def test_output_basis_and_overlap_are_rotation_invariant():
    generator = torch.Generator().manual_seed(4)
    effects = torch.randn(20, 4, generator=generator)
    first = orthonormal_output_basis(effects)
    rotation = torch.linalg.qr(torch.randn(4, 4, generator=generator)).Q
    second = orthonormal_output_basis(effects @ rotation)
    values = overlap_batch(first[None], second[None])
    assert values[0] > 0.999


def test_output_consistency_compares_direct_logit_and_fisher_roles():
    source = Path("scripts/run_bilin18_layer8_prototype_output_consistency.py").read_text()
    for required in ("direct_mlp", "unweighted_logit", "softmax_fisher", "same_route", "different_route", "cross_corpus"):
        assert required in source
