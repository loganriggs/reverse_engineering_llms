import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer17_tail_continuous_fisher import (  # noqa: E402
    canonical_overlap,
    continuous_prediction,
)


def test_continuous_prediction_identity_matches_prefix_form():
    generator = torch.Generator().manual_seed(37)
    z = torch.randn(9, 6, generator=generator, dtype=torch.float64)
    forms = torch.randn(3, 6, 6, generator=generator, dtype=torch.float64)
    forms = 0.5 * (forms + forms.transpose(1, 2))
    q = torch.eye(6, dtype=torch.float64)[:, :3]
    actual = continuous_prediction(z, forms, q)
    expected = torch.einsum("bi,kij,bj->bk", z[:, :3], forms[:, :3, :3], z[:, :3])
    assert torch.allclose(actual, expected)


def test_canonical_overlap_ignores_internal_rotation():
    generator = torch.Generator().manual_seed(41)
    first = torch.linalg.qr(torch.randn(12, 4, generator=generator)).Q
    rotation = torch.linalg.qr(torch.randn(4, 4, generator=generator)).Q
    result = canonical_overlap(first, first @ rotation)
    assert result["mean"] > 0.999
    assert result["minimum"] > 0.999


def test_continuous_fit_is_locked_with_independent_halves():
    config = json.loads(Path("configs/bilin18_layer17_tail_continuous_fisher.json").read_text())
    source = Path("scripts/run_bilin18_layer17_tail_continuous_fisher.py").read_text()
    assert "locked" in config["status"]
    assert config["optimization_steps"] >= 500
    assert config["candidate_pool_rank"] >= 128
    for required in (
        "even_rows", "odd_rows", "optimize_subspace", "continuous_prediction",
        "canonical_overlap", "exact_kl", "frobenius",
    ):
        assert required in source
