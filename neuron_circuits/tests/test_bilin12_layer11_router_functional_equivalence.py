import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin12_layer11_router_functional_equivalence import (  # noqa: E402
    projector_overlap,
    symmetric_kl,
)


def test_projector_overlap_is_one_under_basis_rotation():
    first = torch.linalg.qr(torch.randn(7, 23, 6), mode="reduced").Q
    rotations = torch.linalg.qr(torch.randn(7, 6, 6), mode="reduced").Q
    second = torch.einsum("bdr,brs->bds", first, rotations)
    assert torch.allclose(projector_overlap(first, second), torch.ones(7), atol=1e-5)


def test_symmetric_kl_is_zero_for_identical_logits():
    logits = torch.randn(9, 17)
    assert torch.allclose(symmetric_kl(logits, logits), torch.zeros(9), atol=1e-7)


def test_config_locks_pairwise_function_not_just_performance_tests():
    config = json.loads(Path("configs/bilin12_layer11_router_functional_equivalence.json").read_text())
    assert "locked" in config["status"]
    assert len(config["seeds"]) == 3
    assert len(config["hypotheses"]) == 5
    source = Path("scripts/run_bilin12_layer11_router_functional_equivalence.py").read_text()
    for required in ("relative_write_disagreement", "symmetric_prediction_kl", "native_kl_correlation"):
        assert required in source
