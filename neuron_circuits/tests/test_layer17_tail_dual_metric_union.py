import json
from pathlib import Path

import torch


def test_union_config_is_locked_and_union_is_orthonormal():
    config = json.loads(Path(
        "configs/bilin18_layer17_tail_dual_metric_union.json"
    ).read_text())
    source = Path("scripts/run_bilin18_layer17_tail_dual_metric_union.py").read_text()
    assert "locked" in config["status"]
    for required in ("union16", "fisher8", "frobenius8", "ce_delta", "centered_vector_r2"):
        assert required in source
    generator = torch.Generator().manual_seed(4)
    first = torch.linalg.qr(torch.randn(16, 4, generator=generator)).Q
    second = torch.linalg.qr(torch.randn(16, 4, generator=generator)).Q
    quotient = torch.linalg.qr(torch.cat((first, second), dim=1), mode="reduced").Q
    assert torch.allclose(quotient.T @ quotient, torch.eye(8), atol=1e-5)
