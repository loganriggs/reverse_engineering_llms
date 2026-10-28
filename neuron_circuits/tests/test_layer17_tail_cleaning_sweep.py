import json
import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer17_tail_cleaning_sweep import (  # noqa: E402
    covariance_transforms,
    reconstruct_quadratic,
)


def test_covariance_transform_round_trip():
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(500, 8, generator=generator, dtype=torch.float64)
    square, inverse, _ = covariance_transforms(x, 1e-6)
    assert torch.allclose(square @ inverse, torch.eye(8, dtype=torch.float64), atol=1e-8)


def test_weighted_quadratic_is_exact_before_truncation():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(100, 7, generator=generator, dtype=torch.float64)
    matrix = torch.randn(7, 7, generator=generator, dtype=torch.float64)
    matrix = 0.5 * (matrix + matrix.T)
    square, inverse, _ = covariance_transforms(x, 1e-6)
    z = x @ inverse
    weighted = square @ matrix @ square
    expected = torch.einsum("bi,ij,bj->b", x, matrix, x)
    actual, _, _ = reconstruct_quadratic(z, weighted, rank=7, preserve_trace=True)
    assert np.allclose(actual, expected.numpy(), atol=1e-8)


def test_cleaning_sweep_is_locked_with_matched_nulls():
    config = json.loads(Path("configs/bilin18_layer17_tail_cleaning_sweep.json").read_text())
    source = Path("scripts/run_bilin18_layer17_tail_cleaning_sweep.py").read_text()
    assert "locked" in config["status"]
    assert config["null_sign_flips"] >= 32
    assert set(config["evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    for required in ("trace_clean", "isotropic", "sign_null", "native_score", "clean_score"):
        assert required in source
