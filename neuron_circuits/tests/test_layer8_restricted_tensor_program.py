import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_bilin18_layer8_restricted_tensor_program import (  # noqa: E402
    restricted_tensor,
    ridge_covariance_transforms,
    tucker_prediction,
)


def test_restricted_tensor_matches_explicit_bilinear_derivative():
    generator = torch.Generator().manual_seed(5)
    down = torch.randn(7, 9, generator=generator)
    left = torch.randn(9, 6, generator=generator)
    right = torch.randn(9, 6, generator=generator)
    edit = torch.randn(6, 3, generator=generator)
    tensor = restricted_tensor(down, left, right, edit)
    x = torch.randn(4, 6, generator=generator)
    expected = torch.einsum("oh,ha,nh->nao", down, left @ edit, x @ right.T)
    expected += torch.einsum("oh,ha,nh->nao", down, right @ edit, x @ left.T)
    actual = torch.einsum("oaj,nj->nao", tensor, x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_restricted_program_contains_tucker_and_identity_controls():
    source = Path("scripts/run_bilin18_layer8_restricted_tensor_program.py").read_text()
    for required in ("restricted_tensor", "tucker_prediction", "identity_relative_error", "context_energy", "output_energy", "heldout_r2"):
        assert required in source


def test_ridge_covariance_is_finite_when_rows_are_fewer_than_dimensions():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(5, 11, generator=generator)
    square, inverse, diagnostics = ridge_covariance_transforms(x, 1e-4)
    identity = square @ inverse
    assert torch.isfinite(square).all()
    assert torch.isfinite(inverse).all()
    assert diagnostics["rows"] == 5
    assert diagnostics["dimension"] == 11
    assert diagnostics["ridge_value"] > 0
    assert torch.allclose(identity, torch.eye(11), atol=2e-3, rtol=2e-3)
