from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layer01_native_rotation_tournament import (  # noqa: E402
    inverse_rotate,
    make_rotation,
    rotate,
    selected_rotated,
)


def test_rotation_round_trip_and_inner_products():
    values = torch.randn(7, 32)
    other = torch.randn(7, 32)
    spec = make_rotation(32, 7, 123, "cpu")
    transformed = rotate(values, spec)
    assert torch.allclose(inverse_rotate(transformed, spec), values, atol=2e-6)
    assert torch.allclose(
        (values * other).sum(-1),
        (transformed * rotate(other, spec)).sum(-1),
        atol=2e-5,
    )


def test_full_rotated_selection_reconstructs_native_delta():
    delta = torch.randn(2, 5, 32)
    spec = make_rotation(32, 6, 456, "cpu")
    reconstructed = selected_rotated(delta, torch.arange(32), spec)
    assert torch.allclose(reconstructed, delta, atol=2e-6)
