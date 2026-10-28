from pathlib import Path
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin18_spectral_spike_pilot import (  # noqa: E402
    center_matrix,
    eigensystem,
    interaction_matrix,
    power_law,
    subspace_overlap,
)


def test_interaction_matrix_is_exact_quadratic_form():
    generator = torch.Generator().manual_seed(17)
    left = torch.randn(11, 5, generator=generator)
    right = torch.randn(11, 5, generator=generator)
    coefficients = torch.randn(11, generator=generator)
    inputs = torch.randn(13, 5, generator=generator)
    matrix = interaction_matrix(left, right, coefficients)

    explicit = ((inputs @ left.T) * (inputs @ right.T) * coefficients).sum(dim=1)
    quadratic = torch.einsum("bi,ij,bj->b", inputs, matrix, inputs)
    torch.testing.assert_close(quadratic, explicit, rtol=1e-5, atol=1e-5)


def test_centering_removes_isotropic_trace_and_eigensystem_orders_magnitude():
    matrix = torch.diag(torch.tensor([7.0, -9.0, 2.0, 4.0]))
    centered, tau = center_matrix(matrix)
    assert abs(float(torch.trace(centered))) < 1e-6
    assert float(tau) == 1.0
    values, vectors = eigensystem(centered)
    assert torch.all(values[:-1].abs() >= values[1:].abs())
    torch.testing.assert_close(vectors.T @ vectors, torch.eye(4), atol=1e-6, rtol=1e-6)


def test_subspace_overlap_ignores_rotation_inside_subspace():
    first = torch.eye(7)[:, :3]
    rotation, _ = torch.linalg.qr(torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, -1.0, 2.0],
        [2.0, 1.0, -2.0],
    ]))
    second = first @ rotation
    assert abs(subspace_overlap(first, second) - 1.0) < 1e-6


def test_power_law_recovers_known_exponent():
    ranks = np.arange(1, 1001)
    values = ranks ** -0.73
    fit = power_law(values, 8, 384)
    assert abs(fit["alpha"] - 0.73) < 1e-10
    assert fit["r_squared"] > 0.999999
