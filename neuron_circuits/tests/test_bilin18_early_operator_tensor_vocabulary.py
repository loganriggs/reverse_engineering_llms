from types import SimpleNamespace
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_early_operator_tensor_vocabulary import (  # noqa: E402
    operator_probe_vectors,
    reconstruct,
)
from run_bilin18_operator_probe_convergence import linear_sum_assignment_maximum  # noqa: E402


def test_kronecker_probe_is_exact_jacobian_measurement():
    torch.manual_seed(0)
    dimension, channels, probes = 7, 11, 13
    mlp = SimpleNamespace(
        Left=torch.nn.Linear(dimension, channels, bias=False),
        Right=torch.nn.Linear(dimension, channels, bias=False),
        Down=torch.nn.Linear(channels, dimension, bias=False),
    )
    block = SimpleNamespace(mlp=mlp)
    with torch.no_grad():
        vectors, u, q = operator_probe_vectors(block, probes, 10)
        z = torch.randn(dimension)
        left = mlp.Left.weight @ z
        right = mlp.Right.weight @ z
        jacobian = mlp.Down.weight @ (
            right[:, None] * mlp.Left.weight + left[:, None] * mlp.Right.weight
        )
        explicit = torch.einsum("pd,di,pi->p", u, jacobian, q)
        implicit = vectors @ z
    torch.testing.assert_close(explicit, implicit, rtol=2e-5, atol=2e-6)


def test_full_whitened_basis_reconstructs_state():
    torch.manual_seed(1)
    rows, dimension = 19, 6
    states = torch.randn(rows, dimension)
    mean = torch.randn(dimension)
    vectors = torch.linalg.qr(torch.randn(dimension, dimension)).Q
    rms = torch.rand(dimension) + 0.3
    modes = torch.linalg.qr(torch.randn(dimension, dimension)).Q
    recovered = reconstruct(states, mean, vectors, rms, modes, dimension)
    torch.testing.assert_close(recovered, states, rtol=2e-5, atol=2e-5)


def test_internal_assignment_finds_maximum_weight_permutation():
    score = torch.tensor([[1.0, 9.0, 2.0], [8.0, 2.0, 1.0], [0.0, 3.0, 7.0]]).numpy()
    rows, columns = linear_sum_assignment_maximum(score)
    assert rows.tolist() == [0, 1, 2]
    assert columns.tolist() == [1, 0, 2]
