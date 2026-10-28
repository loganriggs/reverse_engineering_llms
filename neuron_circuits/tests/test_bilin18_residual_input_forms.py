import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_bilin18_residual_input_forms import contract, quadratic, fit_forms
from run_bilin18_residual_input_forms_finite import decoder, decode


def test_contraction_matches_bilinear_native_and_operand_swap():
    torch.manual_seed(7)
    x = torch.randn(21, 9, dtype=torch.float64)
    left, right = torch.randn(2, 17, 9, dtype=torch.float64)
    down = torch.randn(6, 17, dtype=torch.float64)
    u = torch.randn(6, dtype=torch.float64)
    q = contract(left, right, down, u)
    native = ((x @ left.T) * (x @ right.T)) @ down.T @ u
    torch.testing.assert_close(quadratic(x, q), native)
    torch.testing.assert_close(q, contract(right, left, down, u))


def test_full_rank_reconstructs_nonzero_mean_and_shifted_test_distribution():
    torch.manual_seed(8)
    d = torch.randn(100, 9, dtype=torch.float64) + 2
    v = torch.randn(80, 9, dtype=torch.float64) - 1
    q = torch.randn(9, 9, dtype=torch.float64); q = (q + q.T) / 2
    result = fit_forms(d, v, q, [1, 9], 1e-8)
    for method in ("raw", "covariance"):
        assert result["curves"][method]["9"]["r2"] > 1 - 1e-12
        assert result["curves"][method]["9"]["quadratic_variation_r2"] > 1 - 1e-12


def test_executable_decoder_matches_full_form_on_shifted_inputs():
    torch.manual_seed(9)
    train = torch.randn(100, 9, dtype=torch.float64) + 1
    test = torch.randn(50, 9, dtype=torch.float64) - 2
    q = torch.randn(9, 9, dtype=torch.float64); q = (q + q.T) / 2
    torch.testing.assert_close(decode(test, decoder(train, q, 9)), quadratic(test, q))
