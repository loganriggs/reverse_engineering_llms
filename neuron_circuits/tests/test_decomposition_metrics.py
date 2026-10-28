import torch

from neuron_circuits.decomposition_metrics import (
    bilinear_operation_counts,
    combine_effect_agreements,
    effect_agreement,
    fisher_effect_agreement,
    factor_stability_summary,
    linear_cka,
    routing_usage,
    symmetrized_input_inner,
)


def test_effect_agreement_is_shift_invariant_and_exact():
    torch.manual_seed(0)
    base = torch.randn(2, 3, 7)
    changed = base + torch.randn_like(base)
    student_base = base + 4.0
    student_changed = changed + 9.0
    result = combine_effect_agreements([
        effect_agreement(base, changed, student_base, student_changed)
    ])
    assert abs(result["effect_fidelity"] - 1) < 1e-6
    assert abs(result["effect_cosine"] - 1) < 1e-6

    fisher = combine_effect_agreements([
        fisher_effect_agreement(base, changed, student_base, student_changed)
    ])
    assert abs(fisher["effect_fidelity"] - 1) < 1e-6
    assert abs(fisher["effect_cosine"] - 1) < 1e-6


def test_symmetrized_inner_matches_explicit_matrices():
    torch.manual_seed(1)
    la, ra = torch.randn(3, 5), torch.randn(3, 5)
    lb, rb = torch.randn(4, 5), torch.randn(4, 5)
    expected = torch.empty(3, 4)
    for i in range(3):
        qa = 0.5 * (torch.outer(la[i], ra[i]) + torch.outer(ra[i], la[i]))
        for j in range(4):
            qb = 0.5 * (torch.outer(lb[j], rb[j]) + torch.outer(rb[j], lb[j]))
            expected[i, j] = (qa * qb).sum()
    torch.testing.assert_close(symmetrized_input_inner(la, ra, lb, rb), expected)


def test_linear_cka_invariant_to_orthogonal_feature_rotation():
    torch.manual_seed(2)
    values = torch.randn(100, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert abs(linear_cka(values, values @ q) - 1) < 1e-5


def test_usage_and_operation_accounting():
    usage = routing_usage([2, 2, 0, 0])
    assert usage["features_ever_used"] == 2
    assert abs(usage["effective_support_simpson"] - 2) < 1e-9
    counts = bilinear_operation_counts(10, 100, 40, active=5)
    assert counts["native_multiplies"] == 3000
    assert counts["current_routed_multiplies"] == 850
    assert counts["ideal_indexed_multiplies"] == 150


def test_factor_summary_detects_permutation():
    similarity = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    result = factor_stability_summary(similarity)
    assert result["a_to_b_median_best_abs_cosine"] == 1
    assert result["mutual_nearest_fraction"] == 1
