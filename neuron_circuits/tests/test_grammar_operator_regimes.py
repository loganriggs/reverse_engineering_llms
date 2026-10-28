import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from neuron_circuits.grammar_data import VOCAB, build_manifest
from neuron_circuits.grammar_model import GrammarTransformer
from neuron_circuits.grammar_model import ActivationPatch
from run_grammar_exact_factor_chain import local_block_components
from run_grammar_jacobian_gauge_redteam import chart_metrics
from run_grammar_mixed_chart_explanation import feature_sets
from run_grammar_mixed_natural_metric_redteam import matrix_sqrt
from analyze_grammar_channel_coalitions import analyze as analyze_coalitions
from run_grammar_mixed_gate_cross_init import functional_channel_effects
from run_grammar_mixed_address_semantic_residual import grouped_roll
from run_grammar_layer0_token_value_pair_program import canonical_source_positions, relevant_digit_components


CONFIG_OUTPUTS = (
    ("grammar_jacobian_regime_sweep.json", "grammar_jacobian_regime_sweep.json"),
    ("grammar_jacobian_gauge_redteam.json", "grammar_jacobian_gauge_redteam.json"),
    ("grammar_jacobian_capacity_sweep.json", "grammar_jacobian_capacity_sweep.json"),
    ("grammar_mixed_generalization_chart.json", "grammar_mixed_generalization_chart.json"),
    ("grammar_mixed_natural_metric_redteam.json", "grammar_mixed_natural_metric_redteam.json"),
    ("grammar_mixed_chart_explanation.json", "grammar_mixed_chart_explanation.json"),
    ("grammar_mixed_chart_controller.json", "grammar_mixed_chart_controller.json"),
    ("grammar_mixed_chart_controller_same_template.json", "grammar_mixed_chart_controller_same_template.json"),
    ("grammar_exact_factor_chain.json", "grammar_exact_factor_chain.json"),
    ("grammar_last_block_component_chain.json", "grammar_last_block_component_chain.json"),
    ("grammar_last_block_channel_frontier.json", "grammar_last_block_channel_frontier.json"),
    ("grammar_channel_coalition_structure.json", "grammar_channel_coalition_structure.json"),
    ("grammar_finite_path_component_decomposition.json", "grammar_finite_path_component_decomposition.json"),
    ("grammar_mixed_gate_cross_init.json", "grammar_mixed_gate_cross_init.json"),
    ("grammar_mixed_gate_margin_alignment.json", "grammar_mixed_gate_margin_alignment.json"),
    ("grammar_cross_init_linear_conjugacy.json", "grammar_cross_init_linear_conjugacy.json"),
    ("grammar_pure_gate_cross_init.json", "grammar_pure_gate_cross_init.json"),
    ("grammar_pure_cross_init_linear_conjugacy.json", "grammar_pure_cross_init_linear_conjugacy.json"),
    ("grammar_operation_family_tournament.json", "grammar_operation_family_tournament.json"),
    ("grammar_operation_family_effectful_replication.json", "grammar_operation_family_effectful_replication.json"),
    ("grammar_mixed_address_router.json", "grammar_mixed_address_router.json"),
    ("grammar_mixed_address_location_redteam.json", "grammar_mixed_address_location_redteam.json"),
    ("grammar_mixed_attention_address_intervention.json", "grammar_mixed_attention_address_intervention.json"),
    ("grammar_mixed_weight_canonical_address_subspace.json", "grammar_mixed_weight_canonical_address_subspace.json"),
    ("grammar_mixed_address_subspace_redteam.json", "grammar_mixed_address_subspace_redteam.json"),
    ("grammar_mixed_address_semantic_residual.json", "grammar_mixed_address_semantic_residual.json"),
    ("grammar_mixed_address_codeword_swaps.json", "grammar_mixed_address_codeword_swaps.json"),
    ("grammar_final_gate_code_dimension.json", "grammar_final_gate_code_dimension.json"),
    ("grammar_final_gate_lowrank_redteam.json", "grammar_final_gate_lowrank_redteam.json"),
    ("grammar_final_gate_answer_quotient.json", "grammar_final_gate_answer_quotient.json"),
    ("grammar_final_gate_answer_null_distribution.json", "grammar_final_gate_answer_null_distribution.json"),
    ("grammar_answer_quotient_tensor_pairs.json", "grammar_answer_quotient_tensor_pairs.json"),
    ("grammar_block2_gate_bottleneck.json", "grammar_block2_gate_bottleneck.json"),
    ("grammar_gate_bottleneck_depth_telescope.json", "grammar_gate_bottleneck_depth_telescope.json"),
    ("grammar_pure_mixed_gate_depth_comparison.json", "grammar_pure_mixed_gate_depth_comparison.json"),
    ("grammar_pure_mixed_gate_depth_effectful_replication.json", "grammar_pure_mixed_gate_depth_effectful_replication.json"),
    ("grammar_layer0_token_value_pair_program.json", "grammar_layer0_token_value_pair_program.json"),
    ("grammar_layer0_qk_value_identity_redteam.json", "grammar_layer0_qk_value_identity_redteam.json"),
    ("grammar_identity_correction_tensor_paths.json", "grammar_identity_correction_tensor_paths.json"),
    ("grammar_identity_correction_quadrature_redteam.json", "grammar_identity_correction_quadrature_redteam.json"),
)


def test_rms_chart_is_invariant_to_channel_scale_gauge():
    generator = torch.Generator().manual_seed(91)
    train_jacobian = torch.randn(40, 8, 12, generator=generator)
    test_jacobian = torch.randn(20, 8, 12, generator=generator)
    train_activation = torch.randn(40, 12, generator=generator)
    scale = torch.exp(.4 * torch.randn(12, generator=generator))
    original = chart_metrics(train_jacobian, test_jacobian, train_activation, "rms", [0, 4, 8], "cpu")
    transformed = chart_metrics(train_jacobian / scale, test_jacobian / scale,
                                train_activation * scale, "rms", [0, 4, 8], "cpu")
    assert abs(original["effective_rank"] - transformed["effective_rank"]) < 1e-5
    for rank in ("0", "4", "8"):
        assert abs(original["rank_fidelity"][rank] - transformed["rank_fidelity"][rank]) < 1e-6


def test_full_covariance_metric_is_invariant_to_invertible_coordinate_change():
    generator = torch.Generator().manual_seed(92)
    activation = torch.randn(80, 6, generator=generator)
    jacobian = torch.randn(80, 4, 6, generator=generator)
    transform = torch.randn(6, 6, generator=generator) + 2 * torch.eye(6)
    changed_activation = activation @ transform.T
    changed_jacobian = jacobian @ torch.linalg.inv(transform)
    def metric(values, derivatives):
        centered = values - values.mean(0)
        covariance = centered.T @ centered / len(centered)
        return (derivatives @ matrix_sqrt(covariance, 0.0)).flatten(1)
    first = torch.linalg.svdvals(metric(activation, jacobian) - metric(activation, jacobian).mean(0))
    second_values = metric(changed_activation, changed_jacobian)
    second = torch.linalg.svdvals(second_values - second_values.mean(0))
    torch.testing.assert_close(first, second, rtol=2e-4, atol=2e-4)


def test_exact_block_components_match_autograd_and_recompose():
    torch.manual_seed(93)
    model = GrammarTransformer(vocab_size=24, width=8, heads=2, layers=1,
                               mlp_ratio=2, mlp_kind="bilinear", max_length=6).eval()
    block = model.blocks[0]
    hidden = torch.randn(5, 8)
    _, components = local_block_components(block, hidden)
    prefix = hidden[:-1].clone()
    def final_output(final):
        sequence = torch.cat((prefix, final[None]), 0)
        return block(sequence[None])[0][0, -1]
    expected = torch.func.jacrev(final_output)(hidden[-1])
    torch.testing.assert_close(components["A"], expected, rtol=2e-5, atol=2e-6)
    identity = torch.eye(8)
    recomposed = (identity + components["ML"] + components["MR"]) @ (
        identity + components["Q"] + components["V"]
    )
    torch.testing.assert_close(recomposed, components["A"], rtol=1e-6, atol=1e-7)


def test_cross_init_channel_effect_decomposition_matches_autograd():
    torch.manual_seed(94)
    model = GrammarTransformer(vocab_size=VOCAB.size, width=8, heads=2, layers=4,
                               mlp_ratio=2, mlp_kind="bilinear", max_length=6).eval()
    prompt = np.asarray([[1, 2, 3, 4, 5]], dtype=np.int64)
    _, activations = model(torch.as_tensor(prompt), return_activations=True)
    source = activations[0][0]
    answer_ids = torch.as_tensor(VOCAB.answers)
    center = torch.eye(8) - torch.ones(8, 8) / 8
    def answer_logits(patch_value):
        logits = model(torch.as_tensor(prompt), patch=ActivationPatch(0, patch_value))
        return center @ logits[0, -1, answer_ids]
    expected = torch.func.jacrev(answer_logits)(source) @ -source
    channels, complete = functional_channel_effects(model, prompt, "cpu")
    torch.testing.assert_close(complete[0], expected, rtol=3e-4, atol=2e-6)
    assert channels.shape == (1, model.channels, 8)


def test_write_quotient_minimum_norm_gate_reproduces_residual_write():
    torch.manual_seed(95)
    model = GrammarTransformer(vocab_size=VOCAB.size, width=12, heads=3, layers=1,
                               mlp_ratio=2, mlp_kind="bilinear", max_length=6).eval()
    down = model.blocks[0].down.weight.detach()
    target_write = torch.randn(9, model.width)
    target_gate = target_write @ torch.linalg.pinv(down).T
    reconstructed = target_gate @ down.T
    torch.testing.assert_close(reconstructed, target_write, rtol=2e-5, atol=2e-5)


def test_fixed_feature_vocabularies_have_expected_rows_and_finite_values():
    manifest = build_manifest(3, train_per_stratum=2, heldout_pair_per_stratum=1,
                              heldout_entity_per_stratum=1)
    facts = np.asarray(manifest["train_facts"])[:20]
    current = np.asarray(manifest["train_structured"])[:20]
    competitors = (current + 1) % 8
    features = feature_sets(manifest, facts, current, competitors)
    assert set(features) == {"semantic", "output", "address", "semantic_output", "combined"}
    assert all(values.shape[0] == len(facts) for values in features.values())
    assert all(np.isfinite(values).all() for values in features.values())
    assert features["combined"].shape[1] > features["address"].shape[1]


def test_coalition_overlap_counts_intersections_not_boolean_any_overlap():
    row = {
        "site_top32_channels": [list(range(32)), list(range(16, 48)), list(range(32))],
        "evaluation_relations": [0, 0, 1],
        "evaluation_rule_answers": [0, 1, 0],
        "top32_channels": list(range(32)),
    }
    result = analyze_coalitions(row)
    # Pair intersections are 16, 32, and 16, divided by 32.
    assert abs(result["mean_overlap"] - (0.5 + 1.0 + 0.5) / 3) < 1e-8
    assert result["fixed_bank_capture"] == (1 + .5 + 1) / 3


def test_grouped_residual_roll_covers_all_eight_relations():
    relations = np.repeat(np.arange(8), 16)
    labels = np.tile(np.repeat(np.arange(8), 2), 8)
    values = torch.arange(len(relations), dtype=torch.float32)[:, None]
    swapped = grouped_roll(values, relations, labels)
    assert not bool((swapped == values).any())
    for relation in range(8):
        for label in range(8):
            keep = torch.as_tensor((relations == relation) & (labels == label))
            assert sorted(swapped[keep, 0].tolist()) == sorted(values[keep, 0].tolist())


def test_canonical_source_positions_cover_every_prompt_position_once():
    for template in range(8):
        positions = canonical_source_positions(template)
        assert len(positions) == 13
        assert sorted(positions) == list(range(13))


def test_rule_relevant_digit_components_match_grammar_functions():
    expected = (
        {4}, {5}, {8}, {9}, {4, 6, 8}, {5, 9, 10}, {4, 6, 10}, {6, 8, 10},
    )
    assert all(relevant_digit_components(relation) == expected[relation] for relation in range(8))


def test_locked_configs_match_saved_hypothesis_ledgers_if_present():
    for config_name, output_name in CONFIG_OUTPUTS:
        config = json.loads((ROOT / "configs" / config_name).read_text())
        assert config["status"].startswith("hypotheses-locked")
        output_path = ROOT / "artifacts" / "grammar_regimes" / output_name
        if output_path.exists():
            output = json.loads(output_path.read_text())
            assert set(output["hypotheses"]) == set(config["hypotheses"])


def test_reported_operator_regime_results_preserve_positive_and_negative_outcomes():
    path = ROOT / "artifacts" / "grammar_regimes" / "grammar_mixed_natural_metric_redteam.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    assert payload["hypotheses"]["N0_covariance_gap_survives"]["supported"]
    assert payload["hypotheses"]["N1_fisher_gap_survives"]["supported"]
    assert not payload["hypotheses"]["N3_mixed_heldout_matches_mixed_train"]["supported"]
    controller = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                             "grammar_mixed_chart_controller_same_template.json").read_text())
    assert not controller["hypotheses"]["S0_full_context_is_a_compact_within_template_controller"]["supported"]
    finite = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                         "grammar_finite_path_component_decomposition.json").read_text())
    assert finite["hypotheses"]["I0_path_integral_reconstructs_finite_effect"]["supported"]
    assert not finite["hypotheses"]["I3_local_value_path_is_small"]["supported"]
    for row in finite["runs"]:
        for result in row["scales"].values():
            assert result["fidelity"] > .9999
            assert abs(result["projection_sum"] - 1) < 1e-4
    tournament = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                             "grammar_operation_family_effectful_replication.json").read_text())
    assert tournament["hypotheses"]["R0_pure_semantic_program_transfers"]["supported"]
    assert tournament["hypotheses"]["R2_tensor_write_quotient_transfers_semantics"]["supported"]
    assert tournament["hypotheses"]["R4_arbitrary_fact_swaps_do_not_share_one_global_gate_program"]["supported"]
    assert tournament["hypotheses"]["R6_semantic_transfer_is_more_recurrent_than_generic_prompt_swaps"]["supported"]
    assert not tournament["hypotheses"]["R1_semantic_program_is_not_a_neuron_permutation"]["supported"]
    assert not tournament["hypotheses"]["R3_relation_conditioning_improves_semantic_transfer"]["supported"]
    assert not tournament["hypotheses"]["R5_tensor_quotient_recovers_arbitrary_swaps"]["supported"]
    assert tournament["hypotheses"]["R7_own_delta_is_an_effectful_ceiling"]["supported"]
    address = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                          "grammar_mixed_weight_canonical_address_subspace.json").read_text())
    assert address["hypotheses"]["P1_rank48_pca_address_is_sufficient"]["supported"]
    assert address["hypotheses"]["P4_pca_address_beats_random_subspaces"]["supported"]
    shared = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                         "grammar_mixed_address_subspace_redteam.json").read_text())
    assert not shared["hypotheses"]["R2_memorization_has_a_distinct_subspace"]["supported"]
    assert shared["hypotheses"]["R3_one_shared_rank48_manifold_is_sufficient"]["supported"]
    residual = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                           "grammar_mixed_address_semantic_residual.json").read_text())
    assert "correction" in residual
    assert residual["hypotheses"]["D4_residual_identity_is_required_for_precise_effect"]["supported"]
    assert not residual["hypotheses"]["D8_rule_and_arbitrary_facts_use_the_same_decomposition"]["supported"]
    swaps = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                        "grammar_mixed_address_codeword_swaps.json").read_text())
    assert not swaps["hypotheses"]["Q1_same_cell_codewords_are_functionally_equivalent"]["supported"]
    assert swaps["hypotheses"]["Q5_wrong_answer_codeword_overrides_target"]["supported"]
    gate = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                       "grammar_final_gate_lowrank_redteam.json").read_text())
    assert gate["hypotheses"]["M2_eight_dense_gate_directions_are_usable"]["supported"]
    assert gate["hypotheses"]["M5_dense_axes_beat_native_coordinates"]["supported"]
    quotient = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                           "grammar_final_gate_answer_quotient.json").read_text())
    assert quotient["hypotheses"]["E1_answer_centroid_span_is_a_causal_quotient"]["supported"]
    assert not quotient["hypotheses"]["E6_relation_conditioning_adds_precise_effect_information"]["supported"]
    assert not quotient["hypotheses"]["E7_shuffled_answer_span_is_not_causal"]["supported"]
    nulls = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                        "grammar_final_gate_answer_null_distribution.json").read_text())
    assert nulls["hypotheses"]["N0_answer_span_beats_permutation_p99"]["supported"]
    assert nulls["hypotheses"]["N1_answer_span_has_empirical_p_at_most_point01"]["supported"]
    token_pairs = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                              "grammar_layer0_token_value_pair_program.json").read_text())
    assert token_pairs["hypotheses"]["V0_exact_role_pair_recomposition"]["supported"]
    assert token_pairs["hypotheses"]["V3_rule_relevant_value_routes_suffice_for_pure_rules"]["supported"]
    assert token_pairs["hypotheses"]["V4_arbitrary_facts_need_rule_irrelevant_value_routes"]["supported"]
    assert token_pairs["hypotheses"]["V6_cross_token_bilinear_terms_are_material"]["supported"]
    assert not token_pairs["hypotheses"]["V2_mixed_role_pair_program_is_broader"]["supported"]
    assert not token_pairs["hypotheses"]["V7_weight_ranked_pairs_beat_random_pair_banks"]["supported"]
    qkv = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                      "grammar_layer0_qk_value_identity_redteam.json").read_text())
    assert qkv["hypotheses"]["Q0_native_projected_attention_recomposes"]["supported"]
    assert qkv["hypotheses"]["Q4_value_identity_dominates_key_identity"]["supported"]
    assert not qkv["hypotheses"]["Q5_key_identity_is_also_material"]["supported"]
    assert not qkv["hypotheses"]["Q7_mean_and_roll_controls_agree"]["supported"]
    paths8 = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                         "grammar_identity_correction_tensor_paths.json").read_text())
    assert not paths8["hypotheses"]["P0_integrated_tensor_paths_are_exact"]["supported"]
    assert not paths8["hypotheses"]["P2_selected_identity_correction_is_not_a_128_path_program"]["supported"]
    assert paths8["hypotheses"]["P5_bilinear_paths_construct_most_of_the_correction"]["supported"]
    paths32 = json.loads((ROOT / "artifacts" / "grammar_regimes" /
                          "grammar_identity_correction_quadrature_redteam.json").read_text())
    assert all(row["supported"] for row in paths32["hypotheses"].values())
