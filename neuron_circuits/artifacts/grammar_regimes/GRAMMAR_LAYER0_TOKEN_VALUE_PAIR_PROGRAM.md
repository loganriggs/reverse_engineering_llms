# Exact Layer-0 raw-token value-route pair program

At the first block, every attention value is still a linear transform of one raw token state. The final-position attention write is exactly regrouped into thirteen canonical source roles, independent of template order. Together with the final-token boundary this gives 14 additive normalized-input components and 196 ordered bilinear gate terms. Pair means and rankings are fit on a disjoint trained-template panel; all reported cells are literal Layer-0 gate patches with the full suffix recomputed.

## Locked hypotheses

- **V0_exact_role_pair_recomposition: supported.**
- **V1_pure_role_pair_program_is_compact: supported.**
- **V2_mixed_role_pair_program_is_broader: not supported.**
- **V3_rule_relevant_value_routes_suffice_for_pure_rules: supported.**
- **V4_arbitrary_facts_need_rule_irrelevant_value_routes: supported.**
- **V5_arbitrary_load_globally_changes_value_routes: supported.**
- **V6_cross_token_bilinear_terms_are_material: supported.**
- **V7_weight_ranked_pairs_beat_random_pair_banks: not supported.**
- **V8_canonical_role_ranking_transfers_templates: supported.**
- **V9_signal_gate_passes: supported.**

| Regime | Seed | Status | top 16/32/64 | rule-relevant | no identity | identity-only | relation×identity | no cross-source | random64 p95 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| pure | 0 | all | 1.000/1.000/1.000 | 0.999 | 0.996 | 0.699 | 0.411 | 0.995 | 0.998 |
| pure | 101 | all | 1.000/1.000/1.000 | 0.999 | 0.924 | 0.950 | 0.934 | 0.966 | 0.990 |
| pure | 103 | all | 1.000/1.000/1.000 | 0.999 | 0.996 | 0.668 | 0.809 | 0.996 | 0.997 |
| mixed | 0 | unselected | 0.875/0.970/0.995 | 0.568 | 0.187 | 0.935 | 0.642 | 0.278 | 0.738 |
| mixed | 0 | selected | 0.863/0.968/0.994 | 0.328 | 0.014 | 0.942 | 0.606 | 0.192 | 0.651 |
| mixed | 101 | unselected | 0.907/0.997/1.000 | 0.705 | 0.389 | 0.941 | 0.705 | 0.379 | 0.752 |
| mixed | 101 | selected | 0.902/0.997/1.000 | 0.462 | 0.032 | 0.934 | 0.672 | 0.079 | 0.661 |
| mixed | 104 | unselected | 0.717/0.909/0.992 | 0.511 | 0.161 | 0.899 | 0.476 | 0.273 | 0.588 |
| mixed | 104 | selected | 0.501/0.847/0.989 | 0.135 | -0.032 | 0.882 | 0.284 | 0.067 | 0.404 |

The semantic masks isolate value/write routes while retaining the native QK pattern. In particular, rule-irrelevant tokens can still alter every retained source contribution through the softmax denominator. Therefore success of `rule-relevant-only` does not establish complete token irrelevance; a separate QK intervention is required. Conversely, failure can reflect distributed value transport, normalization coupling, or an off-manifold mean fill rather than semantic polysemanticity.
