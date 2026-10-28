# Exact upstream tensor pairs for the final answer quotient

The final normalized MLP input is decomposed into eight additive residual components using one shared RMS scale. The bilinear gate is therefore the exact sum of 64 ordered left-component × right-component terms. Discovery pair means are retained; ranked subsets preserve selected pair variations and mean-fill the rest.

## Locked hypotheses

- **T0_ordered_pair_expansion_is_exact: supported.**
- **T1_sixteen_pair_variations_are_sufficient: supported.**
- **T2_eight_pair_variations_are_insufficient: supported.**
- **T3_ranked_pairs_beat_random_pair_banks: supported.**
- **T4_cross_component_variation_is_essential: supported.**
- **T5_upstream_components_already_carry_most_of_the_program: supported.**
- **T6_arbitrary_facts_use_a_broader_pair_program: not supported.**
- **T7_top_pairs_predict_answer_quotient_coordinates: supported.**
- **T8_pair_ranking_transfers_templates: supported.**
- **T9_signal_gate_passes: supported.**

| Seed | Status | top 4/8/16/32 | self/cross | upstream7x7/final-attn | coord K90 | random16 mean/p95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | unselected | 0.564/0.741/0.869/0.967 | 0.407/0.920 | 0.964/0.127 | 31 | 0.411/0.560 |
| 0 | selected | 0.551/0.701/0.832/0.943 | 0.371/0.889 | 0.950/0.174 | 33 | 0.406/0.635 |
| 101 | unselected | 0.445/0.700/0.847/0.958 | 0.357/0.920 | 0.918/0.223 | 32 | 0.462/0.637 |
| 101 | selected | 0.480/0.644/0.801/0.942 | 0.327/0.875 | 0.827/0.341 | 31 | 0.436/0.601 |
| 104 | unselected | 0.489/0.695/0.852/0.965 | 0.421/0.892 | 0.948/0.174 | 27 | 0.429/0.612 |
| 104 | selected | 0.534/0.717/0.847/0.952 | 0.365/0.841 | 0.942/0.120 | 29 | 0.371/0.607 |

This is an exact affine variation decomposition of one tensor contraction. A compact pair frontier shortens the upstream-to-gate description but does not yet identify which token interactions each residual component computed. A broad frontier would not falsify the seven-dimensional output quotient; it would locate complexity in constructing its coordinates.
