# Occupied-outlier ranking for Layer-17 spelling consumers

Four candidate rankings are compared across all Layer-17 heads: spectral strength, value-axis occupancy, routed rank-1 write norm, and direct output-conditioned tensor score. Rankings are selected on fold 0/split 0; full finite rank-1 head effects on the other five cells are untouched evaluation.

## Locked results

- **D0_selection_is_discovery_frozen: supported.**
- **D1_contextual_occupied_outlier_selects_H2: supported.**
- **D2_output_quotient_selects_H2: supported.**
- **D3_quotient_score_predicts_finite_effects: supported.**
- **D4_rank1_preserves_H2_finite_effect: supported.**
- **D5_context_is_enough_without_semantic_output: supported.**
- **D6_occupied_outlier_discovery_rule: supported.**

| Fold/split | H2 spectral/occupancy/routed/quotient/finite ranks | direct→finite r | H2 rank1/full |
|---|---:|---:|---:|
| 0/0 | 2/1/1/1/1 | 1.000 (0.964 without H2) | 0.932 |
| 0/1 | 2/1/1/1/1 | 1.000 (0.967 without H2) | 0.931 |
| 1/0 | 2/1/1/1/1 | 1.000 (0.939 without H2) | 0.942 |
| 1/1 | 2/1/1/1/1 | 1.000 (0.980 without H2) | 0.947 |
| 2/0 | 2/1/1/1/1 | 1.000 (0.986 without H2) | 0.947 |
| 2/1 | 2/1/1/1/1 | 1.000 (0.945 without H2) | 0.940 |

## Interpretation boundary

Routed write norm uses weights, the counterfactual state change, and the source relation but no output-token semantics. The quotient score additionally uses the declared output contrast while omitting the final-MLP Jacobian. Their comparison states whether contextual tensor occupancy is enough to shortlist the function or whether semantic destination information is essential.
