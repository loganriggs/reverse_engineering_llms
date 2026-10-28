# Output-conditioned suffix atlas at the Layer-16 boundary

A single linear pullback from final output covectors to Layer-16 write adjoints is fit jointly across probability, target-logit, and model-margin queries. Output rank and ridge strength are selected only on a disjoint FineWeb validation slice. The frozen map is evaluated both at the adjoint and after exact contraction through the Layer-16 bilinear Jacobian and natural upstream tangents.

## Locked hypotheses

- **O0_saved_factor_recomposition_is_exact: supported.**
- **O1_shared_static_suffix_predicts_write_adjoints: not supported.**
- **O2_shared_static_suffix_predicts_behavior: not supported.**
- **O3_shared_static_suffix_explains_model_margin: not supported.**
- **O4_shared_map_matches_component_specific_ceiling: not supported.**
- **O5_weighted_map_beats_scaled_identity: not supported.**
- **O6_static_suffix_description_is_rank128_or_smaller: not supported.**
- **O7_shared_suffix_map_transfers_ood: not supported.**
- **O8_output_covector_rank_accounts_for_write_rank: supported.**

Selected shared rank `256` and ridge multiplier `0.01`; mean validation write fidelity `0.321`. Saved-factor recomposition error: `6.50e-07`.

| Corpus | Component | Shared write | Component-specific write | Identity write | Shared behavior | Component-specific behavior | Identity behavior |
|---|---|---:|---:|---:|---:|---:|---:|
| fineweb | ce | 0.237 | 0.205 | 0.649 | -0.127 | 0.144 | 0.181 |
| fineweb | prediction | 0.394 | 0.367 | 0.269 | 0.444 | 0.350 | -0.063 |
| fineweb | target | 0.305 | 0.264 | 0.443 | 0.255 | 0.218 | -0.032 |
| fineweb | margin | 0.216 | 0.220 | 0.578 | -0.131 | 0.142 | 0.355 |
| wikitext | ce | 0.140 | 0.159 | 0.681 | -0.395 | 0.035 | 0.390 |
| wikitext | prediction | 0.393 | 0.354 | 0.283 | 0.457 | 0.382 | -0.068 |
| wikitext | target | 0.256 | 0.227 | 0.494 | 0.278 | 0.215 | 0.024 |
| wikitext | margin | 0.249 | 0.253 | 0.664 | -0.313 | 0.160 | 0.464 |
| tinystories | ce | 0.388 | 0.382 | 0.828 | -0.646 | 0.031 | 0.143 |
| tinystories | prediction | 0.646 | 0.619 | 0.222 | 0.542 | 0.287 | -0.375 |
| tinystories | target | 0.530 | 0.475 | 0.509 | 0.302 | 0.150 | -0.570 |
| tinystories | margin | 0.409 | 0.436 | 0.775 | -0.304 | 0.360 | 0.377 |

A successful shared map would mean broad adjoint samples are mostly many output covectors passed through one stable suffix operator. Failure after validation selection instead favors context-conditioned pullbacks or a nonlinear/routed atlas. Component-specific ridge fits are an overpowered statistical ceiling, not weight-derived circuits.
