# Exact layerwise Shapley decomposition of MLP transport

With all downstream attention responses frozen, the four bilinear MLPs in Layers 7–10 are independently recomputed or frozen across all 16 hybrid worlds. Exact Shapley vectors attribute their finite causal transport while retaining every interaction. The residual response is reported separately.

## Locked hypotheses

- **M0_shapley_recomposition_is_exact: supported.**
- **M1_one_mlp_layer_dominates: not supported.**
- **M2_late_mlp_layers_dominate: not supported.**
- **M3_progressive_unfreezing_is_monotonic: supported.**
- **M4_mlp_layer_interactions_are_material: supported.**
- **M5_layer_profile_transfers_ood: supported.**
- **M6_early_late_cancellation_is_material: not supported.**

| Corpus | Residual projection | L7/L8/L9/L10 projections | Additive interaction norm | Cumulative L7/L8/L9/L10 fidelity |
|---|---:|---:|---:|---:|
| fineweb | 0.707 | 0.061/0.035/0.069/0.128 | 0.320 | 0.790/0.838/0.893/1.000 |
| wikitext | 0.673 | 0.068/0.052/0.075/0.132 | 0.290 | 0.802/0.849/0.900/1.000 |
| tinystories | 0.685 | 0.070/0.048/0.071/0.125 | 0.291 | 0.814/0.860/0.910/1.000 |

Shapley target projections describe signed alignment with the complete attention-frozen response; they are not percentages of stored information. The additive-interaction norm compares the full four-MLP response with a sum of isolated one-MLP responses, exposing composition that no single-layer ranking captures.
