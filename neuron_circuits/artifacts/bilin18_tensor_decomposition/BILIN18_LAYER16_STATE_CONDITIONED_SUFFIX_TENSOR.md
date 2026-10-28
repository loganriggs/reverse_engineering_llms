# State-conditioned CP suffix pullback

A rank-64 CP tensor predicts the write-adjoint correction from a 256D output-covector chart and a 64D centered local-state chart. It is shared across prediction, target, and margin endpoints and sits on top of the fitted scaled identity. Three seeds use validation-only checkpoint selection.

## Locked hypotheses

- **T0_saved_factor_recomposition_is_exact: supported.**
- **T1_state_conditioning_improves_validation_write: not supported.**
- **T2_state_conditioned_tensor_predicts_behavior: not supported.**
- **T3_state_conditioned_tensor_explains_model_margin: not supported.**
- **T4_state_is_causally_specific_for_routing: not supported.**
- **T5_state_conditioned_tensor_beats_static_behavior: not supported.**
- **T6_state_conditioned_tensor_transfers_ood: not supported.**
- **T7_training_seeds_are_stable: supported.**

Parameters: `94,272`. Static validation write fidelity: `0.587`; selected CP validation fidelity: `0.577` (seed `29`).

| Seed | Initial validation | Best validation | Step |
|---:|---:|---:|---:|
| 11 | 0.433 | 0.574 | 975 |
| 29 | 0.433 | 0.577 | 900 |
| 47 | 0.433 | 0.573 | 850 |

| Corpus | Component | Write | Shuffled-state write | Behavior | Shuffled-state behavior | Static behavior |
|---|---|---:|---:|---:|---:|---:|
| fineweb | ce | 0.643 | 0.606 | 0.223 | -0.201 | 0.236 |
| fineweb | prediction | 0.534 | 0.395 | 0.463 | 0.210 | 0.450 |
| fineweb | target | 0.575 | 0.490 | 0.393 | 0.139 | 0.369 |
| fineweb | margin | 0.561 | 0.547 | 0.165 | -0.147 | 0.254 |
| wikitext | ce | 0.608 | 0.615 | -0.360 | -0.598 | 0.238 |
| wikitext | prediction | 0.600 | 0.332 | 0.544 | -0.075 | 0.423 |
| wikitext | target | 0.599 | 0.474 | 0.352 | -0.137 | 0.380 |
| wikitext | margin | 0.604 | 0.591 | -0.085 | -0.390 | 0.190 |
| tinystories | ce | 0.791 | 0.774 | -0.242 | -0.843 | -0.177 |
| tinystories | prediction | 0.730 | 0.477 | 0.539 | 0.148 | 0.539 |
| tinystories | target | 0.762 | 0.633 | 0.325 | -0.138 | 0.373 |
| tinystories | margin | 0.749 | 0.730 | -0.224 | -0.079 | 0.034 |

State-shuffle specificity distinguishes a genuinely routed tensor from an overparameterized q-only map. Failure remains scoped: Layer-17 attention also reads sequence-wide keys, values, and the persistent embedding bus, which are not explicitly supplied by this local-state router.
