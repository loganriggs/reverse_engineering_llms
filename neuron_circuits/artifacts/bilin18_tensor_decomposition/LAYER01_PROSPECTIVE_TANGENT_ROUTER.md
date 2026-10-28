# Prospective tensor-tangent router for Layer-0/1 neurons

The router uses the unedited state, the requested replacement embedding, and a checkpoint-native prefix JVP. It never observes the finite counterfactual product change used in the causal patch. All methods select the same 256 native channels; selected *actual* finite deltas are propagated through the complete suffix and scored in the output Fisher metric.

## Locked hypotheses

- **T0_tangent_predicts_finite_product_direction: not supported.**
- **T1_tangent_router_is_causally_useful: not supported.**
- **T2_tangent_beats_state_free_weights: supported.**
- **T3_tangent_closes_most_of_oracle_gap: not supported.**
- **T4_tangent_router_transfers_ood: supported.**
- **T5_tangent_support_is_predictive: supported.**
- **T6_mechanistic_router_beats_frozen_identity: supported.**
- **T7_base_activity_is_not_enough: not supported.**
- **T8_layer0_is_more_locally_linear: not supported.**

| Layer | Corpus | Pre/product/write cosine | Tangent norm ratio pre/product | Support Jaccard | Random / static / base | Frozen global/family | Tangent activation/state/local | Realized activation/state/local |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.490/0.492/0.486 | 0.985/0.978 | 0.164 | 0.036/0.101/0.437 | 0.149/0.077 | 0.343/0.328/0.408 | 0.686/0.704/0.755 |
| 0 | tinystories | 0.437/0.451/0.433 | 1.010/1.003 | 0.151 | 0.059/0.130/0.389 | 0.179/0.079 | 0.331/0.361/0.355 | 0.687/0.670/0.751 |
| 1 | fineweb | 0.461/0.547/0.547 | 1.334/1.161 | 0.185 | 0.062/0.107/0.283 | 0.148/0.116 | 0.310/0.343/0.371 | 0.631/0.610/0.742 |
| 1 | tinystories | 0.453/0.495/0.490 | 1.359/1.163 | 0.174 | 0.143/0.116/0.387 | 0.182/0.143 | 0.374/0.364/0.406 | 0.677/0.686/0.764 |

`tangent_local` is prospective but edit-aware: it is allowed to see the target token embedding because the intervention to be explained includes that target. `realized_*` methods see the finite counterfactual and are upper controls. Frozen banks see discovery counterfactuals but are fixed before all new lexical pairs.
