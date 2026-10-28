# Which operators create context-dependent transport?

For finite Layer-6 coordinate edits, every downstream attention response and every downstream bilinear-MLP response is independently either recomputed or frozen to its unperturbed value. Exact two-factor Shapley vectors divide the final coordinate response into the always-present residual route, attention response, and MLP response. This is a causal hybrid decomposition, not a probe.

## Locked hypotheses

- **D0_hybrid_recomposition_is_exact: supported.**
- **D1_residual_path_is_major: supported.**
- **D2_attention_response_is_material: not supported.**
- **D3_mlp_response_is_material: supported.**
- **D4_attention_dominates_context_variation: not supported.**
- **D5_operator_interaction_is_material: supported.**
- **D6_operator_profile_transfers_ood: supported.**

| Corpus | Residual/attention/MLP projection | Interaction norm | Freeze-attn/freeze-MLP variance ratio | Residual-only fidelity |
|---|---:|---:|---:|---:|
| fineweb | 0.668/0.039/0.293 | 0.130 | 0.841/0.077 | 0.718 |
| wikitext | 0.634/0.045/0.322 | 0.130 | 0.819/0.085 | 0.725 |
| tinystories | 0.659/0.035/0.306 | 0.109 | 0.866/0.077 | 0.744 |

Here `freeze-attn` is the `mlp_only` world and `freeze-MLP` is the `attention_only` world. Target projections sum to one but may include cancellation. Variance ratios compare across-site Jacobian variability with the native full path; they are descriptive because freezing one operator also changes the inputs received by later operators.
