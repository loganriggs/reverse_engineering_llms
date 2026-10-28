# Full weight-derived QK/value/MLP Jacobian

The normalized squared-attention derivative is implemented explicitly from Q, K, V, rotary, pattern normalization, persistent Layer-0 value mixing, and OV projection. It is composed with the exact RMSNorm and bilinear-MLP derivatives from Layers 7–11. QK-pattern response and value/OV response can be independently enabled, giving an exact two-factor derivative Shapley decomposition.

## Locked hypotheses

- **Q0_manual_attention_trace_is_exact: supported.**
- **Q1_full_weight_jacobian_predicts_native_edits: supported.**
- **Q2_full_jacobian_closes_the_mlp_only_gap: supported.**
- **Q3_full_jacobian_predicts_sparse_grammar: supported.**
- **Q4_qk_response_is_material: supported.**
- **Q5_value_response_is_material: supported.**
- **Q6_qk_value_interaction_is_material: not supported.**
- **Q7_qkov_profile_transfers_ood: supported.**

| Corpus | Full native coord / edge / coeff | MLP-only native | MLP/QK/V projection | QK-V interaction |
|---|---:|---:|---:|---:|
| fineweb | 0.9956 / 0.9961 / 0.9961 | 0.9732 | 0.953/0.020/0.027 | 0.007 |
| wikitext | 0.9965 / 0.9966 / 0.9967 | 0.9712 | 0.944/0.020/0.035 | 0.009 |
| tinystories | 0.9962 / 0.9965 / 0.9964 | 0.9806 | 0.961/0.010/0.029 | 0.005 |

The projections sum to one but are signed and may hide cancellation; norm ratios and the explicit interaction are reported alongside them. This closes a local causal route, not the global task-level explanation for why a text context selects a particular Jacobian.
