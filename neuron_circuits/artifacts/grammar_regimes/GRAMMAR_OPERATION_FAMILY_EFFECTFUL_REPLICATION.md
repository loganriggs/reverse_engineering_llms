# Held-out operation-family alignment tournament

All maps are fitted from unlabelled anchor activations before intervention outcomes are computed. Native permutation, global gate ridge, relation-conditioned ridge, and a weight-quotiented residual-write map compete on finite target-model patches. Fidelity is measured against the target model's own activation delta, only where that own edit has Fisher norm at least 0.1.

## Locked hypotheses

- **R0_pure_semantic_program_transfers: supported.**
- **R1_semantic_program_is_not_a_neuron_permutation: not supported.**
- **R2_tensor_write_quotient_transfers_semantics: supported.**
- **R3_relation_conditioning_improves_semantic_transfer: not supported.**
- **R4_arbitrary_fact_swaps_do_not_share_one_global_gate_program: supported.**
- **R5_tensor_quotient_recovers_arbitrary_swaps: not supported.**
- **R6_semantic_transfer_is_more_recurrent_than_generic_prompt_swaps: supported.**
- **R7_own_delta_is_an_effectful_ceiling: supported.**

## Median effect fidelity across ordered model pairs

| Regime/family/layer | Native | Global ridge | Relation ridge | Write quotient | Source effect |
|---|---:|---:|---:|---:|---:|
| pure / semantic / L1 | 0.295 | 0.990 | 1.000 | 0.923 | 0.551 |
| pure / semantic / L2 | 0.529 | 0.998 | 1.000 | 0.985 | 0.751 |
| pure / semantic / L3 | 0.887 | 0.998 | 1.000 | 0.988 | 0.741 |
| pure / generic_pair / L1 | 0.233 | 0.915 | 0.937 | 0.846 | 0.421 |
| pure / generic_pair / L2 | 0.513 | 0.870 | 0.873 | 0.838 | 0.295 |
| pure / generic_pair / L3 | 0.631 | 0.469 | 0.500 | 0.492 | 0.717 |
| mixed / semantic / L1 | 0.031 | 0.452 | 0.703 | 0.289 | -0.458 |
| mixed / semantic / L2 | 0.043 | 0.482 | 0.681 | 0.337 | -0.070 |
| mixed / semantic / L3 | 0.068 | 0.562 | 0.683 | 0.440 | 0.643 |
| mixed / generic_pair / L1 | 0.008 | 0.109 | 0.200 | 0.071 | -0.094 |
| mixed / generic_pair / L2 | 0.006 | 0.089 | 0.079 | 0.062 | -0.094 |
| mixed / generic_pair / L3 | 0.042 | 0.241 | -0.090 | 0.259 | 0.111 |
| mixed / arbitrary_swap / L1 | 0.004 | 0.105 | 0.156 | 0.047 | -0.025 |
| mixed / arbitrary_swap / L2 | 0.004 | 0.076 | 0.103 | 0.055 | 0.021 |
| mixed / arbitrary_swap / L3 | 0.047 | 0.246 | -0.107 | 0.272 | 0.242 |

The semantic family uses minimal direct-rule transitions, with delta discovery on template 0 and finite evaluation on disjoint identities in template 6. Arbitrary swaps pair two learned randomized facts with the same relation and structured answer but different stored labels. Generic pairs are unrestricted learned prompt swaps. The own target delta is a causal ceiling, not an input to any alignment fit.
