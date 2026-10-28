# Held-out operation-family alignment tournament

All maps are fitted from unlabelled anchor activations before intervention outcomes are computed. Native permutation, global gate ridge, relation-conditioned ridge, and a weight-quotiented residual-write map compete on finite target-model patches. Fidelity is measured against the target model's own activation delta, only where that own edit has Fisher norm at least 0.1.

## Locked hypotheses

- **T0_pure_semantic_program_transfers: not supported.**
- **T1_semantic_program_is_not_a_neuron_permutation: not supported.**
- **T2_tensor_write_quotient_transfers_semantics: not supported.**
- **T3_relation_conditioning_improves_semantic_transfer: not supported.**
- **T4_arbitrary_fact_swaps_do_not_share_one_global_gate_program: supported.**
- **T5_tensor_quotient_recovers_arbitrary_swaps: not supported.**
- **T6_semantic_transfer_is_more_recurrent_than_generic_prompt_swaps: not supported.**
- **T7_own_delta_is_an_effectful_ceiling: not supported.**

## Median effect fidelity across ordered model pairs

| Regime/family/layer | Native | Global ridge | Relation ridge | Write quotient | Source effect |
|---|---:|---:|---:|---:|---:|
| pure / semantic / L1 | 0.274 | 0.987 | 1.000 | 0.906 | 0.541 |
| pure / semantic / L2 | 0.525 | 0.998 | 1.000 | 0.983 | 0.745 |
| pure / semantic / L3 | 0.875 | 0.996 | 0.999 | 0.988 | 0.742 |
| pure / generic_pair / L1 | 0.253 | 0.920 | 0.943 | 0.858 | 0.243 |
| pure / generic_pair / L2 | 0.567 | 0.871 | 0.913 | 0.834 | 0.305 |
| pure / generic_pair / L3 | 0.732 | 0.406 | 0.564 | 0.334 | 0.709 |
| mixed / semantic / L1 | 0.028 | 0.431 | 0.653 | 0.274 | -0.420 |
| mixed / semantic / L2 | 0.045 | 0.449 | 0.647 | 0.314 | -0.081 |
| mixed / semantic / L3 | 0.062 | 0.582 | 0.691 | 0.465 | 0.575 |
| mixed / generic_pair / L1 | 0.009 | 0.050 | 0.172 | 0.050 | -0.116 |
| mixed / generic_pair / L2 | 0.017 | 0.070 | 0.026 | 0.057 | -0.174 |
| mixed / generic_pair / L3 | 0.045 | 0.212 | -0.160 | 0.241 | -0.004 |
| mixed / arbitrary_swap / L1 | 0.004 | 0.085 | 0.153 | 0.041 | -0.079 |
| mixed / arbitrary_swap / L2 | 0.006 | 0.072 | 0.069 | 0.039 | 0.036 |
| mixed / arbitrary_swap / L3 | 0.042 | 0.195 | -0.053 | 0.249 | 0.104 |

The semantic family uses minimal direct-rule transitions, with delta discovery on template 0 and finite evaluation on disjoint identities in template 6. Arbitrary swaps pair two learned randomized facts with the same relation and structured answer but different stored labels. Generic pairs are unrestricted learned prompt swaps. The own target delta is a causal ceiling, not an input to any alignment fit.
