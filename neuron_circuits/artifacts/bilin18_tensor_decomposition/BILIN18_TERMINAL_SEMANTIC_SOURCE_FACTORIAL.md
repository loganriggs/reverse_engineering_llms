# Terminal source factorial across semantic families

Natural cue swaps are decomposed at Layer 17's destination into carried mixed state, boundary attention, nonboundary attention, and the realized final-MLP change. All sixteen coalitions pass through final normalization and the disjoint same-family output contrast; exact finite Shapley values allocate the effect.

## Locked results

- **F0_terminal_source_factorial_recomposes: not supported.**
- **F1_carried_state_is_material_for_transferable_variables: supported.**
- **F2_spelling_uses_more_fresh_attention: supported.**
- **F3_spelling_nonboundary_attention_is_exceptional: supported.**
- **F4_final_MLP_is_not_the_shared_semantic_interface: not supported.**
- **F5_attention_share_predicts_shorter_head_programs: supported.**

Attention-share versus joint-k90 correlation: `-0.919`.

| Family | carried | boundary attention | nonboundary attention | combined attention | final MLP | k90 |
|---|---:|---:|---:|---:|---:|---:|
| spelling | 0.640 | 0.016 | 0.367 | 0.382 | -0.023 | 2.0 |
| sentiment | 0.803 | 0.004 | 0.074 | 0.078 | 0.119 | 5.5 |
| gender | 0.699 | 0.002 | 0.030 | 0.032 | 0.269 | 4.5 |
| tense | 0.936 | 0.001 | 0.014 | 0.015 | 0.048 | 5.0 |
| increasing_scale | 0.850 | 0.002 | 0.041 | 0.044 | 0.106 | 4.5 |
| polarity | 0.827 | 0.000 | 0.015 | 0.015 | 0.158 | 5.0 |

## Numerical closure red team

The locked combined recomposition gate fails because the independently re-summed attention edges differ from the checkpoint's float32 contraction by relative error up to `1.06e-04` (maximum absolute coordinate error `1.56e-02`). The additive final state itself recomposes to about `5e-18`, and finite Shapley output error remains below `2e-7`. The source allocation uses the exact checkpoint attention delta with the independently summed boundary subset and a nonboundary remainder, so the locked edge gate remains failed without compromising final-state closure.

## Interpretation boundary

These projection shares allocate one declared disjoint-family contrast at the final additive boundary. Carried state contains all earlier computation and repeated embedding injection. A broad fresh-attention branch does not imply that the semantic variable itself is distributed there; a small branch cannot be used to infer the organization of the dominant carried mechanism.
