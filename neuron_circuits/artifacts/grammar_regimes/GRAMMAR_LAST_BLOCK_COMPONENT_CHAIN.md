# Exact last-block component chain

The exact last-block local Jacobian is `(I + ML + MR) @ (I + Q + V)`: Q is softmax-pattern response, V is the current-position value response, and ML/MR are the two bilinear product-rule arms. Surrounding A1/A2 factors and final readout remain site-specific while one last-block component is frozen or retained.

## Locked hypotheses

- **D0_component_recomposition_is_exact: supported.**
- **D1_mlp_gating_dominates_mixed_last_block_variation: supported.**
- **D2_mixed_bilinear_arms_are_more_diverse: not supported.**
- **D3_both_bilinear_arms_are_causally_material: supported.**
- **D4_attention_pattern_is_causally_material: supported.**
- **D5_no_single_last_block_component_is_sufficient: supported.**

| Seed | rho | Recompose | rank Q/V/ML/MR | freeze Q/V/ML/MR | only Q/V/ML/MR |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 1.000000 | 6.3/1.4/7.5/7.4 | 0.999/1.000/0.922/0.919 | 0.689/0.688/0.916/0.919 |
| 0 | 0.5 | 1.000000 | 35.0/3.4/32.6/33.5 | 0.935/0.998/0.738/0.744 | 0.207/0.185/0.688/0.687 |
| 1 | 0.0 | 1.000000 | 9.0/2.4/15.8/14.6 | 0.989/1.000/0.903/0.894 | 0.635/0.624/0.883/0.891 |
| 1 | 0.5 | 1.000000 | 35.3/3.4/31.4/30.8 | 0.931/0.995/0.739/0.739 | 0.186/0.166/0.683/0.682 |

Freeze-one fidelity localizes which state-conditioned algebraic component is necessary inside the exact tangent chain. It is an operator recombination, not yet a finite activation ablation.
