# Final-source factorial for the L8H2 spelling edit

At Layer 17's destination, the exact changed output residual is partitioned into carried mixed-state change, boundary-source attention edges, nonboundary attention edges, and the realized final-MLP write change. All sixteen additive source coalitions pass through final RMSNorm, unembedding, and logit capping; finite Shapley values exactly allocate the declared contrast.

## Locked results

- **F0_final_source_factorial_recomposes: not supported.**
- **F1_nonboundary_attention_exceeds_boundary_attention: supported.**
- **F2_final_attention_is_material: supported.**
- **F3_carried_residual_is_material: supported.**
- **F4_final_MLP_is_material: not supported.**
- **F5_terminal_use_is_multi_branch: not supported.**

| Fold/split | total win/mean | carried share | boundary-attn share | nonboundary-attn share | MLP share |
|---|---:|---:|---:|---:|---:|
| 0/0 | 0.914/+0.09846 | 0.706 | 0.069 | 0.225 | 0.000 |
| 0/1 | 0.805/+0.06959 | 0.743 | 0.075 | 0.191 | -0.009 |
| 1/0 | 0.914/+0.08336 | 0.723 | 0.072 | 0.212 | -0.007 |
| 1/1 | 0.883/+0.06073 | 0.658 | 0.069 | 0.276 | -0.002 |
| 2/0 | 0.922/+0.08936 | 0.680 | 0.082 | 0.224 | 0.014 |
| 2/1 | 0.859/+0.07861 | 0.668 | 0.075 | 0.256 | 0.001 |

## Interpretation boundary

These are exact allocations at the final additive residual boundary, not exclusive storage locations. The carried state already contains earlier attention and MLP computation, while the realized final-MLP delta depends on carried and attention inputs. The source factorial answers which terminal branches express the intervention under the declared output contrast; it complements rather than replaces the path-specific edge freezes.
