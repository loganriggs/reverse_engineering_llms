# Precision red team of the final spelling-source factorial

The initial float32 audit failed its 1e-5 structural closure gate despite exact finite output Shapley closure. This rerun accumulates all residual and edge components in float64, defines nonboundary attention as the exact residual after the boundary-edge component, and compares every behavioral share with the original run.

## Locked results

- **N0_double_source_closure_is_exact: supported.**
- **N1_source_shares_are_precision_stable: supported.**
- **N2_carried_and_attention_conclusion_survives: supported.**
- **N3_nonboundary_attention_conclusion_survives: supported.**
- **N4_final_MLP_is_negligible: supported.**
- **N5_terminal_two_branch_ontology_is_supported: supported.**

Maximum float64 state closure: **4.77e-18**. Maximum output closure: **8.79e-08**. Raw native-attention/edge-sum discrepancy: **0.000375**. Maximum share change: **3.07e-07**.

| Fold/split | carried | boundary attention | nonboundary attention | MLP |
|---|---:|---:|---:|---:|
| 0/0 | 0.706 | 0.069 | 0.225 | 0.000 |
| 0/1 | 0.743 | 0.075 | 0.191 | -0.009 |
| 1/0 | 0.723 | 0.072 | 0.212 | -0.007 |
| 1/1 | 0.658 | 0.069 | 0.276 | -0.002 |
| 2/0 | 0.680 | 0.082 | 0.224 | 0.014 |
| 2/1 | 0.668 | 0.075 | 0.256 | 0.001 |

The raw edge-sum discrepancy is retained as a numerical audit of reproducing the checkpoint's float32 contraction order; it is not silently relabeled exact. Float64 residual closure tests the algebraic source partition, while exact finite output closure tests the behavioral allocation.
