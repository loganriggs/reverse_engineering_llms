# Exact final-query attention routes in Layers 0 and 1

For seventeen held-out lexical substitutions, the final-position attention change is decomposed exactly into native `(QK1)(QK2)V` source edges and heads. QK1, QK2, value, and (at Layer 1) carried-state boundary components are then passed through RMSNorm and the bilinear MLP in every finite coalition. Static head capacity uses only scale-canonicalized QK weight forms and composed OV weights; state and target rankings use the realized edit.

## Locked hypotheses

- **Q0_exact_factorization: not supported.**
- **Q1_layer0_is_single_source_transport: supported.**
- **Q2_layer1_recruits_earlier_context: not supported.**
- **Q3_layer0_value_path_dominates: supported.**
- **Q4_layer1_routing_is_material: not supported.**
- **Q5_three_state_selected_heads_are_sufficient: supported.**
- **Q6_state_selection_beats_static_capacity: supported.**
- **Q7_static_capacity_is_not_a_usage_map: supported.**
- **Q8_no_single_QKOV_factor_is_sufficient: supported.**
- **Q9_factor_ordering_transfers_ood: supported.**

| Layer | Corpus | Source before/edit/after | Full-write factor projection | Head fidelity static/state/target | Static↔usage Spearman |
|---:|---|---:|---:|---:|---:|
| 0 | fineweb | 0.000/1.000/0.000 | qk1:0.154 / qk2:0.157 / value:0.672 | 0.188/0.930/0.930 | 0.033 |
| 0 | tinystories | 0.000/1.000/0.000 | qk1:0.083 / qk2:0.117 / value:0.800 | 0.115/0.921/0.922 | -0.083 |
| 1 | fineweb | 0.005/0.522/0.434 | carried:0.219 / qk1:0.063 / qk2:0.109 / value:0.523 | 0.144/0.855/0.838 | -0.083 |
| 1 | tinystories | 0.018/0.541/0.431 | carried:0.249 / qk1:0.069 / qk2:0.105 / value:0.549 | 0.125/0.846/0.846 | -0.117 |

Maximum algebraic factor error: **3.99e-06**. Maximum finite-Shapley error: **7.81e-08**. Maximum discrepancy to the checkpoint's native float32 contraction order: **4.33e-05**.

The locked exactness gate retains the native-order discrepancy and therefore remains failed if that value exceeds `2e-5`; the separately reported algebraic identity is a post-run diagnostic, not a rewritten hypothesis.

The static score is a state-free capacity measure, not a claim of unique canonical gauge. The state-selected score uses exact head deltas; the target-selected score is an oracle upper control. Source and factor projections are signed and can exceed one or be negative when routes cancel.
