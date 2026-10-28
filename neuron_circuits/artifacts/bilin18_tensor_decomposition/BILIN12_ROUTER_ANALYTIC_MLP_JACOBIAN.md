# Weight-derived analytic Jacobian chain

For each natural site, the exact RMSNorm derivative and bilinear-MLP derivative are composed from Layers 7–10, with attention responses frozen as in the causal operator audit. Nothing is fit to perturbation outcomes. The resulting tangent prediction is compared with finite edits, the complete native path, and a version forcibly projected into the same fixed 32-dimensional late subspace after every MLP.

## Locked hypotheses

- **A0_weight_jacobian_matches_finite_mlp_path: supported.**
- **A1_weight_jacobian_explains_native_transport: supported.**
- **A2_weight_jacobian_predicts_graph_changes: supported.**
- **A3_fixed_late_subspace_is_sufficient: supported.**
- **A4_off_subspace_routes_are_material: supported.**
- **A5_weight_derived_account_transfers_ood: supported.**

| Corpus | Analytic→MLP path | Analytic→native | Repeated-32D→MLP path | Edge fidelity |
|---|---:|---:|---:|---:|
| fineweb | 1.0000 | 0.9773 | 0.8785 | 1.0000 |
| wikitext | 1.0000 | 0.9743 | 0.8953 | 1.0000 |
| tinystories | 1.0000 | 0.9841 | 0.8947 | 1.0000 |

A successful full-space Jacobian is a mechanistic local explanation: weights plus the current activation determine the causal transport. It is not yet a global semantic explanation, because it does not compress why natural states select those Jacobians. The repeated-projection control tests whether the 32D endpoint code is also an adequate internal carrier through the whole computation.
