# Can the local residual predict the Layer-6 transport Jacobian?

Small finite edits identify one 32-by-32 causal transport matrix per site. Linear models then predict that matrix from PCA coordinates of the unperturbed 768-dimensional local residual. They are fit on FineWeb and frozen before four-times-larger disjoint and OOD edits. The oracle uses each evaluation site's own small-edit matrix and therefore tests whether a sitewise Jacobian is sufficient at all.

## Locked hypotheses

- **J0_sitewise_jacobian_is_the_correct_local_object: supported.**
- **J1_full_context_materially_improves: supported.**
- **J2_full_context_explains_transport: not supported.**
- **J3_full_context_explains_graph_changes: not supported.**
- **J4_full_context_beats_late_coordinates: not supported.**
- **J5_context_rule_transfers_ood: supported.**
- **J6_transport_has_a_low_dimensional_context_code: not supported.**

| Corpus | Global | Late-32 context | Full r32 | Full r64 | Full r128 | Oracle | r64 edge |
|---|---:|---:|---:|---:|---:|---:|---:|
| fineweb | 0.777 | 0.838 | 0.836 | 0.845 | 0.851 | 1.000 | 0.865 |
| wikitext | 0.788 | 0.844 | 0.844 | 0.855 | 0.860 | 1.000 | 0.866 |
| tinystories | 0.799 | 0.855 | 0.858 | 0.866 | 0.870 | 1.000 | 0.884 |

If the oracle succeeds but full-state prediction fails, the computation remains locally simple but its routing rule is not a low-dimensional linear function of the local residual. That would point next to nonlocal attention/value-bus context or a nonlinear partition of state space—not to unstructured noise.
