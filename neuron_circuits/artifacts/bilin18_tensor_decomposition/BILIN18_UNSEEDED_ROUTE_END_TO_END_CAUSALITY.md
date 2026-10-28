# End-to-end causality of task-agnostic route representations

Fresh FineWeb sequences are run to four prospectively selected layers. Only the destination's realized nine-head source routes are replaced; the target-layer MLP and all later layers are recomputed. Fidelity is measured relative to the complete finite effect of deleting those destination routes.

## Locked hypotheses

- **E0_native_path_is_exact: supported.** Observed: `{'maximum_logit_error': 4.673004150390625e-05}`
- **E1_continuous_beats_discrete_end_to_end: supported.** Observed: `{'layers': [1, 8, 13, 17]}`
- **E2_semantic_modes_have_finite_causal_coverage: supported.** Observed: `{'by_layer': {'1': 0.8132047437761047, '8': 0.45820668397122866, '13': 0.059393003678892, '17': 0.24294464991771625}, 'mean': 0.3934372703359854}`
- **E3_semantic_binding_beats_permuted_modes: supported.** Observed: `{'observed': 0.3934372703359854, 'null_means': [0.00027985036876915226, 0.23424478725547726, 0.07305115866291217, 0.07020547511900874]}`
- **E4_local_write_fidelity_predicts_suffix_fidelity: supported.** Observed: `{'correlation': 0.8320804949121852, 'cells': 12}`
- **E5_logit_coverage_is_not_only_noise: not supported.** Observed: `{'layers': [1, 8]}`

| Layer | Native/deleted CE | PCA fidelity/CE | Centroid fidelity/CE | Semantic fidelity/CE | Semantic CE recovery | Best permuted fidelity |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.816/4.305 | 0.984/3.834 | 0.896/3.857 | 0.813/3.894 | 0.840 | 0.753 |
| 8 | 3.816/3.881 | 0.804/3.810 | 0.590/3.830 | 0.458/3.840 | 0.630 | 0.054 |
| 13 | 3.816/3.857 | 0.824/3.823 | 0.323/3.836 | 0.059/3.870 | -0.326 | 0.031 |
| 17 | 3.816/3.849 | 0.850/3.820 | 0.584/3.825 | 0.243/3.855 | -0.196 | 0.214 |

## Guardrails

This is finite end-to-end evidence for one destination-route intervention, not a fraction of total model computation. Full-logit fidelity weights all vocabulary coordinates; CE tests the observed next token and can disagree when the removed native route was locally harmful. The four layers were chosen after the local coverage audit, so conclusions about all layers require another predeclared replication.
