# CE-adjoint residual route atlas

Every realized source route is treated as an independent local variable while downstream values and MLP states recompute and downstream QK routes stay frozen. Previously validated semantic modes are subtracted. The highest absolute residual next-token CE effects are retained without token labels on two disjoint FineWeb panels.

## Locked hypotheses

- **A0_selection_is_task_and_label_blind: supported.** Observed: `{'ranges': [[3072, 3135], [3328, 3391]]}`
- **A1_known_semantic_modes_leave_most_ce_sensitivity_unexplained: supported.** Observed: `{'panels': {'discovery': {'semantic_effect_fidelity': 0.189887713774199, 'full_square': 68.7950802306732, 'residual_square': 55.731739726758065}, 'replication': {'semantic_effect_fidelity': 0.12016565702601778, 'full_square': 49.330224007266224, 'residual_square': 43.40242522819244}}}`
- **A2_ce_important_residual_subspaces_recur: supported.** Observed: `{'layers': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}`
- **A3_recurrence_beats_destroyed_head_binding: supported.** Observed: `{'observed': 0.913655698851049, 'null_mean': 0.5926926986713308, 'null_p95': 0.6233936492053754}`
- **A4_residual_geometry_is_better_continuous_than_discrete: supported.** Observed: `{'layers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}`
- **A5_residual_edges_are_not_token_lookup_artifacts: supported.** Observed: `{'layers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}`

| L | Semantic CE fidelity D/R | Residual subspace cosine | k4 center cosine | PCA/k4 reconstruction | Distinct sources D/R |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000/0.000 | 0.934 | 0.750 | 0.747/0.490 | 346/327 |
| 1 | 0.352/0.237 | 0.994 | 0.849 | 0.725/0.392 | 327/321 |
| 2 | 0.000/0.000 | 0.949 | 0.971 | 0.742/0.527 | 335/320 |
| 3 | 0.000/0.000 | 0.983 | 0.927 | 0.705/0.429 | 344/319 |
| 4 | 0.503/0.462 | 0.984 | 0.971 | 0.690/0.296 | 342/303 |
| 5 | 0.000/0.000 | 0.933 | 0.923 | 0.752/0.529 | 319/313 |
| 6 | 0.034/0.049 | 0.766 | 0.669 | 0.667/0.285 | 356/337 |
| 7 | 0.000/0.000 | 0.936 | 0.945 | 0.788/0.563 | 342/327 |
| 8 | 0.795/0.484 | 0.864 | 0.913 | 0.704/0.227 | 379/363 |
| 9 | 0.000/0.000 | 0.912 | 0.848 | 0.689/0.372 | 305/293 |
| 10 | 0.000/0.000 | 0.983 | 0.941 | 0.794/0.554 | 387/360 |
| 11 | 0.356/0.519 | 0.908 | 0.851 | 0.708/0.311 | 341/326 |
| 12 | 0.000/0.000 | 0.835 | 0.781 | 0.607/0.228 | 346/357 |
| 13 | 0.199/0.247 | 0.933 | 0.896 | 0.659/0.285 | 395/377 |
| 14 | 0.061/0.063 | 0.853 | 0.910 | 0.761/0.480 | 428/404 |
| 15 | 0.000/0.000 | 0.894 | 0.978 | 0.716/0.506 | 393/354 |
| 16 | 0.000/0.000 | 0.916 | 0.716 | 0.753/0.531 | 416/395 |
| 17 | 0.129/0.023 | 0.868 | 0.593 | 0.763/0.529 | 354/316 |

## Guardrails

The adjoint is local to the realized route variables and freezes downstream QK routes; it is not a finite deletion and not an additive attribution of total loss across layers. Top-residual selection conditions the atlas on large first-order effects. Recurrence and null comparisons test structure in that selected residual, not the fraction of the entire model explained.
