# Known-ontology residual atlas

A declared bank of fifty token features is used only after collecting random FineWeb MLP writes. Linear models predict each layer's exact write from current-token features, next-token features, or both plus position. PCA is then fit to the unexplained write residual on discovery and evaluated on a disjoint panel. Exact current-token lookup is a high-capacity lexical control.

## Locked hypotheses

- **K0_panels_are_fresh_and_task_blind_before_residualization: supported.** Observed: `{'ranges': [[15000, 15255], [15256, 15511]]}`
- **K1_declared_features_explain_substantial_write_variance: not supported.** Observed: `{'passing': 1}`
- **K2_exact_current_token_is_not_a_complete_ontology: supported.** Observed: `{'r2': {'1': 0.2511439919471741, '8': 0.045963943004608154, '12': 0.052363038063049316, '17': 0.36319559812545776}}`
- **K3_residual_subspaces_recur: supported.** Observed: `{'passing': 3}`
- **K4_residual_coordinates_add_material_coverage: supported.** Observed: `{'passing': 4}`
- **K5_stable_residual_axes_exist: supported.** Observed: `{'eligible': 23}`
- **K6_residual_axes_escape_the_declared_vocabulary: not supported.** Observed: `{'passing': 3, 'selected': 8}`
- **K7_late_known_plus_residual_code_is_compact: supported.** Observed: `{'r2': 0.9645846984022707}`

| Layer | Current50 | Next50 | Declared+position | Token lookup | Natural PCA16/64 | Residual overlap16/64 | Known+residual total R2 16/64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.163 | 0.045 | 0.176 | 0.251 | 0.302/0.480 | 0.903/0.845 | 0.351/0.497 |
| 8 | 0.030 | 0.025 | 0.040 | 0.046 | 0.161/0.303 | 0.732/0.744 | 0.166/0.302 |
| 12 | 0.044 | 0.034 | 0.054 | 0.052 | 0.169/0.282 | 0.694/0.668 | 0.167/0.277 |
| 17 | 0.335 | 0.341 | 0.450 | 0.363 | 0.963/0.975 | 0.938/0.848 | 0.965/0.975 |

## Frozen residual-axis candidates

| Candidate | Axis cosine | Residual variance | Best declared feature | AUROC |
|---|---:|---:|---|---:|
| L17A0 | +0.974 | 0.327 | next:slash | 0.770 |
| L17A1 | +0.929 | 0.270 | current:weekday | 0.803 |
| L17A2 | +0.944 | 0.196 | current:slash | 0.834 |
| L17A3 | -0.987 | 0.064 | current:month | 0.837 |
| L12A0 | -0.983 | 0.042 | current:newline | 0.750 |
| L17A4 | +0.990 | 0.036 | current:newline | 0.780 |
| L1A0 | -0.986 | 0.032 | current:month | 0.745 |
| L1A1 | -0.970 | 0.029 | next:weekday | 0.696 |

## Guardrails

The fifty labels are an anchor ontology, not exhaustive ground truth. Next-token labels are diagnostic oracle metadata and do not define an executable forward router. Residual PCA axes are candidate coordinates: subspace recurrence is stronger evidence than individual-axis recurrence, and semantic names require held-out causal transfer. High token-lookup R2 would indicate lexical inheritance, not a complete explanation of contextual use.
