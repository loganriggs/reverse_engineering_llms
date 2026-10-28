# Finite causality of known semantic and residual MLP codes

On a fourth FineWeb panel, the destination MLP write is replaced by: the fifty-feature semantic prediction, the recursive coarse-interaction prediction, either prediction plus rank-16/64 residual coordinates, or ordinary natural-write PCA. The complete nonlinear suffix recomputes. Because next-token labels enter the semantic prediction, it is a hindsight explanatory code rather than an executable model-internal router.

## Locked hypotheses

- **C0_native_replay_is_exact: supported.** Observed: `{'maximum_logit_error': 5.7220458984375e-05}`
- **C1_declared_semantics_have_late_causal_value: supported.** Observed: `{'fidelity': 0.8610770768376621}`
- **C2_residual_coordinates_add_finite_value: not supported.** Observed: `{'passing': 3}`
- **C3_recursive_semantics_add_finite_value: not supported.** Observed: `{'passing': 1}`
- **C4_late_semantic_plus_residual_is_compact: supported.** Observed: `{'fidelity': 0.9825610554378613}`
- **C5_residual_coordinates_beat_random: supported.** Observed: `{'passing': 4}`
- **C6_natural_and_semantic_residual_codes_are_equivalent: supported.** Observed: `{'passing': 4}`
- **C7_write_reconstruction_predicts_finite_behavior: supported.** Observed: `{'correlation': 0.8766055270275022}`

| Layer | Semantic | Recursive | Semantic+res16/64 | Recursive+res16/64 | Natural PCA16/64 | Random res64 p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.515 | -0.470 | 0.063/0.602 | 0.152/0.626 | -0.206/0.528 | -0.244 |
| 8 | -0.025 | -0.022 | 0.126/0.248 | 0.126/0.246 | 0.100/0.261 | 0.067 |
| 12 | -0.049 | -0.053 | 0.088/0.197 | 0.077/0.192 | 0.082/0.196 | 0.017 |
| 17 | 0.861 | 0.867 | 0.983/0.989 | 0.983/0.989 | 0.981/0.989 | 0.877 |

## Guardrails

These are sufficiency tests relative to deleting one MLP write. Semantic predictions use the true next token and therefore measure explanatory compression, not a deployable forward computation. Residual projections observe the native write to encode its coordinates. Similar fidelity between natural PCA and semantic-plus-residual codes establishes behavioral equivalence at this rank, not identical semantic axes.
