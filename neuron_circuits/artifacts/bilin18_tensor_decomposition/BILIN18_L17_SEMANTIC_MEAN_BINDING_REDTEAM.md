# Layer-17 semantic mean/binding red team

The late semantic write is compared with its discovery mean, current-feature-only and next-feature-only predictions, and 32 row-permuted semantic writes with the same empirical distribution but wrong context binding. The complete final normalization and unembedding are recomputed.

## Locked hypotheses

- **B0_native_replay_is_exact: supported.** Observed: `{'maximum_logit_error': 2.9087066650390625e-05}`
- **B1_semantics_beats_mean_write: supported.** Observed: `{'logit_advantage': 0.11606554487133558, 'ce_advantage': 0.24801771365799608}`
- **B2_semantics_beats_permuted_binding: supported.** Observed: `{'semantic_logit': 0.8858966242991206, 'permuted_logit_p95': 0.7140664494359393, 'semantic_ce': 0.8373193285615933, 'permuted_ce_p95': 0.6056285048998871}`
- **B3_joint_semantics_beats_single_role_labels: not supported.** Observed: `{'advantage': 0.038550385434046985}`
- **B4_correct_binding_closes_material_remaining_gap: supported.** Observed: `{'fraction': 0.5042624546476087}`
- **B5_write_r2_and_finite_rank_controls_agree: supported.** Observed: `{'observational_order': ['mean', 'current50', 'next50', 'semantic'], 'finite_order': ['mean', 'current50', 'next50', 'semantic']}`

| Method | Write R2 | Full-logit fidelity | CE-effect fidelity |
|---|---:|---:|---:|
| mean | 0.000 | 0.770 | 0.589 |
| current50 | 0.345 | 0.839 | 0.729 |
| next50 | 0.352 | 0.847 | 0.803 |
| semantic | 0.480 | 0.886 | 0.837 |
| permuted p95 | — | 0.714 | 0.606 |

## Guardrail

The semantic predictor uses the true next-token feature labels and remains a hindsight explanation. Beating the mean and row permutation establishes context-bound information beyond a shared output prior; it does not show that the model internally computes these exact fifty labels or that the semantic basis is unique.
