# Coefficient-matched head-binding red team

The matching chart's head coefficients are preserved exactly but reassigned to random native head identities. Joint, read-only, and write-only permutations are tested with full suffix recomputation on fresh FineWeb.

## Locked hypotheses

- **P0_native_replay_is_exact: supported.** Observed: `{'maximum_logit_error': 4.38690185546875e-05}`
- **P1_native_chart_beats_joint_head_permutation: supported.** Observed: `{'by_target': [True, True]}`
- **P2_native_read_binding_is_specific: supported.** Observed: `{'by_target': [True, True]}`
- **P3_native_write_binding_is_specific: supported.** Observed: `{'by_target': [True, True]}`
- **P4_tensor_head_binding_survives_geometry_redteam: supported.** Observed: `{'by_target': [True, True]}`

| Target | Matching logit/CE | Joint null p95 logit/CE | Read null p95 logit/CE | Write null p95 logit/CE |
|---|---:|---:|---:|---:|
| L12 next:punctuation | 0.473/0.652 | 0.425/0.648 | 0.175/0.345 | 0.114/0.417 |
| L3 loss:hard | 0.685/0.670 | 0.514/0.662 | 0.151/0.136 | 0.194/0.092 |

## Interpretation boundary

Column permutation preserves rank, orthonormality, coefficient values, and factor budget while destroying native head identity. A native advantage supports learned binding to the model's head-indexed QK routes and contextual OV writes. It does not isolate static QK weights from the input-conditioned route values or static OV from the value state.
