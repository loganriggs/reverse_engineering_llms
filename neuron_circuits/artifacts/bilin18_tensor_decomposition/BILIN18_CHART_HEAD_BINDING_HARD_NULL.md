# Discovery-selected hard head-binding null

For each target, 4,096 coefficient-preserving head permutations are screened on discovery frozen-route CE fidelity. The best 32 joint, read-only, and write-only permutations are frozen and evaluated with full suffix recomputation on new FineWeb.

## Locked hypotheses

- **U0_native_replay_is_exact: supported.** Observed: `{'maximum_logit_error': 4.1484832763671875e-05}`
- **U1_joint_binding_beats_discovery_selected_nulls: not supported.** Observed: `{'by_target': [False, True]}`
- **U2_read_binding_beats_discovery_selected_nulls: not supported.** Observed: `{'by_target': [False, True]}`
- **U3_write_binding_beats_discovery_selected_nulls: not supported.** Observed: `{'by_target': [False, True]}`
- **U4_native_binding_has_a_material_hard_null_margin: not supported.** Observed: `{'by_target': [False, False]}`

| Target | Matching logit/CE | Joint best + margins | Read best + margins | Write best + margins |
|---|---:|---:|---:|---:|
| L12 next:punctuation | 0.444/-0.151 | 0.422/0.778 (+0.023/-0.929) | 0.390/0.633 (+0.055/-0.785) | 0.389/0.032 (+0.055/-0.184) |
| L3 loss:hard | 0.695/0.755 | 0.654/0.694 (+0.041/+0.061) | 0.607/0.669 (+0.088/+0.086) | 0.588/0.738 (+0.107/+0.017) |

## Guardrail

The hard nulls are selected on a local frozen-route CE metric, not on finite validation logits. This is a strong multiple-comparison control for prediction-relevant binding but not an exhaustive search over all 9! permutations or all possible orthogonal head remappings.
