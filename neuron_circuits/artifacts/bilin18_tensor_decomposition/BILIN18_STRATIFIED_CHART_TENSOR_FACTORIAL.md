# Route-read × head-write tensor factorial

For each specific natural chart, the matching and strongest alternative rank-two bases are Procrustes-aligned and crossed. `MA` reads matching route coordinates but writes them through the alternative head/OV coalition; `AM` does the reverse. Only selected residual source edges are replaced.

## Locked hypotheses

- **Q0_native_replay_is_exact: supported.** Observed: `{'maximum_logit_error': 2.9087066650390625e-05}`
- **Q1_tensor_factors_are_behaviorally_distinguishable: supported.** Observed: `{'by_target': [True, False]}`
- **Q2_matching_factor_adds_to_the_wrong_chart: not supported.** Observed: `{'by_target': [True, False]}`
- **Q3_full_matching_chart_is_not_reducible_to_one_matching_factor: supported.** Observed: `{'by_target': [True, True]}`
- **Q4_factor_roles_need_not_match_across_regimes: not supported.** Observed: `{'larger_roles': {'logit': ['write', 'write'], 'ce': ['read', 'read']}}`

| Target | Principal cosines | Logit MM/AA/MA/AM | Read/write/joint/interaction gain | CE MM/AA/MA/AM | Read/write/joint/interaction gain |
|---|---:|---:|---:|---:|---:|
| L12 next:punctuation | [0.712, 0.267] | 0.566/0.497/0.254/0.377 | -0.243/-0.120/+0.069/+0.432 | 0.782/0.287/0.669/0.208 | +0.381/-0.079/+0.495/+0.192 |
| L3 loss:hard | [0.967, 0.49] | 0.712/0.600/0.559/0.594 | -0.041/-0.006/+0.112/+0.159 | 0.771/0.744/0.705/0.702 | -0.039/-0.042/+0.027/+0.108 |

## Interpretation boundary

The read factor acts on realized head-route coefficients produced by QK; it is not a pure intervention on QK weights. The write factor remixes those coordinates into native heads whose contextual writes include OV and value state. The factorial therefore localizes specificity to the route-read versus head-payload interface, not to a single static matrix.
