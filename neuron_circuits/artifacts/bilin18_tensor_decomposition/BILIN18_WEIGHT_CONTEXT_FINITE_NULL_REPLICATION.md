# Finite random-basis red team for weight/context route codes

Static weight and CE-conditioned rank-four bases are compared with 32 frozen Haar-random rank-four bases in the same nine-head coefficient space. Known semantic modes are retained equally in every condition; the residual route is projected, and the target-layer MLP plus suffix recompute on a fresh FineWeb panel.

## Locked hypotheses

- **R0_native_path_is_exact: supported.** Observed: `{'maximum_logit_error': 6.747245788574219e-05}`
- **R1_weight_bases_are_finitely_specific: not supported.** Observed: `{'layers': [1, 15]}`
- **R2_context_bases_are_finitely_specific: supported.** Observed: `{'layers': [4, 11, 15]}`
- **R3_L15_static_regime_beats_random: supported.** Observed: `{}`
- **R4_L1_static_regime_beats_random: supported.** Observed: `{}`
- **R5_ce_effect_specificity_agrees_with_logit_specificity: not supported.** Observed: `{'weight_layers': [15], 'context_layers': [4, 11, 15]}`

| L | Weight/context logit | Random logit mean/p95 | Weight/context CE | Random CE mean/p95 |
|---:|---:|---:|---:|---:|
| 1 | 0.985/0.977 | 0.962/0.982 | 0.977/0.875 | 0.955/0.987 |
| 4 | 0.804/0.869 | 0.703/0.829 | 0.806/0.917 | 0.775/0.893 |
| 11 | 0.724/0.777 | 0.656/0.737 | 0.724/0.748 | 0.602/0.731 |
| 15 | 0.809/0.811 | 0.537/0.689 | 0.864/0.808 | 0.504/0.700 |

## Guardrail

Random rank four is a demanding null because the head space has dimension nine. Passing establishes basis specificity under finite behavior; failing means high absolute fidelity is not enough to prefer that ontology. The four layers were selected from earlier regime results, so this tests those regimes rather than estimating prevalence across all depth.
