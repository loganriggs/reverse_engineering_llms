# Finite random-basis red team for weight/context route codes

Static weight and CE-conditioned rank-four bases are compared with 32 frozen Haar-random rank-four bases in the same nine-head coefficient space. Known semantic modes are retained equally in every condition; the residual route is projected, and the target-layer MLP plus suffix recompute on a fresh FineWeb panel.

## Locked hypotheses

- **R0_native_path_is_exact: supported.** Observed: `{'maximum_logit_error': 7.152557373046875e-05}`
- **R1_weight_bases_are_finitely_specific: not supported.** Observed: `{'layers': [1, 15]}`
- **R2_context_bases_are_finitely_specific: supported.** Observed: `{'layers': [1, 4, 11, 15]}`
- **R3_L15_static_regime_beats_random: supported.** Observed: `{}`
- **R4_L1_static_regime_beats_random: supported.** Observed: `{}`
- **R5_ce_effect_specificity_agrees_with_logit_specificity: not supported.** Observed: `{'weight_layers': [4, 15], 'context_layers': [4, 15]}`

| L | Weight/context logit | Random logit mean/p95 | Weight/context CE | Random CE mean/p95 |
|---:|---:|---:|---:|---:|
| 1 | 0.988/0.983 | 0.949/0.983 | 0.988/0.981 | 0.953/0.989 |
| 4 | 0.859/0.898 | 0.722/0.860 | 0.979/0.978 | 0.874/0.977 |
| 11 | 0.734/0.793 | 0.636/0.740 | 0.582/0.649 | 0.668/0.813 |
| 15 | 0.844/0.848 | 0.568/0.769 | 0.852/0.862 | 0.613/0.811 |

## Guardrail

Random rank four is a demanding null because the head space has dimension nine. Passing establishes basis specificity under finite behavior; failing means high absolute fidelity is not enough to prefer that ontology. The four layers were selected from earlier regime results, so this tests those regimes rather than estimating prevalence across all depth.
