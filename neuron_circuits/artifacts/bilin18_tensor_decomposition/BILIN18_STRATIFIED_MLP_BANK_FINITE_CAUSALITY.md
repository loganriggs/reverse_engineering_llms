# Finite causality of natural-stratum MLP channel banks

At each selected natural site, all destination-position MLP products outside a frozen 128-channel bank are zeroed and the complete nonlinear suffix recomputes. Specialized, global, and random banks have identical channel budgets.

## Locked hypotheses

- **N0_native_replay_is_exact: supported.** Observed: `{'maximum_logit_error': 3.147125244140625e-05}`
- **N1_specialized_banks_improve_finite_logits: not supported.** Observed: `{'wins': 3, 'mean_advantage': 0.01422822240300961}`
- **N2_specialized_banks_improve_finite_ce: not supported.** Observed: `{'wins': 2, 'mean_advantage': 0.035996283801092405}`
- **N3_specialized_banks_beat_random_controls: not supported.** Observed: `{'wins': 3, 'targets': 4}`
- **N4_fixed_128_channel_banks_are_often_sufficient: not supported.** Observed: `{'passing': 1}`
- **N5_local_linear_advantage_predicts_finite_advantage: not supported.** Observed: `{'correlation': -0.3453309852700405}`

| Target | Examples | Local/global logit | Advantage | Local/global CE | Advantage | Random logit p95 | Local linear advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1:current:function | 72 | 0.115/0.066 | +0.048 | 0.065/-0.126 | +0.191 | 0.210 | +0.178 |
| L8:next:punctuation | 50 | 0.088/0.081 | +0.006 | 0.100/0.132 | -0.033 | 0.055 | +0.189 |
| L12:next:capitalized | 20 | 0.115/0.121 | -0.006 | 0.386/0.416 | -0.030 | 0.052 | +0.274 |
| L17:next:continuation | 20 | 0.922/0.914 | +0.009 | 0.988/0.972 | +0.016 | 0.213 | +0.136 |

## Guardrail

The bank keeps each selected channel's native product and deletes all other destination products. This tests sufficiency relative to deleting that position's complete MLP write, not necessity in the intact model. Candidates were selected after the census but frozen before this text panel. Four positive cells cannot establish prevalence across all strata.
