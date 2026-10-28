# Orthogonal-pairing red team for chart tensor factors

Every rotation and reflection in a 32-point O(2) grid is allowed to pair the matching and alternative rank-two axes. Best hybrid performance is selected on evaluation itself, making this an intentionally generous oracle test of the one-matching-factor explanation.

## Locked hypotheses

- **R0_native_replay_is_exact: supported.** Observed: `{'maximum_logit_error': 3.337860107421875e-05}`
- **R1_procrustes_conclusion_is_not_assumed_invariant: supported.** Observed: `{'by_target': [True, True]}`
- **R2_full_matching_chart_beats_oracle_hybrids: supported.** Observed: `{'by_target': [True, True]}`
- **R3_no_single_factor_explanation_is_universal: supported.** Observed: `{'jointly_close_by_target': [False, False]}`

| Target | MM logit/CE | AA logit/CE | Hybrid logit range / best gap | Hybrid CE range / best gap | Jointly close hybrids |
|---|---:|---:|---:|---:|---:|
| L12 next:punctuation | 0.492/0.691 | 0.364/0.411 | -0.937..0.228 / +0.264 | -1.307..0.403 / +0.288 | 0 |
| L3 loss:hard | 0.697/0.689 | 0.498/0.495 | -2.203..0.539 / +0.158 | -2.685..0.607 / +0.082 | 0 |

## Guardrail

The grid is an evaluation-selected oracle and therefore cannot validate a deployable axis pairing. It can only falsify claims that interaction is unavoidable under every internal chart symmetry. A surviving gap is strong evidence for paired read/write structure; a rescued hybrid means the previous Procrustes interaction was coordinate-dependent.
