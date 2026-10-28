# Large-panel L12 punctuation replication

After one 38-example panel produced negative signed CE-effect fidelity, the frozen punctuation and numeric charts are compared on 256 new FineWeb rows. Uncertainty resamples FineWeb rows rather than treating prediction positions as independent.

## Locked hypotheses

- **V0_native_replay_is_exact: supported.** Observed: `{'maximum_logit_error': 2.0265579223632812e-05}`
- **V1_matching_chart_has_stable_logit_coverage: supported.** Observed: `{'fidelity': 0.6091296080632673}`
- **V2_matching_chart_has_positive_ce_coverage: supported.** Observed: `{'fidelity': 0.691690662172828}`
- **V3_ce_coverage_survives_row_cluster_uncertainty: supported.** Observed: `{'bootstrap_mean': 0.6827566481938555, 'p05': 0.5279316689252107, 'p95': 0.8011150211619287, 'rows': 132, 'examples': 159}`
- **V4_punctuation_chart_beats_numeric_chart: supported.** Observed: `{'logit_advantage': 0.13548471215133484, 'ce_advantage': 0.36688246370977007}`

| Method | Examples | Full-logit fidelity | CE-effect fidelity | CE cosine | CE norm ratio |
|---|---:|---:|---:|---:|---:|
| matching | 159 | 0.609 | 0.692 | 0.835 | 0.759 |
| alternative | 159 | 0.474 | 0.325 | 0.570 | 0.564 |

## Guardrail

A larger aggregate can establish typical transfer while retaining the earlier negative panel as real heterogeneity. The row bootstrap measures sampling uncertainty over this panel; it does not erase distribution shift or prove that every punctuation context uses the chart.
