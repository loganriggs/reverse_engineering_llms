# Finite causality of natural-stratum route charts

For each frozen target, only residual route components belonging to the selected natural stratum are replaced. All other source edges remain native. Local and global rank-two charts, deletion, and sixteen random charts are propagated through the full suffix.

## Locked hypotheses

- **F0_native_replay_is_exact: supported.** Observed: `{'maximum_logit_error': 3.6835670471191406e-05}`
- **F1_specialized_charts_improve_finite_logit_recovery: supported.** Observed: `{'wins': 4, 'mean_advantage': 0.25867414509757947, 'by_target': [0.2980802258269367, 0.39945090236825265, 0.20805399991914486, 0.12911145227598353]}`
- **F2_specialized_charts_improve_finite_ce_recovery: supported.** Observed: `{'wins': 4, 'mean_advantage': 0.17922634353583658}`
- **F3_specialized_charts_beat_equal_rank_random_controls: supported.** Observed: `{'wins': 4, 'targets': 4}`
- **F4_local_geometry_predicts_finite_advantage: supported.** Observed: `{'correlation': 0.7353690566873688}`

| Target | Examples | Local/global logit fidelity | Advantage | Local/global CE fidelity | Advantage | Random p95 | Observational advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| L8:source:function | 256 | 0.496/0.198 | +0.298 | 0.631/0.227 | +0.403 | 0.226 | +0.897 |
| L6:distance:mid17-64 | 256 | 0.471/0.071 | +0.399 | 0.199/0.061 | +0.139 | 0.354 | +0.720 |
| L12:next:punctuation | 36 | 0.652/0.444 | +0.208 | 0.568/0.452 | +0.116 | 0.422 | +0.684 |
| L3:loss:hard | 62 | 0.623/0.494 | +0.129 | 0.486/0.426 | +0.059 | 0.511 | +0.123 |

## Guardrails

The four targets were selected after the observational census but before this fresh panel. This validates the selected cells, not the prevalence of specialization across every cell. Deleting a stratum's residual route component defines a conditional route intervention, not a complete causal decomposition of the model. Loss strata use discovery-panel quartiles and are selected using native loss before intervention.
