# Stratified natural route census

Every source edge at four prediction positions on two fresh FineWeb panels is retained. Previously named modes are frozen; all other bases are fitted without task labels. Equal-rank global and stratum-specific residual charts are compared on the replication panel.

## Locked hypotheses

- **B0_panels_are_fresh_and_label_blind: supported.** Observed: `{'ranges': [[6144, 6207], [6400, 6463]]}`
- **B1_known_semantic_coverage_is_heterogeneous: supported.** Observed: `{'layers': [1, 4, 6, 8, 11, 13, 14, 17], 'ranges': {'0': 0.0, '1': 0.884186863899231, '2': 0.0, '3': 0.0, '4': 0.7942094802856445, '5': 0.0, '6': 0.28732502460479736, '7': 0.0, '8': 0.8548812866210938, '9': 0.0, '10': 0.0, '11': 0.6420229077339172, '12': 0.0, '13': 0.49367755651474, '14': 0.4591671824455261, '15': 0.0, '16': 0.0, '17': 0.8367285132408142}}`
- **B2_one_global_low_rank_chart_is_not_universal: supported.** Observed: `{'passing_cells': 143, 'eligible_cells': 414, 'fraction': 0.34541062801932365}`
- **B3_stratum_specific_structure_replicates: supported.** Observed: `{'observed': 0.7821840234078122, 'null_p95': 0.6163140327225151}`
- **B4_extra_rank_reduces_regime_dependence: not supported.** Observed: `{'rank2_median_advantage': 0.014111818641582063, 'rank4_median_advantage': 0.019952509228683968}`
- **B5_current_named_modes_are_not_close_to_complete: supported.** Observed: `{'complete_layers': []}`

## Rank-two stratum summary

| Stratum | Cells | Semantic fidelity median [min,max] | Local/global residual fidelity | Local advantage | Recurrence | Local beats random p95 |
|---|---:|---:|---:|---:|---:|---:|
| distance:far65+ | 18 | 0.000 [-0.000,0.890] | 0.658/0.428 | +0.229 | 0.801 | 1.00 |
| distance:local1-4 | 18 | 0.000 [0.000,0.622] | 0.695/0.683 | +0.012 | 0.853 | 0.89 |
| distance:mid17-64 | 18 | 0.000 [-0.000,0.531] | 0.706/0.467 | +0.240 | 0.895 | 0.94 |
| distance:near5-16 | 18 | 0.000 [0.000,0.217] | 0.662/0.490 | +0.172 | 0.772 | 0.94 |
| distance:self | 18 | 0.000 [-0.128,0.885] | 0.691/0.588 | +0.103 | 0.855 | 1.00 |
| loss:easy | 18 | 0.000 [0.000,0.924] | 0.624/0.635 | -0.011 | 0.758 | 0.72 |
| loss:hard | 18 | 0.000 [0.000,0.518] | 0.724/0.728 | -0.004 | 0.803 | 0.94 |
| loss:medium-high | 18 | 0.000 [0.000,0.682] | 0.647/0.637 | +0.010 | 0.845 | 0.83 |
| loss:medium-low | 18 | 0.000 [-0.030,0.450] | 0.602/0.624 | -0.022 | 0.793 | 0.72 |
| next:capitalized | 18 | 0.000 [-0.008,0.906] | 0.727/0.672 | +0.055 | 0.801 | 0.94 |
| next:continuation | 18 | 0.000 [0.000,0.787] | 0.611/0.641 | -0.031 | 0.759 | 0.67 |
| next:function | 18 | 0.000 [0.000,0.290] | 0.630/0.614 | +0.016 | 0.814 | 0.83 |
| next:numeric | 18 | 0.000 [-0.004,0.108] | 0.684/0.776 | -0.093 | 0.613 | 0.61 |
| next:other | 18 | 0.000 [0.000,0.580] | 0.722/0.710 | +0.011 | 0.832 | 0.94 |
| next:punctuation | 18 | 0.000 [0.000,0.455] | 0.620/0.523 | +0.096 | 0.828 | 0.78 |
| next:specialized | 18 | 0.000 [-0.527,0.707] | 0.374/0.424 | -0.050 | 0.560 | 0.39 |
| source:capitalized | 18 | 0.000 [-0.001,0.874] | 0.725/0.702 | +0.023 | 0.821 | 0.78 |
| source:continuation | 18 | 0.000 [0.000,0.834] | 0.714/0.679 | +0.034 | 0.811 | 0.83 |
| source:function | 18 | 0.000 [-0.007,0.562] | 0.723/0.619 | +0.104 | 0.818 | 0.83 |
| source:numeric | 18 | 0.000 [-0.007,0.377] | 0.561/0.466 | +0.095 | 0.644 | 0.61 |
| source:other | 18 | 0.000 [0.000,0.539] | 0.720/0.703 | +0.017 | 0.839 | 1.00 |
| source:punctuation | 18 | 0.000 [-0.000,0.489] | 0.646/0.453 | +0.193 | 0.785 | 0.72 |
| source:specialized | 18 | 0.000 [-0.018,0.787] | 0.565/0.490 | +0.075 | 0.689 | 0.56 |

## Largest departures from the global rank-two chart

| L | Stratum | Local/global | Advantage | Semantic fidelity | Recurrence |
|---:|---|---:|---:|---:|---:|
| 2 | source:numeric | 0.549/-0.926 | +1.475 | 0.000 | 0.776 |
| 8 | source:function | 0.923/0.026 | +0.897 | 0.069 | 0.882 |
| 6 | distance:mid17-64 | 0.836/0.116 | +0.720 | 0.089 | 0.950 |
| 12 | next:punctuation | 0.800/0.116 | +0.684 | 0.000 | 0.921 |
| 6 | distance:far65+ | 0.705/0.066 | +0.639 | 0.288 | 0.739 |
| 8 | distance:near5-16 | 0.736/0.109 | +0.627 | 0.217 | 0.741 |
| 13 | source:specialized | 0.824/0.200 | +0.624 | -0.000 | 0.908 |
| 8 | distance:mid17-64 | 0.675/0.119 | +0.556 | 0.531 | 0.661 |
| 2 | distance:far65+ | 0.743/0.214 | +0.530 | 0.000 | 0.898 |
| 2 | distance:mid17-64 | 0.600/0.109 | +0.491 | 0.000 | 0.950 |
| 12 | source:punctuation | 0.617/0.126 | +0.491 | 0.000 | 0.966 |
| 13 | next:continuation | 0.889/0.398 | +0.490 | 0.000 | 0.962 |
| 3 | source:numeric | 0.029/0.687 | -0.658 | 0.000 | 0.490 |
| 17 | next:continuation | 0.125/0.635 | -0.510 | 0.217 | 0.405 |
| 0 | next:numeric | 0.444/0.936 | -0.492 | 0.000 | 0.577 |
| 16 | source:punctuation | 0.271/0.724 | -0.454 | 0.000 | 0.491 |
| 15 | loss:easy | 0.291/0.721 | -0.430 | 0.000 | 0.541 |
| 13 | distance:local1-4 | 0.349/0.716 | -0.366 | 0.099 | 0.592 |
| 1 | next:numeric | 0.462/0.814 | -0.352 | 0.070 | 0.477 |
| 8 | source:specialized | 0.277/0.605 | -0.329 | 0.774 | 0.541 |
| 7 | next:specialized | 0.039/0.362 | -0.322 | 0.000 | 0.480 |
| 13 | next:specialized | 0.006/0.311 | -0.304 | 0.209 | 0.508 |
| 7 | source:numeric | 0.405/0.681 | -0.275 | 0.000 | 0.539 |
| 7 | next:numeric | 0.572/0.832 | -0.261 | 0.000 | 0.533 |

## Guardrails

The target is the local frozen-route CE adjoint, not a complete attribution of model loss. Coarse token-form and distance strata are observational partitions, not semantic tasks. A stratum-specific basis can exploit stable distributional covariance without constituting a human-readable feature; finite interventions and held-out semantic predictions are required before naming it. Weighted PCA is a declared compression heuristic rather than the mathematically optimal subspace for the varying per-edge gradient metric.
