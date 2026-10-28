# Full-space red team for the task-blind MLP atlas

The rank-64 atlas is rerun directly in each complete view: 4,608 canonical product-allocation coordinates, the exact 1,152-dimensional residual write, and 4,608 channelwise CE contributions. This removes the discovery-PCA transfer bottleneck.

## Locked hypotheses

- **R0_fullspace_clusters_recur: supported.** Observed: `{'layers_beating_null': {'canonical_allocation': 8, 'exact_write': 8, 'ce_allocation': 8}}`
- **R1_rank64_was_adequate: not supported.** Observed: `{'passing': 5, 'cells': 24}`
- **R2_one_fullspace_ontology_spans_views: not supported.** Observed: `{'layer_median_ari': {'0': 0.11731432027857494, '1': 0.0009273175398284339, '3': 0.0008885384672686025, '6': 0.09653124432619688, '8': 0.009607001359697677, '12': 0.13716933322127897, '16': 0.05705685556039082, '17': 0.1132454096145044}}`
- **R3_discrete_fullspace_codes_beat_continuous_subspaces: not supported.** Observed: `{'cluster_wins': 0, 'cells': 24}`
- **R4_fullspace_candidates_exist: supported.** Observed: `{'eligible': 37}`
- **R5_late_layers_are_more_canonical: supported.** Observed: `{'late_minus_early': {'canonical_allocation': 0.3732428352038065, 'exact_write': 0.290768618384997, 'ce_allocation': 0.5484584818283718}}`

| Layer | View | Center/null | Margin D/R | Cluster/PCA fidelity |
|---:|---|---:|---:|---:|
| 0 | canonical_allocation | 0.726/0.021 | 0.308/0.270 | -0.496/0.101 |
| 0 | exact_write | 0.731/0.045 | 0.465/0.382 | -0.122/0.263 |
| 0 | ce_allocation | 0.504/0.020 | 0.135/0.131 | -0.459/0.152 |
| 1 | canonical_allocation | 0.523/0.022 | 0.229/0.211 | -0.683/0.051 |
| 1 | exact_write | 0.740/0.048 | 0.321/0.331 | -0.405/0.154 |
| 1 | ce_allocation | 0.264/0.018 | 0.174/0.166 | -0.541/0.118 |
| 3 | canonical_allocation | 0.472/0.020 | 0.180/0.172 | -0.774/0.025 |
| 3 | exact_write | 0.703/0.043 | 0.264/0.259 | -0.560/0.090 |
| 3 | ce_allocation | 0.213/0.018 | 0.208/0.113 | -0.614/0.099 |
| 6 | canonical_allocation | 0.742/0.014 | 0.209/0.168 | -0.436/0.141 |
| 6 | exact_write | 0.694/0.047 | 0.177/0.210 | -0.313/0.199 |
| 6 | ce_allocation | 0.634/0.015 | 0.138/0.136 | -0.496/0.145 |
| 8 | canonical_allocation | 0.262/0.020 | 0.153/0.130 | -0.833/0.022 |
| 8 | exact_write | 0.429/0.041 | 0.207/0.188 | -0.635/0.087 |
| 8 | ce_allocation | 0.193/0.017 | 0.215/0.124 | -0.665/0.093 |
| 12 | canonical_allocation | 0.507/0.021 | 0.194/0.165 | -0.729/0.042 |
| 12 | exact_write | 0.618/0.041 | 0.269/0.232 | -0.523/0.121 |
| 12 | ce_allocation | 0.508/0.019 | 0.160/0.124 | -0.608/0.111 |
| 16 | canonical_allocation | 0.900/0.023 | 0.324/0.363 | -0.063/0.343 |
| 16 | exact_write | 0.937/0.022 | 0.448/0.517 | 0.464/0.741 |
| 16 | ce_allocation | 0.939/0.031 | 0.219/0.182 | -0.050/0.358 |
| 17 | canonical_allocation | 0.924/0.028 | 0.404/0.426 | 0.236/0.614 |
| 17 | exact_write | 0.949/0.014 | 0.444/0.429 | 0.487/0.858 |
| 17 | ce_allocation | 0.930/0.021 | 0.259/0.236 | -0.155/0.305 |

## Cross-view agreement

| Layer | Allocation↔write | Allocation↔CE | Write↔CE |
|---:|---:|---:|---:|
| 0 | 0.768 | 0.115 | 0.117 |
| 1 | 0.508 | -0.008 | 0.001 |
| 3 | 0.371 | 0.001 | 0.001 |
| 6 | 0.442 | 0.097 | 0.031 |
| 8 | 0.182 | 0.010 | -0.003 |
| 12 | 0.471 | 0.130 | 0.137 |
| 16 | 0.842 | 0.057 | 0.053 |
| 17 | 0.416 | 0.113 | 0.092 |

## Interpretation guardrail

Beating a coordinate-permutation null establishes recurrence in native tensor coordinates, not a discrete vocabulary. The rank-matched PCA comparison tests discreteness, and cross-view ARI tests whether internal allocation, communicated write, and current-prediction effect share one partition. Candidate counts are descriptive until finite interventions and semantic transfer are run.
