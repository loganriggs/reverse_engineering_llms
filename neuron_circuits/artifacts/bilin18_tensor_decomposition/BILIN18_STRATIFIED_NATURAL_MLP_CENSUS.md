# Stratified natural bilinear-MLP channel census

Exact gradients of every native post-product channel are collected at four ordinary FineWeb prediction positions. Fixed channel banks are selected without hand-authored task labels from current-token form, next-token form, loss quartile, or position strata. CE contribution and local MLP-write reconstruction are evaluated separately.

## Locked hypotheses

- **M0_panels_are_fresh_and_task_blind: supported.** Observed: `{'ranges': [[8704, 8767], [8960, 9023]]}`
- **M1_natural_strata_select_specialized_channel_banks: not supported.** Observed: `{'passing': 24, 'eligible': 144, 'fraction': 0.16666666666666666}`
- **M2_selected_banks_beat_random_channels: not supported.** Observed: `{'passing': 65, 'eligible': 144, 'fraction': 0.4513888888888889}`
- **M3_channel_banks_recur: supported.** Observed: `{'observed': 0.15647915570446128, 'null_p95': 0.017050931949308595}`
- **M4_ce_and_write_ontologies_differ: supported.** Observed: `{'opposite': 63, 'eligible': 144, 'fraction': 0.4375}`
- **M5_examplewise_native_channel_support_is_broad: supported.** Observed: `{'layers': [0, 1, 3, 6, 8, 12, 16, 17], 'oracle': {'0': {'median_k90': 1782.5, 'q10': 1658.0, 'q90': 1862.0}, '1': {'median_k90': 1812.5, 'q10': 1744.5, 'q90': 1865.5}, '3': {'median_k90': 1838.5, 'q10': 1772.0, 'q90': 1882.0}, '6': {'median_k90': 1869.0, 'q10': 1812.5, 'q90': 1909.0}, '8': {'median_k90': 1885.5, 'q10': 1830.5, 'q90': 1927.0}, '12': {'median_k90': 1892.0, 'q10': 1844.0, 'q90': 1928.5}, '16': {'median_k90': 1760.0, 'q10': 1521.5, 'q90': 1857.5}, '17': {'median_k90': 1696.0, 'q10': 1416.5, 'q90': 1818.0}}}`

## Size-128 summary

| Stratum | Cells | Local/global CE | CE advantage | Local/global write | Write advantage | Jaccard | Beats random p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| current:capitalized | 8 | 0.246/0.315 | -0.069 | 0.382/0.389 | -0.007 | 0.130 | 0.38 |
| current:continuation | 8 | 0.252/0.304 | -0.052 | 0.350/0.366 | -0.016 | 0.108 | 0.50 |
| current:function | 8 | 0.262/0.250 | +0.011 | 0.402/0.384 | +0.019 | 0.186 | 0.50 |
| current:other | 8 | 0.221/0.251 | -0.030 | 0.365/0.366 | -0.001 | 0.159 | 0.25 |
| current:punctuation | 8 | 0.332/0.272 | +0.060 | 0.412/0.366 | +0.046 | 0.213 | 0.62 |
| loss:easy | 8 | 0.186/0.377 | -0.190 | 0.361/0.377 | -0.015 | 0.139 | 0.62 |
| loss:hard | 8 | 0.283/0.301 | -0.017 | 0.366/0.375 | -0.009 | 0.183 | 0.50 |
| loss:medium-high | 8 | 0.200/0.200 | -0.001 | 0.367/0.379 | -0.013 | 0.170 | 0.50 |
| loss:medium-low | 8 | 0.250/0.288 | -0.038 | 0.347/0.364 | -0.017 | 0.143 | 0.38 |
| next:capitalized | 8 | 0.259/0.266 | -0.006 | 0.382/0.378 | +0.004 | 0.146 | 0.38 |
| next:continuation | 8 | 0.328/0.320 | +0.008 | 0.386/0.401 | -0.015 | 0.089 | 0.50 |
| next:function | 8 | 0.284/0.304 | -0.021 | 0.357/0.365 | -0.008 | 0.178 | 0.38 |
| next:other | 8 | 0.212/0.257 | -0.045 | 0.379/0.380 | -0.001 | 0.175 | 0.38 |
| next:punctuation | 8 | 0.337/0.277 | +0.059 | 0.349/0.363 | -0.014 | 0.167 | 0.62 |
| position:p126 | 8 | 0.244/0.243 | +0.001 | 0.361/0.373 | -0.012 | 0.158 | 0.50 |
| position:p31 | 8 | 0.283/0.284 | -0.000 | 0.370/0.379 | -0.009 | 0.159 | 0.38 |
| position:p63 | 8 | 0.237/0.267 | -0.030 | 0.354/0.375 | -0.021 | 0.151 | 0.38 |
| position:p95 | 8 | 0.263/0.296 | -0.034 | 0.356/0.374 | -0.018 | 0.164 | 0.38 |

## Examplewise oracle breadth

| Layer | Median K90 | q10--q90 |
|---:|---:|---:|
| 0 | 1782 | 1658--1862 |
| 1 | 1812 | 1744--1866 |
| 3 | 1838 | 1772--1882 |
| 6 | 1869 | 1812--1909 |
| 8 | 1886 | 1830--1927 |
| 12 | 1892 | 1844--1928 |
| 16 | 1760 | 1522--1858 |
| 17 | 1696 | 1416--1818 |

## Guardrails

CE banks are selected from activation-times-total-gradient contributions and are prediction-objective coordinates, not monosemantic neurons. Local-write fidelity asks a different question and uses the same CE-selected bank without refitting. Examplewise K90 is an oracle descriptive breadth statistic based on absolute contribution mass; it is not a prospective sufficient circuit size. Finite suffix replacement is required before treating local scalar fidelity as causal sufficiency.
