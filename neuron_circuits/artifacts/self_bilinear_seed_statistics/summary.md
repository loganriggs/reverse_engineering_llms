# Self-bilinear seed statistics

This study uses 20 independent seeds at 18432 computation tasks. Each seed changes fact values, sampled train/test subsets, and parameter initialization. JSON mean intervals are 10,000-resample bootstrap 95% confidence intervals across seeds; convergence intervals are Wilson binomial intervals. Tables show mean ± sample SD.

## Architecture outcomes

| architecture | converged | lookup mean ± SD | parity 2 | parity 4 | parity 8 |
|---|---:|---:|---:|---:|---:|
| full | 19/20 (95.0%) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.991 ± 0.039 | 0.977 ± 0.098 |
| no bias | 20/20 (100.0%) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | 0.915 ± 0.184 |
| no residual | 10/20 (50.0%) | 0.883 ± 0.208 | 0.876 ± 0.220 | 0.873 ± 0.219 | 0.769 ± 0.245 |
| neither | 0/20 (0.0%) | 0.511 ± 0.041 | 0.502 ± 0.017 | 0.503 ± 0.014 | 0.506 ± 0.014 |

## Held-out cross-entropy variance

CE is heavy-tailed, so both mean ± SD and median are shown. Unlike accuracy, CE penalizes rare confident errors without bound.

| architecture | lookup | parity 2 | parity 4 | parity 8 |
|---|---:|---:|---:|---:|
| full | 1.77e-07 ± 5.75e-07 (med 0) | 9.52e-07 ± 2.25e-06 (med 0) | 0.152 ± 0.678 (med 0) | 1.35 ± 4.36 (med 0.000103) |
| no bias | 1.89e-06 ± 3.13e-06 (med 2.9e-07) | 2.48e-06 ± 3.12e-06 (med 1.19e-06) | 0.00166 ± 0.00632 (med 1.11e-05) | 3.85 ± 9.77 (med 0.00264) |
| no residual | 0.174 ± 0.307 (med 2.12e-06) | 0.174 ± 0.307 (med 1.37e-05) | 0.24 ± 0.382 (med 0.00266) | 5.49 ± 8.37 (med 0.694) |
| neither | 0.693 ± 0 (med 0.693) | 0.693 ± 0 (med 0.693) | 0.693 ± 0 (med 0.693) | 0.693 ± 0 (med 0.693) |

## Full-model circuit-size variance

Counts are shared task-family masks. Values are mean ± SD across independently trained models.

| family | neurons for 90% accuracy | neurons for 90% CE recovery | CE/margin rank correlation |
|---|---:|---:|---:|
| lookup | 25.7 ± 21.3 | 110.3 ± 30.6 | 0.415 ± 0.278 |
| parity_2 | 58.5 ± 34.5 | 124.6 ± 46.4 | 0.502 ± 0.262 |
| parity_4 | 70.7 ± 21.5 | 104.4 ± 25.8 | 0.771 ± 0.107 |
| parity_8 | 88.4 ± 18.8 | 113.8 ± 25.1; 18/19 attained | 0.853 ± 0.068 |

The complete JSON contains 80%, 90%, and 99% thresholds, confidence intervals, per-seed curves, losses, steps, and accuracies. Failed seeds are retained in unconditional architecture means rather than silently discarded.
