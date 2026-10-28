# Shared linear context versus quadratic context computation

| Layer | Mean-linear | Ridge .001 | Ridge .01 | Ridge .1 | Cross16 | Cross64 | Cross128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 0.064 | -1.040 | -0.156 | 0.134 | 0.327 | 0.545 | 0.674 |
| L8 | -0.004 | -1.176 | -0.813 | -0.405 | 0.100 | 0.267 | 0.408 |
| L17 | 0.325 | 0.550 | 0.735 | 0.795 | 0.957 | 0.971 | 0.979 |

Locked gates: {'late_mean_linear': False, 'late_learned_linear': False, 'late_compact_quadratic': False, 'middle_mean_insufficient': True}.

Full1152-dimensional deltas and exact RMS endpoints retained. PCA context_only need not integrate to a state function; cross and linear do. All native weights remain for context rules; learned linear stores1152x1152 coefficients. No semantic identification, no nonlinear suffix test, no total-rank-read compression claim. One model/cached corpus; discovery/test complete-document hashes disjoint.
