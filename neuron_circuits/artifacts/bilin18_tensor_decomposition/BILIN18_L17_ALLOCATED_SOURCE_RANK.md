## H509 — Unequal source-rank allocation improves the same36-coordinate budget

Replace the uniform four-coordinates-per-head rule with a training-only allocation at the same total budget36. For each head, diagonalize C_h^(1/2) F_h C_h^(1/2), using the frozen calibration covariance and H507 full-path Fisher metric from64 training receivers. Select the largest36 eigenvalues globally. Because each head's spectrum is ordered, this selects a prefix of its eigenvectors. This minimizes the sum of per-head calibration-local errors under the budget; it ignores cross-head correlations and finite nonlinearity.

The allocation and compiled maps were saved and hashed before finite evaluation. No evaluation-outcome rank search or best-head selection was performed. Compare joint own-source replacement on the same512 natural contexts, using both MLP-only and full-pre-MLP paths. This remains a diagnostic on reused data.

| Head | Uniform rank | Allocated rank |
|---|---:|---:|
|0|4|10|
|1|4|3|
|2|4|16|
|3|4|1|
|4|4|1|
|5|4|1|
|6|4|1|
|7|4|1|
|8|4|2|

The training metric allocates10 coordinates to head0,16 to head2,3 to head1,2 to head8, and1 each to heads3–7. Total remains36. The summed discarded training weighted energy falls from0.006441 to0.005174, a19.67% reduction. These are local objective units before the common one-half KL factor, not observed finite KL. The allocation reflects absolute behavioral error under the training distribution, not each head's normalized recovery or semantic importance.

| Evaluation path | Source map | Mean KL | Recovery vs mean replacement | Observed-token ΔCE (paired SE) |
|---|---|---:|---:|---|
|MLP_only|PCA|0.001354|76.45%|-0.000375 (0.002445)|
|MLP_only|full_metric|0.001284|77.68%|+0.000004 (0.002358)|
|MLP_only|allocated|0.000951|83.47%|-0.000819 (0.002036)|
|full_pre_MLP|PCA|0.005837|50.62%|+0.005061 (0.004711)|
|full_pre_MLP|full_metric|0.005747|51.39%|+0.003690 (0.004945)|
|full_pre_MLP|allocated|0.004627|60.86%|+0.003003 (0.004170)|

Full-path KL falls from0.005747 for uniform full-path maps to0.004627 with the allocation, a19.48% relative error reduction. Normalized recovery rises from51.39% to60.86%. PCA's uniform-rank full-path error is0.005837, so the allocated map is about20.73% lower on this reused panel. MLP-only KL also falls from0.001284 to0.000951, despite allocation using the full-path metric. Observed-token CE changes remain too uncertain for a task-loss improvement claim.

This supports heterogeneous source-rank needs as a substantial limitation of the earlier fixed4-per-head construction. It does not show that the exact rank allocation is canonical, stable across sampling, or optimal under actual joint finite KL. It also does not resolve the remaining roughly39% of mean-replacement effect KL, establish semantic axes, or replace the native MLP. Total source-coordinate count is controlled, but deployment memory/FLOP savings have not been measured.

Audits: the selected indices are a prefix for each head and sum exactly to36; training objective is no worse than the uniform allocation; artifact hashes freeze the allocation before evaluation; all replayed baseline conditions agree with H507 within prescribed tolerances; all per-case outputs are finite and aggregate KL metrics replay. No new documents were consumed; tracked total remains19,456.

Next H510: freeze the allocated36 map, uniform full-path map, and PCA comparator, then evaluate natural joint full-pre-MLP replacement on512 fresh complete-document-disjoint contexts. Use the same frozen means and construction. Prescribe two primary comparisons: allocated versus uniform full-path, and allocated versus uniform PCA, measured as paired mean-KL reduction. Use one simultaneous95% bootstrap band across those two gains, resampling independent documents. Report absolute KL, normalized recovery, and observed-token CE descriptively; CE is not a co-primary improvement claim. Do not alter allocation after outcomes. Include all prior document panels, especially H499receivers and H500donors, in the exclusion audit. This tests whether a fixed-budget representation chosen from data and a downstream metric improves behavior preservation on new natural inputs.
