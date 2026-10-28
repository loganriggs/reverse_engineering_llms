## H520 — Training receiver sample curve for behavioral coordinates

Following H519's metric-generalization diagnostic, fit the same Fisher-weighted source reconstruction family using nested64/256/1,024 training receivers. The initial64 are exactly H516's contexts and ordering; the remaining training indices are appended in ascending order. All contexts belong to the established first1,024 training contexts. The evaluation metric from H519 is not used for fitting. Fix the1,023-dimensional basis, source covariance estimated from1,024 separate calibration documents, shared reconstruction mean, and prescribed ranks36/144/576. All nine maps are saved before evaluation.

Evaluate actual finite full-path replacements on H512's reused512 documents, measuring original-to-replacement prediction KL and observed-token CE change. These are descriptive repeated-panel results, not fresh confirmation. The nested curve changes sample count and context composition together; it cannot establish an asymptotic sample-size law or identify pure estimation variance. No sample size or rank is selected by evaluation outcomes.

| Rank | Metric contexts | Prediction KL | Recovery vs mean replacement | Gain over PCA (pp) | Error reduction vs64 | CE change ± SE |
|---|---:|---:|---:|---:|---:|---:|
|36|64|0.003585555|69.90%|1.042|0.00%|+0.004692 ± 0.003950|
|36|256|0.003549998|70.20%|1.340|0.99%|+0.005706 ± 0.003866|
|36|1024|0.003509706|70.54%|1.679|2.12%|+0.004810 ± 0.003917|
|144|64|0.001631563|86.30%|0.467|0.00%|-0.000079 ± 0.002505|
|144|256|0.001551220|86.98%|1.141|4.92%|+0.001357 ± 0.002541|
|144|1024|0.001517467|87.26%|1.424|6.99%|-0.000134 ± 0.002532|
|576|64|0.000233424|98.04%|0.259|0.00%|+0.000734 ± 0.000963|
|576|256|0.000216534|98.18%|0.401|7.24%|+0.001027 ± 0.000934|
|576|1024|0.000208623|98.25%|0.467|10.63%|+0.001092 ± 0.000871|

Recovery means one minus replacement KL divided by mean-source-replacement KL. It is not model accuracy. Positive CE change means worse observed-token loss. Small/noisy CE estimates do not establish equivalence or improvement. SE values describe this reused panel and are not selection-adjusted confidence intervals.

All three ranks improve monotonically in mean prediction KL as the training metric grows from64 to256 to1,024 contexts. At1,024 contexts, KL error is2.12%,6.99%,10.63% lower than the original64-context fit at ranks36,144,576 respectively. Recovery increases by0.64,0.96,0.21 percentage points. The same fixed source covariance was used throughout, so this improvement is attributable to changing the receiver-metric sample within this construction, rather than additional source dimensions or source calibration data.

The improvement supports metric sample sensitivity as a contributor to the earlier generalization gap. It does not eliminate the broad behavioral rank spectrum: the1,024-context maps recover70.54%,87.26%,98.25% at the three ranks. Compared with PCA, their gains are1.68,1.42,0.47 percentage points. More accurate behavioral weighting helps, but it does not reveal a tiny exact representation of the natural attention contribution. The rank36 gain over64 is small relative to its paired sampling error; repeated evaluation and inspecting several comparisons require fresh confirmation before inferential claims.

Observed-token CE remains inconclusive. The1,024-context maps change CE by+0.00481±0.00392,−0.000134±0.00253, and+0.00109±0.00087 at ranks36,144,576. These were KL-oriented maps; the experiment does not optimize CE or establish loss neutrality. This reinforces the need to treat prediction fidelity and task loss as distinct objectives.

Next H521: freeze the1,024-context maps at all three prescribed ranks and compare them with the original64-context maps and PCA on512 fresh documents. Prespecify six paired mean-KL gains (two comparators at each rank), simultaneous95% intervals across that family, and descriptive CE/recovery reporting. Use4,000 document bootstrap resamples with fixed seed20280104 and a centered max-absolute standardized statistic with fixed original paired SEs; derive simultaneous two-sided bounds and require a positive lower bound for each individual supported superiority claim. Do not select ranks or maps on the fresh panel. Include H514's1,024 calibration documents among all prior document/window exclusions and independently audit disjointness. This tests whether the metric-sample improvement generalizes before building a more elaborate conditional metric. A refreshed twenty-hypothesis assessment is saved alongside this report.


Audits: all frozen input hashes checked; training indices unique and exactly the established1,024-context pool, with the original64 prefix; full-path derivative checked by central differences; the64-context Fisher metric and finite KL endpoints replay H516 within declared tolerances; all per-document measurements finite. Source geometry, reconstruction mean, evaluation panel, and replacement path are fixed. All fitted matrices/eigenvectors and per-document KL/CE are archived. No documents added: total21,504. Native attention/MLP computations remain in the evaluator, so these results do not establish deployment speedups or a standalone replacement of the MLP.
