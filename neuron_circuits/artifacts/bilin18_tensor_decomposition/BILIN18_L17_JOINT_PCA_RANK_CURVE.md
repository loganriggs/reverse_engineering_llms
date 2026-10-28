## H515 — Behavioral fidelity requires a broad source-rank spectrum

Use the frozen joint PCA basis fitted on1,024 new calibration documents in H514, retaining the old shared reconstruction mean. Evaluate the prescribed nested ranks36,72,144,288,576,1,023 on the unchanged H512 natural panel. Each map compresses the summed attention write before both the final MLP and residual addition; all other computation is native. No rank is selected after outcomes, and this reused-panel curve is diagnostic rather than confirmation of a chosen fidelity threshold.

KL recovery is1−replacement KL / mean-source-replacement KL. The mean-replacement KL on this panel is0.011913. Source retention is1−squared source reconstruction error / squared centered-source energy. These are different metrics, neither of which is token accuracy or a fraction of all model computation.

| Rank | Raw source retention | Mean KL | KL recovery | Observed-token ΔCE (paired SE) |
|---|---:|---:|---:|---|
|36|80.796%|0.0037097|68.859%|+0.006163 (0.004033)|
|72|85.748%|0.0025864|78.288%|+0.001623 (0.003159)|
|144|90.314%|0.0016872|85.837%|-0.001704 (0.002576)|
|288|94.657%|0.0008769|92.639%|-0.000402 (0.001801)|
|576|98.290%|0.0002643|97.781%|+0.000998 (0.000938)|
|1023|99.917%|0.0000126|99.894%|-0.000057 (0.000218)|

Recovery rises smoothly from68.86% at rank36 to78.29% at72,85.84% at144,92.64% at288,97.78% at576, and99.894% at1,023. On this prescribed grid, rank576 is the first point above95% recovery, but that is a descriptive threshold crossing on reused data, not a selected or confirmed operating point. The underlying attention-write space has1,152 dimensions;576 is a substantial fraction of that space. There is no evidence here for a tiny set of coordinates preserving nearly all output-distribution behavior.

The full empirical-span endpoint has KL0.0000126, only about0.34% of the rank36 error. This strongly suggests that the large rank36 gap is predominantly truncation within supported source directions, rather than a lack of calibration-span coverage. The endpoint is a high-rank control, not a compact representation, and is not a strict lower bound for every other source map.

Low-variance directions continue to matter behaviorally. At rank144, raw source retention is90.31% but KL recovery is85.84%; at288,94.66% versus92.64%. These differences are less dramatic than some earlier conditional response decompositions, but still show why raw energy should not substitute for a behavioral fidelity measurement. Both metrics happen to improve monotonically along this curve; nested PCA only guarantees monotone raw source error in general, not finite KL monotonicity.

Observed-token CE changes are small and noisy, with signs that do not follow KL ordering monotonically. No task-loss equivalence or improvement claim is made. The rank curve characterizes this model, source distribution, layer/path, and PCA family; it is not a lower bound on the number of fundamental features or on nonlinear/metric-optimized representations.

Audits: rank36 per-case KL/CE reproduces H514; frozen basis/data hashes were checked before execution; all rows are finite and aggregate metrics replay. Rank1,023 uses the full archived supported basis and its source residual matches H514's outside-span energy. No new documents were added; tracked total remains21,504. A static figure of the raw and behavioral retention curves is saved with this report.

Next H516: distinguish a genuinely broad behavioral spectrum from an inefficient PCA ordering under the expanded calibration covariance. Fit a joint full-path Fisher metric in the pooled1,024-document supported basis, using the same64 metric-training receivers as before. Evaluate the same prescribed ranks36/72/144/288/576/1,023 and shared reconstruction mean, without choosing a rank from outcomes. Compare each Fisher map with PCA at equal rank on the reused H512 panel; audit oblique-map amplification, covariance conditioning, full-rank reconstruction, and derivative consistency. The older435-dimensional covariance already showed only a small unconfirmed Fisher gain at rank36; the larger-support comparison will test whether that conclusion persists across rank and calibration support. This is a diagnostic, not justification to declare a low-dimensional canonical basis after choosing a favorable point.
