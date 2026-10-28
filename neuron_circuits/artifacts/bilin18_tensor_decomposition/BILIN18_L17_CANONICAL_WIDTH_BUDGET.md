## H472 — Input width versus optimization budget, with equal-size compiled functions

Factorialize64/128 empirical bilinear input coordinates and8/32 full-gradient finite-response-KL steps, output rank16 fixed. Kernel support768, training/validation/test splits, source assignments, normalized joint initialization recipe, step grid and validation-only prefix selection stay fixed. Rank64 replays H471 exactly through its8-step selected model. Rank128 contains the original64 feature span numerically and starts from its own normalized joint regression. This compares complete width-specific pipelines, not identical initial functions. No new documents; all test results remain conditional on a heavily reused research panel.

| Input width / optimization budget | Validation-selected iteration | Negative KL / raw-response recovery | Positive KL / raw-response recovery | Intact KL |
|---|---:|---|---|---:|
|64 / 8|8|59.67% / 93.13%|59.97% / 92.90%|0.144965|
|64 / 32|32|61.99% / 92.56%|61.80% / 92.26%|0.155192|
|128 / 8|8|61.86% / 93.68%|61.65% / 93.57%|0.128920|
|128 / 32|14|63.91% / 93.14%|63.39% / 92.97%|0.136341|

| Frozen primary statistic | Estimate | Joint95% interval | Pass |
|---|---:|---|---|
|width_gain_negative|+0.019183|[-0.003463, +0.041829]|False|
|width_gain_positive|+0.015959|[-0.007197, +0.039115]|False|
|budget_gain_r64_negative|+0.023173|[-0.035083, +0.081429]|False|
|budget_gain_r64_positive|+0.018241|[-0.038536, +0.075018]|False|
|budget_gain_r128_negative|+0.020549|[-0.035856, +0.076953]|False|
|budget_gain_r128_positive|+0.017383|[-0.035836, +0.070601]|False|
|r128_b32_absolute_negative|+0.639099|[+0.436812, +0.841386]|False|
|r128_b32_absolute_positive|+0.633927|[+0.474949, +0.792905]|False|

All eight gates fail. The first six require>5pp gains; the final two require>80% absolute recovery lower bounds. At32-step budgets, width128-minus64 improves recovery1.918/1.596pp on point estimates, joint intervals[−.346,4.183]/[−.720,3.912]pp. Extra steps give1.7–2.3pp point gains, but all corresponding intervals include zero. Rank128/budget32 reaches63.91/63.39% recovery with joint intervals[43.68,84.14]/[47.49,79.29]%. Neither intervention sign reaches the80% lower-bound target.4000 paired test-document bootstrap replicates and a joint8 standardized maximum-deviation band; calibration/source assignment and validation selection remain fixed. No passing thresholds were substituted.

Rank64 selects iteration32; rank128 selects iteration14 from its32-step trajectory. Rank64 training response KL falls from.00080403 to.00030935 and validation from.00135359 to.00101316. Rank128 final training loss is.00027702, while its best validation iterate occurs earlier than the endpoint. This weakens a simple universal 'more steps solves it' explanation, but neither run proves global convergence or an input-rank lower bound. The width-by-budget interaction is saved as a descriptive contrast. Large raw-response recovery (~92–94%) alongside~62–64% probability-response recovery remains.

### A concrete functional-equivalence result

The learned map has the form normalized_write = centered_kernel(x) E C + intercept, with E of size768xq and C of sizeqx1152, q=64 or128. C has rank16. Factor C=U S V and combine E with U S. The resulting read matrix is768x16 and write matrix16x1152, independent of q. Kernel centering is folded into the constant intercept and an explicit row-mean correction; it is not silently discarded. Thus both training widths compile to the same final architecture and storage size, although they represent different fitted functions.

For each selected model, store shared support768x1152=884736 scalars, shared scale1, and31888 per-map scalars: read12288, write18432, intercept1152 and row-mean read correction16. A single compiled map totals916625 scalars. This is much more than '16 neurons'; the dictionary dominates cost. All six initial/selected maps were compiled and tested on128 held-out query inputs. Relative write-prediction differences are4.3e−9 to5.0e−9 in float64; rank16 truncation of the saved float32 effective matrices discards only~1.0–1.2e−8 relative matrix norm. This verifies the algebraic collapse within numerical precision. It does not verify identical downstream probabilities after arbitrary low-precision implementations, establish a runtime speedup, remove sign/degenerate-subspace ambiguity, or prove a unique global canonical form.

This directly illustrates the user's distinction: a wider training parameterization can describe the same type of final function, and unnecessary internal factorization can be removed without changing predictions. The performance experiment remains separate: same final size does not imply same function or sufficient faithfulness.

Next avoid interpreting more reused-panel tuning as confirmation. Freeze the validation-selected64/32 and128/32 maps, collect a genuinely new receiving-document panel, and evaluate intact predictions and both signed response changes with unchanged sources. Track absolute effects and error concentration as well as recovery. This separates sampling/generalization uncertainty from the narrow width/step gains before choosing another optimization branch. New source/distribution transfer is an additional question, not supplied by fresh receivers alone. Canonical-form CE comparison, source rank1/tail tests and broader-head transfer remain unresolved.

Numerical nested-feature and rank64-prefix replay audits pass. Compiled models, unfused models, complete optimization histories, per-case scores and bootstrap draws are preserved. Total12992 documents; no claim that the broad research goal is complete.
