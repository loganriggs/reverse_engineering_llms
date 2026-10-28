## H519 — Receiver metric generalization limits the Fisher advantage

We keep the H516 representations fixed and compute the average full-path prediction Fisher metric over all 512 H512 evaluation receivers, in the same 1,023-dimensional supported basis. The Fisher metric measures local sensitivity of the model's output probabilities to a change in the layer input; the full path includes both the residual bypass and the MLP. This evaluation metric is a diagnostic, not a fitted replacement model. No representation is optimized or selected using it.

Cross training64/evaluation512 receiver metrics with calibration1024/evaluation512 source panels around the same frozen reconstruction mean. The quadratic error is one-half E[e^T F e]. Averaging F separately from source errors computes the exact empirical independent-pair expectation within the supported basis. The naturally paired calculation instead measures each evaluation error under its own receiver metric. These are local prediction errors, not observed-token CE, finite interventions, or fresh confirmation.

| Representation | Train metric + calibration sources | Train metric + evaluation sources | Evaluation metric + calibration sources | Evaluation metric + evaluation sources | Natural evaluation pairing |
|---|---:|---:|---:|---:|---:|
|PCA36|75.26%|72.04%|75.44%|72.02%|68.29%|
|Fisher36|78.63%|75.89%|76.24%|73.16%|69.32%|
|PCA144|91.55%|86.05%|91.60%|86.13%|85.75%|
|Fisher144|93.72%|89.62%|91.62%|86.37%|86.20%|
|PCA576|99.73%|97.50%|99.73%|97.52%|97.87%|
|Fisher576|99.86%|98.69%|99.75%|97.72%|98.14%|

Values above are recovery relative to mean-source replacement under each corresponding metric/panel/pairing. Each column has its own denominator. Recovery is not accuracy or the fraction of all model computation explained.

The key interaction is metric generalization. On fixed evaluation sources, replacing the training receiver metric with the evaluation metric leaves PCA recovery almost unchanged: 72.04% to 72.02% at rank36, 86.05% to 86.13% at144, and97.50% to97.52% at576. Fisher recovery instead decreases from75.89% to73.16%,89.62% to86.37%, and98.69% to97.72%. Its advantage over PCA therefore falls from3.85 to1.13 percentage points at36,3.57 to0.25 at144, and1.19 to0.20 at576. The same pattern occurs on calibration sources. This supports sensitivity to the sampled training receiver metric, beyond the source-moment generalization effect isolated in H518. It does not separate finite metric-estimation error from a change in the context distribution, and does not prove that more metric samples will solve the problem.

Natural pairing has a distinct, rank-dependent effect. Relative to exact independent pairing under the evaluation metric, natural pairing increases Fisher36 absolute local error by about12%, barely changes Fisher144 error, and reduces Fisher576 error by about20%. Thus pairing cannot be represented by a universal penalty. It affects both the error numerator and the baseline denominator. Associations may concern vector magnitude, direction, receiver scale, or semantic context; this experiment does not isolate their causes.

| Representation | Train metric / evaluation sources error | Evaluation metric / evaluation sources error | Natural supported error |
|---|---:|---:|---:|
|mean|0.009643026|0.011846227|0.011654763|
|PCA36|0.002695904|0.003314039|0.003695167|
|Fisher36|0.002324744|0.003179781|0.003575240|
|PCA144|0.001345413|0.001643225|0.001660884|
|Fisher144|0.001001001|0.001614137|0.001607810|
|PCA576|0.000241198|0.000294322|0.000248741|
|Fisher576|0.000126465|0.000270115|0.000216633|

The supported-space natural errors are below H517 full-vector natural errors by about0.8–0.9% at rank36,1.5% at144, and5.9–7.2% at576. This is a net difference, including cross terms, not an additive allocation of error to the omitted directions. H517 used float32 directional derivatives whereas H519 uses double algebra with float32 vocabulary multiplication/Gram accumulation, so the comparison also includes numerical implementation differences. It should not be treated as an exact support-only decomposition. The tiny raw outside-span source energy is consequently not sufficient evidence that its behavioral contribution is negligible at high rank.

Audits: frozen inputs hash checked; central-difference derivative relative errors1.29e−8 and3.21e−9; average metric exactly symmetric with positive minimum eigenvalue8.41e−11; all archived errors finite and nonnegative within numerical tolerance; every training-metric/source-panel score replays H518 within2.23e−16 relative error. All512 evaluation receivers were used, with no permutation sampling. No new documents: total21,504.

Implication for canonical forms: empirical behavioral coordinates depend on two sampled objects—what activations occur and which activation changes affect predictions in their contexts. A more behavior-aware objective does not automatically yield a more reproducible representation. The broad rank spectrum and modest held-out Fisher advantage persist after separating these effects. No unique feature axes, global equivalence, or whole-layer canonical form has been established.

Next H520: hold the source covariance, shared mean, supported basis, and ranks36/144/576 fixed. Fit Fisher maps with nested64/256/1,024 training receivers from the already established first1,024 training contexts. Keep evaluation receivers out of the metric fit. Use the original64 ordering as the prefix and append the remaining training indices deterministically; predeclare the256 prefix. Compare frozen maps on the same H512 panel using actual finite prediction KL and observed-token CE; report all ranks and sample sizes without choosing a winner on this panel. Replay the64-context endpoint. This discriminates metric sample sensitivity from persistent limitations of the independent average-metric construction. A subsequent fresh-document comparison is required before claiming a generalization improvement. Larger sample size cannot by itself distinguish variance from context-distribution mismatch, and CE remains descriptive unless explicitly optimized and confirmed.
