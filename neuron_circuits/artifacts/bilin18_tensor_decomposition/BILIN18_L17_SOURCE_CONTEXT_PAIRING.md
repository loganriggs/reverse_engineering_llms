## H517 — Source/context pairing matters, but does not reveal a large hidden Fisher advantage

Use the same frozen source reconstruction errors from the H512 natural panel: mean-source replacement and PCA/Fisher maps at prescribed ranks36,144,576. Compute local full-path output-Fisher error at each receiver under its natural source-error pairing, then under eight fixed random permutations of those error vectors. Every permutation preserves each method's exact marginal error-vector pool and the receiver-context pool. Use the same permutation for every method to preserve pairing of method comparisons. No map is refitted, and no permutation is selected by outcomes.

These are artificial error-vector injections evaluated locally, not natural task-loss or semantic substitution experiments. Changes may reflect source magnitude, direction, receiver scale, context sensitivity, or their associations. They cannot isolate semantic dependence. Random permutations retain occasional fixed points; all are complete permutations rather than forced derangements. The panel is reused and the eight shuffles are a diagnostic, not a distributional equivalence test.

| Rank | Natural PCA recovery | Natural Fisher recovery | Natural Fisher gain (pp) | Shuffled Fisher gain range (pp) |
|---|---:|---:|---:|---|
|36|68.12%|69.17%|+1.05|[+0.53, +1.97]|
|144|85.57%|86.04%|+0.47|[-0.29, +0.78]|
|576|97.74%|98.00%|+0.26|[+0.02, +0.29]|

| Error map | Natural mean local KL | Pooled shuffled mean local KL | Natural recovery | Pooled shuffled recovery |
|---|---:|---:|---:|---:|
|mean|0.011690|0.011740|0.00%|0.00%|
|PCA36|0.003727|0.003265|68.12%|72.19%|
|Fisher36|0.003604|0.003118|69.17%|73.44%|
|PCA144|0.001687|0.001626|85.57%|86.15%|
|Fisher144|0.001632|0.001582|86.04%|86.53%|
|PCA576|0.000264|0.000308|97.74%|97.38%|
|Fisher576|0.000233|0.000289|98.00%|97.54%|

At rank36, shuffling generally increases the local recovery of both methods, but Fisher's incremental gain stays modest: natural1.05 percentage points, shuffled0.53–1.97 points. At144 the gain ranges−0.29 to+0.78 points across shuffles, compared with natural+0.47. At576 the gain remains positive but small,0.02–0.29 points, versus natural0.26. Thus source/context association influences the error, but removing natural pairing does not yield a dramatic Fisher advantage in these controls. This does not prove pairing is irrelevant or fully explain the training/evaluation objective gap.

Source error magnitude and receiver scale are associated. The correlation between log error squared norm and log receiver squared norm is about0.65–0.78 for the compressed maps, but only0.13 for the mean-source baseline. Shuffling preserves the norm distribution but breaks its pairing with receiver scale. Therefore changes in normalized recovery combine changes in numerator and baseline denominator, and should be read alongside absolute local errors. The comparison does not isolate directional or semantic effects from scale effects.

Audits: all nine index arrays (identity plus eight shuffles) contain every receiver index exactly once; the permutations were saved before evaluation and shared across methods. All local errors are finite and nonnegative. Per-case local errors, original error norms, receiver norms, and permutation arrays are archived. The local derivative follows the full-path formula audited in H507/H516. No documents were added; total remains21,504. No finite-KL, CE, fresh-confirmation, or causal semantic claims are introduced by this experiment.

Next H518 decomposes another source of the objective mismatch without fitting new maps. The expanded covariance was centered on the new1,024-document calibration mean, while source reconstruction intentionally retained the old shared mean. Under the fixed training Fisher metric, compare (1) centered calibration covariance, (2) calibration second moments around the shared reconstruction mean, and (3) held-out source second moments around that same mean. For each prescribed rank, separate covariance and mean-offset contributions exactly in the supported basis. Compare these predictions with H517's natural and shuffled local outcomes, while clearly marking that the latter use evaluation receiver sensitivities and include outside-span components. This distinguishes omitted mean-offset effects from source-moment shift before attributing the remaining discrepancy to receiver metric estimation. It is an offline diagnostic, not a new optimization or confirmation claim.
