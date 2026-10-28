## H522 — Source-calibration sample curve with fixed receiver metric and basis

Hold the1,024-context training Fisher metric, original1,023-dimensional supported basis, shared reconstruction mean, and ranks36/144/576 fixed. Add2,048 new source-calibration documents to the previous1,024, forming nested pools1,024/2,048/3,072. Fit each source covariance around its own pool mean; replacement continues using the same old shared mean. The expanded covariances are full matrices in the old basis, not diagonal approximations. Build symmetric covariance square roots and inverses, then Fisher maps C^(1/2) U U^T C^(−1/2). PCA maps use leading covariance eigenvectors in the same fixed basis.

All18 maps are saved before evaluation. Test on both already-used512-document H512 and H521 panels; neither panel supplies fit data. These are repeated-panel diagnostics, not fresh confirmation. The original basis was fitted on the first1,024 calibration examples, so the comparison is conditional on that basis and does not test expanding empirical support. Sample count and pool composition change together. Native downstream computation remains intact.

### H512: finite prediction fidelity

| Rank | Method | Source samples | Mean prediction KL | Recovery | Error reduction vs1,024 |
|---|---|---:|---:|---:|---:|
|36|PCA|1024|0.003709683|68.86%|0.00%|
|36|PCA|2048|0.003596114|69.81%|3.06%|
|36|PCA|3072|0.003527327|70.39%|4.92%|
|36|Fisher|1024|0.003509706|70.54%|0.00%|
|36|Fisher|2048|0.003373910|71.68%|3.87%|
|36|Fisher|3072|0.003334299|72.01%|5.00%|
|144|PCA|1024|0.001687154|85.84%|0.00%|
|144|PCA|2048|0.001502175|87.39%|10.96%|
|144|PCA|3072|0.001456014|87.78%|13.70%|
|144|Fisher|1024|0.001517467|87.26%|0.00%|
|144|Fisher|2048|0.001358847|88.59%|10.45%|
|144|Fisher|3072|0.001320727|88.91%|12.97%|
|576|PCA|1024|0.000264291|97.78%|0.00%|
|576|PCA|2048|0.000191648|98.39%|27.49%|
|576|PCA|3072|0.000170040|98.57%|35.66%|
|576|Fisher|1024|0.000208623|98.25%|0.00%|
|576|Fisher|2048|0.000141985|98.81%|31.94%|
|576|Fisher|3072|0.000128268|98.92%|38.52%|

### H521: finite prediction fidelity

| Rank | Method | Source samples | Mean prediction KL | Recovery | Error reduction vs1,024 |
|---|---|---:|---:|---:|---:|
|36|PCA|1024|0.003762451|70.82%|0.00%|
|36|PCA|2048|0.003680151|71.46%|2.19%|
|36|PCA|3072|0.003610387|72.00%|4.04%|
|36|Fisher|1024|0.003595400|72.11%|0.00%|
|36|Fisher|2048|0.003507163|72.80%|2.45%|
|36|Fisher|3072|0.003452669|73.22%|3.97%|
|144|PCA|1024|0.001729058|86.59%|0.00%|
|144|PCA|2048|0.001588508|87.68%|8.13%|
|144|PCA|3072|0.001546471|88.01%|10.56%|
|144|Fisher|1024|0.001605549|87.55%|0.00%|
|144|Fisher|2048|0.001472801|88.58%|8.27%|
|144|Fisher|3072|0.001419668|88.99%|11.58%|
|576|PCA|1024|0.000278303|97.84%|0.00%|
|576|PCA|2048|0.000206238|98.40%|25.89%|
|576|PCA|3072|0.000183687|98.58%|34.00%|
|576|Fisher|1024|0.000219649|98.30%|0.00%|
|576|Fisher|2048|0.000162462|98.74%|26.04%|
|576|Fisher|3072|0.000146091|98.87%|33.49%|

Both representation families improve monotonically with additional source calibration at every tested rank on both reused panels. Moving from 1,024 to 3,072 sources reduces Fisher error by 5.00%, 12.97%, 38.52% on H512 and 3.97%, 11.58%, 33.49% on H521 at ranks 36, 144, 576. PCA reductions are 4.92%, 13.70%, 35.66% and 4.04%, 10.56%, 34.00%. The gain is shared by raw covariance and behavioral weighting, consistent with a source-statistics limitation rather than exclusively an unstable Fisher reading rule. These are descriptive effects on reused panels, not a fresh significance claim.

The covariance condition number falls from 5.69e7 to 2.03e6 to 1.34e6. This is consistent with better-supported weak directions, but condition number alone does not identify the cause of the behavioral gain. Source-distribution composition and sampling error both change. No clipping, ridge or shrinkage was used.

The broad behavioral spectrum persists. The 3,072-source Fisher maps recover about 72.01/88.91/98.92% on H512 and 73.22/88.99/98.87% on H521. Larger samples reduce the high-rank residual, yet do not reveal a tiny near-exact representation. The original residual spectrum should not be treated as an intrinsic property of model computation, because its shape depends on how well source covariance is estimated.

Together with H520-521, the leading account includes estimation of both source statistics and downstream sensitivity. Next we should distinguish better average fidelity from more reproducible representations. Earlier equal-quality fits disagreed in subspace and predictions. Improved approximation alone is not evidence of unique or stable computational units.


### Observed-token loss at3,072 calibration samples

| Panel | Map | CE change ± SE |
|---|---|---:|
|H512|PCA36|+0.005151 ± 0.003885|
|H512|PCA144|-0.002194 ± 0.002295|
|H512|PCA576|-0.000059 ± 0.000820|
|H512|Fisher36|+0.003738 ± 0.003755|
|H512|Fisher144|-0.001150 ± 0.002279|
|H512|Fisher576|-0.000355 ± 0.000698|
|H521|PCA36|+0.001419 ± 0.003624|
|H521|PCA144|-0.000403 ± 0.002446|
|H521|PCA576|-0.000138 ± 0.000876|
|H521|Fisher36|+0.002450 ± 0.003581|
|H521|Fisher144|-0.000085 ± 0.002174|
|H521|Fisher576|+0.000387 ± 0.000763|

Recovery is relative to mean-source replacement for the relevant panel, not model accuracy. Positive CE change means worse observed-token loss. CE is descriptive and does not establish loss neutrality or improvement. No rank or sample size is selected by these evaluations.

Audits: frozen artifacts hash checked; independent document rehash/window verification finds2,048 new unique documents disjoint from all22,016 previously tracked documents, total24,064. Every covariance is positive definite; square-root/inverse and idempotence checks pass. The1,024-source PCA and Fisher operators reproduce their original counterparts, and finite KL on H521 reproduces the original endpoints within prescribed1e−4 relative tolerance. All archived per-document measurements are finite. This establishes numerical implementation fidelity, not population uniqueness or causal semantics. Full audit values are in the accompanying JSON.

Next H523: test split-fit stability using the pooled 3,072 calibration activations. Prespecify four random complementary 1,536/1,536 partitions (seed 20280106), holding the old supported basis, shared reconstruction mean, 1,024-context metric and ranks 36/144/576 fixed. Fit full-covariance PCA and Fisher maps on each half; save every map before evaluation. Compare corresponding halves using write-subspace principal angles, coordinate-operator differences, finite prediction KL to the native model, and symmetric prediction KL between fits on H512 and H521. Use average squared principal cosine rather than individual eigenvector matching; report read/write operator differences separately because Fisher maps are oblique. Compare disagreement with average approximation error, not only the native mean-removal baseline. This is conditional stability under shared basis/metric, not independent end-to-end canonicalization, and remains a reused-panel diagnostic. Do not claim fresh confirmation of the 3,072-source gains or functional equivalence from similar fidelity. Test reproducibility before adding model complexity or another performance-only confirmation.

