## H514 — More calibration improves stability and support, but only modestly improves rank36 fidelity

Collect1,024 new calibration documents, disjoint by complete-document hash from all20,480 tracked prior documents. Fit joint PCA36 on the first512, the last512, and their pooled1,024 sample. Each covariance is centered around its own fitting-pool mean; all reconstructions retain the old436-document mean to isolate basis variation. Compare all three fits with the original436-document basis on the unchanged H512 natural evaluation panel. No model is selected by outcomes.

The new512-document calibration pools are disjoint, unlike the overlapping repeated split designs of H513. They still share a frozen reconstruction mean, so this is not total estimator uncertainty. The pooled fit reuses both halves; it is not an independent replication. Evaluation is reused, so these are diagnostic results, not fresh confidence claims.

| Calibration fit | Mean KL | Recovery vs mean replacement | Observed-token ΔCE (paired SE) |
|---|---:|---:|---|
|reference436|0.003897|67.29%|+0.006329 (0.004119)|
|left512|0.003893|67.32%|+0.006552 (0.004210)|
|right512|0.003952|66.82%|+0.005056 (0.004135)|
|pooled1024|0.003710|68.86%|+0.006163 (0.004033)|

The pooled1,024-document fit has KL0.003710, about4.80% lower than the original436-document fit's0.003897. The two512 fits are close to the original in aggregate fidelity. The pooled improvement is modest, and comparisons with the older calibration pool mix sample size and sample composition; they do not establish a pure sample-size causal effect.

The independent512-fit subspaces agree more than the earlier218-fit subspaces did, but remain different. Mean squared principal cosine is0.661, median principal angle27.95°, and maximum angle88.49°. Their symmetric output KL is0.000715, or18.22% of their average approximation error. H513's218-fit pairs had overlap0.509–0.530, median angles38–41°, and disagreement about24% of approximation error. This is consistent with improved reproducibility as calibration grows, not a controlled estimate of convergence rate or proof of a unique limiting subspace.

All pairwise comparisons are retained, including reference and pooled fits:

| Pair | Mean squared principal cosine | Median / maximum angle | Symmetric output KL | Disagreement / average map error |
|---|---:|---|---:|---:|
|reference436__left512|0.636|27.85° / 89.80°|0.000758|19.45%|
|reference436__right512|0.672|26.58° / 89.90°|0.000684|17.42%|
|reference436__pooled1024|0.703|24.12° / 87.33°|0.000616|16.20%|
|left512__right512|0.661|27.95° / 88.49°|0.000715|18.22%|
|left512__pooled1024|0.843|13.31° / 88.27°|0.000308|8.11%|
|right512__pooled1024|0.875|11.62° / 87.08°|0.000250|6.51%|

The empirical support expands much more dramatically than rank36 fidelity:

| Fit | Empirical covariance rank | Held-out energy outside full empirical span | Calibration energy in top36 |
|---|---:|---:|---:|
|left512|511|3.360%|84.12%|
|right512|511|3.271%|84.30%|
|pooled1024|1023|0.083%|83.01%|

For the pooled1,024 sample, only0.0829% of held-out source energy lies outside the full1,023-dimensional empirical span, compared with4.55% outside the older435-dimensional span on the same evaluation panel. This is raw source support coverage, not probability faithfulness. Retaining only36 coordinates still leaves substantial behavioral error. The result motivates measuring a rank/fidelity curve before attributing the remaining gap mainly to missing calibration support.

CE changes remain descriptive and uncertain; no loss-neutral or improved-model claim is made. The shared old reconstruction mean means new mean-estimation benefits or risks are intentionally omitted. The source maps still compress the summed attention write before the final native MLP and residual addition, not the whole model.

Audits: an independent source-cache rehash verified all20,480 excluded documents and1,024 new calibration documents, zero overlap, and exact window matches. Total tracked documents is21,504. Frozen source artifacts were hash-checked; original436-fit per-case KL/CE replays H512; all rows are finite and aggregates replay. Full covariance bases, spectra, pool means, summed calibration source writes, and calibration receiver states are saved. The prescribed first/last512 split was not chosen by outcomes. New calibration documents are training data, not independent evaluation evidence.

Next H515: use the fixed pooled1,024 calibration PCA basis and old reconstruction mean to evaluate a prescribed nested rank grid36,72,144,288,576,1,023 on the same H512 panel. Compare raw source retention, native full-path KL, and observed-token CE. Include the rank1,023 empirical-span endpoint as a support diagnostic, not a compact matched-budget model. Do not select ranks after outcomes and claim confirmation. This will distinguish the effect of rank truncation from the remaining empirical-support limitation, and estimate the effective dimension needed for specified behavioral fidelity on this distribution. Low raw discarded energy must not be presumed low KL error. Any chosen compact operating point requires fresh confirmation before a generalization claim.
