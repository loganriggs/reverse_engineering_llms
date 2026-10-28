

## H310 — Response magnitude concentration explains much of PCA fitting inflation

Same H308 cached vectors, source hash verified. Original four256-document blocks,18 head/payload groups,1024 responses per block/group. Fit uncentered PCA on raw responses or each response divided by its Euclidean norm. All responses nonzero. Evaluate each basis under BOTH raw-energy capture and equal-response directional capture, every within/cross block pair, ranks32/128. No new model execution or fresh data. Equal-response scoring changes the question; raw-energy evaluation remains separately reported.

| Rank | Fitting metric | Evaluation metric | Within | Cross | Gap |
|---|---|---|---:|---:|---:|
|32|raw|raw|34.74%|7.87%|26.87%|
|32|raw|unit|11.87%|6.64%|5.23%|
|32|unit|raw|17.74%|9.21%|8.53%|
|32|unit|unit|17.46%|9.24%|8.22%|
|128|raw|raw|68.62%|23.94%|44.68%|
|128|raw|unit|38.42%|22.42%|16.00%|
|128|unit|raw|48.80%|26.11%|22.69%|
|128|unit|unit|48.32%|26.13%|22.19%|

Energy concentration across72 block/group combinations: top11/1024 responses (~1%) carry14.09% energy on average (range5.19–33.25%); top103/1024 (~10%) carry48.86% (29.59–83.36%). The maximum single-response share averages2.67% and reaches13.27%. Energy participation count (sum energy)^2/sum energy² averages309.6 of1024 (range49.9–572.6). This is a concentration statistic, NOT an estimated statistical effective sample size.

Locked diagnostic gap-reduction gate PASSES: raw-fit/raw-score gap minus unit-fit/unit-score gap exceeds10pp at both ranks. This comparison changes both fitting and evaluation metric and therefore cannot establish improved raw reconstruction. Locked raw-transfer improvement>=5pp at both ranks FAILS: unit fitting improves raw cross capture by only1.33pp at rank32 and2.17pp at rank128.

**Matched-metric inference.** Holding raw evaluation fixed, unit fitting reduces the rank32 gap26.87→8.53pp and rank128 gap44.68→22.69pp, largely by reducing within-sample capture while modestly raising transfer. Holding the raw-fitted basis fixed but scoring directions equally reduces the gaps to5.23/16.00pp. Large sample magnitudes therefore contribute substantially to PCA's optimistic within-sample concentration. This is stronger evidence for an amplitude-weighting explanation than the earlier Gaussian/effective-sample analogy, because the response vectors and partitions are held fixed.

However, unit-fit/unit-score cross capture remains only9.24% at rank32 and26.13% at128. Removing amplitude concentration does not reveal a compact globally shared direction space. These are residual cross responses after an existing approximation, not all model features. Normalization is a diagnostic, not evidence of better behavior or an identified semantic unit. Three other explanations can coexist: noisy covariance estimation, context-dependent directions and genuinely broad fixed linear output requirements.

**Audit.** Raw-fit/raw-score arrays reproduce H306 for every block pair/group within1e-10. Both-fit/both-score arrays have4×4×18 shape and captures in[0,1]. The source hash, nonzero norm assertion and existing layout preserve identical responses, document partitions and groups. All jobs terminal; native model/compiler/base unchanged.

**Next discriminator.** Use the equal-norm response geometry to test a coordinate-free, source/context-conditioned prediction of directions, with frozen labels and held-out documents. Before choosing semantic labels, measure whether nearby normalized input/context states predict response-direction similarity better than random matched pairs and same-head identity alone. If similarity is predictable, local charts may have a meaningful conditioning variable; if not, a small global output space remains unsupported. Preserve raw-energy and equal-response evaluation as separate endpoints. Broad research goal active.
