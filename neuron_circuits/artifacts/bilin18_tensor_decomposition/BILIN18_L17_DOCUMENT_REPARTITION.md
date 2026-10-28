

## H308 — Empirical whole-document repartition control

Cache exact missing cross-response vectors after frozen H300 refined64,18 groups×1024 documents×4 positions×1152 outputs. Reuse original discovery documents and payload seed. Three seeded permutations assign complete documents to four256-document blocks, same assignments across groups. Every document appears once per partition; positions remain together. Fit specialist PCA within each block and evaluate all block pairs at ranks32/128. This preserves empirical tails, amplitudes and within-document dependence. It changes block membership, not response vectors or the frozen approximation.

| Rank | Partition | Within capture | Cross capture | Gap |
|---|---|---:|---:|---:|
|32|Original|34.74%|7.87%|26.87pp|
|32|Shuffle1|34.75%|7.94%|26.82pp|
|32|Shuffle2|34.67%|7.91%|26.76pp|
|32|Shuffle3|34.68%|7.88%|26.80pp|
|128|Original|68.62%|23.94%|44.68pp|
|128|Shuffle1|68.65%|24.01%|44.64pp|
|128|Shuffle2|68.63%|23.99%|44.64pp|
|128|Shuffle3|68.63%|23.94%|44.69pp|

Locked boundary-effect gate: FAIL. Requires original gap exceed largest repartition gap by>=5pp at BOTH ranks. Only three repetitions: these are descriptive controls, not a calibrated permutation p-value.

Observed rank32 gap26.87pp is reproduced by whole-document shuffle gaps26.76–26.82pp; rank128 observed44.68pp is reproduced by44.64–44.69pp. Thus the original block boundaries are unnecessary for the large transfer gap in these three controls. This weakens a block-specific population-shift explanation while retaining document dependence and empirical high-dimensional variation.

**Interpretation.** Compare the joint within/cross capture levels, not only their difference. If repartition preserves both, the original block boundaries are not needed to reproduce weak transfer. This narrows the explanation toward finite-sample variation in the empirical response distribution, which includes document-specific correlations and non-Gaussian heterogeneity. It does not exclude context-dependent computation within documents or prove a shared Gaussian population. No semantic-context labels were tested.

The residual approximation already used the original first768 documents. Repartition mixes those fitting statuses and is not an end-to-end retraining experiment. Exchangeability is an assumption for any inferential permutation claim, which we do not make here. Output-PCA capture remains an oracle projection, not executable prediction or downstream behavior.

**Audit.** Each partition contains all1024 documents exactly once, hence disjoint equal-size blocks. All result arrays have4×4×18 shape with captures in[0,1]. Cached vectors reproduce original block0/group0 covariance to relative<1e-6; H306 separately replayed aggregate H304 moments. Whole-document indexing preserves all four positions. Responses are saved to allow dependence controls without additional native-model runs. All jobs terminal; no model weights changed.

**Next discriminator.** Repartition individual positions instead of whole documents, holding the empirical response vectors fixed. Cross-block evaluation then intentionally shares document context; it is a dependence diagnostic, not valid fresh-document generalization. A large transfer increase would identify document clustering as a driver of the gap. Compare with the whole-document controls and quantify the difference before interpreting semantic context or estimating effective sample size. Broad research goal remains active.
