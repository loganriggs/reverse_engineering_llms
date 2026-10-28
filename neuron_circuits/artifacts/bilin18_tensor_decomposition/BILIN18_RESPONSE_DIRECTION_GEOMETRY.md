## H826 — Downstream response directions preserve only part of edit geometry

Reanalyze H825's saved full-vocabulary logits for all56 frozen receivers, without further model forwards or selection changes. The preceding goal turn completed663 forwards and established mixed donor transfer but no consistent equal-norm near/far effect-size ordering. Here test angular relationships rather than magnitude alone.

### Measurement

Each receiver has six common-norm edits: near/far crossed with block/PCA/random. Subtract that receiver's saved zero-edit logits from each edited output. For native recipient probabilities p and logit change d, form sqrt(p)*(d-E_p[d]). This removes irrelevant constant logit shifts and weights changes by the recipient distribution. Its inner product is the local categorical Fisher metric; these are finite interventions, so this is not an exact finite KL distance. Also compute uniformly centered logit-change cosines as a secondary measure.

For each receiver compare all15 pairs of edit directions with their output-response directions. Report Pearson correlation across those15 angles. Calculate the participation rank of the Gram matrix of six normalized directions, (trace G)^2 / sum(G^2). This measures concentration among the six sampled directions, not the rank of the full model Jacobian, the number of semantic features, or the dimension of a manifold.

### Results

| Site | Receivers | Median input direction rank | Fisher output rank | Uniform output rank | Median input/output angle correlation |
|---|---:|---:|---:|---:|---:|
| residual0 |24|4.445|2.655|3.457|.503|
| residual1 |32|4.705|2.799|3.276|.468|

The sampled output effects are more angularly concentrated than the edit directions. Input geometry contains information about output geometry, but is far from a complete description: median absolute input/output cosine distortion is .384 at residual0 and .330 at residual1. Probability weighting contributes to concentration; uniformly weighted output ranks also fall below input ranks, though less strongly. These are descriptive receiver medians, not held-out prediction scores or independent-replication estimates.

Near/far block edit cosines have medians .285/.255 at residual0/residual1. Their Fisher response cosines are .378/-.047, and uniformly centered response cosines .291/.078. The corresponding PCA Fisher response cosines are .285/.400 and random .423/.342. These different input-direction distributions prevent interpreting the raw response-cosine ordering as a matched test of featurizer quality. In particular, residual1 block near/far effects need not point in a similar output direction even when their input directions have modest positive alignment. Median comparisons do not establish a specific pairwise reversal or semantic opposition.

### Interpretation

These results are consistent with anisotropic downstream sensitivity: some combinations of edit directions affect similar output directions, others are reoriented. A simple fixed linear map with unequal singular values could produce this, so curvature or nonlinear manifold computation is not required. Euclidean block geometry alone is therefore an incomplete guide to the output consequences in this panel.

Recipient-dependent sensitivity remains a hypothesis, not an established result: each receiver was given different donor-derived directions. We have not crossed identical edits across recipients, and the Fisher metric itself changes with recipient probabilities. Six directions also provide limited coverage of a1152-dimensional activation space. The result does not distinguish fixed anisotropy from context-dependent routing, finite-dose nonlinearities, or discrete contextual states.

No validated semantic manifold, hierarchy or computational DAG follows. Retain flat and grouped explanations. H825's donor-transfer evidence and H826's geometry distortion are compatible: a representation can carry contextual information without preserving Euclidean distances or angles downstream.

### Audits

All840 pairwise rows are retained. Every response exceeds10 times the weighted zero-edit/native replay discrepancy; the minimum ratio is1399.3. That discrepancy is a numerical replay proxy, not a bound on biological/statistical noise or total numerical error. Common edit norms agree within6.66e-16 relative error; all input hashes unchanged and metrics finite. Independent checks verify invariance to adding a constant logit shift (9.71e-16 max discrepancy) and the weighted-covariance inner-product identity (1.11e-16). Zero new whole-model forwards, no new documents or reserved-test evaluations; natural-document pool26,112.

### Next discriminating experiment

Cross identical frozen edit directions across multiple same-token/same-position recipients within each selected block. Use a small fixed amplitude ladder and both signs to estimate local response linearity. Compare uniformly centered output responses and a COMMON reference probability metric, as well as recipient-specific Fisher weighting, to separate changes in downstream sensitivity from changes in metric. Freeze recipient/direction selection before outputs and use unused occurrences from the H824 panel rather than recycle the56 evaluated receivers. If identical directions transfer poorly even at small doses and under a common metric, context-conditioned computation becomes a stronger explanation. If they transfer well, a common anisotropic response subspace may explain much of the apparent geometric complexity more simply.
