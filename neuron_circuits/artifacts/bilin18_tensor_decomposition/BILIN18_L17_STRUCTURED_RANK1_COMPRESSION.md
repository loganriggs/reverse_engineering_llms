## H487 — Data-weighted structured rank1 compression

Exploit H486's exact source-conditioned response structure rather than fit an arbitrary global quadratic-coordinate map. For fixed source direction b, write N1=M_b p with M_b=−D[diag(Rb)L+diag(Lb)R], fixed N2=D[(Lb)⊙(Rb)] and d(a)=d0+a d1+a²d2. Approximate only M_b p, retaining the scalar denominator, fixed quadratic vector and native bias-free baseline write w0:

Δw_hat = a M_hat p/d(a) + a² N2/d(a) + [d0/d(a)−1]w0.

Training uses the same1024 receivers and32 fixed centered-source donors, both signs. Build the matrix of exact cross-term contributions a M_b p/d(a), then take its uncentered output SVD. For each rank r, M_hat=V_r^T V_r M_b. This minimizes training cross-term squared projection error at that rank, not finite KL. The compiled read M_b^T V_r^T and write V_r avoid retaining the full cross operator at inference. No validation/confirmation input enters the SVD.

Select the smallest rank in[4,8,16,32,64,128] with validation rank1 KL recovery>.95 at both signs, otherwise use128 and flag the criterion unmet. The256 validation receivers select rank32; their rank1 recoveries are96.09/96.59%. Select and save the compiled map before collecting outcomes on512 new complete-document-hash-disjoint confirmation receivers. Independent audit rehashes16832 excluded documents and512 new ones, verifies all129-token windows and finds zero overlap. Total tracked documents17344. Donor pool and calibration remain fixed; this is continuation freshness, not a model-training holdout.

| Model | Rank1-target KL recovery −/+ | Full-source KL recovery −/+ | Rank1 raw recovery −/+ |
|---|---|---|---|
|structured|96.288% / 95.097%|92.326% / 90.382%|99.606% / 99.614%|
|omit_norm_correction|5.608% / -13.094%|-1.284% / -19.641%|63.594% / 60.504%|
|exact_rank1|100.000% / 100.000%|96.302% / 95.828%|100.000% / 100.000%|

The normalization ablation omits only [d0/d(a)−1]w0; it still uses d(a) in the cross and quadratic terms. It is not identical to H486's frozen-denominator control. Exact_rank1 keeps the full native cross operator while dropping the source tail; it isolates the source-truncation limitation. Full-source and rank1-target recoveries have different native effect denominators and must not be subtracted as additive error shares.

| Predeclared primary statistic | Estimate | Joint95% interval | Threshold | Pass |
|---|---:|---|---:|---|
|absolute_negative|0.9629|[0.9506, 0.9751]|0.95|True|
|norm_gain_negative|0.9068|[0.6296, 1.1840]|0|True|
|absolute_positive|0.9510|[0.9219, 0.9801]|0.95|False|
|norm_gain_positive|1.0819|[0.7044, 1.4594]|0|True|

Three of four gates pass. The negative-sign rank1 absolute lower bound .9506 exceeds.95; the positive-sign lower bound .9219 does not. Both normalization gains have joint lower bounds above zero. Gain intervals may exceed1 because the ablation can be worse than the zero-response baseline; these are differences in error-normalized recoveries, not probabilities. No full-source95% gate was specified or met. The source tail remains a limitation: exact rank1 recovers96.30/95.83% of full-source KL effects on this panel.

Four simultaneous standardized bands use4000 paired within-donor receiver resamples, eight receivers per64 fixed donors. They condition on the training basis, validation rank selection, donor pool and calibration. No claim of training-sample uncertainty or cross-head robustness. The baseline write is provided exactly by the native model, as in earlier response-isolation comparisons; this method does not independently replace a whole layer or establish CE preservation.

Storage: rank32 compiled read/write matrices contain73728 scalars. Adding b, N2, d2 and epsilon gives76034 stored scalars. This boundary expects receiver p, source scalar a and the native bias-free baseline write w0 as inputs. It excludes the native computation needed to produce p/w0 and the bias subtraction if only a biased baseline write is available. Therefore the count is for the conditional response program, not the complete language model, and no wall-clock speedup is claimed. Its rank and exact structured components differ from the previous PCA20 predictor, so gains across their separate panels cannot establish a matched architecture advantage.

Audits: cross-operator matrix identity agrees below1e−10; exact-rank1 response replay exceeds1−1e−7 raw recovery on both signs; all outcomes finite; rank-selection record and compiled factors saved before confirmation. Independent document exclusion/window audit passed. Compiled factors, selection grid, per-case effects/errors and confirmation baseline inputs archived. Native weights inform the cross operator; this is a data-weighted factorization of known computation, not a claim to infer weights from observations alone.

Interpretation: preserving known normalization structure enables a compact conditional response representation with high fresh rank1 fidelity. The important unit here is a fixed source direction acting through a receiver-linear operator plus a normalization correction, not a global static output feature. The formula and factorization do not establish a unique model-wide ontology.

Next run a matched diagnostic on these same receivers: evaluate the frozen earlier PCA20/kernel256 predictor under both full and rank1 sources, alongside the structured map, using the same native baseline, target definitions and signs. Include a normalization-aware hybrid of the earlier predictor only if its decomposition is explicitly defined; do not infer an advantage by comparing separate cohorts. Report parameter/storage and exact-baseline differences. This will quantify how much of the observed progress reflects source truncation, panel differences or the explicit structure. The comparison will be post-hoc; subsequent learned claims require fresh confirmation. CE objectives, full-layer replacement and cross-head/layer transfer remain open.
