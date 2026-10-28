## H708 — Stable contextual coordinates do not yet outperform matched causal controls

Test all eight H707selected stable residual1directions. For each, choose four deterministic hash-selected natural validation pairs with the same token and sampled position, different documents, bothTop32active. Apply both swap directions:64directed edits over32unordered pairs and64unique endpoints. Swap the single decoded coordinate contribution at block1output, preserving the receiver's other residual components and value bus.

Controls: global training PCA1and one fixed random direction receive projected donor–receiver deltas rescaled to the coordinate-edit norm. An additional unscaled Euclidean projection onto the learned decoder direction distinguishes that operation from oblique dual-coordinate substitution. Full donor residual replacement is a reference, not an upper bound. All edits act on receiver prefixes at the final sampled position. No correctness target is defined for these natural context pairs.

Donorward gain=KL(Pdonor||Pnative_receiver)−KL(Pdonor||Pedited_receiver). Positive means movement toward the donor distribution, not improved next-token loss or task correctness. Receiver KL measures change from the native receiver distribution separately.

| Edit | Mean donor KL gain | Median donor KL gain | Positive directions, out of64 | Mean receiver KL |
|---|---:|---:|---:|---:|
|coordinate|0.009434|0.000816|37|0.015073|
|decoder_projection|-0.003030|0.000080|34|0.000860|
|pca|0.013063|-0.000057|31|0.002593|
|random|0.007056|0.000214|43|0.000278|
|full|0.213009|0.077294|50|0.053508|

Learned-coordinate swaps have no clear advantage over norm-matched controls on this panel. Their mean gain minus PCA is−0.00363 (paired bootstrap95%interval−0.0182to0.0125); minus random is0.00238 (−0.0158to0.0255). Bootstrap resamples32unordered pairs, keeping both swap directions together; these are descriptive intervals conditional on selected directions, not broad population guarantees. A single random control direction and small per-feature panels limit conclusions.

Direction121has mean coordinate gain0.10253, but its matched PCA control also gains0.08450. Excluding121leaves mean coordinate gain−0.00386. Direction208is counterproductive on average (−0.03636). These posthoc observations guide red-team analysis, not discovery-confirmed feature labels. The full-residual reference has larger mean gain0.213, but it changes a much larger vector and the donor's other prefix state remains absent.

Coordinate substitutions and Euclidean projections agree in sign on only42of64edits. Their mean norms also differ (about539versus403in native activation units). The projection control is deliberately unscaled; its different outcome cannot be attributed solely to direction semantics. Norm-matched PCA/random edit sizes agree with coordinate sizes to relative discrepancy below1.91e-7.

The result establishes causal sensitivity, but weakens a claim that these stable scalar coordinates individually isolate reusable context computation. Stability, same-token variation and reconstruction gains were insufficient. It does not reject meaningful multi-direction subspaces, nonlinear manifolds or task-specific units; those hypotheses require separate tests. The existing BSF interventions used different directions, ranks and context panels, so their larger gains are not a matched head-to-head comparison.

Next inspect the positive and negative candidate contexts and output changes to identify concrete hypotheses, then evaluate any proposed interpretation on diverse, competence-checked task prompts and matched controls. Also test whether training-derived coactivation groups give more stable/selective subspace effects than individual axes, without assuming that grouping is inherently better. Preserve the negative scalar-control result in the research ledger. Broad goal active.

Audits: all data, model checkpoint, loader, forward and analysis hashes unchanged;384whole-model prefix evaluations,64native endpoints,64directed pairs. Native block1states replay cached activations within1.67e-6relative error. Finite metrics; test evaluations0; no new documents. Validation panel is exploratory and selected directions were identified using validation, so no independent-confirmation claim is made.
