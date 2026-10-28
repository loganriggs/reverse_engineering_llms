

## H331 — Direct residual and MLP behavioral interaction

Reuse H327’s128 documents and18 head/payload groups. Split source-induced final-state change into direct residual change, compiler MLP response and remaining MLP response. Remove all seven nonempty combinations from the native edited state. In parallel compute the local probability-weighted, centered logit-derivative Gram matrix at that state, differentiating final RMS normalization, unembedding and softcap. Its quadratic prediction for removing a sum of terms is half the corresponding Gram-block sum. No new fitting.

|Target|Direct–MLP Fisher cosine [95%]|Direct–compiled cosine|Direct–remaining cosine|Finite joint-vs-sum change|
|---|---:|---:|---:|---:|
|actual|-0.494 [-0.530,-0.455]|-0.364|-0.335|-42.68%|
|angle|-0.295 [-0.326,-0.265]|-0.192|-0.184|-26.89%|

Finite joint-vs-sum change=(KL(source)−KL(direct)−KL(MLP))/(KL(direct)+KL(MLP)); it is an interaction statistic, not an additive attribution. The Fisher cosine measures alignment in local probability-sensitive geometry. Negative values mean opposing local effects, not necessarily learned repair or a semantic cleanup function.

|Target|Removed path(s)|Finite KL|Local quadratic KL|
|---|---|---:|---:|
|actual|direct|0.00164038|0.00162002|
|actual|compiled|0.000409201|0.000427406|
|actual|residual|0.000212992|0.000212404|
|actual|direct_compiled|0.00149151|0.00144105|
|actual|direct_residual|0.00145731|0.00143895|
|actual|mlp|0.00061479|0.000633308|
|actual|source|0.0012926|0.00125349|
|angle|direct|0.00169685|0.00174814|
|angle|compiled|0.000647389|0.000633105|
|angle|residual|0.000264018|0.000266147|
|angle|direct_compiled|0.00194156|0.0019769|
|angle|direct_residual|0.00169458|0.00176364|
|angle|mlp|0.000717663|0.000705956|
|angle|source|0.00176532|0.00179911|

Locked gates: both_cancel: PASS, actual_more_cancel: PASS. both_cancel requires direct–MLP cosine<=−0.1 and paired95% upper bound<0 in both families. actual_more_cancel requires angle cosine−actual cosine>=0.1; it is a point-estimate gate, not a significance claim. Bootstrap resamples128 documents, conditional on the fixed angle realization and chosen decomposition.

**Audit.** Recomputed H330 KL endpoints differ by at most2.44e-07 relative, allowing different float32 subtraction order. Full per-document Fisher Grams and finite Kls retained. The derivative formula has a separate synthetic finite-difference audit. Native weights unchanged.

**Limits and next discriminator.** This is local output-distribution geometry and finite algebraic mediation on a reused panel, not proof of semantic feedback. If native edits show stronger opposition, test source/context shuffles with matched edit geometry and fresh documents. Compare compiled and remaining paths before assigning a cancellation role to compact units. Broad research goal active.
