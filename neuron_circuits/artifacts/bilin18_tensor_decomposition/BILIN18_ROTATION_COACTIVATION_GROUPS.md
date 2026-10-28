## H711 — Token/position-conditioned groups show weak validation repeatability

Use H706seed7061Top32binary activities. Within each joint(token identity,sampled position)stratum, subtract the stratum's mean activity. Estimate correlations separately in the two1024document training halves and256document validation set; retain strata with at least8examples. For each of eight previously stable H707anchors, select seven neighbors by the minimum of their two training-half conditional correlations, requiring positive scores. This yields overlapping eight-direction candidate groups. Neighbor selection uses training only; anchors were previously selected using validation stability.

Validate mean anchor-neighbor and all-within-group conditional correlations. Compare with100independent per-feature shuffles within each eligible validation stratum, preserving each feature's conditional marginal distribution while removing joint coactivation. This shuffle does not preserve the global Top32constraint or all document dependencies, so it is a descriptive null, not an exhaustive causal control.

The value0control has exactly zero residual activity variance after conditioning, as expected for a token-only representation. Residual1has variance in all256directions. Eligible examples are4180/4104in the training halves and495in validation; validation has48eligible strata. These are subsets dominated by sufficiently frequent token/position combinations, not all contexts.

| Anchor | Validation anchor-neighbor correlation | Mean all-pair correlation | Shuffle95%range | Shuffles at least as large, of100 |
|---|---:|---:|---|---:|
|97|0.0582|0.0199|[-0.0346, 0.0386]|0|
|216|0.0332|0.0212|[-0.0314, 0.0376]|4|
|7|0.0125|-0.0027|[-0.0335, 0.0276]|31|
|208|0.0339|0.0084|[-0.0332, 0.0348]|4|
|166|0.0218|0.0145|[-0.0312, 0.0345]|6|
|121|-0.0040|0.0025|[-0.0390, 0.0336]|55|
|200|0.0138|0.0110|[-0.0363, 0.0252]|15|
|184|-0.0037|-0.0029|[-0.0355, 0.0309]|55|

Only anchor97clearly exceeds its individual central95%shuffle range; its conditional correlation is0.0582and all-pair correlation0.0199. None of100shuffles exceed it, which is not a zero p-value: an add-one randomization estimate would be1/101before multiple-comparison considerations. Eight candidate groups were examined, so this is not strong corrected confirmation. Anchors216/208are borderline by individual shuffle tail counts; the remaining groups are weak, and121/184have slightly negative validation anchor correlations.

The neighborhoods were deliberately not selected using donorward intervention gains. In particular, scalar candidate121does not gain validation coactivation support from this procedure. Multiple coactive directions cannot be presumed a better computational unit simply because single-coordinate swaps were inconclusive. The fixed eight-direction neighborhood size is a heuristic, not an inferred manifold dimension; no semantic labels or global group-gate model has been learned.

Next freeze all eight memberships and the conditional-correlation metric, then evaluate them on additional document-disjoint contexts from the available corpus pool, excluding H681training/validation/reserved-test documents. This increases repeated-token/position coverage without changing groups based on new results. Keep the reserved test split sealed and document research reuse versus representation-fitting independence. If repeatability remains weak, do not advance these groups as verified manifolds; reconsider the activity statistic or group hypothesis explicitly. Broad goal active.

Audits: source hashes unchanged, finite metrics, training-only neighbor selection, test evaluations0, whole-model forwards0, no new documents. A development-time validation-variance filter was removed before the final rerun to keep neighbor selection training-only; final source and results reflect that correction. Anchor selection still reused prior validation, as disclosed. No causal group intervention claim is made.
