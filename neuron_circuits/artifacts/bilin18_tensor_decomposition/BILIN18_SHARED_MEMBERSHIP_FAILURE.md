## H690 — Learned overlap can lose capacity and drift away from a successful fit

H689's failed learned memberships could reflect poor continuous weights or a support structure unable to represent the data. We distinguish these by calculating per-support rank bounds, then continuing each fit for 6000 additional steps with either frozen membership or live membership. All six continuations use the same H689 training data and batch stream within seed, fresh Adam at learning rate0.001, and supplied true group gates. This is not an exact optimizer resume. No heldout checkpoint selection.

For a support with k independent unit-variance teacher directions and only r selected dictionary atoms, any reconstruction confined to those atoms has expected relative squared error at least max(k−r,0)/k. This holds even with an optimal encoder and decoder. Bias cannot remove independent zero-mean missing variance. Exact population errors are computed analytically, and every initial/final support error respects this lower bound.

Seed11 has no rank deficit. Seed12 has deficits on one training pair and three held supports: for all four groups, it selects7atoms for8independent directions. Seed13 has deficits on four training pairs and all five held supports: all four groups select6atoms for8directions. Thus the failed memberships have provable capacity limitations. Rank counts are necessary, not sufficient, for recovery and do not explain all residual error.

| Seed | Initial held relative MSE | After frozen membership | After live membership |
|---|---:|---:|---:|
|11|8.23922e-09|6.36305e-08|0.163317|
|12|0.110975|0.111847|0.250942|
|13|0.217279|0.217592|0.255741|

Frozen memberships preserve the successful fit and do not repair the failed ones. Live continuation changes membership in all three seeds and worsens all three fits, including seed11's previously near-exact solution. Its training relative error rises from2.17e-7 to0.232, so this is not merely heldout overfitting. The straight-through top-k membership optimizer is not a reliable monotonic search over discrete group structure. This experiment does not isolate optimizer restart, step size, surrogate gradients, or discrete switches as the sole cause; all remain candidate contributors.

The useful next comparison is training-only accepted discrete membership changes with continuous refits, against validated best-checkpoint straight-through training and a relaxed/annealed variant. Select by training or independent validation, never these heldout compositions. We should measure recovery reliability across seeds, not report only the successful seed. Thereafter, genuinely unsupervised group gates, trained hierarchy/computation benchmarks, matched budgets, and early-layer manifold/causal validation remain required.

This result does not reject shared dictionaries: known memberships already compose accurately. It identifies a concrete bottleneck in learning reusable groups. No LLM forwards, no new documents, and no access to the reserved LLM test split. All input/code hashes unchanged. Broad goal remains active.
