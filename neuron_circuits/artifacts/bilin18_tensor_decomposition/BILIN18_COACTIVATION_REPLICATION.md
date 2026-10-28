## H712 — Frozen conditional coactivation groups replicate on more disjoint contexts

Freeze all eight H711group memberships and their evaluation metric. Deterministically select2048documents from the existing document-hashed corpus pool, excluding every H681training/validation/reserved-test document and every previously selected128token window hash. No group refit or new neighbor selection. Capture value0/residual1at the same16positions using the first two model blocks. These documents are disjoint from representation fitting and prior group evaluation, but reused from the broader research corpus and not established as model-training holdout. Near duplicates beyond exact hashes may remain.

Repeat within(token,sampledposition)mean subtraction, requiring at least8examples per stratum. The new panel supplies10755eligible positions across424strata, versus495positions/48strata in H711validation. Compare199independent within-stratum per-feature shuffles. For each shuffle, take the maximum anchor-neighbor correlation over all eight frozen groups, providing a multiple-comparison-aware max-statistic reference under the stated shuffle assumptions.

| Anchor | Earlier validation correlation | Replication correlation | Replication all-pair correlation | Shuffle panel maxima at least as large, of199 |
|---|---:|---:|---:|---:|
|97|0.0582|0.0241|0.0139|0|
|216|0.0332|0.0482|0.0291|0|
|7|0.0125|0.0169|0.0035|0|
|208|0.0339|0.0621|0.0280|0|
|166|0.0218|0.0573|0.0299|0|
|121|-0.0040|0.0375|0.0136|0|
|200|0.0138|0.0166|0.0041|0|
|184|-0.0037|0.0272|0.0129|0|

All eight frozen groups show positive conditional coactivation, and every observed anchor statistic exceeds all199null-panel maxima. The add-one max-statistic tail estimate is1/200=0.005for each, not zero. The token-only value0control again has exactly zero residual activity variance, while residual1has variation in all256directions. This is stronger evidence that the neighborhoods capture some recurring context-dependent joint activity than H711's small validation panel provided.

Effect sizes remain modest: anchor-neighbor correlations0.0166–0.0621and all-pair correlations0.00352–0.02995. Groups are anchor-centered neighborhoods, not necessarily cohesive low-dimensional manifolds. Shared membership, activity-budget effects, broader context statistics or correlated downstream roles could produce these associations. The shuffle preserves token/position feature marginals but not the global Top32constraint or all document dependence; the empirical tail estimates are conditional on that null, not universal significance guarantees.

The replication supports advancing these fixed groups to causal testing, not assigning semantic or computational labels. Next use small-dose group-coordinate swaps on this disjoint panel, with rank8PCA and random-subspace edits matched for norm and the previously informative temperature control. Keep all eight groups rather than selecting only the largest replication correlation. Separate donorward output change, observed continuation loss and task correctness. New diverse competence-checked prompts will still be required for any interpretation that survives.

Audits: all source/model/group hashes unchanged; corpus source hashes unchanged; document and128token window disjointness asserted.2048two-block prefix evaluations, whole-model forwards0, test evaluations0, new unique research documents0. The original reserved test split remains untouched. New activation and provenance manifests saved for paired follow-up tests. Broad goal active.
