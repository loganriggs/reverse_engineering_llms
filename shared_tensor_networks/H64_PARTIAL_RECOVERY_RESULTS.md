# h64 recovery: broad mismatch versus a few incomplete features

This is a post-hoc descriptive analysis of the60already-audited h64fits, with no new training or seed selection. The original full-recovery criterion requires all64 matched Gaussian quadratic-feature cosines to exceed.99. That binary criterion conceals distinct failures.

| Inputs | Tasks | Sorted number of features recovered across10random starts | Full recovery |
|---:|---:|---|---:|
|8192|520|0,0,0,0,0,0,0,0,0,0|0/10|
|8192|2080|0,0,0,0,59,60,61,61,61,64|1/10|
|32768|520|60,61,61,61,61,62,64,64,64,64|4/10|
|32768|2080|58,59,60,61,62,62,62,64,64,64|3/10|

At8192inputs/fulltasks, both random starts fail broadly for two teachers, while both starts recover59–64features for the other three. This is teacher-associated heterogeneity in this panel, not proof of a general phase transition. At32768inputs all runs recover most features. Relaxing.99to.95 changes none of the high-input counts, so their remaining misses are not close threshold crossings. No uncertainty estimate treats features or starts as independent teacher replications.

## Are only the axes wrong, or is the span wrong too?

For each unit-norm teacher quadratic, compute its squared population L2 projection residual onto the span of all learned quadratics. This permits arbitrary linear combinations of learned features; it is more permissive than one-to-one matching. Gaussian inner products use E[(xᵀQx)(xᵀRx)]=2tr(QR)+tr(Q)tr(R). Gram eigenvalues below1e-12times the maximum are omitted.

| Inputs | Tasks | Median teacher-mean residual, averaged over all64features |
|---:|---:|---:|
|8192|520|.95518|
|8192|2080|.01869|
|32768|520|.007742|
|32768|2080|.012421|

For high-input runs with any misses, the mean residual among their missed teacher directions ranges.268–.331with520tasks and.254–.330with2080tasks. Thus around25–33% of the missed quadratics' squared population magnitude cannot be recovered by a linear combination of the learned quadratics. The small all-feature average reflects how few directions are affected. It does not mean every missed direction has the same residual.

Every random learned dictionary has rank64under the stated cutoff, including broad failures. Calling this numerical rank collapse would be incorrect. The diagnostics distinguish broadly wrong full-rank dictionaries from nearly recovered dictionaries with several partially missing teacher directions. They do not identify why those directions were missed, prove a local minimum, or establish that the resulting product-task function span has the same residual. Linear feature-span error and downstream quartic function error are different quantities.

## Verification and research consequence

All input records are tied to the passed h64audit; all60checkpoint hashes were verified. Existing Gaussian feature-matching values replay exactly within tolerance. Span projection was independently replayed as Euclidean least squares on the embedding [sqrt(2)vec(Q),tr(Q)], with maximum residual discrepancy 6.19e-14. Synthetic invertible mixing preserves the span, while dropping a direction produces a positive residual; both controls pass. These are algebraic/numerical diagnostics, not additional independent learning replications.

The frozen h128experiment is unchanged. Its already-planned partial recovery distributions will help distinguish broad failure from a small number of missed directions. A mechanistic follow-up could test whether missing directions are weakly supervised, ill-conditioned, or trapped by finite-budget optimization; these descriptive results alone do not choose among those hypotheses. No optimizer rescue is introduced.

Artifacts: results/h64_partial_recovery/analysis.json and results/h64_feature_span_verified/analysis.json. The initial projection-only calculation and its exact source are retained in results/h64_feature_span; the verified calculation is authoritative.
