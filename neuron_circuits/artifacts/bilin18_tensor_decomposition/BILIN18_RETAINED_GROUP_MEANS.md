## H903 — Retaining cluster means does not beat nearby-capacity flat approximations

The previous turn made progress: H902 showed that excessive or output-based clustering can move number information into group means, degrading residual-only approximation. H903 tests whether restoring those means repairs the failure. Keep H902 input-cluster assignments and discovery data frozen for K=2,4,6,12; compute each input centroid and contextual-output mean from its assigned discovery endpoints.

For new inputs, route to the nearest normalized-input centroid. Compare cluster mean only; cluster mean plus a single shared residual direction; the prior global-mean residual approximation; and flat PCA rank K+1. Conditional coefficients are exactly compiled from the fixed bilinear weights, avoiding explicit C computation for the approximation. The exact C is computed for intervention/verification controls.

For number interventions, compare recomputing the route on the changed number coordinate (live conditional approximation) with keeping the native route (a diagnostic intervention protocol). Both keep the exact context-only and quadratic MLP terms. The exact contextual write C(w) is invariant to changing only the number coordinate, since w stays fixed. Thus router changes are an additional approximation behavior, not an inherent change in this exact component.

### Competent grammar fidelity

All fits are evaluated on the reused64-context H893 panel. G passes the previous grammar gate, H fails and remains separate. No best K or fit is selected. Native KL ratio uses removal of the same cross term as denominator. Effect RMSE compares exact versus approximate number-flip margin changes relative to each condition's own native baseline.

|K|Global-mean residual effect RMSE|Conditional live effect RMSE|Fixed-route effect RMSE|Flat rank K+1 effect RMSE|Conditional native KL/removal|Flat native KL/removal|
|---|---:|---:|---:|---:|---:|---:|
|2|0.255|0.238|0.238|0.197|0.233|0.224|
|4|0.189|0.179|0.179|0.172|0.268|0.209|
|6|0.178|0.180|0.182|0.151|0.266|0.199|
|12|0.312|0.298|0.303|0.101|0.286|0.175|

Retaining cluster means slightly improves some intervention errors, but does not repair the K12 failure: competent G effect RMSE0.312→0.298, compared with flat rank13 at0.101. Conditional native KL/removal worsens0.260→0.286 there. Cluster means alone perform poorly, with G effect RMSE1.184–1.353, versus removal0.688.

Flat rank K+1 beats the conditional variant on both competent G native KL and effect RMSE at every tested K. The same direction holds for pooled grammar: K12 conditional effect RMSE0.231 versus flat0.074. Copying offers no strong contrary evidence: conditional K12 native KL/removal0.778 versus flat0.583, effect RMSE0.046 versus0.039. Small copy effects still do not imply recovery of its mechanism.

### Capacity and routing accounting

The conditional approximation stores K input centroids, K output means, one decoder and one encoder direction, plus scalar offsets/distance biases: approximately(2K+2)d+K+1 coefficients. Flat rank K+1 stores approximately(2K+3)d+K+1. With d=1152, conditional/flat counts are6915/8067,11525/12677,16135/17287,29965/31117. The flat comparator has1152 extra coefficients, so this is nearby-capacity evidence, not an exactly matched parameter or FLOP comparison. Common number-axis/origin/quadratic components and full-model evaluation machinery are excluded from both counts. Training costs and hard-routing costs are not matched.

Under the number flip, nearest-centroid routing changes on0/64,1/64,2/64 and57/64 contexts for K2/4/6/12 respectively. Yet fixing native routes produces only small changes in effect RMSE and does not repair K12. This rules out route switching as the sole explanation of poor conditional fidelity. It does not prove that every switch is harmful or that better routing cannot help a different representation.

### Research decision

Do not keep adding routing complexity to this isolated component without new evidence. Algebraically C(w) is affine in w for a fixed number axis; the full cross term is t times that affine map. A flat approximation is therefore well matched to this particular subproblem. The result does not refute nonlinear grouped featurizers for the full bilinear MLP, shared-atom representations across multiple axes, or computational DAGs.

The useful finding is narrower: grouping exposed a causally relevant direction hidden by pooled variance, but hard conditional means do not provide a reliable capacity advantage here. Next broaden the diagnostic to existing competent language and binding contexts, using the cached domain-specific captures/fits to identify whether contextual output dimension and causal selectivity resemble the number case. Keep input-axis choice, reconstruction, native fidelity, and intervention fidelity separate. This is a better next discriminator than further tuning a router on the repeatedly reused grammar panel. Shared dictionary/manifold validation and diverse-task reuse remain outstanding.

### Verification

2,880 full-model forwards. Recomputed input centroids reproduce every frozen discovery cluster assignment. Exact native replacement is an identity with zero logit difference. Compiled coefficients match explicit projections under both native and changed coordinates to relative9.21e−15. All source/model/frozen fits and newly saved fits unchanged during evaluation; covariance/mean replay errors below1e−15. All KL finite. No natural or reserved-test evaluation; document pool remains26,112. Goal active.
