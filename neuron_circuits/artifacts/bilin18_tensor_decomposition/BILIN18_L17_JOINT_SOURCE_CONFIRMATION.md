## H512 — Joint source representation confirms; incremental Fisher gain does not

Freeze H511's joint PCA36 and joint Fisher36 maps and H509's per-head allocated36 comparator. Evaluate full attention-write replacement before both the layer17 MLP and residual addition on512 fresh complete-document-disjoint contexts. Preserve every fitted mean, covariance, metric, basis, rank, and allocation. Two comparisons were prescribed: joint PCA versus the per-head allocation, and joint Fisher versus joint PCA. Use one simultaneous95% bootstrap band across their paired mean-KL reductions, with4,000 iid document resamples. CE remains descriptive.

| Representation | Mean KL | Recovery vs mean replacement | Observed-token ΔCE (paired SE) |
|---|---:|---:|---|
|mean|0.011913|0.00%|+0.026886 (0.007296)|
|allocated|0.005004|58.00%|+0.006844 (0.004894)|
|joint_PCA|0.003897|67.29%|+0.006329 (0.004119)|
|joint_Fisher|0.003866|67.55%|+0.004865 (0.003985)|
|empirical_span|0.000732|93.86%|+0.000134 (0.001544)|

| Prescribed improvement | Mean KL reduction | Simultaneous95% interval | Positive lower bound |
|---|---:|---|---|
|joint_PCA_over_allocated|0.001107|[0.000806, 0.001408]|True|
|joint_Fisher_over_joint_PCA|0.000031|[-0.000091, 0.000153]|False|

Joint PCA improves on the per-head allocation: KL falls from0.005004 to0.003897, a22.12% relative reduction. Its simultaneous lower bound for absolute mean-KL reduction is0.000806, above zero. This confirms that the joint summed-source construction can outperform this head-partitioned allocation at the same36-coordinate budget on new natural contexts.

Joint Fisher's additional point improvement is only0.0000313 KL, about0.80% relative to joint PCA. Its simultaneous interval spans−0.0000908 to+0.0001534, so the prescribed improvement test fails. This does not establish equivalence, no benefit, or no value for Fisher metrics elsewhere. Earlier per-head and read/write comparisons tested different representation families and objectives; they are not contradicted by an inconclusive incremental gain here.

The evidence therefore favors joint representation of the summed attention write as the main demonstrated improvement in this stage. It does not isolate a single mechanism: joint covariance modeling, flexible coordinate allocation across mixed head directions, and the constraint of observing the summed write all change relative to the per-head map. Same coordinate count is not a deployment cost comparison. Architectural head identity is not required by the downstream native computation, which receives their sum, but this does not imply heads never have distinct causal or semantic roles.

The finite calibration-support limitation persists. Held-out source energy outside the435-dimensional calibration span is4.55%. The high-rank empirical-span control leaves KL0.000732. This control is not a strict oracle error floor; it uses orthogonal projection and can differ from other compensating maps. Empirical nullspace should not be equated with universally unused computation. Observed-token CE changes remain positive/noisy and were not subjected to a loss-neutrality or improvement test.

Audits: independent rehashing verifies512 fresh complete documents with no overlap against19,968 prior documents; all saved token windows match their manifest source caches. Total tracked documents is20,480. Frozen artifact hashes were checked before evaluation; head-write summation passed1e-5; token position63 predicts document token64; arrays are finite and aggregate outcomes replay. Bootstrap critical value is2.2131. Intervals condition on the frozen representation fit and sampled document population, not model-training or calibration uncertainty. This is not a model-pretraining holdout claim.

Next H513 tests calibration-sample stability of joint PCA36, the simpler construction whose benefit was confirmed. Use four prescribed complementary random splits of the original436 calibration documents (218 per half); fit a rank36 PCA basis independently in each half, retaining the full-calibration mean for all reconstructions so the diagnostic isolates basis variation. Compute each half covariance around its own training-half mean. Compare the paired subspaces using principal angles and compare actual reconstructed output distributions on the now-reused H512 panel; retain the original436-document basis as reference. Do not select the best split or align axes using evaluation outcomes. Report all fits/pairs and distinguish scalar fidelity, subspace stability, and prediction agreement. The halves within each split are disjoint, while splits overlap; this measures basis sensitivity conditional on the shared global mean, not total estimator uncertainty. It does not add new documents or constitute another fresh confirmation.
