## H511 — Joint attention-write coordinates improve on a head-partitioned representation

Construct a single covariance of the summed centered attention write using the same436 calibration documents as the earlier per-head maps. Its mean equals the sum of the head means. Fit a joint rank36 PCA map and a joint rank36 full-path Fisher map using the same64 training receivers as H507. The full-path derivative includes I+J_MLP. These maps operate on the summed attention contribution, allowing coordinates to span architectural heads, rather than allocating separate coordinates to each head.

Compare with the frozen H509 per-head allocation [10,3,16,1,1,1,1,1,2], also totaling36 coordinates, on the now-reused H510 natural panel. All maps replace the entire attention contribution before the final MLP and residual addition. No new documents, outcome-based rank choice, or fresh confirmation are introduced. Same coordinate count does not establish identical parameter count, encoder access pattern, or implementation cost: the per-head map reads individual head outputs, while the joint map reads their sum. The downstream native computation depends on that sum.

| Representation | Source rank | Mean KL to original | Recovery vs mean replacement | Observed-token ΔCE (paired SE) |
|---|---:|---:|---:|---|
|Per-head allocated|36|0.004431|63.42%|+0.009818 (0.004927)|
|joint_PCA|36|0.003664|69.75%|+0.009184 (0.004196)|
|joint_Fisher|36|0.003607|70.22%|+0.008529 (0.004230)|
|empirical_span|435|0.000650|94.63%|+0.001490 (0.001441)|

Joint PCA reduces KL from0.004431 for the per-head allocation to0.003664, about17.31%. Joint Fisher reduces it further to0.003607, only1.56% below joint PCA. Thus most of the improvement in this comparison comes from representing the summed source jointly, with a smaller additional effect from probability weighting. This is evidence against treating architectural head boundaries as necessary boundaries for this low-rank data-based representation. It does not identify semantic units or establish a unique joint basis.

Calibration support is an important limitation. The436 centered calibration samples span435 dimensions of the1152-dimensional attention-write space. The basis exactly reconstructs calibration sources to numerical tolerance, but4.29% of held-out summed source energy lies outside that span. Even retaining all435 supported dimensions leaves mean KL0.000650 on the natural evaluation panel. This is a high-rank support control, not a matched-budget result or a strict lower bound for every possible oblique map; compensating for omitted components could change its behavioral error.

Consequently, the empirical covariance nullspace cannot be treated as universally unused computation. It is absent from this finite calibration sample, not necessarily from the underlying data distribution. A data-based canonical form must specify both the sampled support and the desired behavior metric. The joint calibration covariance condition number over retained dimensions is about84,232; the prescribed singular cutoff is1e-10 relative to the largest singular value. No outcome-based regularization or rank adjustment was made.

The joint Fisher training-local recovery is80.89%, compared with77.95% for joint PCA on the same training objective. Their finite held-out recovery difference is much smaller. Possible limitations include metric estimation noise, finite source/receiver correlations, empirical covariance estimation, and nonlinear effects. CE changes remain positive/noisy; these are fidelity maps, not demonstrated task-loss improvements.

Audits: source head summation relative error4.31e-7; composed full-path derivative central-difference relative errors below1.07e-8; supported calibration reconstruction relative error7.10e-14; mean-replacement outcomes replay H510 to maximum absolute error9.16e-12. The Fisher training objective is no worse than PCA at rank36. Models were saved before finite evaluation; rows are finite and aggregates replay. No additional documents were consumed; total remains19,968.

Next H512: freeze both joint maps and the per-head allocated comparator, then test them on512 new complete-document-disjoint natural contexts. Prescribe two primary paired mean-KL reductions: joint PCA versus per-head allocation, and joint Fisher versus joint PCA. Use one simultaneous95% document-bootstrap band for these two tests. This separates the benefit of the joint representation from the incremental benefit of probability weighting. Report all absolute KL and CE outcomes without changing rank, covariance, metric, or maps after outcomes. Keep the empirical-span limitation visible; do not infer universal equivalence from low finite-sample KL.
