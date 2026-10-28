## H518 — Mean offset is small; calibration source moments underestimate residual error

Hold the H516 training Fisher metric and all PCA/Fisher maps fixed. In its1,023-dimensional supported basis, measure one-half E[e^T F e] for three source panels: calibration sources centered on their own new mean; calibration sources relative to the old shared reconstruction mean; and held-out sources relative to that same shared mean. Decompose each expectation exactly into a centered covariance term and a mean-offset term. Use empirical covariance normalized by n, so covariance plus mean outer product equals the empirical second moment exactly; H516's n−1 normalization differs by a common factor for the centered calibration panel and leaves recovery ratios unchanged.

No network forward pass, representation fit, source selection, or metric update is performed. This diagnostic changes only the source moments under a fixed metric. It does not measure new natural task performance. Components outside the supported basis are excluded here and reported separately; comparisons to H517's actual-pair local scores require that caveat.

| Rank | PCA calibration recovery | PCA held-out-source recovery | Fisher calibration recovery | Fisher held-out-source recovery | Held-out / calibration error: PCA / Fisher |
|---|---:|---:|---:|---:|---|
|36|75.26%|72.04%|78.63%|75.89%|1.09 / 1.09|
|72|84.05%|79.60%|86.87%|83.13%|1.24 / 1.24|
|144|91.55%|86.05%|93.72%|89.62%|1.59 / 1.60|
|288|97.21%|92.23%|98.21%|95.00%|2.69 / 2.70|
|576|99.73%|97.50%|99.86%|98.69%|8.97 / 8.97|

Changing the reconstruction centering has little effect on predicted recovery. For Fisher rank36, the centered-calibration prediction is78.67% and the shared-mean calibration prediction78.63%; at rank144,93.74% versus93.72%; at576,99.862% versus99.859%. On held-out sources, mean-offset terms account for about0.42%,0.38%,0.41% of Fisher error at those ranks. Thus the shared-mean choice is not the main explanation of the observed objective mismatch in the supported-space calculation.

Source moments matter much more. Under exactly the same training receiver metric, held-out-source error is about1.09 times calibration error at rank36,1.60 at144,2.70 at288, and8.97 at576 for Fisher. PCA has nearly identical ratios. The high-rank multiplicative gap concerns a small residual error, not a ninefold total model loss. For Fisher576, the absolute predicted KL rises from0.0000141 to0.0001265. This points to optimism in the calibration residual statistics shared by both representation families, not only Fisher-specific oblique amplification.

These panels differ in sampled source moments, so this result cannot distinguish finite covariance-estimation effects from genuine source-distribution shift. The high-dimensional covariance was estimated from1,024 documents, making small residual variance estimates particularly uncertain. No claim of a specific population shift or of calibrated out-of-distribution robustness follows.

Even replacing calibration moments with held-out moments does not fully reproduce H517's actual receiver-weighted results. For example, fixed-training-metric held-out Fisher36 recovery is75.89%, while naturally paired local evaluation is69.17%. These quantities differ in receiver sensitivity, source/receiver pairing, and treatment of outside-span components. The present decomposition cannot assign their difference to any one cause. H517 already showed that breaking source/context pairing alone does not expose a large universal Fisher advantage.

Support caveat: outside-span raw source fractions are essentially zero for centered calibration,0.000275% for calibration around the shared mean, and0.0829% for held-out sources. Small raw fractions do not prove negligible behavioral effects, especially near full rank. The rank1,023 supported-space errors vanish by construction here and must not be confused with H515/H516's nonzero native full-span residual KL.

Audits: every per-source quadratic score is finite; covariance and mean-offset terms reconstruct the total within1e-10 relative tolerance; centered-calibration recovery replays every stored H516 training-objective ratio within1e-9. Frozen artifacts are hash-checked. All source panels and map ranks are prescribed, with no refitting or selection. No new documents were used; tracked total remains21,504.

Next H519: compute the mean full-path Fisher metric over all512 evaluation receivers in the same fixed supported basis, for diagnostic use only. Cross the two metrics (training64 versus evaluation512) with the two shared-mean source panels (calibration versus held-out), keeping every representation map fixed. In the same pass, compute naturally paired supported-space local errors for PCA/Fisher ranks36/144/576 and mean replacement. This yields exact supported-space independent-pair expectations from source moments, allowing receiver-metric differences and natural pairing effects to be separated without Monte Carlo permutation noise or an outside-span mismatch. Do not refit maps using this evaluation metric or call it a deployable model. Record its diagnostic/oracle role, derivative audits, and comparison with H517's full-vector local scores. The objective is to identify which approximation limits behavioral coordinate selection, not to optimize on the evaluation set.
