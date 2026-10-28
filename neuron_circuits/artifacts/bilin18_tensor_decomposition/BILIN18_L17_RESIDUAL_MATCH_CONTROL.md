

## H360 — Confidence-balanced alternatives to residual-similar donors

Reuse H359 six residual-similar donors per recipient/position. From the remaining18 of the same24 confidence candidates, assign six distinct alternatives minimizing total standardized entropy/top1-log-odds distance to those six donors. Residual direction does not enter assignment. This is exact rectangular minimum-cost assignment via dynamic programming, checked against exhaustive enumeration on ten random3×5 problems. Then evaluate unchanged native responses under alternative donor metrics.

|Head|Moment residual matched|Moment confidence control [joint95%]|Donor residual matched|Donor confidence control [joint95%]|
|---|---:|---:|---:|---:|
|0|0.959|0.858 [0.771, 0.954]|0.965|0.872 [0.785, 0.970]|
|1|1.061|1.100 [0.989, 1.223]|1.058|1.064 [0.957, 1.183]|
|6|1.203|1.432 [1.288, 1.592]|1.185|1.372 [1.234, 1.525]|

Head6 control/residual pairing ratios are1.190 [1.098,1.290] and1.158 [1.068,1.254], with joint2 bands. Both exceed1, but the frozen >=10% effect threshold (lower bound>1.1 for both) FAILS. Confidence-balance and residual-separation gates PASS. Do not round1.098 to1.10 and call it a pass.

Matching audit: mean recipient confidence distances0.451 versus0.440 (difference0.0115), while mean residual cosine0.694 versus0.280. Mean target-donor/alternative confidence error0.128, maximum2.469. This improves control of confidence differences, but does not exactly match every recipient or all context properties. Excluding high-similarity targets also changes the alternative pool; the contrast is diagnostic, not a randomized intervention on residual direction alone.

The confidence-control head6 gap returns toward its previous level, and head0 again points below1. This supports residual geometry as a contributor to the H359 attenuation, without establishing the stronger magnitude claim or semantic coordination.

Audit and run history: builder initially shadowed a path variable and failed before completing config; first launch lacked config. After repair, missing scipy stopped import before model computation. Both failures preserved in work/h360_setup_failure.log and work/h360_dependency_failure.log. Replaced unavailable assignment solver with independently checked exact dynamic programming for the same minimum-cost objective; no outcome-driven threshold change. Native response hashes and H359 donor identities verified, alternatives exclude target donors and self. Three heads completed successfully.

Uncertainty:4000 paired document/control-realization/donor-rank resamples. Joint6 new-condition bands and separate joint2 head6 comparison bands. Fixed matching and donor dependence not reconstructed; heads and panel reused.

Next: replace nearest-neighbor matching with direct geometric control of the output metric, separating its RMS radial projector from its logit-probability weighting. That intervention can distinguish residual-direction geometry from correlated token identity without searching for progressively better donor matches. Broad research goal remains active.
