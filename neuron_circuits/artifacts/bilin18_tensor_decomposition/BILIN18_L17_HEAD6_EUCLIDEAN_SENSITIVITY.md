

## H347 — Output-metric dependence of local alignment

Reuse the H346128-document panel and exact six moment/six donor payloads. Change only the compiler output metric from the intact-state Fisher metric to Euclidean residual-output distance, M=outputs @ outputs.T. Keep the same sensitivity factorization, statistics, thresholds and bootstrap draws. This is a controlled metric comparison on reused documents, not a fresh replication. Gain g=E/P divides tangent output energy by projection energy into the active input span; pooled gain is the active-energy-weighted average. Separate local gain from weighting using the same exact symmetric identity as H345.

Also compare equal-site averages of g/λ_max, where the largest sensitivity eigenvalue is common to actual/control at each intact context. This removes both active-energy weighting and overall spectral scale.

|Statistic|Moment estimate [95% interval]|Donor estimate [95% interval]|
|---|---:|---:|
|Geometric mean site gain ratio|1.21 [1.17, 1.25]|1.20 [1.16, 1.24]|
|Sites with control gain > actual|84.38% [73.43, 85.35]|82.42% [70.70, 83.79]|
|Equal-site spectrum-normalized gain ratio|1.24 [1.20, 1.29]|1.23 [1.19, 1.29]|
|Local-gain share of pooled gain excess|103.30% [97.74, 112.26]|105.03% [100.35, 112.96]|
|Weighting share of pooled gain excess|-3.30% [-12.26, 2.26]|-5.03% [-12.96, -0.35]|

The local-majority and majority-sites gates pass, but the equal-site gain2 gate FAILS for both controls. The previously replicated Fisher ratios were2.862 [2.526,3.433] for moment controls and2.787 [2.462,3.302] for donors. Euclidean ratios are only1.238 and1.233. This weakens a metric-independent large amplification explanation: a modest residual-output gain difference is much larger under the downstream probability metric. Ratios normalized separately by each metric’s largest eigenvalue are not an exact decomposition of the Fisher/Euclidean difference. No causal percentage is inferred.

Uncertainty: 4000 joint document resamples, independently resampling six global realizations per control family. Pointwise percentile intervals condition on frozen model/compiler, fixed donor assignments, this panel and six sampled realizations. Donor document dependence is not fully reconstructed. The nonlinear ratio of averaged control energies is distinct from the mean of individual control ratios. This is a metric-robustness follow-up, not a fresh confirmation or semantic-feature identification. Intervals are not simultaneous across metrics; the frozen decision requires both control families to pass each threshold.

Audit: active projection/input-energy replay against H346 1.24e-13; payload norm error 6.48e-07; exact coefficient-term replay error 6.93e-06; spectral/tangent response energy error 1.5e-14. Exact local-plus-weighting identity asserted in estimates and all bootstrap evaluations. Per-site sufficient statistics now saved.

Next: directly test finite prediction changes from the compiled response, comparing natural and matched control edits. Preserve the distinction between isolated compiler-path interventions and the complete native source intervention, whose direct and remaining paths can cancel. Broad research goal remains active.
