

## H346 — Fresh confirmation of local sensitivity alignment

Use128 fresh documents excluding4288 complete-document hashes from listed prior panels, with six new moment draws and six new donor assignments. Freeze the H345 statistic, compiler and thresholds. Recompute the intact-state output metric on the new panel. Independently audit complete-document hashes and exact token windows. Freshness is relative to listed panels, not model training or all historical studies. Average the six control realizations at each site before forming gains. Let P be active projection energy, E tangent response energy, g=E/P, and w=P/ΣP. The pooled gain is Σwg. Split its control-minus-actual change exactly as Σ[(w_c+w_a)/2](g_c−g_a) + Σ(w_c−w_a)[(g_c+g_a)/2]. These are local-gain and weighting contributions, an algebraic attribution rather than causal interventions.

Also compare equal-site averages of g/λ_max, where the largest sensitivity eigenvalue is common to actual/control at each intact context. This removes both active-energy weighting and overall spectral scale.

|Statistic|Moment estimate [95% interval]|Donor estimate [95% interval]|
|---|---:|---:|
|Geometric mean site gain ratio|3.17 [2.47, 3.34]|2.88 [2.26, 3.03]|
|Sites with control gain > actual|87.11% [77.93, 88.67]|84.57% [74.80, 85.94]|
|Equal-site spectrum-normalized gain ratio|2.86 [2.53, 3.43]|2.79 [2.46, 3.30]|
|Local-gain share of pooled gain excess|102.06% [98.10, 110.81]|103.76% [100.21, 111.42]|
|Weighting share of pooled gain excess|-2.06% [-10.81, 1.90]|-3.76% [-11.42, -0.21]|

All three prespecified gates pass: local-gain share lower bound>50%, equal-site normalized gain ratio lower bound>2, and majority of sites show larger control gain, for both control families. Values above100% arise because weighting slightly offsets the local-gain increase. Within-context gain differences account for essentially all pooled gain excess under this decomposition.

Uncertainty: 4000 joint document resamples, independently resampling six global realizations per control family. Pointwise percentile intervals condition on frozen model/compiler, fixed donor assignments, this panel and six sampled realizations. Donor document dependence is not fully reconstructed. The nonlinear ratio of averaged control energies is distinct from the mean of individual control ratios. This is a fresh-document confirmation of the frozen head6 statistic, not semantic-feature identification. Intervals are not simultaneous across metrics; the frozen decision requires both control families to pass each threshold.

Audit: Payload norm error 6.48e-07; exact coefficient-term replay error 6.93e-06; spectral/tangent response energy error 2.38e-14. Exact local-plus-weighting identity asserted in estimates and all bootstrap evaluations. Per-site sufficient statistics now saved.

Next: test whether the aligned low-gain geometry is specific to the Fisher output metric or also holds under Euclidean residual-output geometry. Then test finite behavior and cross-head generalization before interpreting local sensitivity directions as computation units. Broad research goal remains active.
