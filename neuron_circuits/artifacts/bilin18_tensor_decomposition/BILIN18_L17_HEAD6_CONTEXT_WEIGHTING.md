

## H345 — Within-context sensitivity versus context weighting

Recover per-site H344 statistics on the same H339 panel and exactly replay saved document aggregates. Average the six control realizations at each site before forming gains. Let P be active projection energy, E tangent response energy, g=E/P, and w=P/ΣP. The pooled gain is Σwg. Split its control-minus-actual change exactly as Σ[(w_c+w_a)/2](g_c−g_a) + Σ(w_c−w_a)[(g_c+g_a)/2]. These are local-gain and weighting contributions, an algebraic attribution rather than causal interventions.

Also compare equal-site averages of g/λ_max, where the largest sensitivity eigenvalue is common to actual/control at each intact context. This removes both active-energy weighting and overall spectral scale.

|Statistic|Moment estimate [95% interval]|Donor estimate [95% interval]|
|---|---:|---:|
|Geometric mean site gain ratio|3.19 [2.53, 3.35]|3.03 [2.35, 3.17]|
|Sites with control gain > actual|89.26% [78.71, 90.43]|85.35% [74.02, 86.13]|
|Equal-site spectrum-normalized gain ratio|3.02 [2.74, 3.51]|2.91 [2.61, 3.40]|
|Local-gain share of pooled gain excess|101.51% [99.15, 107.59]|102.61% [99.44, 108.04]|
|Weighting share of pooled gain excess|-1.51% [-7.59, 0.85]|-2.61% [-8.04, 0.56]|

All three prespecified gates pass: local-gain share lower bound>50%, equal-site normalized gain ratio lower bound>2, and majority of sites show larger control gain, for both control families. Values above100% arise because weighting slightly offsets the local-gain increase. Within-context gain differences account for essentially all pooled gain excess under this decomposition.

Uncertainty: 4000 joint document resamples, independently resampling six global realizations per control family. Pointwise percentile intervals condition on frozen model/compiler, fixed donor assignments, this panel and six sampled realizations. Donor document dependence is not fully reconstructed. The nonlinear ratio of averaged control energies is distinct from the mean of individual control ratios. This remains exploratory, not a fresh replication or semantic-feature identification.

Audit: H344 document aggregate replay relative error 0; spectral/tangent response energy error 1.89e-14. Exact local-plus-weighting identity asserted in estimates and all bootstrap evaluations. Per-site sufficient statistics now saved.

Next: freeze the equal-site spectrum-normalized gain statistic and replicate head6 on fresh documents with new moment draws and donor assignments. This is needed before expanding a local alignment result into a stable computation-unit claim. Broad research goal remains active.
