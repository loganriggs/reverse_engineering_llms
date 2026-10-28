

## H334 — Common intact-state output metric

Freeze H333 source moments, recipient documents and all actual, isotropic, donor, mean-only, centered-covariance and mean-plus-covariance payloads. Change only the Fisher metric anchor: every condition uses the same intact native final state for a given recipient position. Final RMS derivative, unembedding softcap derivative and probability weighting all use that intact anchor. Finite removal KLs retain their original edited-state definition and replay H333 unchanged. No fitting or tuning.

Measure all seven finite removal combinations of direct residual/compiler/remaining MLP paths, and the Fisher Gram at the shared intact native state.

|Condition|Direct–MLP cosine [95%]|Direct–compiled|Direct–remaining|Finite joint-vs-sum change|
|---|---:|---:|---:|---:|
|actual|-0.460 [-0.499,-0.419]|-0.312|-0.340|-42.68%|
|angle|-0.263 [-0.301,-0.227]|-0.154|-0.201|-26.89%|
|donor1|-0.470 [-0.516,-0.421]|-0.386|-0.167|-49.11%|
|mean|-0.536 [-0.575,-0.495]|-0.454|-0.209|-55.82%|
|centered|-0.426 [-0.468,-0.384]|-0.336|-0.165|-43.29%|
|mean_cov|-0.461 [-0.507,-0.410]|-0.372|-0.203|-47.16%|

|Condition|Comparison|Cosine gap [paired95%]|
|---|---|---:|
|donor1|vs actual|-0.010 [-0.050,0.027]|
|donor1|vs angle|-0.206 [-0.263,-0.150]|
|mean|vs actual|-0.076 [-0.111,-0.044]|
|mean|vs angle|-0.272 [-0.317,-0.228]|
|centered|vs actual|0.033 [-0.010,0.079]|
|centered|vs angle|-0.163 [-0.204,-0.119]|
|mean_cov|vs actual|-0.001 [-0.041,0.040]|
|mean_cov|vs angle|-0.197 [-0.242,-0.149]|

Locked gates: mean_matches_actual5: FAIL, centered_beats_angle10: PASS, mean_cov_matches_donor5: PASS. mean_matches_actual5 and mean_cov_matches_donor5 require absolute cosine gaps<=0.05; these point-estimate closeness gates do not establish statistical equivalence. centered_beats_angle10 requires centered−angle<=−0.1 with paired95 upper<0.

**Metric comparison.**

|Condition|Edited-anchor cosine (H333)|Intact-anchor cosine (H334)|
|---|---:|---:|
|actual|-0.494|-0.460|
|angle|-0.295|-0.263|
|donor1|-0.518|-0.470|
|mean|-0.582|-0.536|
|centered|-0.454|-0.426|
|mean_cov|-0.498|-0.461|

The intact-anchor quadratic values are diagnostics at a different state, not local expansions at each edited endpoint.

**Interpretation limits.** Matching norm and angle does not match every property of naturally occurring source vectors. Moment-generated perpendicular directions preserve selected distributional structure; these are synthetic interventions, not natural context swaps. One Gaussian draw is shared between centered and mean-plus-covariance conditions. All conditions now share an intact-state metric. Their MLP responses still depend on the edited input; this control removes metric-anchor variation but does not make the nonlinear responses identical. Cosine opposition does not establish semantic repair. Permutations reuse the same documents; bootstrap treats recipient documents as units conditional on donor construction and does not capture general donor-population uncertainty. No fresh confirmation.

**Audit.** Norm/dot matching: {'max_norm_relative': 6.146427722342196e-07, 'max_dot_scaled': 4.458744626845146e-07}. All54 groups ×7 finite KL endpoints replay H333 within0 relative. Per-document Grams and finite KLs saved; native weights unchanged.

**Next discriminator.** Test whether the source-moment result survives independent Gaussian draws and fresh documents. Avoid inferring semantic repair from cancellation explained by low-order source statistics. Broad goal active.
