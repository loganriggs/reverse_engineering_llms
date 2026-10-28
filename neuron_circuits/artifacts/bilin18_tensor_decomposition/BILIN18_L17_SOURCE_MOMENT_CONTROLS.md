

## H333 — Source mean and covariance controls

Estimate means and centered covariance square roots separately for9 heads×4 positions using768 discovery documents. On reused H327128 recipients, construct three source directions: mean only, centered covariance-Gaussian, and mean plus the same Gaussian draw. Match each recipient edit norm and context dot via the same source-span construction. Compare actual, isotropic-angle and donor1 controls. No response fitting; moment training documents are disjoint from recipients. Recipient-specific norm/dot remain known, so these are moment-based directional controls rather than fully context-free edits.

Measure all seven finite removal combinations of direct residual/compiler/remaining MLP paths, and the Fisher Gram at each edited native state as in H332.

|Condition|Direct–MLP cosine [95%]|Direct–compiled|Direct–remaining|Finite joint-vs-sum change|
|---|---:|---:|---:|---:|
|actual|-0.494 [-0.528,-0.457]|-0.364|-0.335|-42.68%|
|angle|-0.295 [-0.328,-0.264]|-0.192|-0.184|-26.89%|
|donor1|-0.518 [-0.561,-0.471]|-0.443|-0.144|-49.11%|
|mean|-0.582 [-0.618,-0.545]|-0.509|-0.181|-55.82%|
|centered|-0.454 [-0.492,-0.417]|-0.373|-0.142|-43.29%|
|mean_cov|-0.498 [-0.538,-0.451]|-0.416|-0.189|-47.16%|

|Condition|Comparison|Cosine gap [paired95%]|
|---|---|---:|
|donor1|vs actual|-0.024 [-0.063,0.017]|
|donor1|vs angle|-0.223 [-0.273,-0.170]|
|mean|vs actual|-0.089 [-0.123,-0.053]|
|mean|vs angle|-0.288 [-0.326,-0.249]|
|centered|vs actual|0.040 [0.001,0.079]|
|centered|vs angle|-0.159 [-0.193,-0.125]|
|mean_cov|vs actual|-0.004 [-0.042,0.034]|
|mean_cov|vs angle|-0.203 [-0.242,-0.162]|

Locked gates: mean_matches_actual5: FAIL, centered_beats_angle10: PASS, mean_cov_matches_donor5: PASS. mean_matches_actual5 and mean_cov_matches_donor5 require absolute cosine gaps<=0.05; these point-estimate closeness gates do not establish statistical equivalence. centered_beats_angle10 requires centered−angle<=−0.1 with paired95 upper<0.

**Interpretation limits.** Matching norm and angle does not match every property of naturally occurring source vectors. Moment-generated perpendicular directions preserve selected distributional structure; these are synthetic interventions, not natural context swaps. One Gaussian draw is shared between centered and mean-plus-covariance conditions. Each condition has its own edited-state output metric, so the comparison includes local metric changes as well as response changes. Cosine opposition does not establish semantic repair. Permutations reuse the same documents; bootstrap treats recipient documents as units conditional on donor construction and does not capture general donor-population uncertainty. No fresh confirmation.

**Audit.** Norm/dot matching: {'max_norm_relative': 6.146427722342196e-07, 'max_dot_scaled': 4.458744626845146e-07}. H330 endpoints for actual/random-angle replay within2.44e-07 relative. Per-document Grams and finite KLs saved; native weights unchanged.

**Next discriminator.** Test whether the mean and covariance effects survive a fixed output metric, independent draws and fresh documents. Avoid inferring semantic repair from cancellation explained by low-order source statistics. Broad goal active.
