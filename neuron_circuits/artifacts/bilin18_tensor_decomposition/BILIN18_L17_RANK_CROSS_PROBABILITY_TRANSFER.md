

## H369 — Shared probability geometry preserves substantial cancellation, with uncertain head-specific attenuation

Hold the H367 finite post-RMS response split fixed on the same 128-document panel, heads0/1/6 and 13 actual/control conditions. Omit the cap derivative to isolate probability weighting, while using probabilities from the intact capped model. Compare own probabilities, six H365 confidence-matched other documents at the same token position, and a leave-recipient-document-out average of the other127 probability distributions at that position. Native weights and response vectors are unchanged. These are metric-transfer diagnostics, not edited model states.

For confidence matching, average six covariance energy estimates before taking C/E. For the shared condition, average probability distributions before computing covariance. These differ: covariance under a mixture includes a between-distribution mean term that averaging separate covariance metrics omits. Confidence matching also preserves probability concentration only approximately; the shared distribution generally has different concentration. Therefore this comparison cannot isolate token identity from entropy or mixture effects.

|Head|Family|Own C/E|Matched C/E|Shared C/E|Shared / own (joint95% interval)|
|---|---|---:|---:|---:|---|
|0|actual|-0.719|-0.584|-0.733|1.020 [0.595, 1.445]|
|0|moment|-0.616|-0.472|-0.611|0.991 [0.566, 1.416]|
|0|donor|-0.540|-0.396|-0.524|0.971 [0.546, 1.396]|
|1|actual|-0.979|-1.030|-1.199|1.224 [0.799, 1.650]|
|1|moment|-0.856|-0.900|-1.059|1.238 [0.812, 1.663]|
|1|donor|-0.894|-0.994|-1.136|1.271 [0.846, 1.696]|
|6|actual|-1.096|-0.872|-0.918|0.838 [0.413, 1.263]|
|6|moment|-0.728|-0.370|-0.406|0.558 [0.133, 0.984]|
|6|donor|-0.696|-0.363|-0.386|0.554 [0.129, 0.979]|

All three frozen gates fail: {'shared_cancellation': False, 'matched_attenuation': False, 'shared_majority': False}. Shared cancellation has a negative point estimate in all9 groups and a negative joint upper bound in8/9; the remaining head6 donor upper bound is +0.00315, so the all9 criterion fails. This is uncertainty around the strict criterion, not evidence that shared cancellation is absent. Shared/own ratios range0.554–1.271, but the all9 >0.5 lower-bound requirement fails. Confidence-matched minus own contrasts do not meet the all9 >0.1 requirement; every joint contrast interval includes zero. No post-hoc gate relaxation.

Exploratory pattern: shared distributions retain approximately97–102% of own C/E magnitude for head0,122–127% for head1, and55–84% for head6. Head6’s moment/donor shared/own joint intervals have upper bounds0.984/0.979, suggesting a head-specific reduction; these were not the frozen primary gates and need fresh confirmation. This pattern argues against treating the cancellation as uniformly dependent on exact context pairing.

Audit: {'h368_replay_relative_to_global_max': 0.0, 'energy_identity_max_relative': 1.641232759527752e-13, 'shared_probability_sum_error': 2.220446049250313e-16}. Own-probability energies replay H368 exactly. Donor indices verified to exclude the recipient document, preserve token position, and match across heads. Shared distributions sum to1 and are nonnegative. Energy identity and finite-value checks pass. Save every site’s energy under all8 metrics and bootstrap arrays.

Uncertainty: 4,000 bootstrap replicates resample recipient documents, control realizations and matched donor ranks. Neighbors and shared distributions are held fixed, so the intervals are conditional and do not include their re-estimation uncertainty or all donor reuse dependence. Joint27 cross-statistic bands and separate joint9 contrast/ratio bands are reported. This remains an exploratory sequence on a reused panel.

Hypothesis update: a shared anisotropic token metric remains plausible; universal dependence on each context’s exact probabilities is weakened by the point estimates but not decisively rejected. Head6-specific context enhancement remains plausible. A covariance-of-mixture versus mean-covariance distinction is a live alternative explanation that should be isolated algebraically.

Next: separate those two averaging operations using the law of total covariance on the same donor distributions, then freeze a shared distribution from this panel and test on fresh documents. This distinguishes generic token weighting, between-context means and recipient-specific pairing without fitting response outcomes. Broader goal remains active.
