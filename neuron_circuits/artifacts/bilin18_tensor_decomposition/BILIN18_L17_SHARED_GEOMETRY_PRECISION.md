

## H372 — Fixed 512-document confirmation supports shared geometry plus head-specific context enhancement

Prespecify512 entirely new documents, excluding4800 complete-document hashes including H371. Retain the H371 axes, discovery probability summaries, three selected heads, metric definitions and thresholds. Use new moment seeds and donor assignments. Analyze once after the full panel completes; no interim inferential looks or optional stopping. Each document supplies four positions, yielding2048 sites and13 response conditions per head. This is independent of listed analysis panels, not a model-training holdout.

As before, C/E is the signed top2/complement cross term divided by total response energy under an uncapped-derivative covariance metric. The own metric uses each new recipient’s intact capped probabilities; the frozen mixture and mean-within metrics use H370 discovery summaries unchanged. These are geometry diagnostics on native MLP-only responses, not an alternative MLP implementation.

|Head|Family|Own C/E|Frozen mixture C/E|Frozen within C/E|Mixture / own (joint95% interval)|
|---|---|---:|---:|---:|---|
|0|actual|-0.773|-0.771|-0.653|0.998 [0.825, 1.170]|
|0|moment|-0.601|-0.573|-0.420|0.954 [0.781, 1.126]|
|0|donor|-0.583|-0.556|-0.408|0.954 [0.781, 1.126]|
|1|actual|-1.032|-1.275|-0.986|1.235 [1.063, 1.408]|
|1|moment|-1.010|-1.175|-0.936|1.163 [0.991, 1.336]|
|1|donor|-1.027|-1.241|-0.982|1.209 [1.036, 1.381]|
|6|actual|-1.288|-0.785|-0.578|0.609 [0.437, 0.782]|
|6|moment|-0.982|-0.419|-0.383|0.427 [0.254, 0.599]|
|6|donor|-0.910|-0.377|-0.351|0.415 [0.242, 0.587]|

All three unchanged gates pass: {'fresh_within_cancellation': True, 'fresh_own_cancellation': True, 'head6_mixture_attenuation': True}. Own and frozen mean-within cancellation each have negative joint upper bounds in all9 head/family groups. Head6 moment/donor frozen-mixture ratios are0.427/0.415, with joint upper bounds0.599/0.587. Thus the head6 control attenuation is considerably better bounded than H371’s nearly-one upper bounds. The claim applies to the normalized C/E diagnostic and these three heads, not a fraction of total causal behavior or all model layers.

The frozen mixture also has negative upper bounds in all9 groups, descriptively. Head0’s mixture ratios are near1, while head1’s are above1; head1 actual/donor lower ratio bounds exceed1, a secondary observation rather than a newly declared primary gate. Shared geometry and context-dependent modulation coexist. Context-specificity should not be described as uniform across heads.

Prespecified influence checks, across the three primary metrics: the largest single-document total-energy share is2.53%, and the largest top-five share is11.51%. The maximum leave-one-document-out change in C/E is0.0391. No document was excluded from any primary estimate. These checks argue against domination by one document, while not ruling out heavier tails or collective subgroup effects. Exact influential indices and all leave-one-out values are saved. They do not retroactively explain the prior panel’s uncertainty.

Audit: {'payload_geometry_max_relative': 4.77240632790199e-07, 'energy_identity_max_relative': 6.603744206529584e-13, 'minimum_self_energy': 6.34707828933063e-14}. Full-document hash and exact-window reconstruction checks pass for all512 rows; frozen metric hashes verified; source-control norm/context-dot geometry preserved within4.8e-7; covariance identities and nonnegative self-energy checks pass. New native response arrays, per-site energies, influence arrays and4,000 bootstrap samples are saved. Native weights unchanged.

Uncertainty: joint27 cross-statistic bands and separate joint9 ratio bands resample recipient documents and control realizations. Discovery probability summaries remain fixed; empirical donor dependencies are not rebuilt. Compared with H371, the cross-band half-width decreases from approximately0.523 to0.204 on a different independent panel. This comparison is descriptive rather than an isolated causal effect of sample size. The successful test does not erase the earlier failed universal gate.

Research update: weight-orthogonal components interfere under both transferable probability geometry and context-dependent geometry. A unit defined only by Euclidean orthogonality is therefore not automatically behaviorally independent. We have not established semantic features or compressed the native MLP.

Next highest-information test: construct output axes from the frozen mean-within probability metric, then compare a prespecified rank ladder against weight-only axes using finite distribution-recovery interventions. That tests whether choosing a behavior-relevant shared geometry yields genuinely more self-contained output components, rather than only better explanatory statistics. Keep basis construction independent of target responses and retain finite and local checks. Broader goal remains active.
