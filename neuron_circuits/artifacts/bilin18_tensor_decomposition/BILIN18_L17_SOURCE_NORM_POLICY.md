## H464 — Source normalization changes apparent covariance sufficiency

Reuse H463 head6 basis,64 Gaussian directions and natural signals,128 receiving contexts. Compare natural, independent paired norm, sorted norm and global RMS scale. Exact retained128 response plus native omitted response to rank16 reconstruction. Eight primary contrasts: sorted/global minus natural equivalence within2pp each sign; sorted/global minus paired improvement above2pp each sign. Joint8 fixed-policy paired source/context bootstrap; no fresh corpus or intervention contexts. Policies fixed before behavioral outcomes; conditional on realized sorted assignment/global scale.

This is a mechanistic comparison on the H463 panel, not fresh confirmation. Natural sources and paired controls replay exactly. Sorted matching assigns the observed natural norms to Gaussian directions ordered by their original norms; global scaling multiplies every Gaussian signal by one constant to match total natural source energy. All three Gaussian policies use the same directions. Sorted matching preserves the empirical natural norm distribution; global scaling preserves the Gaussian sample covariance shape up to scale. Neither preserves both constraints.

| Family | Negative data16 / retained recovery | Positive data16 / retained recovery | Rank16 source energy loss |
|---|---|---|---:|
|natural|99.386% / 85.980%|99.327% / 84.570%|1.561%|
|paired_norm|94.651% / 79.024%|95.335% / 80.914%|12.454%|
|sorted_norm|99.362% / 84.809%|99.363% / 84.904%|1.443%|
|global_scale|99.434% / 84.930%|99.438% / 85.087%|1.297%|

| Frozen contrast | Estimate | Joint95% interval | Pass |
|---|---:|---|---|
|sorted_norm_minus_natural_negative|-0.000247|[-0.033423, +0.032929]|False|
|sorted_norm_minus_natural_positive|+0.000363|[-0.032813, +0.033539]|False|
|global_scale_minus_natural_negative|+0.000480|[-0.032696, +0.033657]|False|
|global_scale_minus_natural_positive|+0.001110|[-0.032066, +0.034286]|False|
|sorted_norm_minus_paired_norm_negative|+0.047111|[+0.013935, +0.080288]|False|
|sorted_norm_minus_paired_norm_positive|+0.040286|[+0.007109, +0.073462]|False|
|global_scale_minus_paired_norm_negative|+0.047838|[+0.014662, +0.081015]|False|
|global_scale_minus_paired_norm_positive|+0.041033|[+0.007856, +0.074209]|False|

The first four gates require the full interval inside[-.02,+.02]. The last four require lower>.02.4000 paired source-index/context bootstrap replicates use one unstudentized max-absolute-deviation radius across all eight contrasts. Inference conditions on the realized source draws, calibration, sorted norm assignment and global scale; normalization is not recomputed inside resampling. No gate or interval method was changed after outcomes.

Sorted/global scaling restores point recovery to the natural-source level: negative/positive sorted99.362%/99.363%, global99.434%/99.438%, natural99.386%/99.327%. Source rank16 energy loss is1.443% sorted and1.297% global versus12.454% under independent pairing. The magnitude policy alone changes the observed difficulty of the same Gaussian directions. This is evidence for a direction/magnitude coupling confound, not evidence that semantic content is required for the original high recovery.

All eight frozen gates nevertheless fail. Sorted-minus-natural differences−0.0247/+0.0363pp and global-minus-natural+0.0480/+0.1110pp have joint intervals extending roughly3.3pp in each direction, failing±2pp equivalence. Sorted-minus-paired gains4.711/4.029pp and global-minus-paired4.784/4.103pp have intervals[1.393,8.029]/[0.711,7.346]/[1.466,8.101]/[0.786,7.421]pp. All four exclude zero under the frozen joint interval, supporting an improvement, but none has its lower bound above the prespecified2pp minimum. Failure to establish that minimum does not mean no improvement. No gates are relabeled successful.

The common absolute-radius simultaneous band is dominated by paired-control comparisons: bootstrap standard deviations are about1.52–1.71pp for these gains, versus0.14–0.17pp for the natural-equivalence contrasts. This is a descriptive precision diagnosis, not permission to substitute post-hoc intervals or discard tests. The fixed normalization policy is conditioned on in this bootstrap; uncertainty from choosing and estimating the policy is not covered. Total finite KL effects remain comparable: negative natural/sorted/global24.999/23.687/23.619; positive22.277/23.785/23.837. Equal source energy does not guarantee equal behavioral effects, and these sums are not semantic attribution.

Update to the broader explanation: covariance-shaped random sources, when their magnitudes are handled without independently amplifying weak directions, can reproduce the high compact-response point scores. Thus the success of this rank16 predictor does not require natural donor content in this panel. It remains possible that the naturally occupied directions carry meaningful variables; the experiment concerns sufficiency for a prediction metric, not the absence of semantics. Head6 remains a selected, unusually compact source, and no claim of whole-model universality follows.

Next run an independent confirmation with new Gaussian draws and the remaining128 intervention-unused receiving contexts. Freeze the H463 basis/calibration and natural evaluation donors; compare natural signals with unnormalized covariance-Gaussian signals (their expected covariance already matches calibration) and a prespecified single global scale, keeping magnitude/effect audits. Use many independent Gaussian draws. Prespecify contrast-scaled simultaneous inference, or separate explicitly budgeted families, before outcomes so noisy auxiliary comparisons do not impose their absolute uncertainty on equivalence tests. Calibration uncertainty remains a separate limitation. Only after confirming this distributional account should we spend further experiments assigning semantic names or exporting the operation to other heads/distributions. Preserve H464's failed gates and distinguish fresh receiving sites/new random draws from reused natural donor pool and calibration.

Audit: {"energy_relative_error": 2.220446049250313e-16, "sorted_norm_max_abs": 1.8189894035458565e-12, "direction_max_abs": 2.220446049250313e-16, "replay_max_abs": 0.0, "effect_replay_max_abs": 0.0, "numerical_gates": {"energy": true, "sorted_norm": true, "direction": true, "replay": true, "effect_replay": true}}. Recovery cells are independently checked by scalar sums in the analyzer; the table has16 signed/family/method cells. No source or context exclusions.

Scope: baseline states and exact native retained128-factor responses remain in the predictor. No semantic coordinate naming, whole-model replacement, multi-head transfer, or runtime compression is established. Total corpus documents remains12992;128 H454 receiving contexts remain intervention-unused after H463.
