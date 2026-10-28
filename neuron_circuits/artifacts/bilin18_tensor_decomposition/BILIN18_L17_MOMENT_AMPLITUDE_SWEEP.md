

## H422 — Amplitude dependence of source moment equivalence

Reuse the H421 donor mappings and the exact Gaussian moment factors and random seeds, verified by file hashes. Same H420 recipients, disjoint H413 donors, head6 and H410 core64/added64/union128. Scale each source edit by 0.25, 1, 2 and 4. The direct residual path stays fixed; edits change the MLP path. No new documents, no ranking, no geometry matching. Recompute amplitude1 for an exact saved-row replay.

Freeze a joint family of twelve Gaussian-minus-raw contrasts: finite combination score S=R_union−R_core−R_added, union recovery R, and log(native effect KL ratio), at all four amplitudes. Equivalence tolerances are ±0.05, ±0.05 and ±0.10 respectively (the last is a multiplicative effect interval of exp(±0.10)). A standardized max-absolute joint95% band uses 4000 paired document/draw bootstrap samples, conditional on the donor calibration pool. Meaningful difference requires an interval entirely beyond its tolerance; failing equivalence alone does not establish a difference. Including the added effect-size statistic and all amplitudes makes this a different simultaneous family from H421; prior equivalence results are not retroactively replaced.

|Amplitude|Family|Union recovery|Combination score|Native effect KL|Effect / quadratic prediction|
|---:|---|---:|---:|---:|---:|
|0.25|Actual|0.812720|0.240653|0.00013946|0.983614|
|0.25|Raw donor|0.810232|0.247347|0.00016089|1.006505|
|0.25|Gaussian|0.809638|0.254473|0.00015800|1.021665|
|1|Actual|0.803000|0.278367|0.00209933|0.925407|
|1|Raw donor|0.814691|0.237521|0.00263482|1.030166|
|1|Gaussian|0.816614|0.229538|0.00267942|1.082856|
|2|Actual|0.786592|0.324035|0.00767951|0.846301|
|2|Raw donor|0.822365|0.214437|0.01088233|1.063695|
|2|Gaussian|0.827267|0.188736|0.01145874|1.157729|
|4|Actual|0.764023|0.340600|0.02761769|0.760884|
|4|Raw donor|0.840200|0.155717|0.04605372|1.125383|
|4|Gaussian|0.855890|0.097565|0.04893669|1.236075|

The last column is a descriptive check against 0.5 × amplitude² × H421 full native Fisher energy; it was derived from saved arrays after the sweep and has no additional confirmatory interval.

|Amplitude|Gaussian-minus-raw statistic|Estimate|Joint95% interval|Equivalent|Beyond tolerance|
|---:|---|---:|---|---|---|
|0.25|combination score difference|0.007127|[-0.065437, 0.079691]|False|False|
|0.25|union recovery difference|-0.000594|[-0.073158, 0.071970]|False|False|
|0.25|log effect ratio|-0.018150|[-0.163278, 0.126978]|False|False|
|1|combination score difference|-0.007983|[-0.080547, 0.064581]|False|False|
|1|union recovery difference|0.001923|[-0.070641, 0.074487]|False|False|
|1|log effect ratio|0.016783|[-0.128345, 0.161911]|False|False|
|2|combination score difference|-0.025702|[-0.098266, 0.046862]|False|False|
|2|union recovery difference|0.004902|[-0.067663, 0.077466]|False|False|
|2|log effect ratio|0.051613|[-0.093516, 0.196741]|False|False|
|4|combination score difference|-0.058151|[-0.130715, 0.014413]|False|False|
|4|union recovery difference|0.015691|[-0.056873, 0.088255]|False|False|
|4|log effect ratio|0.060719|[-0.084409, 0.205847]|False|False|

Outcome: all twelve equivalence gates fail, and all twelve beyond-tolerance difference gates also fail. The joint band is too wide to establish either claim. This is not evidence that every amplitude differs, and it does not negate H421's narrower frozen family. Unit prediction and effect arrays replay H421 exactly.

Point estimates suggest an amplitude-dependent departure in group relationships. Gaussian-minus-raw combination score changes from +0.007127 at quarter strength to −0.058151 at fourfold strength; union recovery differences remain smaller (−0.000594 to +0.015691). Native Gaussian/raw effect ratios range from about0.982 to1.063. Actual-source behavior differs descriptively from both shuffled families: actual union recovery falls0.813→0.764 while raw donor recovery rises0.810→0.840 and Gaussian recovery rises0.810→0.856. These latter contrasts were not frozen primary tests, and do not establish a semantic interpretation.

Post-result precision diagnosis (separate JSON, not a replacement confirmatory family): the log-effect contrasts determine the standardized bootstrap maximum in3508/4000 replicates (87.7%). This broad family therefore sacrifices precision on combination/recovery to cover absolute effect size. Maximum recipient-document shares of pooled effect are1.73–3.17% for raw donors and1.74–4.30% for Gaussian sources across amplitudes; no rows are excluded. A paired amplitude interaction, (Gaussian−raw combination score at4)−(Gaussian−raw at0.25), is−0.065278 with exploratory pointwise interval[−0.104149,−0.026407]. It supports selecting an amplitude interaction for a future fresh test, but this post-result interval is not one of the original twelve gates and does not make them pass.

Current interpretation: unit-amplitude aggregate moment equivalence remains supported within H421's scope. The present sweep suggests a possible nonlinear dependence on source distribution, but has not established an effect beyond a chosen practical tolerance under its frozen family. Even a confirmed difference would identify distributional dependence, not a semantic feature. Finite sample moment errors, Gaussian tail behavior, recipient-conditional geometry and higher-order natural source statistics remain competing explanations.

Next high-information test: fresh recipient documents, frozen groups and a predeclared paired amplitude interaction in the combination score. Use quarter and fourfold amplitudes with the same source sample across amplitudes. Preserve a separate donor calibration panel and quantify Monte Carlo uncertainty; the decisive primary contrast should target the amplitude interaction selected here. Keep absolute effect-size diagnostics separate from that narrow confirmatory family rather than retroactively removing them from H422. A subsequent conditional-moment or exact-moment control is needed to distinguish higher-order effects from small finite-sample first/second-moment differences. Avoid simply increasing a large omnibus family until a preferred gate passes.

Hypothesis priorities: #12 (operator/output alignment) and #20 (context-conditioned response operators) remain the leading descriptions. #17 (local explanations extending to finite edits) now has a possible amplitude boundary to test. #13 (specific source-recipient pairing) remains unnecessary for the unit raw-donor relationship but may affect amplitude dependence, as suggested by the distinct actual-source curves. #19 (semantic identities) remains unsupported. Broadly distributed residual factors and context-dependent compact subsets remain viable; this source-distribution experiment does not decide between them.


Audits: {'unit_prediction_replay_max_abs': 0.0, 'unit_effect_replay_max_abs': 0.0}. Independent saved-array checks reproduce all pooled scores and joint intervals. The broad goal remains open; unique completed documents remain 9408.
