

## H423 — Fresh confirmation of the amplitude interaction

Fresh512 recipient documents independently exclude all9408 completed documents, bringing the total to9920. The donor calibration pool remains H413; its H421 mean/covariance factors are hash-frozen, as are H410 core64/added64/union128. Use six new donor permutations and six new unmatched Gaussian draws, preserving each source sample across amplitudes0.25 and4. Head6, four positions, MLP-path intervention with direct residual held fixed. No group reselection or geometry matching.

The single frozen primary is I=[S_Gaussian(4)−S_raw(4)]−[S_Gaussian(0.25)−S_raw(0.25)], where S=R_union−R_core−R_added and R=1−pooled KL(native edited||group prediction)/pooled KL(native edited||intact). A symmetric95% interval uses4000 paired document/draw bootstrap samples. Negative interaction requires upper<0; a meaningful negative interaction requires upper<−0.02. Equivalence requires the whole interval within±0.02. These are alternative conclusions from the same scalar interval. The fresh test follows the contrast selected exploratorily in H422; it does not change H422’s failed joint12 gates. Uncertainty conditions on the fixed donor calibration pool.

|Amplitude|Family|Core recovery|Added recovery|Union recovery|Combination score|Native effect KL|
|---:|---|---:|---:|---:|---:|---:|
|0.25|Actual|0.660522|-0.101034|0.814808|0.255319|0.00013097|
|0.25|Raw donor|0.686218|-0.151217|0.820315|0.285315|0.00015003|
|0.25|Gaussian|0.682439|-0.145887|0.820617|0.284065|0.00014310|
|4|Actual|0.584977|-0.157633|0.773533|0.346190|0.02608352|
|4|Raw donor|0.724660|-0.019694|0.853500|0.148533|0.04629220|
|4|Gaussian|0.755149|0.016735|0.868903|0.097020|0.04497453|

Primary interaction: **-0.050264**,95% interval **[-0.097458, -0.003071]**. Gates: {'disjointness': True, 'group_partition': True, 'negative_interaction': True, 'meaningful_negative_interaction': False, 'equivalence02': False}.

Outcome: the fresh primary is -0.050264, 95% interval [-0.097458, -0.003071]. The negative-interaction gate passes. The meaningful-negative threshold (upper < -0.02) fails, and equivalence within +/-0.02 also fails. Document disjointness and group partition pass. Thus the direction replicates, while the size is not resolved tightly enough to establish a change exceeding 0.02. H422's original failed gates remain failed.

Gaussian-minus-raw combination scores are -0.001249 at quarter strength and -0.051514 at fourfold strength. Raw donor combination score falls 0.285315 to 0.148533; Gaussian falls 0.284065 to 0.097020. Union recovery still rises in both donor families: raw 0.820315 to 0.853500, Gaussian 0.820617 to 0.868903. Actual-source combination score rises 0.255319 to 0.346190 while its union recovery falls 0.814808 to 0.773533; those actual-versus-donor comparisons are descriptive, not the fresh primary. Group-response partition error is 8.88e-7 relative to union output norm.

Interpretation: the selected group-score relationship shows a replicating amplitude dependence on source distribution. This sets a boundary on treating H421's unit-amplitude aggregate moment equivalence as universal. It does not show semantic features, unique causal units, or a higher-order mechanism by itself. The empirical Gaussian samples match moments in expectation, not exactly; the donor pool is fixed; nonlinear input RMS, bilinear responses and output scoring can all contribute. Statistical sign confirmation is weaker than the preregistered practical-size claim, which remains unconfirmed.

Next high-information experiment: separate nonlinear response generation from nonlinear output scoring using the SAME frozen source samples and recipients. Cross two response choices (native finite MLP response versus amplitude times its native local derivative) with two metrics (exact native output KL versus fixed baseline output Fisher quadratic energy). Compute the same combination-score amplitude interaction in all four conditions. The fully linear-response/quadratic-metric condition must be amplitude invariant, providing an architectural audit. The native-response/quadratic-metric condition tests whether response generation alone produces the trend; the linear-response/exact-KL condition tests whether output nonlinearity alone can do so. Their factorial interaction is diagnostic accounting, not a unique mediation percentage. This split can distinguish concrete mechanisms more directly than another omnibus source-distribution comparison.

Separately, exact empirical mean/covariance matching of Gaussian-derived batches can test finite-sample moment mismatch. Whitening/recoloring batches changes their joint sampling distribution, so describe these as Gaussian-derived controls rather than independent Gaussian draws. Recipient-dependent operators still require paired assignment controls; global batch moment equality does not force exact equality at every recipient.

Hypothesis update: strengthen #17's claim that finite behavior has an amplitude-dependent boundary beyond local moment accounting, but only for the sign of this selected contrast. #12 (operator/output alignment) remains leading; the next factorial test distinguishes response generation and scoring contributions. #5 (compensating combinations) remains a score-level relationship, not an identified semantic pair. #19 (semantic units) remains unsupported. #20 (context-conditioned operators) remains a descriptive candidate requiring predictive transfer and compression tests. No universal feature ontology has been established.


Audit: {'group_partition_relative_error': 8.879593451638357e-07}. Independent array checks reproduce all pooled scores and the primary interval; all donor maps are bijections. No completed-process restart was needed. The broad research goal remains open; next fresh panels must exclude9920 documents.
