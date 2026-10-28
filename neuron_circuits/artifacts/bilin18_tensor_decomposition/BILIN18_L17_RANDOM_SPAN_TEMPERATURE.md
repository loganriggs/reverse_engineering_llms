

## H379 — Controlled output temperature strongly changes arbitrary-span oracle recovery

Use H377’s first fixed random basis (seed20270602), the same512 documents, native activation-response vectors and rank16/64 spans. Divide capped logits by temperatureT in {0.5,1,2} for baseline, target and patched distributions. Refit the local Fisher oracle for each temperature, including the cap derivative and1/T scaling, then apply the unchanged finite-KL step grid. Token ordering and activation responses remain fixed. This intervention changes diagnostic output distributions and fitting/scoring geometry, not the native model’s hidden computation.

|Head|Family|Random64 T=0.5|Random64 T=1|Random64 T=2|Cold-minus-hot (joint95% interval)|
|---|---|---:|---:|---:|---|
|0|actual|0.987|0.810|0.499|0.488 [0.450, 0.526]|
|0|moment|0.984|0.725|0.278|0.706 [0.668, 0.744]|
|0|donor|0.984|0.730|0.294|0.690 [0.651, 0.728]|
|1|actual|0.977|0.768|0.334|0.644 [0.606, 0.682]|
|1|moment|0.977|0.750|0.314|0.663 [0.625, 0.701]|
|1|donor|0.977|0.753|0.312|0.666 [0.627, 0.704]|
|6|actual|0.978|0.778|0.491|0.487 [0.449, 0.526]|
|6|moment|0.985|0.822|0.677|0.308 [0.270, 0.347]|
|6|donor|0.984|0.824|0.688|0.296 [0.258, 0.334]|

All frozen gates pass: {'cold_hot_advantage': True, 'temperature_monotone': True, 'entropy_order': True}. Recovery falls monotonically across the three temperatures in every head/family/rank point estimate. Every cold-minus-hot joint lower bound exceeds15pp, with the smallest lower bound25.8pp. Entropy rises at every site; mean entropy is1.012,3.148,8.607nats at T=0.5,1,2. Random64 recovery is97.7–98.7% atT=0.5 versus27.8–68.8% atT=2. Random16 similarly changes from88.5–92.6% to13.4–41.2%.

Interpretation: a fixed arbitrary span can appear nearly sufficient when evaluation emphasizes a concentrated set of token outcomes. This demonstrates a controlled temperature effect on the diagnostic pipeline, beyond the between-context entropy association in H378. It does not show that the model has changed its internal representational dimension, or that98% of native-temperature behavior is captured. It also does not make entropy a uniquely identified causal mediator: temperature changes the entire probability weighting, fitting objective, finite safeguard choice and KL denominator together.

The result cautions against using high oracle KL recovery alone as evidence for fundamental feature directions. Both full native response access and the output metric can make arbitrary directions effective for the scored task. Native-temperature results remain those atT=1. AtT=2 substantial head-dependent differences persist, so concentration is not the only determinant of recovery.

Audits: temperature1 exactly replays the H377 first-seed guarded finite and local errors. All16/64 Gram directions are retained at every temperature; local oracle errors never exceed Euclidean errors; normal-equation residuals are below7.7e-12. All finite values and safeguard inequalities pass. Entropy ordering verified per site. Save temperature-specific bases, coefficients, local/finite/grid errors, entropy, choices and bootstrap arrays. Native weights and activation responses unchanged.

Uncertainty:4,000 paired document/control-draw resamples give joint54 recovery bands and separate joint18 cold-minus-hot bands. One frozen random orientation only, reused panel, fixed discovery data and empirical donor assignments; not a random-orientation population estimate or new-document replication. Monotonicity gate is pointwise over group means, not a claim that every individual response is monotonic.

Next discriminating factorial: cross the temperature used to fit coefficients with the temperature used to score them, initially at0.5 and2 with the native-temperature reference. Keep coefficient vectors and selected step fixed when changing only the score. This separates changing which errors count from changing the fitted response, and tests whether high cold-temperature recovery requires context-specific adaptation or mostly a forgiving output metric. Broader goal remains active.
