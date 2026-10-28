

## H378 — Random-span recovery is strongly associated with prediction concentration

Compute intact capped-model entropy at all2048 H372 sites. Freeze bins [0,1),[1,3),[3,5),[5,infinity) nats before stratified outcomes. Reuse H376 probability-axis and H377 three-fixed-random-span guarded errors. Within each bin, compute pooled target-to-patch KL recovery using that bin’s baseline KL denominator. Random spans are averaged in error space before recovery. This is an association analysis on an already-used panel, not an entropy intervention or an independent confirmation.

Bin site counts are304,659,735,350, respectively; distinct recipient documents are238,402,422,263. A document can contribute sites to several bins. Both extreme bins exceed the frozen50-document support requirement.

|Head|Family|Random64 entropy<1|Random64 entropy≥5|Low-minus-high (joint95% interval)|
|---|---|---:|---:|---|
|0|actual|0.930|0.684|0.246 [0.200, 0.292]|
|0|moment|0.923|0.522|0.401 [0.355, 0.447]|
|0|donor|0.930|0.521|0.409 [0.363, 0.455]|
|1|actual|0.949|0.570|0.380 [0.334, 0.426]|
|1|moment|0.941|0.549|0.391 [0.345, 0.437]|
|1|donor|0.947|0.556|0.391 [0.345, 0.437]|
|6|actual|0.931|0.680|0.250 [0.204, 0.296]|
|6|moment|0.944|0.754|0.191 [0.145, 0.237]|
|6|donor|0.944|0.758|0.186 [0.140, 0.232]|

Frozen gates: {'adequate_extremes': True, 'random64_low_high': False, 'random64_monotone': True}. Random64 recovery is pointwise nonincreasing across all four entropy bins in every head/family group. Extreme-bin recovery ranges92.3–94.9% at entropy<1 and52.1–75.8% at entropy≥5. All9 simultaneous low-minus-high lower bounds are positive. The stronger frozen >15pp requirement fails for head6 moment/donor controls, whose lower bounds are14.46/14.00pp; do not replace that failed criterion with a weaker one. Support and pointwise monotonicity gates pass.

Probability-axis64 recovery also falls across bins, and its advantage is generally larger in the high-entropy bin. In low-entropy head6 controls, the mean random estimate is slightly higher than the probability-axis estimate. These are descriptive comparisons; no bin-specific basis-advantage gate was declared. All rank16/64 and both-basis estimates are saved.

Interpretation: arbitrary64-dimensional spans with response-dependent fitting recover most prediction changes when the intact distribution is concentrated. This supports concentration as a plausible reason why oracle recovery can look impressive without identifying special feature directions. However entropy covaries with token identity, context, response magnitude and geometry. The pooled statistic also weights sites by baseline KL. This analysis does not show that changing entropy alone causes the recovery difference.

Audit: {'site_partition_total': 2048, 'recombined_random_error_relative': 1.1463951381967674e-15, 'entropy_range': [0.0007375502429619184, 8.53199691592662]}. Entropy spans0.000738–8.532nats. The four bins partition all2048 sites, and recombining bin error sums replays the full random-error sums within1.2e-15 relative error. Input artifact hashes verified; saved intact entropy, inverse probability second moment, top probability and bin assignment. Native weights unchanged.

Uncertainty:4,000 document/control-draw bootstrap resamples preserve sites within each sampled document, condition on the fixed bin labels and three random orientations, and do not rebuild empirical donor controls. Separate joint36 random64 bin bands and joint9 extreme-bin contrast bands. Monotonicity gate uses point estimates, not uncertainty on every adjacent contrast. No response fitting or step selection was redone in this analysis.

Next discriminating intervention: vary softmax temperature within each fixed context while preserving token ordering, native activation-response vectors and random subspaces. Refit the same oracle under each modified probability metric and compare local/finite recovery. This isolates a controlled change in output-distribution concentration from between-context covariates, while explicitly treating temperature-modified outputs as diagnostic distributions rather than the unchanged native model. Broader goal remains active.
