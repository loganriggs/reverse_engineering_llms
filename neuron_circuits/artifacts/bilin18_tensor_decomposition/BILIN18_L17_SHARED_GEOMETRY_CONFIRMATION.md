

## H371 — Fresh documents confirm own-context cancellation; shared geometry transfers incompletely under frozen gates

Sample128 new documents excluding4672 complete-document hashes from listed previous panels. Independently audit full-document hashes and every sampled token window. Compute new native MLP-only response vectors for heads0/1/6, with six new moment draws and six new empirical donor assignments. Freeze H364’s top2 unembedding eigenvectors and H370’s probability summaries from the earlier128 documents before outcomes. Fresh relative to listed experiments, not model-training holdout.

Split each new finite post-RMS response into the frozen top2 component and its complement. Evaluate uncapped-derivative covariance metrics using (1) the new recipient’s own intact capped-model probabilities, (2) the frozen position-mean probability distribution, and (3) frozen mean-within covariance, obtained by subtracting the frozen between-document expected-unembedding covariance from the mixture metric. Unlike H370 leave-one-out sets, the frozen metrics use all128 discovery documents, none of which overlaps the new panel. This is a downstream geometry diagnostic, not a replacement native computation or an uncapped model.

|Head|Family|Own C/E|Frozen mixture C/E|Frozen within C/E|Mixture / own (joint95% interval)|
|---|---|---:|---:|---:|---|
|0|actual|-0.742|-0.761|-0.642|1.026 [0.616, 1.436]|
|0|moment|-0.696|-0.582|-0.423|0.837 [0.427, 1.247]|
|0|donor|-0.651|-0.532|-0.393|0.817 [0.407, 1.226]|
|1|actual|-0.933|-1.185|-0.920|1.270 [0.860, 1.680]|
|1|moment|-0.874|-1.192|-0.938|1.364 [0.954, 1.774]|
|1|donor|-0.869|-1.243|-0.986|1.432 [1.022, 1.841]|
|6|actual|-1.072|-0.701|-0.522|0.654 [0.244, 1.063]|
|6|moment|-0.787|-0.461|-0.412|0.586 [0.176, 0.996]|
|6|donor|-0.768|-0.452|-0.398|0.589 [0.179, 0.999]|

Frozen gates: {'fresh_within_cancellation': False, 'fresh_own_cancellation': True, 'head6_mixture_attenuation': True}. Own-context cancellation passes all9 simultaneous upper-bound tests. Frozen within covariance is negative in all9 point estimates but only4/9 joint upper bounds; the universal transfer gate fails. Head6 control mixture/own ratios are0.586 and0.589, with joint upper bounds0.9956 and0.9986: the frozen attenuation gate passes narrowly. Do not describe this as a robustly bounded 41% reduction; 41% is the point estimate and the interval admits almost no reduction. The all-group bands are wide.

The broad interpretation remains conditional: interference under actual probabilities repeats independently, and the head6 attenuation pattern passes its predeclared test. A universal shared within-context explanation is not confirmed at this precision. Negative point estimates alone do not establish the failed universal claim, while intervals including zero do not establish absence. H370’s stronger fixed-donor in-panel statement does not override this new result.

Audit: {'payload_geometry_max_relative': 6.949048838578165e-07, 'energy_identity_max_relative': 7.961454223378946e-14, 'minimum_self_energy': 4.1034428644386115e-13}. Full-document/window/hash checks pass; frozen metric files verified by SHA256; source-control norms and context dot products preserved within7e-7 relative error; covariance energy identity passes; self-energies nonnegative. Save each new head response array, site-level energies and bootstrap results. No old recipient response cache reused and no native weights changed.

Bootstrap4,000 fresh recipient-document and control-realization resamples. Cross-statistic bands are simultaneous over27 head/family/metric estimates, ratio bands separately over9. Frozen discovery metrics are conditioned on rather than resampled; empirical control donors remain dependent and are not rebuilt. Head choices and hypotheses came from the older panel. No threshold changed after outcomes.

Next: increase precision with a prespecified fixed-size independent panel before making a universal or sharply head-specific claim. Retain the same axes, frozen discovery distributions, response definitions and thresholds; avoid repeated optional stopping at a passing interval. Inspect document-level influence as a diagnostic, without excluding difficult examples from the primary statistic. Broader research goal remains active.
