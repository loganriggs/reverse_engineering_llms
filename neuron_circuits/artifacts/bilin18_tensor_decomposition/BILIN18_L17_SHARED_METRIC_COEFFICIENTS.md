

## H381 — Recipient-specific probability fitting adds substantial recovery at fixed native scoring

Keep native-temperature finite target/scoring distributions fixed. In both H373 probability-defined and H377 first-random spans, compare rank16/64 coefficients fitted using (1) frozen global discovery mean-within covariance K without the cap derivative, (2) recipient probability covariance without the cap derivative, and (3) cached recipient probability/cap Fisher fits. Apply the identical finite grid safeguard to all. Fit1 is a fixed linear map of the native activation delta; fit2 adds recipient-dependent probability weighting; fit3 adds the cap derivative.

All fits still require the full native response. Even the shared-metric condition chooses its final step using the native target, so it is not fully context-independent or executable from compressed inputs. The own-uncapped comparison isolates probability weighting from the cap derivative, but not concentration from token identity or every difference between shared and own distributions.

|Basis|Head|Family|Shared64|Own uncapped64|Own capped64|
|---|---|---|---:|---:|---:|
|probability|0|actual|0.706|0.860|0.861|
|probability|0|moment|0.463|0.744|0.747|
|probability|0|donor|0.475|0.751|0.753|
|probability|1|actual|0.539|0.797|0.799|
|probability|1|moment|0.497|0.781|0.784|
|probability|1|donor|0.500|0.782|0.786|
|probability|6|actual|0.555|0.820|0.824|
|probability|6|moment|0.506|0.840|0.846|
|probability|6|donor|0.508|0.842|0.848|
|random|0|actual|0.545|0.809|0.810|
|random|0|moment|0.326|0.722|0.725|
|random|0|donor|0.338|0.728|0.730|
|random|1|actual|0.428|0.765|0.768|
|random|1|moment|0.378|0.746|0.750|
|random|1|donor|0.379|0.749|0.753|
|random|6|actual|0.434|0.775|0.778|
|random|6|moment|0.395|0.815|0.822|
|random|6|donor|0.398|0.816|0.824|

Frozen gates: {'probability_adaptation10': True, 'cap_equivalent5': True, 'shared64_half': False}. Own-uncapped minus shared recovery is15.4–42.0pp across36 basis/head/family/rank combinations; every joint lower bound exceeds10pp (minimum11.28pp). Adding the cap derivative changes recovery by0.12–1.48pp, with all joint intervals inside the frozen±5pp equivalence margin (overall bounds−0.38 to1.98pp). The universal shared64 >50% lower-bound criterion fails.

At rank64, shared-metric probability axes recover46.3–70.6%, and shared-metric random axes32.6–54.5%. Recipient-uncapped fitting raises these to74.4–86.0% and72.2–81.6%, respectively. Thus much of the strong random-span oracle result depends on recipient-specific coefficient adaptation. The output basis still matters, but a fixed subspace and a fixed coefficient map should not be conflated with the adaptive oracle.

For the probability basis, the shared-metric fit algebraically equals Euclidean projection because its columns are eigenvectors of K. The implementation verifies this to6.6e-16 coefficient-map error, and its unguarded finite scores replay H373 within2.7e-15 relative error. Guarded shared results differ from H373’s unguarded results because step selection is now allowed. This gives a useful exact control for the fixed-map comparison.

Other audits: all16/64 Gram directions retained; normal-equation residuals below9.5e-13; recipient-uncapped local errors never exceed shared-fit errors under that same recipient metric. Finite values and positivity checks pass. Save per-site metric/rank/grid scores and choices, bootstrap arrays, script/config and source hashes. Native weights unchanged.

Uncertainty:4,000 paired recipient-document/control-draw bootstrap resamples. Separate joint36 bands for shared recovery, probability-adaptation contrasts and cap contrasts. Fixed discovery K, one random orientation, reused512-document panel and fixed empirical donor assignments. Figure shows point estimates. These results do not establish fresh-document generalization of the coefficient comparison.

Hypothesis update: recipient-dependent probability geometry is a major contributor to oracle coefficient quality; cap-derivative adaptation is a small contributor for this guarded finite diagnostic. This strengthens the interpretation of previous high recovery as an adaptive readout result rather than a discovery of fixed computational units. Concentration versus specific likely-token identity remains unresolved.

Next discriminator: fit coefficients using probability distributions from different documents matched on entropy and top-token probability, then score against the unchanged recipient target. Hold the cap derivative absent in both own and donor fits and retain the same safeguard. If confidence-matched donor metrics lose most of the gain, recipient-specific likely-token alignment matters beyond concentration alone. Freeze matching and gates before outcomes. Broader goal remains active.
