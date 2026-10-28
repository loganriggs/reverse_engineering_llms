

## H362 — Radial fraction alone predicts the wrong pattern

Before spending a new confirmation panel, test the simplest fixed geometric model against the saved H356/357 native responses: assume isotropic downstream sensitivity, so output Rayleigh after RMS projection is proportional to1−cos(response,residual)². The scalar RMS factor cancels in each control/actual site ratio. Apply the same mean-control, geometric own/matched statistic across all9 heads and2 control families. No coefficients, head labels, signs or scales are fitted.

|Control family|Zero-baseline log R²|Log correlation|Correct signs|Head6 prediction|Head6 measured|
|---|---:|---:|---:|---:|---:|
|moment|-1.050|-0.916|1/9|0.875|1.365|
|donor|-1.002|-0.902|3/9|0.888|1.306|

Both frozen descriptive gates FAIL: log R²>0.5 and at least7/9 signs correct, both families. R² uses a zero-log-pairing baseline without fitting an intercept; negative values mean worse squared log error than predicting no pairing everywhere. Several head effects are near zero, so sign accuracy is descriptive rather than a significance test. The strong negative correlations are also descriptive across only9 heads.

H361 showed that changing the RMS projector carries the pairing contrast while swapping the downstream metric does little. H362 shows that this does not make the downstream metric irrelevant: applying that projector under isotropic output geometry reverses much of the pattern. The projector interacting with anisotropic downstream sensitivity is necessary to explain these results. Radial fraction is not a sufficient scalar account. This is a model falsification, not evidence for a particular semantic unit.

Audit: all native response cache hashes verified; own and confidence-donor residual directions reconstructed; minimum predicted tangential energy fraction 0.127513>0. No fitted parameters or sign flips. Per-site predicted Rayleigh values saved. Native weights unchanged.

Limits: reused panel and native response inputs; this is an inexpensive screening/falsification stage, not independent predictive confirmation. The proposed geometry model assumes isotropic sensitivity after normalization and neglects tiny RMS epsilon effects; neither assumption is made by the measured Fisher pipeline.

Next: retain a fixed anisotropic downstream metric, starting with the centered unembedding Gram, and test whether it predicts the head pattern without context-dependent probability weights. Only a candidate that passes this screening should be frozen and tested on independent data. Broad research goal remains active.
