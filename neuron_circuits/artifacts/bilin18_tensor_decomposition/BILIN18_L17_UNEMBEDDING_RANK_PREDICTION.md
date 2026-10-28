

## H364 — Weight-ranked low-dimensional output geometry

Diagonalize the fixed centered-unembedding covariance from H363. Retain leading eigenvectors at ranks1,2,4,8,16,32,64,128,256,512,1152. Axes and ordering use weights only; no native response fitting. Predict each native unit response after its own or donor RMS projector using this truncated metric. Rank16 is the prespecified diagnostic; compare one sign-corrected Gaussian-QR random16 orientation with identical eigenvalues. Other ranks are an exploratory fixed ladder.

|Rank|Moment actual log R²|Donor actual log R²|Moment fidelity to full metric|Donor fidelity to full metric|
|---|---:|---:|---:|---:|
|1|0.766|0.713|-3.889|-4.016|
|2|0.853|0.861|-0.421|-0.746|
|4|0.797|0.806|0.006|-0.302|
|8|0.758|0.760|0.276|-0.004|
|16|0.726|0.733|0.558|0.357|
|32|0.730|0.733|0.629|0.492|
|64|0.727|0.733|0.699|0.605|
|128|0.739|0.753|0.734|0.678|
|256|0.748|0.769|0.766|0.745|
|512|0.748|0.778|0.791|0.784|
|1152|0.583|0.611|1.000|1.000|
|random16|-0.766|-0.516|-2.777|-1.971|

Frozen rank16 outcomes: {'rank16_fidelity95': False, 'rank16_actual_half': True, 'rank16_orientation_gain20': True}. Rank16 predicts the measured pattern better than the full uniform-unembedding metric and substantially better than the single random orientation. It fails95% fidelity to the full metric’s log-pairing predictions. Thus this is not faithful compression of that metric: dropping directions changes the prediction in a way that happens to improve agreement with the probability-sensitive target.

Rank2 gives exploratory log R²0.853/0.861 and7/9 signs correct in both families, versus full metric0.583/0.611. Its axes remain weight-only, but selecting rank2 for follow-up uses these outcomes. Freeze rank2 now and carry rank16/full/random controls into independent validation; do not claim heldout selection here.

A rank2 metric requires2×1152 direction coefficients plus2 eigenvalues, but the predictor still consumes full native response vectors and context residuals. This does not compress the native MLP, identify two semantic features, or show that all computations occur in two dimensions. It is a compact downstream geometric predictor.

Audit: rank1152 predicted energies replay H363 within relative error 1.71e-15; metric eigenvectors/eigenvalues and random basis saved; source hashes verified. One random orientation is a diagnostic, not a null distribution. No fitted response parameters or native weight edits.

Limits: reused H356/357 panel,9 correlated head outcomes per family, no new uncertainty analysis for this screening. Full-metric and actual-target fidelity measure distinct objectives.

Next: independent-document confirmation with rank2 frozen from this ladder, prespecified rank16, full rank and the fixed random16 control, preserving both error and sign criteria. Broad research goal remains active.
