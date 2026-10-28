

## H363 — Fixed unembedding geometry recovers much of the pattern

Reuse H356/357 native responses and recipient/confidence-donor residual directions. Use a fixed downstream metric K equal to the covariance of unembedding rows across the vocabulary (uniform weights, centered rows). No context-dependent probabilities, cap derivative, fit, scale correction or sign adjustment. For unit response d and unit residual u, predicted normalized output energy is dᵀKd−2(d·u)dᵀKu+(d·u)²uᵀKu. Apply the same mean-control and geometric own/matched statistic. Scalar RMS factors cancel per site.

|Control family|Zero-baseline log R²|Log correlation|Correct signs|Head6 prediction|Head6 measured|
|---|---:|---:|---:|---:|---:|
|moment|0.583|0.940|8/9|1.145|1.365|
|donor|0.611|0.946|6/9|1.120|1.306|

The frozen R²>0.5 gate passes for both families; the at-least7/9-sign gate fails because donors have6/9 correct signs. R² is zero-baseline log prediction fidelity without an intercept. High correlation captures ordering but does not imply calibrated magnitude. Head6 predicted1.145/1.120 versus measured1.365/1.306, so the fixed metric substantially underestimates its effect.

Replacing isotropic geometry with the fixed unembedding metric reverses the prior model failure and recovers much of the head ordering without fitting. Shared anisotropic output geometry interacting with the residual projector is therefore a viable partial account. Context-dependent probabilities, cap effects or other geometry may explain the remaining magnitude. No feature semantics or compact input computation is identified.

Audit: analytic quadratic energies match direct centered-vocabulary logit variance on a fixed small subset for every head within relative error 2.94e-15. Minimum predicted energy 0.00104667>0; source hashes verified. Fixed1152×1152 metric and per-site predictions saved. Native weights unchanged.

Limits: existing-panel screening, no independent confirmation; only9 heads per family, correlated outcomes and small near-zero effects make descriptive sign/correlation statistics fragile. The predictor requires native response vectors and residual states; it predicts their downstream geometry, not the native MLP computation itself. The full metric is not a compact learned circuit.

Next: freeze this no-fit metric and assess transfer on independent documents, retaining both error and sign criteria. Separately test whether a low-rank weight-only approximation preserves its predictive value before claiming a compact account. Broad research goal remains active.
