## H769 — A scalar sensitivity estimate transfers to withheld probe directions

Previous goal turn was progress: H768 established context heterogeneity without categorical sampling noise, but did not establish consistent special alignment between errors and their original contexts. This experiment tests whether overall sensitivity inferred from one probe bank transfers to another bank on separate documents.

Design: 32 training documents and 16 validation documents, each at prefix lengths 32 and 120. Training documents exclude the H764 gradient-estimation panel; validation documents exclude the H765/H767 validation panel. All documents were already in the natural research pool, so these are newly probed contexts, not a new corpus or untouched reserved test. Twenty-four fixed independent isotropic random directions are shared across contexts. Their common norm is fixed from the training median budget-8 adaptive write-PCA reconstruction error. Probe responses are local Fisher curvature estimated by central logit differences and full-vocabulary probability variance.

The first 12 probes are observed at an evaluation context; the remaining 12 responses are withheld there. Target directions are measured on training documents, so this is prediction at new contexts for known directions, not extrapolation to never-measured directions.

Four predictors are fitted using training documents only. Static predicts each target direction's training mean. Scalar multiplies that mean by the observed probes' average response relative to their training means. Two richer predictors fit residual directional variation after dividing out this scalar: a rank-two ridge map and a full ridge map. Ridge strength .1 and rank two were fixed before outcomes; no validation selection was performed. These are continuous regression controls, not discrete semantic regimes or groups.

| Predictor | Validation relative RMS error | Median relative error |
|---|---:|---:|
| static | 82.3% | 108.6% |
| scalar | 54.6% | 42.6% |
| scalar_rank2 | 54.2% | 42.2% |
| scalar_full_ridge | 57.1% | 41.3% |

The scalar estimate improves substantially over a context-independent prediction. Rank-two residual structure provides only a small pooled gain, and full residual regression performs worse on validation despite better training fit. Scalar improvement appears at both lengths: relative RMS error falls from 77.9% to 61.0% at length 32 and from 89.2% to 42.0% at length 120. The rank-two correction improves length 32 slightly but worsens length 120.

A descriptive 10,000-draw paired bootstrap resamples the 16 validation documents, averaging the two lengths and 12 target directions within each document. The static-minus-scalar MSE interval is entirely positive. Rank-two-minus-scalar and full-ridge-minus-scalar intervals both include zero. These intervals omit fitted-estimator uncertainty and multiple-comparison correction; they support prioritization, not a broad statistical claim.

The remaining 54.6% scalar RMS prediction error is substantial. A scalar does not adequately describe all directional geometry. Nor does this experiment show that richer geometry cannot be learned with more data, a better estimator, or a different probe bank. It shows that this small directional regression does not yet justify additional complexity over the scalar control.

Crucially, all predictors receive model-derived probe responses at evaluation time. This is not yet an activation-only sensitivity estimator, a new featurizer, or proof that scalar-weighted feature learning improves causal fidelity. Random probes also differ from reconstruction errors and task interventions. Any learned feature objective must be validated on actual KL and causal task edits, not merely probe prediction.

Audits: 11,520 forward evaluations; finite curvatures; two numerical steps agree within .42% relative RMS (median relative difference .32%). Training/validation documents are disjoint and excluded from the specified earlier panels. Hashes unchanged; neither directional predictor required clipping a negative validation factor. Original reserved test untouched; unique natural document pool remains 26,112.

Next: determine whether simple activation-derived quantities predict the scalar gain on held-out documents, using a fixed small predictor and reporting both gain prediction and target-probe errors. If that succeeds, compare sensitivity-weighted and unweighted feature learning at matched capacity. If it fails, obtain training-only probe weights directly rather than claiming a usable activation predictor. Preserve scalar weighting as a control when testing richer conditional geometry. The ultimate gate remains manifold stability, held-out combination transfer and selective causal effects across diverse competent tasks.
