## H770 — Small activation-only predictors do not recover the useful sensitivity gain

Previous goal turn was progress: H769 established a scalar probe-based predictor that transferred to withheld directions on separate documents. H770 tests whether inexpensive features of the local activation recover that scalar without evaluation-time probes.

Use exactly the H769 document split: 32 training and 16 validation documents, each at lengths 32 and 120. Labels are the scalar gains measured from the first 12 probes. Three prespecified ridge regressors predict log gain: prefix length only; five statistics (log length, log head norm, log centered head norm, log centered output-write norm, and top-eight write-coordinate energy fraction); and those five statistics plus the first eight signed write-PCA coordinates. Standardization, regression, log-to-linear mean calibration and fixed ridge .1 all use training data only. The original write-PCA basis was also fit on training data. No hyperparameters or features were selected on validation outcomes.

| Method | Validation gain relative RMS error | Validation target-probe relative RMS error |
|---|---:|---:|
| static | 72.9% | 82.3% |
| probe_scalar | 0.0% | 54.6% |
| length_only | 73.7% | 83.9% |
| five_stats | 75.8% | 85.6% |
| five_stats_plus_eight_write_PCs | 91.6% | 96.3% |

None of the tested activation predictors improves target-probe prediction over the static training mean. Adding principal coordinates improves log-gain training fit but worsens validation performance, consistent with insufficiently generalizable signal at this sample size. The probe-derived scalar remains substantially better. Zero gain error for that reference is by construction: its input is the measured gain itself; target-probe error is the independent part of that comparison.

Descriptive paired document bootstrap intervals for predictor-minus-static target MSE all include zero. They do not establish a reliable worsening or improvement; the important operational result is that no usable improvement was demonstrated. This rejects adoption of these particular small predictors, not the existence of an activation-derived sensitivity estimator. Downstream state, richer local features, larger training data, and a different regression objective remain possibilities. Optimizing log gain also differs from directly optimizing curvature MSE.

No full-model forwards were needed: all activation features and targets were cached. A validation-label mutation check multiplies every validation gain by 1,000 and verifies fitted predictions are exactly unchanged. Hashes unchanged; all features/predictions finite. Training and validation remain split by document. No reserved-test evaluation; unique natural document pool remains 26,112.

Next: do not use these failed predictors to create weights over the full activation corpus. Instead use directly measured **training-only** sensitivity labels for a controlled weighted-versus-unweighted feature fit on the same activation sample. Compare equal sample count, basis dimension, mean handling and regularization, and retain the larger unweighted natural-data fit as a reference. Evaluate actual reconstruction KL and subsequently causal transfer. This isolates whether weighting itself helps before investing in a cheap sensitivity estimator. A small labeled sample is a limitation to quantify, not a reason to quietly compare against a larger differently trained baseline.

The long-term objective is unchanged: better featurizers must preserve computation before their groups can be interpreted as meaningful manifolds. This negative result prevents an unsupported shortcut from probe diagnostics to manifold claims.
