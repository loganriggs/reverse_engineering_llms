

## H313 — Scaling helps, but the fixed predicted directions impose a low ceiling

Reuse H311 joint16 neighbor indices and H308 response vectors, both source hashes locked. Reference first768 documents unchanged. Split the previously reused256 discovery-validation documents: first128 fit one nonnegative scalar per family and estimator; last128 evaluate. This is disjoint marginal calibration/evaluation, not fresh or end-to-end independence. Two estimators: mean RAW neighbor responses; normalized mean UNIT neighbor responses times the mean neighbor norm. Neither amplitude uses the query target. No intercept or per-head scalar.

Scalar minimizing calibration squared error is max(0,sum predictor·target / sum||predictor||²). Separately compute per-query best NONNEGATIVE amplitude along the frozen predicted direction: captured energy=max(predictor·target,0)²/||predictor||². This uses query targets and is explicitly a nonexecutable oracle upper bound for that fixed direction, not model performance.

| Family | Estimator | Scalar | Unscaled fidelity | Calibrated fidelity [95%] | Oracle amplitude bound |
|---|---|---:|---:|---:|---:|
|actual|raw_mean|0.6477|1.68%|2.67% [2.44,2.90]|3.79%|
|actual|unit_direction_mean_norm|0.2005|-45.55%|2.78% [2.54,3.02]|3.97%|
|angle|raw_mean|0.4294|-1.17%|1.36% [1.21,1.51]|1.99%|
|angle|unit_direction_mean_norm|0.1322|-63.12%|1.43% [1.27,1.59]|2.07%|

Both locked gates FAIL: calibrated unit-direction predictor does not reach10% fidelity in both families; its oracle directional bound does not reach50%. Paired document bootstrap2000 resamples of the128 evaluation documents supplies conditional calibrated-fidelity intervals. Scalars remain fixed across resamples; intervals omit calibration uncertainty.

**Inference.** Scalar shrinkage fixes much of the raw-mean overshoot, especially for angle controls, but leaves low fidelity. Unit-direction and raw-mean directions perform similarly after calibration. Even perfect target-dependent positive scaling along the unit-average directions recovers only3.97% actual and2.07% angle energy on these evaluation documents. Thus most error cannot be corrected by amplitude-only changes to these fixed directions. This is a ceiling for the tested directional estimator on this sample, not for other directions, multiple-vector combinations, signed flips, nonlinear predictors or the underlying model.

Unit-direction mean cosine is.209 actual/.124 angle, versus raw-mean.194/.108. Positive mean cosine therefore coexists with low recoverable squared-error energy. Equal-site directional averages and energy-weighted fidelity answer different questions. The unscaled unit-direction×neighbor-mean-norm predictor badly overshoots (−45.55/−63.12% fidelity), so assigning a full typical norm to a weakly predicted direction is not justified.

**Audit.** Independently replayed closed-form scalars, calibration sufficient statistics and evaluation fidelity from saved rows; calibrated fidelity is below the per-query oracle bound in both estimators/families. References and target indices remain the saved disjoint H311 split. No native model execution or weights changed; all jobs terminal. H312 fresh panel was not used for calibration.

**Next discriminator.** Stop tuning amplitude for this neighbor rule. Compare an explicit bilinear interaction predictor with additive context/intervention predictors at controlled data and representation cost. Use the known quadratic structure to construct a cross-input feature kernel or restricted tensor features, fit on discovery references, and reserve new documents for frozen confirmation. The key test is whether representing the interaction improves response-vector direction and fidelity beyond nearest-neighbor geometry, not whether another cosine threshold can be passed. Broad research goal active.
