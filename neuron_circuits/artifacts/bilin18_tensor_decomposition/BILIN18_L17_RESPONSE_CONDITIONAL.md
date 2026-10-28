

## H292 — Conditional covariance geometry yields a small, unconfirmed improvement

H291's raw residual cross-covariance directions did not generalize better than natural PCA. H292 conditions outside input-change covariance and input-target cross-covariance on the existing64 reads via a training-only Schur complement. Within the1088-dimensional orthogonal complement, take the inverse square root of the conditional covariance plus beta times its mean eigenvalue, multiply by conditional target cross-covariance, and retain leading left singular directions. Map them back into the input complement, QR and scale by training natural-state standard deviation. Existing64 reads remain available to the final ridge readout.

Fixed grid: beta0/.001/.01/.1/1, added rank32/64, and eight final ridge strengths from H290.80 candidate readouts. Eigenvalues have a relative1e-12 floor before inverse square root; this makes beta0 numerically defined but does not establish a population inverse. All basis moments come from the768 training documents. Validation selects the final candidate; no validation covariance enters directions. This is a conditional-covariance basis followed by a separately penalized refit, not a claim of exact joint reduced-rank optimality after QR/scaling/ridge. The larger search grid is a selection advantage over natural PCA, requiring independent confirmation.

A focused numerical test matches Schur-complement statistics to explicit sample residualization and verifies complement orthogonality. All added bases have relative overlap<1e-12 with existing reads;32-direction maps are exact prefixes of64-direction maps. Full candidate normal-equation residuals and covariance spectra are saved. Native model/compiler/base and discovery samples remain unchanged. No fresh documents or intact-KL evaluation in this screen.

| Basis beta | Added directions | Selected output ridge | Training response error/base | Validation response error/base |
|---|---:|---:|---:|---:|
|0|32|0.1|0.930856|0.964087|
|0|64|1|0.949947|0.965277|
|0.001|32|0.1|0.930815|0.964103|
|0.001|64|1|0.949847|0.965262|
|0.01|32|0.1|0.930550|0.964263|
|0.01|64|1|0.949056|0.965158|
|0.1|32|0.1|0.930909|0.965906|
|0.1|64|1|0.945565|0.965272|
|1|32|1|0.951907|0.968209|
|1|64|1|0.944593|0.968002|

Winner: b0_rank32_a0.1 (96 total reads). Validation response-MSE reduction is0.428% versus best natural-PCA expansion and0.511% versus best regularized64. Both preset2% gates FAIL. No thresholds relaxed. These are reused-validation-selected numbers without fresh confidence intervals, not confirmed generalization gains.

**Inference.** Accounting for covariance geometry changes the direction-selection result from a small loss to a small validation gain. This makes raw covariance ranking an inadequate basis for declaring all outside information unhelpful. However, the gain remains below the practical threshold while the training advantage is much larger. Estimation, repeated validation selection and limited feature utility remain plausible. The96-read winner outperforms128-read candidates under this grid; this does not establish intrinsic dimension96. Intact accuracy is unknown for the winner and cannot be inferred from response-only MSE.

**Next discriminator.** Stop tuning this validation panel. Freeze the conditional96 winner, best natural expansion, best regularized64 and H287 joint fit, then compare them on a new complete-document-disjoint panel. Measure actual/angle response MSE, conditional downstream KL and ordinary intact KL, with paired document uncertainty. Predeclare the2% response improvement and2% intact-retention thresholds. Confirmation would support a modest geometry-dependent outside-span benefit; failure would weaken the apparent gain and increase priority of disjoint basis/readout fitting or more discovery documents. No semantic-unit claim follows either way.

Storage: added32/64 read vectors plus110,592/147,456 active output coefficients and the fixed base; the80-model research grid is not inference storage. All matrices and spectra retained under bilin18_l17_response_conditional*. All jobs terminal. Research goal remains active.
