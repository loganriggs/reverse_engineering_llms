## H462 — Weight spectrum versus exercised source activity

Reuse H461 centered natural donor signals and128 receiving contexts,±1. Keep the centered-data rank16 bases calibrated on donors excluding evaluation IDs. For each head, take the top16 left singular directions of its1152x128 output matrix. Draw12 isotropic Gaussian latent128 vectors per head, multiply by that native matrix, then rescale each output vector to the paired natural centered signal’s norm. Compare data16, weight16 and retained-only predictors using exact native omitted response to the reconstructed source. No context-input/output compression; total12992 documents.

The synthetic control changes source orientation while matching its output norm; it does not preserve residual angle, covariance or semantic content. A transfer failure indicates dependence on the naturally exercised distribution, not automatically semantic specificity. Spectral capture ratios are expectations before per-sample norm matching.

| Head | Weight16 spectral energy | Data16 isotropic energy capture | Natural data16 / weight16 / retained recovery | Isotropic data16 / weight16 / retained recovery |
|---|---:|---:|---|---|
|0|21.60%|19.48%|64.18% / 59.10% / 23.99%|32.54% / 29.84% / 14.57%|
|1|21.22%|19.27%|90.22% / 86.34% / 50.50%|57.31% / 59.42% / 48.68%|
|2|25.80%|22.00%|86.14% / 78.22% / 34.01%|58.05% / 54.52% / 38.12%|
|3|19.92%|16.44%|69.81% / 59.11% / 42.91%|51.36% / 55.42% / 37.55%|
|4|18.70%|15.73%|83.26% / 50.93% / 32.51%|33.39% / 33.63% / 20.62%|
|5|19.39%|15.89%|53.83% / 58.61% / 23.26%|34.31% / 33.13% / 24.28%|
|6|20.60%|18.21%|98.71% / 97.63% / 80.61%|55.57% / 51.51% / 37.50%|
|7|19.00%|16.28%|58.25% / 47.47% / 30.79%|34.08% / 40.86% / 25.59%|
|8|21.09%|18.62%|83.12% / 77.72% / 57.91%|52.81% / 57.40% / 41.97%|

Table uses positive amplitude. Weight energy is the fraction of output-matrix squared singular values in the top16; data capture is ||B_data^T W_head||_F²/||W_head||_F². Both characterize the linear output matrix for isotropic latent coefficients, not natural source variance or probability behavior.

| Frozen head6 contrast | Estimate | Joint95% interval | Required lower | Pass |
|---|---:|---|---:|---|
|head6_weight_minus_data_natural_negative|-0.012952|[-0.175760, +0.149855]|-0.02|False|
|head6_weight_minus_data_natural_positive|-0.010844|[-0.173652, +0.151963]|-0.02|False|
|head6_isotropic_minus_natural_data_negative|-0.469869|[-0.632677, -0.307061]|-0.05|False|
|head6_isotropic_minus_natural_data_positive|-0.431362|[-0.594169, -0.268554]|-0.05|False|

Natural-source weight16 versus data16 allows2pp loss at each sign; isotropic-source data16 versus natural-source data16 allows5pp loss at each sign.4000 paired donor/control and receiving-context bootstrap replicates, joint95% max-absolute-deviation intervals. Covariance/basis uncertainty is fixed. Other heads are descriptive, not selected substitutes for primary failures. Effect sizes and energy denominators are saved for both families.

Audits: source norm-match maximum absolute difference9.09e-13; natural data16 reference replay0; native effect replay0. All numerical gates pass and arrays finite; no exclusions.

Head6's natural centered activity is compact, but its full output-weight map is not approximately rank16 in Frobenius energy: the best16 weight directions capture20.60%, and the centered-data16 directions capture18.21% of expected isotropic output energy before norm matching. Natural data16 recovery is98.541%/98.710% at negative/positive amplitude, versus51.554%/55.574% on norm-matched isotropic latent controls. The isotropic-minus-natural differences are−46.987/−43.136pp, with joint95% intervals[−63.268,−30.706]/[−59.417,−26.855]pp. Both5pp noninferiority gates fail, and these intervals support a substantial deficit, not merely lack of evidence for equivalence.

Weight16 natural recovery is97.246%/97.625%, versus data16 98.541%/98.710%. Differences−1.295/−1.084pp have joint95% intervals[−17.576,+14.986]/[−17.365,+15.196]pp; both2pp noninferiority gates fail. These wide bands do not establish data16 superiority or weight16 equivalence. The joint band covers all four contrasts, including variable synthetic-source contrasts.

A post-hoc robustness audit, explicitly descriptive and using the same panel, finds synthetic/native total effect KL ratios0.949/0.911 at negative/positive amplitude. Synthetic effects are not larger in aggregate despite worse recovery. Equal-donor mean recoveries are natural96.708%/96.824% versus synthetic45.586%/47.333%. Removing any one paired donor leaves the synthetic-minus-natural pooled recovery gap between−57.176 and−41.814pp at negative amplitude, and−50.914 and−38.521pp at positive amplitude. Removing any single receiving context also preserves large gaps. Every synthetic donor has lower recovery than every natural donor within each sign. However, pooled effect effective donor counts are only about4–5 out of12, so larger independent control samples remain valuable. These sensitivity checks are not new confirmatory tests and do not incorporate calibration uncertainty.

The supported explanation is restricted naturally exercised source activity, not an inherently rank16 output matrix. It does not yet distinguish a covariance-level restriction from higher-order structure or semantic variables. A covariance-matched Gaussian can occupy the same narrow directions without carrying donor-specific semantic information.

Next experiment: freeze the existing centered-data16 basis and calibrate centered latent covariance on the disjoint donor calibration pool. Compare fresh centered natural donors, covariance-matched Gaussian signals, and isotropic latent controls, with the same output-norm matching and signs. Use unused receiving contexts and substantially more independent source draws. Predeclare natural-versus-covariance-matched recovery equivalence and covariance-matched-versus-isotropic superiority, alongside source reconstruction and absolute probability effects. If covariance matching restores recovery, second-order activity geometry suffices for this result; if it does not, test residual angle, higher moments and natural source/context dependence before semantic interpretation. Matching covariance in expectation before sample norm matching is not exact covariance matching afterward; record both empirical covariances and do not conflate these controls.

This diagnoses where source compactness comes from. It does not identify semantic coordinates or prove that arbitrary latent coefficients are naturally reachable. The exact retained response and original dense omitted computation remain in evaluation; no new compact multi-head program or runtime acceleration is claimed.
