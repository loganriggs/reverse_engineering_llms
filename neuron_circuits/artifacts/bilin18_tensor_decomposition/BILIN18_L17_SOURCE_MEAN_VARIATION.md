## H461 — Mean source signal versus varying donor information

Reuse H459108 donor/head signals and128 receiving contexts,±1. Calibrate head means and centered source SVD16 using H460’s donor split, excluding evaluation donor IDs. Keep uncentered16 bases frozen from H460. For full-source h, compare omitted-response reconstructions from mean mu, uncentered16 projection, and mu+centered16(h−mu). For centered-deviation interventions h−mu, compare zero omitted correction, uncentered16 projection of the deviation and centered16 projection of the deviation. Exact retained response is recomputed for the corresponding full or centered target. No context-input/output compression or outcome-based fitting; total12992 documents.

| Head | Calibration mean-energy share | Full mean-only / uncentered16 / affine16 recovery | Deviation retained-only / uncentered16 / centered16 recovery | Deviation/full effect KL ratio |
|---|---:|---|---|---:|
|0|5.88%|29.96% / 67.07% / 66.96%|23.99% / 64.36% / 64.18%|0.92154|
|1|59.37%|84.23% / 96.89% / 96.89%|50.50% / 90.17% / 90.22%|0.29412|
|2|15.97%|42.96% / 87.98% / 88.01%|34.01% / 86.15% / 86.14%|0.85492|
|3|4.93%|52.43% / 75.35% / 74.81%|42.91% / 70.32% / 69.81%|0.83215|
|4|1.73%|16.64% / 79.27% / 79.33%|32.51% / 83.26% / 83.26%|1.23305|
|5|1.52%|31.43% / 59.22% / 58.78%|23.26% / 55.11% / 53.83%|0.89268|
|6|25.35%|92.05% / 99.48% / 99.48%|80.61% / 98.70% / 98.71%|0.40066|
|7|2.85%|44.94% / 66.79% / 66.80%|30.79% / 58.44% / 58.25%|0.79629|
|8|15.47%|81.93% / 92.68% / 92.74%|57.91% / 82.98% / 83.12%|0.42814|

Table uses positive amplitude. Mean-energy share is squared calibration mean norm divided by mean squared source norm. Effect ratios compare sums of native target-versus-baseline KL across matched cases, not an additive decomposition: mean and deviation responses can interact nonlinearly. Recovery denominators differ by source family. Absolute pooled effects and source energies are saved.

| Frozen head6 contrast | Estimate | Joint95% interval | Required lower | Pass |
|---|---:|---|---:|---|
|head6_mean_minus_uncentered_negative|-0.078238|[-0.113267, -0.043209]|-0.02|False|
|head6_mean_minus_uncentered_positive|-0.074266|[-0.109296, -0.039237]|-0.02|False|
|head6_centered_minus_affine_full_negative|-0.008728|[-0.043757, +0.026301]|-0.05|True|
|head6_centered_minus_affine_full_positive|-0.007744|[-0.042773, +0.027286]|-0.05|True|

Primary mean-only versus uncentered16 allows2pp loss at each sign. Centered-deviation centered16 versus full-source affine16 recovery allows5pp loss at each sign. These compare prediction quality, not equality of the underlying intervention effects.4000 paired donor/context bootstrap replicates, joint95% max-absolute-deviation bands; calibration uncertainty fixed. Other heads are descriptive.

Audit: full-source uncentered16 replays H460 with max absolute score error0; native effect replay0; both below1e-6. Calibration/evaluation donor IDs remain disjoint, arrays finite and no exclusions.

Head6 compactness is not explained by a mean-only omitted predictor. At−1/+1, mean-only recovery91.586%/92.055% falls below uncentered16 99.410%/99.482% by7.824pp/7.427pp. Joint95% intervals[−11.327,−4.321] and[−10.930,−3.924]pp fail both2pp noninferiority gates. Remember exact retained-factor responses remain in mean-only predictions, so even these mean-only scores do not attribute all effects to a constant.

Centered donor variations remain compact. Head6 centered16 predicts98.541%/98.710% of centered-deviation effects, versus full-source affine16 99.414%/99.484%. Differences−0.873pp/−0.774pp have joint95% intervals[−4.376,+2.630] and[−4.277,+2.729]pp; both frozen5pp noninferiority gates pass. This supports compact prediction of donor-to-donor variation at the specified tolerance, not equality. Centered-source reconstruction loses2.642% of held-out deviation energy.

Head6 calibration mean energy is25.35% of total uncentered source energy. Positive centered-deviation effect KL sums to40.07% of the full-donor effect on this panel, so the centered test is not merely a negligible perturbation denominator. This ratio is not an additive attribution because normalization/bilinear responses interact. Other heads differ: head1 mean energy59.37%, centered16 recovery90.22%; head8 centered16 recovery83.12%; broader heads0/3/5/7 remain around64.18/69.81/53.83/58.25%. These comparisons are descriptive, not new cross-head tests.

The updated explanation is heterogeneous varying-source dimensionality, rather than a universal mean-dominated interface. Head6 contains a genuinely low-dimensional distribution of varying donor signals in the tested sense, but this alone does not identify semantic variables or prove its computation is low-dimensional for arbitrary inputs. Uncentered and affine16 full-source results are nearly identical, indicating that explicitly separating the mean does not radically change the earlier ranking.

Next distinguish low-rank output weights from low-dimensional exercised input activity. Audit each attention head’s output matrix spectrum, then compare weight-selected rank16 output directions with centered-data rank16 on natural centered donor writes. Add norm-matched isotropic latent head-coefficient controls passed through each head’s native output matrix. If head6 data directions retain coverage on those controls, weight geometry may explain compactness; if natural variation is compact but isotropic latent inputs are not, the occupied activity distribution is the key restriction. Freeze spectra/bases before intervention outcomes, preserve source effect magnitudes and matched receiving contexts, and avoid turning geometric compactness into semantic labels prematurely.

A mean-only omitted predictor still retains the original source’s exact128-factor response, so its score must not be described as explaining the entire intervention by a constant. Affine16 uses16 varying coordinates plus a stored mean; it is not identical in storage to an uncentered16 linear space. No semantic unit or intact-model replacement is established.
