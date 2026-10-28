## H440 — Activity compression overfits baseline contexts

H439 validated exact fixed maps but weight-only output compression lost finite behavioral coverage. H440 uses exactly the same16 compiled maps and storage budgets, replacing the output basis with an uncentered PCA basis of T_H*x over the32 older H434 baseline states and16 source directions. It does not fit finite targets, intervention errors, or H438 states. The512 activity vectors come from only32 independent baseline contexts, not512 independent examples. Input RMS is not used to reweight the calibration Gram.

Evaluate the same output ranks16,32,64,128,256 on the H438 transfer panel, preserving the exact radial term or projecting both terms. This is a mechanism diagnostic using previously inspected transfer outcomes, not new independent confirmation. No new documents; total12480.

| Frozen rank64 radial-exact contrast | Estimate | Joint95% interval | Gate passes |
|---|---:|---|---|
|activity_minus_weight_low|-0.018572|[-0.069352, +0.032208]|False|
|activity_minus_weight_median|-0.010632|[-0.061412, +0.040147]|False|
|activity_minus_exact_low|-0.066052|[-0.116832, -0.015272]|False|
|activity_minus_exact_median|-0.045852|[-0.096632, +0.004928]|False|

All four gates fail: neither >2-point improvement over the weight-only basis nor within2-point noninferiority to exact compiled recovery is established in either stratum. Point estimates favor the weight-only basis, but their difference intervals include zero; do not claim established superiority of weight-only compression. The low-stratum activity-versus-exact interval excludes zero, supporting a loss there, without establishing a particular minimum2-point deficit. Joint bands use4000 paired source/document resamples conditional on the fixed calibration basis and donor pool.

| Output rank | Calibration activity energy retained | Transfer activity energy retained | Radial-exact full finite-KL recovery low / median |
|---|---:|---:|---|
|16|93.34%|81.89%|90.55% / 93.15%|
|32|98.40%|83.38%|91.19% / 94.11%|
|64|99.18%|85.47%|91.60% / 94.83%|
|128|99.55%|87.02%|92.95% / 95.27%|
|256|99.85%|89.32%|94.74% / 96.22%|

At rank64, calibration activity coverage99.18% drops to85.47% on transfer contexts. Full finite-KL recovery is91.60%/94.83%, compared with93.46%/95.89% for the same-budget weight basis and98.21%/99.41% for exact maps. Even rank256 transfers only89.32% of unweighted map activity despite99.85% calibration coverage. Therefore ordinary activity generalization already fails; output-probability weighting is not the only reason the behavioral compression underperforms.

A post-primary diagnostic compares compressed corrections to the exact shared16 correction, independently of the full finite model response. Rank64 raw correction error is23.46% in low-RMS and14.91% in median contexts. These ratios differ from pooled T_H*x coverage because actual source coefficients, input RMS and radial terms enter the correction. They do not represent accuracy or semantic attribution.

The uncompressed path still replays exact shared16 predictions within4.3e-15 relatively and prior scored errors within0 absolutely. Matched output ranks have the same map/basis storage as H439, so added capacity does not explain the comparison. Radial-exact handling is retained for primary contrasts. All failed gates are preserved.

Interpretation: the transferred small source basis does not imply that its context-dependent output writes share a comparably small fixed output basis. A basis can describe a small calibration set almost perfectly while missing directions visited in other states. This constrains the current compilation approach, not the possibility of compact computational units in general. It also reinforces the distinction between32 baseline states and512 correlated map outputs.

Next distinguish insufficient calibration coverage from an intrinsically broad fixed output representation. Compare calibration-state counts under a fixed selection rule, and regularize the input-state covariance toward the isotropic weight-only objective. Choose covariance mixing using held-out baseline map-activity prediction within the calibration corpus, not H438 finite errors. Evaluate the resulting fixed maps on the transfer panel as a diagnostic, then independently confirm any selected design. Count state coverage, basis storage and radial/baseline dependencies; do not optimize only calibration energy.
