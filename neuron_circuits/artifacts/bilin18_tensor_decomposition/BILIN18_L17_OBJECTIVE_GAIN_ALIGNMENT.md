## H525 — Activation, prediction-KL and observed-token-CE objectives differ locally

Freeze the pooled3,072-source Fisher rank36 read/write factors. Introduce36 multiplicative coordinate gains initialized to1. This is a restricted family with rank at most36, not an optimization of the entire bilinear layer. Gains do not generally preserve projection idempotence. Within this fixed parameterization, compute per-document gradients for raw source reconstruction MSE, original-to-replacement prediction KL, and observed-token CE change, on both H512 and H521 panels. Both panels are reused diagnostics.

Every gradient is evaluated at the same baseline gains. Gradient cosine describes local objective alignment in Euclidean gain coordinates; it is parameterization-dependent and is not a global equivalence claim. Bootstrap2,048 paired document resamples within each panel, independently between panels, using seed20280107. These are descriptive marginal percentile intervals, not simultaneous or selection-adjusted tests.

| Gradient comparison | Cosine | Bootstrap95% range |
|---|---:|---:|
|H512_activation_MSE__prediction_KL|-0.417|[-0.740, 0.459]|
|H512_activation_MSE__observed_CE|0.815|[-0.225, 0.872]|
|H512_prediction_KL__observed_CE|-0.287|[-0.591, 0.351]|
|H521_activation_MSE__prediction_KL|-0.672|[-0.848, 0.772]|
|H521_activation_MSE__observed_CE|-0.283|[-0.723, 0.675]|
|H521_prediction_KL__observed_CE|0.301|[-0.653, 0.716]|
|cross_panel_activation_MSE|0.376|[-0.798, 0.853]|
|cross_panel_prediction_KL|0.916|[0.486, 0.948]|
|cross_panel_observed_CE|-0.336|[-0.660, 0.467]|

Prediction-KL gradients are strongly aligned across panels (cosine0.916; bootstrap range0.486–0.948). Observed-token CE gradients are not reproducible here: cross-panel point cosine−0.336 with a wide interval spanning both signs. Activation-MSE alignment is also uncertain. Within-panel KL/CE cosine changes sign (−0.287 versus+0.301), so these data do not establish a stable population tradeoff between KL and CE.

Apply every prescribed step gains=1−alpha*6*g/||g|| for alpha0.01/0.03 and every source-panel/objective direction. Report all results without selection. Negative changes below indicate improvement relative to the frozen baseline map. The table shows cross-panel transfers; all own-panel results and linear predictions are archived.

| Direction from | Objective | Alpha | Evaluated on | MSE change | Prediction KL change | CE change ± paired SE |
|---|---|---:|---|---:|---:|---:|
|H512|activation_MSE|0.01|H521|+18.5553|+0.00002490|+0.0000775 ± 0.0001186|
|H512|activation_MSE|0.03|H521|+171.0369|+0.00010183|+0.0002543 ± 0.0003534|
|H512|prediction_KL|0.01|H521|+24.4733|-0.00002596|-0.0000428 ± 0.0001418|
|H512|prediction_KL|0.03|H521|+213.3345|-0.00004632|-0.0001042 ± 0.0004274|
|H512|observed_CE|0.01|H521|+20.1359|+0.00002168|+0.0000837 ± 0.0001244|
|H512|observed_CE|0.03|H521|+185.3688|+0.00009494|+0.0002742 ± 0.0003707|
|H521|activation_MSE|0.01|H512|+22.1552|+0.00002047|-0.0001541 ± 0.0001287|
|H521|activation_MSE|0.03|H512|+211.3760|+0.00008918|-0.0004394 ± 0.0003864|
|H521|prediction_KL|0.01|H512|+35.3343|-0.00001604|+0.0002062 ± 0.0001504|
|H521|prediction_KL|0.03|H512|+298.9651|-0.00000931|+0.0006519 ± 0.0004512|
|H521|observed_CE|0.01|H512|+13.7364|-0.00000233|+0.0001399 ± 0.0000949|
|H521|observed_CE|0.03|H512|+113.7223|+0.00000831|+0.0004342 ± 0.0002846|

KL directions lower KL on both their source panel and the opposite panel at both scales. Their effects are small, approximately0.3–1.4% of baseline prediction KL across these conditions. CE directions lower CE on the source panel but raise it on the other panel in every tested transfer point estimate; uncertainties remain wide. A same-panel CE decrease is expected from choosing its negative gradient and does not establish generalization. The transferable KL direction and unstable CE direction support investigating label-gradient noise before claiming that CE demands a fundamentally different computation.

The activation-MSE steps overshoot even their source-panel minima. Raw MSE is exactly quadratic in coordinate gains, so the negative derivative is compatible with a positive finite-step change. Independent replay of the quadratic coefficient from alpha0.01 and0.03 agrees within1.4e−15 relative error. Along each source panel's own normalized MSE direction, the exact optimal alpha is0.001412 for H512 and0.000340 for H521, both below the prescribed scales. We have not evaluated KL/CE at those optima here. Therefore the failed finite MSE steps must not be read as an impossible MSE improvement or a reliable objective conflict. Euclidean normalization of gradients does not equalize objective curvature.

Observed-token CE differs from teacher-distribution cross-entropy. Using the original predicted probabilities as targets gives the same gradient as prediction KL; using one observed token introduces a label-dependent gradient contrast. That contrast can contain a real correction to the original model as well as sampling variation. Present uncertainty does not distinguish them.

Audits: fixed model and data hashes checked; reconstructed rank36 map exactly replays H522; baseline finite KL replays to1.30e−6 relative or better despite a normalization implementation change to explicit RMS arithmetic. A double-precision directional finite-difference audit with the original float32 normalization epsilon agrees with autograd within7.93e−9 relative across all three objectives at two step sizes. Per-document gradients and losses are finite and saved; summary means replay. No new documents:24,064. No model is selected, deployed, or claimed to improve task performance.

Next H526: quantify the observed-label gradient contrast d=g_CE−g_KL using the saved per-document gradients. Separate mean-direction estimates from their sampling variability: report independent-panel dot products, paired document bootstrap distributions, and the unbiased squared-mean estimator (||sum d||^2−sum||d_i||^2)/(n(n−1)), which may be negative from sampling noise and must not be clipped. Compare with the same calculations for KL gradients; inspect contribution concentration without discarding observations. Also evaluate the exact source-panel MSE-optimal steps (computed from the fixed quadratic, not selected by target outcomes) on both panels, cross-reporting all objectives. This distinguishes the current step-size failure from genuine objective disagreement, while assessing whether CE's extra direction is detectable before further optimization. Reused-panel diagnostic; any task-improvement claim needs a separate training/validation design and fresh confirmation.
