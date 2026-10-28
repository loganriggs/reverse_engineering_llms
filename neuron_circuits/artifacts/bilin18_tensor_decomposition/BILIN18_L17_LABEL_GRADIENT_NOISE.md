## H526 — Label-gradient contrast is uncertain; corrected MSE steps are tiny

Use the same frozen36 coordinate gains and H525 per-document gradients. The contrast d=g_CE−g_KL is the part of the observed-label gradient beyond the original-model teacher-distribution gradient. It may contain a real task-loss correction and sampling variation; subtracting KL does not make it pure noise. Analyze KL and this contrast in the same gain coordinates.

For n documents, report squared sample mean, trace of sample gradient covariance divided by n, and their difference. The difference equals (||sum d_i||²−sum||d_i||²)/(n(n−1)), an unbiased estimator of squared population mean under independent identically distributed sampling. It may be negative and is not clipped. Independence/distribution assumptions are not proven by document hash separation alone. No stable nonzero population contrast is inferred from a single noisy estimator.

| Panel | Gradient | Squared sample mean | Estimated mean sampling variance | Unbiased squared-mean estimate |
|---|---|---:|---:|---:|
|H512|KL|1.6748812e-07|2.8581968e-08|1.3890615e-07|
|H512|CE_minus_KL|4.8365471e-05|2.3468735e-05|2.4896736e-05|
|H521|KL|3.2319398e-07|2.5650836e-08|2.9754314e-07|
|H521|CE_minus_KL|1.4622407e-05|2.5342695e-05|-1.0720288e-05|

Estimated sampling variance is about49% of the contrast's naive squared mean on H512 and173% on H521, versus about17% and8% for KL. The contrast estimate is positive on H512 but negative on H521. This does not prove a zero task signal; it shows that treating the observed mean norm as reliable evidence would be premature.

Independent-panel mean-gradient dot products provide a separate reproducibility check. KL dot product is2.13e−7, with a descriptive document-bootstrap95% range[8.67e−8,3.47e−7]. The label contrast is−7.27e−6, range[−4.66e−5,2.71e−5]. No consistent label-correction direction is established. Bootstrap distributions for the unbiased squared-mean estimates are archived, but are not used as calibrated zero-signal tests: a squared-mean statistic is nonregular near zero and a naive empirical bootstrap can be misleading. All panels are reused and these are diagnostics, not fresh inferential confirmation.

Gradient concentration alone does not explain the difference. The largest six document gradient norms contribute37–39% of total squared-gradient energy for KL but14–19% for the contrast. KL directions nevertheless reproduce more clearly. Maximum leave-one-out mean changes are10–16% of the mean norm for KL and11–31% for the contrast. No observations are removed. These are vector-energy concentration statistics, not shares of task loss or proof of outlier causality.

Correct the MSE overshoot with each source panel's exact line minimum along its prescribed negative-gradient direction. Alpha is computed from the source quadratic alone, not target outcomes:0.001412 for H512 and0.000340 for H521. All read/write directions stay frozen and rank remains at most36.

| Step derived on | Evaluated on | MSE change | KL change | CE change ± paired SE |
|---|---|---:|---:|---:|
|H512|H512|-0.374992|+0.000001523|-0.00004699 ± 0.00001766|
|H512|H521|+0.288330|+0.000002970|+0.00001049 ± 0.00001680|
|H521|H512|-0.039973|+0.000000546|-0.00000534 ± 0.00000438|
|H521|H521|-0.030468|+0.000000786|+0.00000230 ± 0.00000502|

Both source-panel MSE changes are negative, as required by the exact quadratic minima. They are tiny relative to baseline MSE around5,400: about0.007% and0.0006%. Transfer is mixed: the H512-derived step increases H521 MSE; the H521-derived step decreases H512 MSE slightly. KL rises in all four evaluations by about0.0000005–0.0000030. This establishes the measured local difference between objectives at these gains, but not a stable population frontier or substantial task-performance consequence. Own-panel step selection makes its MSE improvement expected, not independent evidence. CE effects remain small and uncertain, with no task-improvement claim.

Audits: input hashes checked; unbiased estimator equals squared sample mean minus estimated mean sampling variance; no negative estimates clipped; all archived arrays finite; exact MSE quadratic minima replay within7.48e−13 relative error. The original objective-gradient and endpoint audits are in H525. No new documents; total24,064.

Next H527: obtain2,048 fresh documents disjoint from every tracked panel, including the H522 source calibration. Freeze the existing rank36 read/write map and two unit directions before collecting new gradients: pooled H512/H521 mean KL gradient and pooled mean contrast g_CE−g_KL. Compute the same per-document gradients on the fresh panel. Prespecify two primary statistics: mean projection onto each corresponding frozen direction, with a positive value indicating reproduced local descent direction. Use4,000 document bootstrap resamples (seed20280109), fixed original SEs and centered max-absolute standardized deviations for a simultaneous95% two-statistic interval family. Report four sequential512-document block means and their dispersion descriptively, with no block selection. No gain optimization or step-size tuning. Reproducibility of a contrast direction would justify a task-loss optimization branch; failure would bound what this sample can resolve, not prove that CE and KL have identical population optima. Audit complete-document/window disjointness and update the unique total only after successful allocation; extend the local text cache if fewer than2,048 eligible unused documents remain, rather than recycling used documents. This tests whether the apparent CE-specific direction persists with more information.
