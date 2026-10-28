## H469 — Empirical measure versus fitting distribution

Factorialize two choices at fixed64 bilinear input coordinates and rank16 output map: choose the coordinate measure from ordinary training states or ordinary plus intervened states; fit the write map on ordinary states or that augmented set. H466256 training and128 test contexts are frozen. Augmentation adds−1/+1 centered head6 source interventions, cycling the first32 donor vectors across training contexts. Test uses the previous64 donor assignments, permitting descriptive trained/untrained source-index strata. All target writes use native weights; no test data enters coordinate selection or fitting. Output objective stays write MSE, and no validation selection occurs.

Each fit uses its own centered feature Gram matrix to compute exact reduced-rank least squares, dropping eigenvalues below1e−10 of maximum. All four retain full64 fit-Gram rank. Natural reference support has256 vectors; augmented support has768. Thus latent ranks match but total storage does not: support alone costs294912 versus884736 scalars, and kernel encoders16384 versus49152. Each output rank16 map could be stored in19456 factor coefficients, plus its intercept. Current files also store redundant effective maps for auditing. No total-budget equality claim is made.

Repeating the same natural observations three times preserves their empirical probability measure and regression objective. The nonzero normalized centered-kernel spectra are numerically unchanged within1e−10. This rules out duplicated row count as the explanation; it does not replace a genuinely enlarged independent-natural-sample control, which has not been run.

| Coordinate measure / fit data | Intact KL | Negative delta KL / raw recovery | Positive delta KL / raw recovery |
|---|---:|---|---|
|natural_natural|0.125486|-34.07% / 63.74%|-29.38% / 60.78%|
|natural_augmented|0.125657|-24.45% / 67.52%|-20.48% / 64.92%|
|augmented_natural|0.125274|-32.22% / 65.56%|-27.58% / 62.81%|
|augmented_augmented|0.125443|-20.75% / 69.90%|-16.84% / 67.56%|

| Frozen comparison: both augmented versus both natural | Estimate | Joint95% interval | Pass |
|---|---:|---|---|
|augmented_both_minus_natural_both_deltaKL_negative|+0.133119|[+0.049293, +0.216945]|False|
|augmented_both_minus_natural_both_deltaKL_positive|+0.125344|[+0.061984, +0.188704]|False|
|augmented_both_minus_natural_both_intactKL|-0.000043|[-0.000885, +0.000799]|True|

The first two gates require improvement>.10 in delta-KL recovery at each sign. Both fail: gains13.312/12.534pp have joint95% intervals[4.929,21.694]/[6.198,18.870]pp. Both intervals exclude zero, supporting some improvement while leaving the prespecified minimum unproven. The intact-KL increase gate upper<.01 passes: change−.0000428, interval[−.0008851,+.0007994]. The bands use4000 paired test-document bootstrap replicates and a joint3 standardized maximum deviation; fitted calibration and deterministic donor assignment remain fixed. This is a reused research panel, not fresh corpus confirmation.

Absolute delta-KL recovery remains negative on point estimates for all four models. Both-augmented reaches−20.75%/−16.84%, with marginal intervals[−99.40,+26.43]%/[−73.56,+22.69]%. These broad intervals do not establish significantly worse performance than zero change, but provide no high response coverage. Raw-write delta recovery increases to69.90%/67.56%. Fitting augmented outputs alone helps more than selecting augmented coordinates alone on these point estimates; these individual factorial contrasts were not separately confirmed. Both-augmented untrained-source-index KL recoveries remain negative,−32.89%/−21.74%; small strata and fixed assignments prohibit a strong source-generalization claim.

Interpretation: exposing the canonical representation to intervened states helps without material intact-KL damage, but does not solve response faithfulness. The distinction between state fitting and contrast fitting is now important. Augmented-state MSE still emphasizes large whole writes; it need not emphasize a comparatively small difference between neighboring states. This is a mechanistic hypothesis, not something established merely by the failure of this loss.

Next discriminator: hold the augmented input coordinates fixed and fit explicitly to native write differences ΔY using coordinate differences ΔZ. Compare joint state/response fitting with response-only fitting at the same output rank16; include an unrestricted rank64 response-fit capacity witness to identify output-rank versus input-span limitations. Normalize state and response losses by their respective training target energies before choosing fixed weights, and report natural-state quality alongside response predictions. Training and test donors/context roles stay documented. This isolates the preserved behavior in the proposed equivalence relation before changing the kernel or adding more optimization heuristics.

Numerical gates pass: normalized repeated-kernel spectrum invariance and natural/natural replay of H468. All arrays finite; full saved model coefficients, intercepts, supports, errors and bootstrap samples retained. Total12992 corpus documents; broad goal remains active.
