## H443 — Context-conditioned projection helps learned and random correction spans

H442 left fixed-span capacity versus fixed-projection failure unresolved. H443 fixes the selected learned rank64 decoder span and a seeded rank64 random orthonormal output span. At each receiving context, solve a regularized full baseline-Fisher projection of the analytic shared16 linear-map response. The right-hand side comes from baseline state, source and weights, not the full finite target. Retain the exact radial correction and exact native union128 response in every condition. Thus random-span results concern only a correction component within a structured predictor, not a wholly random model or64 random features explaining the model.

Primary context solves use ridge1e-6 times mean within-span Fisher eigenvalue; ridge.01 is a frozen secondary stability control. Learned fixed projection reuses H442; random fixed projection is ordinary Euclidean projection. All coefficient choices precede finite target scoring. This is an expensive context-specific diagnostic on reused H438 data, not compiled acceleration or new independent confirmation. No new documents; total12480.

| Method | Low-RMS full finite-KL recovery | Median-RMS full finite-KL recovery |
|---|---:|---:|
|learned_fixed|93.268%|97.160%|
|learned_context|97.329%|98.427%|
|random_fixed|85.443%|83.808%|
|random_context|95.784%|94.098%|
|learned_context_strong_ridge|97.389%|98.370%|
|random_context_strong_ridge|95.851%|94.092%|

| Frozen contrast | Estimate | Joint95% interval | Lower > .02 |
|---|---:|---|---|
|context_minus_fixed_low|+0.040609|[-0.028341, +0.109559]|False|
|context_minus_fixed_median|+0.012668|[-0.056282, +0.081619]|False|
|learned_minus_random_context_low|+0.015455|[-0.053495, +0.084406]|False|
|learned_minus_random_context_median|+0.043294|[-0.025657, +0.112244]|False|

All four gates fail, and all difference intervals include zero. The point gain from contextual learned coefficients is4.06 points low/1.27 points median; the learned-over-random contextual gap is1.55/4.33 points. These are useful mechanistic estimates, not established advantages. Joint bands resample12 paired source draws and selected contexts, conditional on fixed calibration and one random span; random-span population variability is not represented.

Random-span finite recovery improves from85.44%/83.81% with fixed coefficients to95.78%/94.10% with contextual coefficients. The important control is that context-sensitive coefficient computation can produce a large gain even in an unlearned correction span. It does not show equality between learned and random spans, nor negate semantic circuits elsewhere. The native retained factors and radial term remain important structured components.

Local Fisher and finite-KL scores are close in this test: contextual learned local recovery97.33%/98.50% versus finite97.33%/98.43%; contextual random local95.98%/94.24% versus finite95.78%/94.10%. Tiny-ridge projected linear-vector norm ratios stay below1.49 of the analytic linear term; stronger ridge limits them to about1.10 learned/1.05 random without materially improving finite recovery. This tested set does not exhibit the catastrophic finite failure that can occur with ill-conditioned local fits.

Numerical gates pass: relative linear-solve residual1.28e-15 below1e-8; fixed learned scores replay H442 within1.73e-17 absolute error. Conditioning spectra and prediction norm ratios are saved. Full vocabulary probabilities define the metric, so the prior sampled-metric approximation is not used in the contextual solve.

Cost and interpretation: evaluating full baseline metric images of64 basis vectors and solving a64x64 system per context is substantial work. The analytic linear-map responses are also available to the solve. This improves the diagnosis of a fixed span’s capacity but does not yet provide a cheap predictive unit. Small apparent output dimensionality must be interpreted alongside adaptive coefficient computation, metric geometry, structured baseline components and storage/construction costs.

Next test transfer of the frozen projection rule on receiving contexts not used in this diagnostic, with several predeclared random spans and fresh sources. Preserve the learned basis, ridge choices, exact radial term and retained factors. Report continuous paired uncertainty rather than relabeling the current failed gates. Include radial-term ablation or separate accounting so high recovery is not attributed entirely to the random correction span. If the contextual rule generalizes, only then try a cheap baseline-only approximation to its solve; fitting coefficients to finite targets would return to an oracle rather than a computational explanation.
