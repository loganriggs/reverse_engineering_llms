## H445 — Likely-token constraints and an interpolation-boundary failure

Reuse H444’s128 receiving contexts,24 source draws, learned rank64 correction span and four random rank64 spans. No new corpus documents or independent confirmation: total remains12480. Preserve the native union128 response and exact radial correction. Fit coefficients to the analytic linear response using baseline probabilities only, never the finite intervention target. Evaluate every condition with full-vocabulary finite KL.

Compare full vocabulary with top32/64/128/256/512 baseline tokens. Truncated weights are renormalized and derivatives recentered within the selected conditional distribution. Controls keep the top128 support but shuffle probability weights, or retain the sorted weights but assign them to128 random vocabulary identities. Thus truncation changes centering as well as support; its result cannot be attributed to token count alone.

| Constraint condition | Learned finite recovery | Random mean finite recovery | Learned local Fisher recovery | Random mean local Fisher recovery |
|---|---:|---:|---:|---:|
|full|98.493%|96.231%|98.707%|96.634%|
|top32|95.567%|93.512%|96.005%|94.228%|
|top64|44.927%|-262.272%|54.992%|-151.913%|
|top128|98.085%|94.949%|98.324%|95.403%|
|top256|98.390%|95.896%|98.598%|96.289%|
|top512|98.465%|96.129%|98.673%|96.518%|
|shuffle_weights128|96.857%|89.396%|97.281%|90.884%|
|random_identities128|93.546%|70.198%|94.654%|74.871%|

| Frozen contrast | Estimate | Joint95% interval | Required lower bound | Pass |
|---|---:|---|---:|---|
|top128_minus_full_learned|-0.004078|[-0.075417, +0.067261]|-0.02|False|
|top128_minus_full_random_mean|-0.012822|[-0.084160, +0.058517]|-0.02|False|
|top128_minus_random_identities_learned|+0.045387|[-0.025952, +0.116726]|+0.02|False|
|top128_minus_random_identities_random_mean|+0.247502|[+0.176163, +0.318841]|+0.02|True|

Only the random-span likely-token-versus-random-identity advantage passes the frozen gate:24.75 percentage points, joint95% interval17.62–31.88. Learned advantage is4.54 points with uncertainty including zero. Top128 loses0.41 points learned and1.28 points random on point estimates, but neither frozen noninferiority test passes. Bands use4000 paired source/receiving-context bootstraps, with resampling of four random spans and a joint maximum-absolute-deviation radius across all four contrasts. The identity contrast contributes to the wide common band; do not replace the frozen inference with a more favorable one.

Top512 point recovery is98.465% learned and96.129% random, versus98.493% and96.231% full. This is descriptive closeness, not a passed prospective equivalence test. Median captured baseline probability mass is85.01/89.76/93.57/95.89/97.37% for32/64/128/256/512 tokens; top128 mass ranges45.75% to99.9997%. Shuffling weights on the same top128 support reduces recovery to96.857% learned and89.396% random. Both support and weight assignment are plausible contributors; only the frozen random-identity comparison has the stated primary inference.

Unexpectedly, top64 performs much worse than top32 or top128:44.93% learned and−262.27% random finite recovery. Negative recovery means worse error than predicting no intervention effect. The local Fisher metric also deteriorates, so this is not solely finite-output curvature. Centering k token constraints limits their rank to k−1; k=64 therefore lies just below the64-coordinate correction dimension. Poorly constrained directions near this interpolation boundary are a leading hypothesis, not yet verified by spectra or coefficient norms. Top32 may implicitly suppress more directions while top128 overdetermines them. Ridge scaling and low-probability extrapolation are alternatives to distinguish. No failing cases are excluded.

Numerical audit passes: full-vocabulary replay max absolute error4e-17; maximum relative solve residual2.59e-14; all errors finite and frozen input hashes verified. Small solve residual certifies solving the regularized system, not good conditioning or good prediction.

Next discriminate ill-conditioning from irreducible missing-token information. Freeze top32/64/128 and ridge multipliers1e-6,1e-4,1e-2; keep full-vocabulary1e-6 as reference. Record eigenvalues, effective rank, coefficient/output norms, retained-token versus omitted-token local errors, and full finite KL. A stronger-ridge rescue of64-token failure would favor unstable extrapolation; persistent residual failure after norm stabilization would favor insufficient constraints. This is a mechanistic follow-up on reused data, not fresh confirmation.

Interpretation: a structured, input-derived predictor can often be fitted against a small set of likely outputs, and token identities matter beyond probability concentration. This does not identify semantic features or make the whole model low rank. The retained native factors, exact radial term and dense analytic response remain. The runner computes full tangent matrices before slicing; no runtime savings have been measured or claimed.
