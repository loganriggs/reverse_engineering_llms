

## H374 — Oracle coefficients reveal substantial recoverable information and a local-to-finite failure mode

Reuse H372’s512-document native responses and H373’s frozen probability-defined rank16/64 spans. For each recipient and response, solve the local probability/cap Fisher least-squares normal equations for coefficients in the same span. Eigenvalue cutoff is fixed at1e-12 relative to the maximum Gram eigenvalue; all16/64 directions remain retained everywhere. No finite-KL optimization or target-response basis fitting. Evaluate the resulting post-RMS patches directly through unembedding/cap without another normalization.

This oracle requires the complete native target response and recipient metric. It diagnoses information available within a span; it is not a compressed executable MLP or a learned coefficient predictor. Optimality applies to the local Fisher objective, not finite KL.

|Head|Family|Euclidean64 finite|Oracle16 local|Oracle16 finite|Oracle64 local|Oracle64 finite|
|---|---|---:|---:|---:|---:|---:|
|0|actual|0.643|0.741|0.741|0.862|0.861|
|0|moment|0.404|0.563|0.564|0.746|0.747|
|0|donor|0.411|0.573|0.573|0.753|0.753|
|1|actual|0.458|0.648|0.653|0.799|0.799|
|1|moment|0.407|0.613|0.618|0.782|0.784|
|1|donor|0.414|0.616|0.621|0.784|0.786|
|6|actual|0.307|0.682|0.687|0.825|0.824|
|6|moment|-0.084|0.670|-0.832|0.834|0.837|
|6|donor|-0.072|0.676|-1.164|0.837|0.839|

All frozen gates fail: {'oracle64_local_sufficient': False, 'oracle64_finite_sufficient': False, 'oracle64_finite_gain': False}. Oracle64 finite recovery spans74.7–86.1%, a21.8–92.1 percentage-point improvement over Euclidean64. All nine paired improvement lower bounds exceed zero, but not the frozen20pp universal margin. Oracle64 local recovery is also below80% in several point estimates; universal80% sufficiency is not established.

The finite joint36 recovery band has half-width approximately1.460 because it includes unstable oracle16 results. Consequently its oracle64 lower bounds are very loose. Preserve this predeclared band and failed gate; do not replace it post hoc with a narrower oracle64-only band. The paired gain has a separately prespecified joint9 band, whose positive lower bounds support improvement without proving the failed20pp margin. Local joint36 bands are much tighter. Figure shows point estimates and local/finite comparison, not confidence intervals.

Local/finite discrepancy: head6 control oracle16 has local recovery67.0–67.6%, but finite recovery−83.2%/−116.4%. An oracle minimizing local error can move into a poor finite distribution. Rank64 mostly avoids this aggregate failure, reaching83.7–83.9% finite recovery for these controls. Thus a useful local span and good local coefficients alone do not establish a faithful finite computation.

Post-outcome tail diagnostic (descriptive; no exclusions): document251 contributes20.3%/27.9% of head6 moment/donor oracle16 KL, and the top five documents contribute54.6%/58.7%. Both families’ worst site is1007 (document251, position index3): oracle16 target-to-patch KL is5.40/5.53 nats, versus baseline0.000150/0.000151 and oracle640.000725/0.000811. Oracle16 activation-delta norm is only0.995/0.972 times the full target delta norm there, so this is not explained simply by a huge Euclidean edit norm. Rank64 is still worse than baseline at these particular sites despite strong aggregate recovery. The mechanism of the tail failure remains untested.

Audits: {'euclidean_h373_replay_relative': 3.463995791336576e-16, 'max_local_suboptimality_relative_to_baseline': 0.0, 'max_relative_normal_equation_residual': 7.371160293016838e-13, 'minimum_retained_gram_ranks': [16, 64], 'oracle_norm_ratio_quantiles': [0.23100836500400254, 0.6812593729155156, 0.9390621942957862, 2.496984323242528]}. Euclidean finite/local scores replay H373 to3.5e-16 relative error; normal-equation residual below7.4e-13; local oracle errors never exceed Euclidean errors within the frozen tolerance; no Gram direction was dropped. All KL values finite and nonnegative within tolerance. Oracle edit-norm/target-edit-norm quantiles across ranks/responses/sites are0.231 median,0.681 p95,0.939 p99,2.497 maximum. Save coefficients, per-site finite/local errors, norm ratios and bootstrap arrays. Native weights unchanged.

Uncertainty:4,000 shared document/control-draw bootstrap resamples on the reused panel. Separate joint36 finite/local bands and joint9 finite-gain bands; discovery geometry and empirical donor assignments fixed. No independent confirmation of the oracle findings yet.

Hypothesis update: poor Euclidean coefficient choice explains a substantial part of prior projection failure; missing span remains material because the rank64 local oracle is not universally80% sufficient. A third issue is local-to-finite instability, concentrated in tail examples for rank16. None of these results identifies semantic units or supplies coefficients without the native response.

Next: inspect the worst rank16 sites and a prespecified amplitude sweep, tracing probability mass and logit changes for tokens initially assigned low probability. Compare the local oracle with finite-KL fitting or a trust-region constraint within the same span. This separates low-probability extrapolation, logit-cap effects and intrinsic span limits while retaining all tail examples in primary evaluation. Broader goal remains active.
