## H449 — Shared tail geometry from disjoint baseline contexts

H448 attributed most of the remaining top128 gap to within-tail variation. H449 estimates one shared tail covariance from128 H444 baseline states, disjoint from128 H447 receiving states. The evaluation interventions are reused, so this is a disjoint-baseline calibration diagnostic, not a new independent intervention confirmation. Total corpus documents remains12480. Learned span, eight random spans,24 sources, native union128 and exact radial terms are unchanged.

For each calibration context, exclude the128 most likely baseline tokens and draw128 tokens with replacement from the conditional tail distribution. Form token gradients of capped logits with respect to normalized output state, centered by the exact conditional tail mean. Their sampled second moment estimates the conditional covariance; average over128 contexts. The covariance includes the factor1/2 used in the local Fisher score. No intervention targets or outcomes enter calibration.

At each receiving context, match the shared covariance trace to the exact conditional tail covariance trace, then transport it through the receiving output RMS derivative. Compute both basis Gram matrix and analytic-response cross-covariance using the same surrogate. Keep the exact head covariance, tail mass and between-group mean contrast. Compare isotropic covariance and an orthogonally rotated shared covariance, using a frozen signed coordinate permutation that preserves eigenvalues and trace. All fits use the same baseline condition-cap1000 rule.

| Condition | Learned finite KL recovery | Random mean finite KL recovery | Learned included / omitted local error | Random included / omitted local error |
|---|---:|---:|---|---|
|top128_condition|97.932%|95.294%|0.007523 / 0.012954|0.011993 / 0.034965|
|full_condition|98.440%|96.446%|0.007718 / 0.007008|0.014332 / 0.020566|
|top_plus_between|98.067%|95.345%|0.007416 / 0.011625|0.011986 / 0.034345|
|shared_tail|98.410%|96.338%|0.007773 / 0.007259|0.014494 / 0.021470|
|isotropic_tail|98.414%|96.294%|0.007810 / 0.007163|0.014597 / 0.021732|
|rotated_shared_tail|98.405%|96.242%|0.007804 / 0.007254|0.014749 / 0.022102|

| Frozen contrast | Estimate | Joint95% interval | Required lower | Pass |
|---|---:|---|---:|---|
|shared_minus_full_learned|-0.000308|[-0.000580, -0.000036]|-0.005|True|
|shared_minus_full_random_mean|-0.001083|[-0.001354, -0.000811]|-0.005|True|
|shared_minus_isotropic_learned|-0.000041|[-0.000313, +0.000231]|+0.005|False|
|shared_minus_isotropic_random_mean|+0.000441|[+0.000169, +0.000713]|+0.005|False|

Primary margins are0.5pp: shared-versus-full noninferiority and shared-versus-isotropic advantage, each for learned and random-mean spans. Joint95% max-absolute-deviation bootstrap uses4000 paired source/document draws with eight-span resampling. Calibration-sample uncertainty is not resampled; conclusions are conditional on the frozen covariance estimate. Failed noninferiority is not proof of a difference.

Audits: exact covariance decomposition relative error2.74e-15; top/full/between replay absolute error0; solve residual3.54e-15. All numerical gates pass, saved arrays finite, and calibration/evaluation contexts are disjoint. No outcome exclusions.

Both shared-versus-full noninferiority gates pass at the frozen0.5pp margin. Shared recovery is98.410% learned and96.338% randommean, versus98.440%/96.446% exact full. Differences are−0.0308pp (joint95%[−0.0580,−0.0036]) and−0.1083pp ([−0.1354,−0.0811]). Thus the shared approximation is detectably imperfect under these conditional intervals but comfortably within the prespecified practical margin. This is not equality, and calibration uncertainty is not included.

Neither shared-over-isotropic practical advantage gate passes. Shared minus isotropic is−0.0041pp learned (joint95%[−0.0313,+0.0231]) and+0.0441pp random ([+0.0169,+0.0713]). The small random advantage is distinguishable from zero but far below0.5pp. Isotropic point recovery98.414%/96.294% nearly matches shared; its own noninferiority was not a frozen primary test. Rotating shared covariance gives98.405%/96.242%, again descriptively close. These results weaken a claim that tail-specific anisotropic directions are necessary for this predictor’s quality at the tested tolerance.

A simpler account now leads: broadly distributed within-tail constraints regularize corrections usefully, while the exact likely-token fit, receiving RMS geometry, tail strength and between-group contrast carry much of the remaining contextual information. Isotropic in normalized-output coordinates is not isotropic after transport through the receiving RMS derivative. Nor is this plain ridge: coherent surrogate target cross-covariance pulls the projected analytic response toward its actual normalized-output geometry. The experiment does not show the tail is semantically meaningless or that all output geometry is generic.

Next freeze a factorial using the isotropic surrogate: receiving-specific versus calibration-constant tail covariance trace, crossed with exact versus omitted between-group term. Keep top128 baseline constraints, tail mass, output normalization and condition rule unchanged. This tests whether a reusable scalar is enough and whether exact tail means remain necessary after generic within-tail correction. Confirm on still-unused receiving interventions before claiming simplification transfers. It directly targets the information required by a computational rule; covariance storage can be eliminated only if the isotropic variant is confirmed. Dense analytic-response construction and model-wide/semantic interpretation remain unresolved.

Cost/scope: shared covariance stores1152x1152=1,327,104 scalars (about10.6MB in float64), excluding model/maps and other predictor components. Baseline probabilities, top/tail identity selection, exact tail trace and between-group means remain context-specific. The implementation also evaluates the exact tail for controls. No speedup or fully shared coefficient rule is established. This concerns output constraints in a structured intervention predictor, not a semantic feature ontology.
