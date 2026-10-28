## H447 — Baseline conditioning rule on unused interventions

Test a frozen rule for stabilizing token-constrained coefficient fits. Select128 receiving documents from the352 remaining in H438, excluding the32 H438 intervention sites and128 H444 sites. Baselines were previously collected, interventions unused. Use12 new natural donor draws plus12 Gaussian draws and eight new random rank64 output spans. Learned span and source-direction maps remain fixed. Total corpus documents12480, not128 newly sampled corpus documents; the source donor pool remains fixed.

At each baseline context, compute the64x64 within-span Gram matrix for full vocabulary or top48/64/80/128 tokens. Center within the conditional token distribution. Tiny ridge is1e-6 times mean eigenvalue. The frozen adaptive rule sets lambda=max(tiny,(largest_eigenvalue−1000*smallest_eigenvalue)/999), limiting the regularized condition number to1000. This uses baseline geometry only, never finite intervention outcomes. Native union128 and exact radial corrections remain in every prediction.

| Tokens | Rule | Learned finite KL recovery | Random mean recovery | Maximum learned norm ratio | Maximum random norm ratio |
|---|---|---:|---:|---:|---:|
|full|tiny|98.470%|96.447%|1.612|1.205|
|48|tiny|95.303%|91.859%|2.332|2.876|
|48|condition|96.359%|93.977%|1.366|1.941|
|64|tiny|35.302%|-106.762%|25.891|55.780|
|64|condition|96.886%|93.432%|1.598|2.164|
|80|tiny|95.742%|90.738%|4.038|3.440|
|80|condition|97.249%|93.512%|1.764|2.590|
|128|tiny|97.964%|95.242%|2.169|1.858|
|128|condition|97.932%|95.294%|1.785|1.487|

| Frozen contrast | Estimate | Joint95% interval | Required lower bound | Pass |
|---|---:|---|---:|---|
|condition_rescue_learned|+0.615836|[-0.278464, +1.510136]|+0.02|False|
|condition_rescue_random_mean|+2.001940|[+1.107640, +2.896239]|+0.02|True|
|stabilized64_minus_full_learned|-0.015845|[-0.910145, +0.878455]|-0.02|False|
|stabilized64_minus_full_random_mean|-0.030155|[-0.924455, +0.864145]|-0.02|False|

Use4000 paired source/context bootstrap replicates with resampling of the eight random spans and a joint max-absolute-deviation band for four frozen comparisons. No failed gate is relabeled; simultaneous uncertainty can be dominated by unstable tiny-ridge outcomes. Transfer is conditional on this model, existing corpus panel and fixed donor pool.

Audit: relative solve residual2.75e-14; all arrays finite, condition cap verified, and128 unique recipients disjoint from all160 excluded intervention cases. Input hashes checked by runner. Adaptive condition-number maximum1000.000000. No outcome exclusions.

The baseline-only condition rule transfers its stabilization effect to previously unused receiving interventions. At64 tokens, learned recovery rises35.30%→96.89%, randommean−106.76%→93.43%. Maximum correction norm ratios fall25.89→1.60 learned and55.78→2.16 random. The random-mean rescue is200.19 percentage points, joint95% interval110.76–289.62, passing the >2-point gate. Recovery improvements can exceed100 points because the tiny-ridge reference is negative. Learned rescue61.58 points remains unresolved under the same wide joint band; do not claim both pass.

Both full-vocabulary noninferiority gates fail. Point deficits after stabilization remain1.58 points learned and3.02 points random. Full recovery is98.47%/96.45%, stabilized128 gives97.93%/95.29%. The stable64 condition’s omitted-token Fisher error fraction is2.47% learned/5.79% random, versus included-token0.62%/0.71%. This supports two separable issues: unstable extrapolation causes the sharp64-token collapse, and omitted output constraints remain relevant after stabilization. The latter is a mechanistic lead, not proof of irreducible missing information or a formal noninferiority failure demonstrating difference.

The main scientific update is that a context-dependent numerical rule, without using intervention outcomes, predicts and prevents a major apparent feature-capacity failure. That strengthens the account of output compactness as an interaction among source response, context, probability geometry and coefficient computation. It does not establish a semantic ontology or show random spans are as good as learned spans.

Next discriminate tail-information loss from residual regularization bias at a fixed stable token budget. Keep128 token constraints and the condition rule fixed, then add the omitted tokens’ baseline metric/analytic-response contribution either exactly, with a shared calibration approximation, or with token identities shuffled while preserving weights. Compare included- and omitted-token errors and full finite KL. Correctly include between-group centering terms when recombining conditional metrics; a naive sum would change the objective. This asks what reusable information the coefficient rule needs, rather than continuing to tune ridge solely for higher scores. Dense analytic target construction remains an unresolved computational cost, and broader layers/semantic interventions remain outside current confirmation scope.

This is a test of a coefficient-computation rule, not identification of64 semantic units or a whole-model compression. Dense analytic response construction, native retained factors, exact radial write and baseline forward pass remain. Full tangent matrices are evaluated before token slicing, so no runtime saving is claimed.
