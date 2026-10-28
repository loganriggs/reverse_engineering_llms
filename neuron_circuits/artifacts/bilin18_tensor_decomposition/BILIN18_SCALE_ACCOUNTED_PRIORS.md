## H941 — Scale-accounted hull pilot finds no recovery advantage and exposes optimization confounding

Train42 four-product bilinear students on the SAME512 native states and independently specified interventions fromH939. The objective is native-output plus finite-effect MSE, with a common normalization; no extra output-weight penalty. Compare free unit factors, native-scale convex state factors, native-scale convex intervention-difference factors, and normalized-state-hull factors (explicitly an optimization control becauseH940 proved all directions feasible). Hull factors are capped at unitnorm by shrinkage, valid because the audited hull contains0. Anchor banks are each rescaled to unit RMS anchor norm; scales saved. These choices define a scale-accounted comparison, not an A-SAE replication.

Cross two independent128-anchor banks with two random initializations and three output-matrix Frobenius caps2/32/2048. Free factors have no anchor bank and are trained once per initialization/cap.1500 Adam steps, same minibatches per seed. Teacher rank4 is supplied, true axes are evaluation-only. Hull methods have more trainable parameters than free factors. Anchor randomness is independent of initialization, but effective factor initializations are NOT matched across parameterizations.

| Method | Decorrelated effect error, gain cap2 | Error, cap32 or2048 |
|---|---:|---:|
| Free unit factors |~0–47.5%|~0–39.8%|
| State hull |86.9–87.6%|86.2–90.2%|
| Difference hull |90.6–96.5%|38.2–61.9%|
| Normalized state hull |85.7–87.4%|85.7–87.4%|

Ranges retain all seeds/anchor banks; they are not uncertainty intervals. At cap32, observed gains are only4.6–5.2 for state hull and9.1–11.1 for differences, so cap2048 gives identical trajectories: increasing a nonbinding cap does not repair optimization. Difference factors improve substantially relative state factors at loose caps, but do not consistently match the successful free solution. The normalized state-hull control remains poor despite full directional expressivity. Its failure cannot be attributed to exclusion of the true axes.

Free factor seed0 recovers the exact teacher to displayed precision, while seed1 retains a substantial error/local solution. This instability at the same true rank also prevents interpreting a best seed as reliable recovery. Teacher-pair cosines at cap32 are free .741–1.000, state .518–.709, differences .690–.968, normalizedstate .532–.715. All outputs and curves retained; no validation-based selection or restarted failures.

The pilot gives no evidence to promote a hull prior. It ALSO does not show that such a prior intrinsically excludes useful components: scale-accounted functional realizability, optimization, and starting states differ. A good state-hull solution may require much larger gain than reached here, while normalized factors can represent it without gain expansion but are still hard to optimize. We must not convert this finite-budget result into an architecture impossibility claim or attribute it to manifold semantics.

Verification: independent simplex sums8.88e-16, factor-formula replay exact, finite-difference input-gradient error1.53e-11, saved-checkpoint metric replay1.11e-16. Finalfactor norm<=1+2.3e-16; output cap excess<9e-8. All42 fits finite, hashes unchanged. No original-model or reserved-test evaluations; natural-document pool unchanged.

Bounded next action: one matched-functional-initialization control for free versus normalized hull will isolate the most obvious parameterization confound without pretending the latter is a directional restriction. Use convex witnesses for identical initial unit factors and output weights, both anchor banks/seeds, and do not optimize against evaluation outcomes. Then move the independent-intervention objective to an established real-model site, retaining native/endpoint/effect-loss controls and known mechanisms as explicitly labeled training or evaluation anchors. Do not spend an open-ended sequence of toy tuning runs trying to manufacture a hull advantage.
