## H683 — Longer training preserves the reconstruction ranking

Extend all36 H682 fits from1000 to5000 steps with identical model family, data, seeds, minibatch streams, learning rate and validation cadence. Since the pilot did not save optimizer state, reproduce training from initialization, explicitly compare all weights at step1000, then continue. Save selected weights and separate final-step model/optimizer/minibatch-generator state for exact future resumption. No test examples evaluated.

| Site | Block width | Mean validation relative MSE | Mean training relative MSE | Best steps | Mean median effective rank |
|---|---:|---:|---:|---|---:|
|value0|1|0.2314|0.1401|[5000, 4250, 5000]|1.00|
|value0|4|0.2618|0.1689|[5000, 4500, 5000]|2.91|
|value0|16|0.2725|0.1805|[4500, 4500, 4250]|5.92|
|residual0|1|0.2217|0.1668|[5000, 5000, 5000]|1.00|
|residual0|4|0.2357|0.1872|[5000, 5000, 5000]|2.80|
|residual0|16|0.2621|0.2200|[5000, 5000, 4750]|7.64|
|residual1|1|0.2789|0.2018|[5000, 5000, 5000]|1.00|
|residual1|4|0.3047|0.2313|[5000, 4750, 5000]|3.00|
|residual1|16|0.3345|0.2694|[4750, 4750, 5000]|8.32|
|residual2|1|0.3125|0.2303|[5000, 5000, 5000]|1.00|
|residual2|4|0.3508|0.2684|[5000, 5000, 5000]|3.16|
|residual2|16|0.3937|0.3151|[5000, 4750, 4750]|9.10|

The signed one-dimensional baseline still reconstructs better than widths4 and16 at every activation site. More training improves all families, so the initial pilot was undertrained; the current ranking is not solely a1000-step artifact. It remains specific to this architecture and fixed scalar-sparsity budget. Width16 activates only two groups, and grouped sparsity has different support flexibility than32 independently selected scalar features.

Some selected checkpoints occur before the final step, particularly for the value stream; other runs still select5000. This is not a blanket convergence claim. The next useful question is whether the learned block structures recur across seeds and have consistent activation support, rather than repeatedly extending optimization solely to improve reconstruction. Higher-dimensional code covariance alone does not establish a manifold.

The planned next audit matches learned decoder subspaces across seeds using squared principal-angle overlap with one-to-one assignment, which permits within-block rotations. Compare matched activity support on validation and an independently matched signed-coordinate-scramble reference. A stable subspace with unstable support is weaker evidence than both recurring together. Neither score by itself establishes semantic purity, smooth geometry or causal usefulness. Test remains sealed until fitting/evaluation protocol is frozen.

Audits:36models with5000steps each; pilot1000-step maximum weight replay difference0. All source hashes unchanged, finite results. Zero test evaluations, zero original-model forwards, no new documents. Checkpoints distinguish selected validation-best weights from final optimizer-resume state. Prior turn experimental progress; broad goal active and incomplete.
