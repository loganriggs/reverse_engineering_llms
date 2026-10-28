## H471 — Finite response-KL optimization at fixed coordinates

Freeze H470 augmented64 empirical bilinear input coordinates and output rank16. Start from joint16. Optimize finite native-target KL after adding predicted write changes to the native intact baseline, using both signed training interventions. Each of8 iterations computes a full-training gradient in balanced SVD factors, normalizes the first-order training response change by the norm of the current predicted response, and selects a fraction from[0,.0001,.0003,.001,.003,.01,.03,.1] by training KL. Rebalance the resulting rank16 map by SVD, then select the best iteration on validation response KL, including initialization. The natural-state intercept is anchored to its original training mean. No test selection, no new corpus documents; old splits/source assignments remain fixed.

Validation selects iteration8, the last tested point. Mean training response KL falls.000804032→.000443443 (about44.8% reduction); validation falls.001353586→.001145707 (about15.4%). Training and validation improvement are not an optimization-convergence certificate. Selecting the final tested point specifically leaves longer optimization as a live explanation.

| Map | Negative delta KL / raw recovery | Positive delta KL / raw recovery | Intact native-output KL |
|---|---|---|---:|
|joint16|54.26% / 93.23%|55.46% / 93.03%|0.142229|
|response16|57.44% / 94.14%|61.59% / 93.33%|0.286429|
|kl16|59.67% / 93.13%|59.97% / 92.90%|0.144965|

| Frozen primary statistic | Estimate | Joint95% interval | Pass |
|---|---:|---|---|
|kl16_minus_joint16_deltaKL_negative|+0.054149|[-0.005891, +0.114189]|False|
|kl16_minus_joint16_deltaKL_positive|+0.045099|[-0.014336, +0.104534]|False|
|kl16_minus_joint16_intactKL|+0.002736|[-0.002281, +0.007753]|True|
|kl16_absolute_deltaKL_negative|+0.596743|[+0.369032, +0.824454]|False|
|kl16_absolute_deltaKL_positive|+0.599727|[+0.425990, +0.773465]|False|

The gain gates require>.10 at each sign; actual gains5.415/4.510pp have joint95% intervals[−.589,11.419]/[−1.434,10.453]pp. These fail and include zero, so the test gain is not established. Intact-KL increase.002736 has interval[−.002281,+.007753], passing the predeclared.01 practical-loss gate relative to joint16. This does not establish equivalence to the native model: joint16 already has substantial baseline KL error. Absolute response recovery59.67/59.97% has intervals[36.90,82.45]/[42.60,77.35]%, failing both80% lower-bound gates.4000 paired test-document bootstrap replicates give joint5 standardized bands, conditional on fitted calibration, validation selection and fixed donor assignment. There is no fresh corpus confirmation.

Raw response recovery slightly decreases from joint16's93.23/93.03% to93.13/92.90% while probability recovery point estimates improve. This is consistent with different objectives valuing different errors, but the uncertain test improvement prevents a stronger claim. The response-only write fit remains competitive at57.44/61.59% KL recovery, so this short KL trajectory is not an established superior method. Mean next-token CE and state reconstruction are saved as cross-objective diagnostics, not selection criteria. No CE optimum or semantic coordinate interpretation is claimed.

The native intact baseline remains an evaluation/training device for isolating predicted response changes. The deployable full-state map uses its learned intercept and coefficients; it does not have native retained-factor bypasses. Its context-conditional finite-KL fit is only one member of the data-supported equivalence hierarchy proposed by the user.

Update to the hypothesis shortlist: wrong output weighting can be reduced on training data without changing input coordinates, but this run does not show that it explains the remaining test gap. Input-coordinate limitations, incomplete optimization and generalization remain distinct possibilities. The last selected iterate rules out treating eight steps as a solved optimum. Next compare input ranks64/128 with the same output rank16 and both8/32-step budgets, fixing training/validation selection rules before outcomes. Report the interaction rather than attributing any width gain to representation without a longer64-rank control. Shared kernel support must remain fixed and complete storage counted. A later new-document panel is required to confirm whichever alternative this reused-data diagnostic selects. The separate source-rank1/tail and observed-token CE questions remain pending.

All saved arrays finite; joint16 baseline replays H470 within1e−4. Maps and iteration histories are frozen and hashed. Total12992 corpus documents; broad research goal remains active.
