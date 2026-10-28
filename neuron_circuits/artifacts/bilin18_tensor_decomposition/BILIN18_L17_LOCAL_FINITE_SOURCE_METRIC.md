## H502 — Local output sensitivity explains most of the read/write advantage

Keep the H500 donor/H499 receiver pairs and all three rank-4 source maps fixed: PCA, orthogonal projection into the Fisher write span, and the original oblique Fisher map. For each actual receiver/source pair, differentiate the MLP and final probability readout at the intact baseline. Compute one-half the probability-weighted squared centered-logit tangent for the full source and each reconstruction error. This is the leading quadratic term of KL. Divide error by effect to predict local recovery, then compare with native finite KL at prescribed intervention magnitudes 0.1, 1, and 2, both signs and all nine heads.

The residual bypass remains fixed, matching prior experiments. Derivatives include input RMS normalization, bilinear MLP, final RMS normalization, vocabulary projection, and softcap. The local prediction is sign-independent and scales quadratically with magnitude; finite behavior can depend on sign. The experiment uses local derivatives on the actual evaluation pairs as a diagnostic, not to refit any map. It is exploratory on reused data.

| Head | Local PCA recovery | Local orthogonal-write recovery | Local oblique recovery | Largest finite/local recovery gap at 0.1 / 1 / 2 (pp) |
|---|---:|---:|---:|---|
|0|24.82%|21.54%|24.08%|0.35 / 3.85 / 8.05|
|1|78.90%|78.04%|82.04%|0.08 / 0.85 / 1.78|
|2|38.09%|37.49%|50.93%|0.34 / 3.90 / 7.69|
|3|34.32%|52.19%|66.93%|0.19 / 2.05 / 4.72|
|4|82.98%|82.65%|82.46%|0.04 / 0.47 / 0.84|
|5|21.36%|41.88%|57.82%|0.09 / 0.85 / 1.68|
|6|97.82%|98.31%|98.58%|0.02 / 0.22 / 0.40|
|7|39.32%|43.90%|51.38%|0.04 / 0.46 / 1.00|
|8|69.66%|74.27%|82.56%|0.02 / 0.22 / 0.52|

At magnitude 0.1, finite recovery differs from the local prediction by at most 0.35 percentage points across all heads, methods, and signs; the median difference is 0.06 points. All 18 signed head cases have the same method ordering as the local prediction. Aggregate finite KL divided by its local quadratic prediction ranges from 0.9942 to 1.0064 across effect and method-error terms. Thus the small-scale regime is consistent with the implemented local approximation.

At magnitude 1, the median recovery discrepancy is 0.57 points and the maximum is 3.90 points. At magnitude 2, these become 0.96 and 8.05 points. The largest gaps occur in heads 0 and 2. Aggregate finite/local KL ratios span 0.9465–1.0970 at unit scale and 0.9035–1.2605 at double scale. These pooled quantities do not bound every individual case.

For head 5, local recovery is 21.36% / 41.88% / 57.82% for PCA / orthogonal-write / oblique, closely matching H501's finite values of about 21–22% / 41–43% / 58%. Head 3's local values similarly predict its large two-stage improvement. The strong read/write advantage therefore does not require finite-intervention nonlinearity to explain it in these cases; it is already present in local probability sensitivity.

Finite nonlinearity still matters. At unit and double scales, method ordering changes relative to the local prediction in the positive-sign cases of heads 0, 2, and 4. These are mostly comparisons with small local ordering margins. In particular, local Fisher does not predict a universal benefit: head 0's oblique map is locally slightly worse than PCA, and head 4's local oblique recovery is also below PCA. Because these differences already exist under the evaluation-pair local metric, finite nonlinearity alone cannot explain why the training-derived maps fail to dominate there. Metric estimation from only 16 training receivers, source covariance shift, and source/receiver pairing are remaining possibilities.

This narrows the explanation: downstream probability geometry and the coordinate-reading rule explain much of the observed benefit, with finite nonlinear corrections that vary by head and scale. It does not establish that the resulting axes are unique or stable across independently estimated metrics, nor that the original network implements the approximate map as an isolated internal unit.

Audits: the float32 MLP directional derivative agrees with independent double-precision algebra to maximum relative error 7.81e-7 on the checked source direction for each head. Earlier H498 verified the composed readout derivative by central differences; this experiment additionally checks small-scale native behavior. Every unit-scale error/effect aggregate replays H500/H501 within the 1e-5 recovery tolerance. All rows are finite; the smallest saved KL is positive, about 6.48e-12. Very small individual effects can still be sensitive to floating-point error, so conclusions rely on pooled diagnostics rather than tiny per-row ratios. No documents were added; tracked total remains 19,456.

The twenty-hypothesis tournament is updated: hypothesis4 (local sensitivity explains the large gain) gains support; hypothesis5 (finite nonlinear effects) remains relevant but is not the main explanation for the strongest gains. Hypothesis6 (metric estimation noise) and hypothesis13 (effect concentration) are not resolved. Hypotheses14–15 (stable subspaces/operators versus unstable individual axes) now become especially relevant to claiming a canonical form.

Next H503: fit rank-4 Fisher source maps using several prescribed disjoint groups of 16 training receivers, keeping source covariance fixed, and compare full read/write operators, write subspaces, and finite held-out behavior across all heads. Use existing training contexts only; keep H500/H499 evaluation pairs frozen and treat reuse as diagnostic. Include the original fit as reference without choosing the best refit by evaluation outcomes. This distinguishes reproducible behavior-relevant structure from sensitivity to the small metric-estimation sample. Operator comparisons should allow coordinate rotations; unstable axes alone do not imply unstable computation. A subsequent larger-training-metric confirmation should follow only if this diagnostic supports that explanation.
