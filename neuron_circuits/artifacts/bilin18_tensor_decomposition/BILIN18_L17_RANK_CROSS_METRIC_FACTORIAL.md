

## H368 — Probability weighting explains most rank-2/complement interference

Reuse H367’s post-RMS finite response split u=P₂Δ and v=(I−P₂)Δ, selected heads 0/1/6, 128 documents and actual/six moment/six donor conditions. Cross uniform versus intact-model probability weighting with identity versus logit-cap derivative. Keep the intact capped model probabilities fixed across derivative choices. Thus the probability/no-cap cell is a metric diagnostic, not the behavior of an uncapped model; no second RMS normalization or native weight change.

For each cell, pool the signed cross term C=2uᵀFv and total energy E=(u+v)ᵀF(u+v), then take z=C/E. With cells U0,U1,P0,P1, probability attribution is [(P0−U0)+(P1−U1)]/2 and cap attribution is [(U1−U0)+(P1−P0)]/2. These add exactly to P1−U0. Shares divide by that signed total change. This attributes a normalized diagnostic including its changing denominator, not an absolute amount of causal behavior.

|Head|Family|Uniform, cap C/E|Probability, no cap C/E|Probability, cap C/E|Probability share (joint 95% interval)|
|---|---|---:|---:|---:|---|
|0|actual|-0.0083|-0.719|-0.762|0.966 [0.908, 1.024]|
|0|moment|-0.0024|-0.616|-0.598|1.013 [0.955, 1.071]|
|0|donor|-0.0018|-0.540|-0.528|1.010 [0.952, 1.068]|
|1|actual|-0.0003|-0.979|-0.866|1.065 [1.007, 1.123]|
|1|moment|-0.0007|-0.856|-0.793|1.039 [0.981, 1.097]|
|1|donor|-0.0004|-0.894|-0.822|1.043 [0.985, 1.102]|
|6|actual|-0.0121|-1.096|-1.221|0.944 [0.886, 1.002]|
|6|moment|-0.0091|-0.728|-0.923|0.889 [0.831, 0.947]|
|6|donor|-0.0095|-0.696|-0.887|0.887 [0.829, 0.945]|

Uniform without the cap gives essentially zero interference, as expected from the eigenvector construction. Adding only the cap gives small point estimates (C/E between −0.0122 and −0.0003); these small estimates alone do not establish equivalence. Probability weighting without the cap gives negative interference in all nine simultaneous intervals. Its symmetric attribution share ranges from 88.7% to 106.5%, with every joint lower bound above 82.9%. Values above 100% mean the cap offsets part of the probability-weighting effect. For head6 controls, the cap strengthens cancellation and receives approximately 11% of the total change.

All frozen gates pass: {'uniform_uncapped_orthogonality': True, 'probability_alone_cancellation': True, 'probability_majority': True}. The algebraic null has max site |C|/sqrt(Eu Ev) 7.14e-15. Exact H367 probability/cap energy replay passes; energy identities pass to floating-point precision. Four thousand shared document/control bootstrap resamples yield joint36 cross-statistic bands and separate joint9 attribution-share bands. Fixed donor dependencies remain conditional. All per-site energy cells and bootstrap statistics are saved. This reused-panel diagnostic is not fresh confirmation.

Interpretation: perpendicular directions under the unweighted vocabulary geometry need not be independent when likely tokens receive more weight. In this panel, that reweighting is the main source of interference. This helps explain why weight eigenvectors predict a relative pairing statistic yet do not define self-contained behavioral units. It does not tell us whether particular token meanings, generic probability concentration, or context-specific response/probability alignment cause the reweighting effect.

Hypothesis update: probability weighting rises to the leading explanation of local interference; a cap-only explanation is weakened; finite curvature from H367 remains a separate effect. Next compare own probabilities with confidence-matched donor probabilities and a leave-one-document-out position-average distribution, holding response vectors fixed and omitting the cap derivative. That distinguishes context-specific pairing from a shared anisotropic token metric before pursuing semantic labels. Broader research goal remains active.
