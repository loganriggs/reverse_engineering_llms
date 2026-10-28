## H485 — Refreshed hypotheses before the source decomposition experiment

These hypotheses can coexist. Ranking is qualitative, not a numerical posterior. Evidence through H484 precedes this experiment's outcomes.

| # | Hypothesis | Current evidence and discriminating test |
|---|---|---|
|1|Computational units depend on the fidelity criterion.|Strong: raw and probability-optimal coordinates differ. Compare native coordinates with behavioral compensation.|
|2|Small output components carry large behavioral effects.|Strong: H481 restores~2.1% raw energy with substantial KL benefit. Test learnable sensitivity-selected directions.|
|3|Fixed output spans have unused behavioral capacity.|Strong oracle evidence H476/H480/H483; does not prove coefficient learnability.|
|4|Finite-KL versus local-Fisher objective mismatch dominates learned error.|Weakened for tested methods by H484; alternative optimizers/regularization remain open.|
|5|Receiver-dependent optimal coefficients are not linear in retained kernel features.|Plausible; compare context-conditioned coefficients with matched-capacity controls.|
|6|Retained input coordinates omit important information.|Unresolved: isolated128→256 gain not established. Test targeted missing inputs.|
|7|Training receiver coverage is insufficient.|Plausible: smooth point learning curve, but all H478 joint gains fail. Test repeated independent training draws.|
|8|Regularization/selection limits transfer.|Plausible: strong ridge and very early stopping selected; test training-only risk or stable validation.|
|9|Rare large-effect receivers dominate estimates.|Strong descriptive evidence across panels. Audit concentration and replicated paired gains.|
|10|Euclidean PCA selects wrong output directions for behavior.|Partially weakened: PCA20 has better oracle behavior, but learned transfer fails. Test sensitivity-based basis versus energy-based basis.|
|11|The source is functionally almost rank1.|Geometric support~97.6% energy, behavioral test pending. Rank1 intervention versus full source is today's test.|
|12|Small source tails disproportionately drive behavior.|Plausible by analogy to output remainder, not established. Tail-only intervention and rank1 omission test.|
|13|Source rank1 and tail interact strongly.|Unknown. Compare native full response with the sum of separate responses.|
|14|Receiver dependence converts one source coordinate into many output directions.|Plausible. If rank1 source suffices but outputs remain complex, test response subspace across receivers.|
|15|RMS normalization creates much of the nonlinearity.|Mechanistically plausible; follow source factorial with normalization-frozen controls.|
|16|Bilinear cross-products create non-additive source effects.|Mechanistically plausible; distinguish from normalization via exact fixed-normalizer decomposition.|
|17|Donor semantic content is required for compact behavior.|Weakened for earlier conditional tests by covariance-Gaussian controls; not ruled out globally.|
|18|The dominant source variation is mostly a scale-related direction.|Geometry supports alignment with source mean; test radial versus nonradial effects without inferring semantics.|
|19|Observed-token CE allows qualitatively simpler representatives than KL.|Open; earlier CE improvements did not confirm. Capacity-matched comparison remains pending.|
|20|These units transfer across heads/layers/models.|Unsupported broad claim; previous head heterogeneity argues caution. Replicate only after identifying a useful mechanism.|

Priority now: hypotheses11–16. They offer direct native-model discrimination without another learned compression fit. Freeze top1 from the old disjoint calibration source basis. On already-used receivers, compare full source, its top1 projection and its orthogonal tail at their original amplitudes, both signs. Compare full native response with the sum of separate native responses. No norm matching, semantic label claim, or fresh confirmation. This tests whether the difficult outputs originate in a small source tail or receiver-dependent processing of the dominant direction. Canonical coefficient learning and CE comparisons remain open.
