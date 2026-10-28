## H530 hypothesis refresh — Twenty current explanations and tests

Qualitative assessments, not numerical posterior probabilities. Several explanations can coexist.

| # | Hypothesis | Assessment | Evidence/limitation | Distinguishing test |
|---|---|---|---|---|
|1|Cross-head coordinates capture shared structure|Supported|H512 fresh joint compression gain|Earlier-layer matched joint/head allocation|
|2|Useful source computation has a broad rank spectrum|Supported in this family|H515–530 across samples/positions/history|Compare nonlinear maps at explicit complexity|
|3|Receiver-metric estimation limits fidelity|Fresh support|H521 larger metric beats64 at all ranks|Disjoint metric-fit stability|
|4|Source-covariance estimation limits fidelity|Supported diagnostic|H522 gains across both methods/panels|Fresh fixed maps or controlled sample repeats|
|5|High fidelity identifies a unique map|Weakened|H523 equal fidelity with differing predictions|Independent end-to-end fits|
|6|Axes differ only by rotation inside one subspace|Weakened|H523 differing write subspaces/operators|Compare invariant operators, not axis labels|
|7|Fit-dependent residuals explain all remaining error|Weakened|H524 averaging removes only2–26%|Change family to test shared remainder|
|8|Finite nonlinearities dominate these residual differences|Weakened locally|H524 local predicts finite averaging gains|Larger intervention amplitudes|
|9|Pooling data is competitive with averaging fitted maps|Supported diagnostic|H524 similar fidelity at lower rank|Fresh matched-complexity comparison|
|10|The shared reconstruction mean causes the main failure|Weakened|H518 mean-offset terms small|Mean shift under new distribution|
|11|KL and activation fidelity prefer different local changes|Supported narrowly|H526 exact MSE steps slightly worsen KL|Validated matched objective optimization|
|12|A reproducible CE-specific direction is already identified|Unconfirmed|H527 contrast interval crosseszero|Independent task-loss training/validation|
|13|One-label sampling noise explains CE instability scale|Supported reference comparison|H528 observed/reference variance1.03–1.15|Calibrated model or repeated-label reference|
|14|CE and KL have identical population optima|Unestablished|Low power is not equivalence|Larger properly held-out task-loss study|
|15|A few extreme contexts alone cause all instability|Unestablished|KL more concentrated yet reproducible|Prespecified influence and robust-estimation controls|
|16|Maps are specific to position63|Weakened|H529 transfer at31/95/127|Other corpora and layers|
|17|The position trend is only target-token composition|Weakened|H530 trend persists with same target|Matched history/position interventions|
|18|Longer history adds source components poorly retained at low rank|Leading open mechanism|H530 lower low-rank fidelity with longer history|H531 compress context-induced source deltas|
|19|Most retained activity is shared, while contextual updates are missed|Leading open alternative|Whole-source fidelity does not test increments|H531 delta-specific behavioral recovery|
|20|These maps are standalone fundamental MLP programs|Not established|Native upstream/downstream computation retained|Executable replacement and explicit dependency accounting|

The strongest current account combines distributed source structure with estimation limits in source covariance and receiver sensitivity. It supports useful data-dependent approximations, not unique axes. The next mechanistic distinction is whether frozen coordinates preserve the update induced by additional context, rather than merely the larger shared activation. H531 addresses18–19 at a fixed receiver. The CE branch remains unresolved because the attempted correction is not reliably identified; noise-scale agreement must not be promoted to objective equivalence.
