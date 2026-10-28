## H441 — More baseline coverage helps activity transfer, not enough behavioral fidelity

H440 exposed activity-basis overfitting. H441 recollects baseline states for all512 H434 documents, fixes a seeded384-training/128-validation split, and compares nested32/128/384 training-context sets. For each set, mix the uncentered empirical input covariance with a trace-matched isotropic covariance at activity weights0,.25,.5,.75,1. Weight0 recovers the weight-only objective. Output rank remains64, preserving the1,253,376 map+basis coefficient budget.

Select each training-count candidate and the overall candidate solely by reserved baseline T_H*x activity reconstruction. Freeze selection before loading H438 transfer outcomes. The reserved split validates this output-covariance choice conditional on the already-frozen source maps; some maps’ prior calibration used H434 states, so it is not pristine independent validation of the whole pipeline. H438 finite results were already inspected in prior work; this remains a mechanism diagnostic. No new documents; total12480.

| Training contexts | Selected activity weight | Validation activity coverage | Transfer activity coverage | Full finite-KL recovery low / median |
|---|---:|---:|---:|---|
|32|0.25|92.37%|88.32%|93.36% / 96.57%|
|128|0.25|93.08%|89.00%|93.66% / 96.73%|
|384|0.50|93.55%|89.74%|93.17% / 96.83%|

The validation-selected overall model uses384 contexts and a50/50 covariance mixture. It improves transfer map-activity coverage to89.74% compared with85.47% for H440’s32 selected contexts without shrinkage. Under the current nested random training rule, pure empirical transfer coverage rises from84.79% at32 contexts to89.19% at384; the selected mixture adds a further0.55 points at384. Thus coverage explains part of the activity failure, while covariance regularization helps modestly. The current32 random contexts differ from H440’s selected low/median contexts, so that earlier comparison is not an isolated count effect.

| Frozen overall-selected contrast | Estimate | Joint95% interval | Gate passes |
|---|---:|---|---|
|selected_minus_weight_low|-0.002892|[-0.048015, +0.042231]|False|
|selected_minus_weight_median|+0.009459|[-0.035664, +0.054582]|False|
|selected_minus_exact_low|-0.050372|[-0.095495, -0.005250]|False|
|selected_minus_exact_median|-0.025761|[-0.070883, +0.019362]|False|

All four gates fail: >2-point improvement over weight-only and within2-point noninferiority to exact compiled recovery are not established. At the selected384-context mixture, finite-KL recovery is93.17% low/96.83% median, versus93.46%/95.89% for weight-only and98.21%/99.41% exact. Point improvements differ by stratum, and uncertainty includes either sign relative to weight-only. The low selected-versus-exact interval excludes zero, so a loss remains established there within this conditional diagnostic band. Bootstrap uses4000 paired source/document samples, conditional on the selected basis and fixed donor pool; selection uncertainty is not resampled.

Larger training sets improved baseline validation and transfer activity, but did not monotonically improve low-stratum finite KL: the128-context selected mixture gives93.66%, versus93.17% at384. Do not choose the128-context option using that observation; the overall winner was frozen using baseline validation, and its result remains the reported primary.

Audits: training and validation document indices are disjoint and cover all512 calibration documents; source/corpus/maps hashes are checked; selected basis files are written before transfer evaluation; scored arrays are finite. The fixed-rank storage, source basis, exact radial correction, native baseline and intervention setup match the earlier comparison. No group membership is refitted and no contexts are excluded by outcome.

Interpretation: more baseline examples partly remedy state-coverage overfitting, but unweighted activity reconstruction remains an imperfect selection objective for behavioral fidelity. This does not imply that more data is useless, or prove the current fixed output representation impossible. It isolates a meaningful next distinction: output-metric alignment versus residual state coverage.

Next compare an output-probability-sensitive baseline objective with the unweighted objective at the same rank and calibration split. Construct metrics from baseline states and model weights, without full finite targets or target-fitted coefficients. Validate any regularization on reserved baseline predicted-effect error, then freeze before another transfer test. Keep radial correction exact and report raw reconstruction alongside finite KL. Because the current transfer panel is already part of the research loop, a chosen design still needs independent confirmation.
