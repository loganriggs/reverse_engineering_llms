## H532 — Update-specific calibration helps modestly

Fit source-update maps on the existing3,072 calibration documents, with no document overlap with the2,048 evaluation documents. Collect matched32/128-token histories ending at the same input token and fit the uncentered second moment of their L17 summed-write difference. Keep the old1,023-dimensional basis,1,024-context receiver metric and ranks fixed. No intercept is added; the delta mean is included in the second moment. Save all maps before evaluation on the three existing H531 hybrid update tasks.

| Short history | Map | Whole-source-fit recovery | Update-fit recovery | Relative KL error reduction |
|---|---|---:|---:|---:|
|32|PCA36|57.23%|60.40%|7.42%|
|32|Fisher36|60.76%|62.99%|5.70%|
|32|PCA144|82.25%|83.04%|4.47%|
|32|Fisher144|84.01%|84.68%|4.18%|
|32|PCA576|97.65%|97.77%|5.12%|
|32|Fisher576|98.20%|98.30%|5.48%|
|64|PCA36|58.52%|61.51%|7.21%|
|64|Fisher36|60.93%|63.29%|6.02%|
|64|PCA144|82.65%|83.92%|7.31%|
|64|Fisher144|84.07%|85.37%|8.17%|
|64|PCA576|97.58%|97.74%|6.67%|
|64|Fisher576|98.14%|98.28%|7.43%|
|96|PCA36|58.32%|59.35%|2.48%|
|96|Fisher36|60.31%|61.59%|3.21%|
|96|PCA144|81.68%|82.75%|5.88%|
|96|Fisher144|83.18%|84.37%|7.06%|
|96|PCA576|97.45%|97.58%|4.92%|
|96|Fisher576|97.99%|98.17%|8.65%|

Every tested map improves its point-estimate update KL, but the changes are modest. Fisher36 recovery moves from about60–61% to62–63%; Fisher144 reaches84–85%; Fisher576 about98.2–98.3%. The broad rank spectrum remains. Calibrating to contextual updates helps but does not explain away their distributed representation. These are reused-panel diagnostics, not fresh superiority claims.

The comparison changes source second moments, including mean treatment appropriate to no-intercept deltas; it does not isolate semantic content. The fixed downstream metric still comes from position63 rather than the128-token receiver. Both constructions retain native downstream MLP computation. The hybrid source intervention does not represent the model's total use of added history.

Audits: all frozen artifacts hashed; calibration/evaluation complete-document sets disjoint; covariance positive definite; root/inverse and idempotence checks pass; zero-map baselines match H531 and identity produces zero KL/CE exactly; arrays finite. No new documents:26,112.

Research steering: the user asks to discuss the most likely hypotheses and whether compression directly addresses fundamental units of computation. Do not continue automatically into another compression sweep. This result constrains approximation complexity and objective dependence but does not identify computational meaning. The next experimental design should be chosen from explicit mechanistic hypotheses, with predictions about context/content interventions, causal mediation and transport, rather than treating improved compression as sufficient progress toward interpretable units. Broad goal remains active; no claim of completion or user-requested pause.
