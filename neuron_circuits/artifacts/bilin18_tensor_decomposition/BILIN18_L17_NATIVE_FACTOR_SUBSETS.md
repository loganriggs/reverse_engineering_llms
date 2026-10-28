

## H324 — Native factor subset curve

Rank native factors by their individual squared output contribution on the6144 reference examples: mean(feature_j²)×norm(fitted_readout_j)², accounting for H323 feature normalization. The H323 full fitted readout uses reference labels and a calibration-selected penalty. Thus ranking excludes evaluation data, but is not wholly calibration-independent. Compare nested ranked subsets with one nested random permutation at256,512,1024,2048 factors, and replay all4608. Refit outputs for every subset; independently select a common family ridge using the same seven-point grid and discovery768:896 calibration. Evaluate896:1024. No new documents or semantic labels are used.

| Factors | Selection | Ridge | Actual fidelity | Angle fidelity | Stored scalars incl. geometry |
|---:|---|---:|---:|---:|---:|
|256|ranked|100.0|3.01%|3.02%|1,070,210|
|512|ranked|10.0|8.39%|7.88%|1,954,946|
|1024|ranked|10.0|14.13%|13.40%|3,724,418|
|2048|ranked|1.0|33.50%|31.19%|7,263,362|
|4608|ranked|0.001|90.69%|89.97%|16,110,722|
|256|random|100.0|1.61%|1.56%|1,070,210|
|512|random|10.0|3.34%|2.85%|1,954,946|
|1024|random|10.0|9.52%|8.99%|3,724,418|
|2048|random|1.0|23.90%|21.46%|7,263,362|

Locked gates: ranked1024_80: FAIL, ranked1024_gain10: FAIL. The1024-factor accuracy gate requires80% fidelity in both families; selection gain requires>=10pp over random1024 with positive paired95% lower bounds.

|1024-factor contrast|Gain (pp)|Conditional paired95% (pp)|
|---|---:|---:|
|actual|4.60|[4.42,4.80]|
|angle|4.41|[4.24,4.58]|

**Interpretation limits.** Ranking measures individual contribution energy, not unique or causal importance: correlations and cancellation can make it suboptimal. A failed subset curve does not rule out better sparse selection or low-rank mixtures. A successful subset is a response predictor, not an intact replacement or a semantic ontology. One random permutation does not establish robustness across random seeds. Evaluation documents have prior adaptive research use. Bootstrap conditions on all fitting and selection.

**Storage.** Each subset stores two k×1152 factor matrices and a k×1152 output readout, plus shared core96, partner64, mean and two scalar normalizers:3×k×1152+185474 scalars. Integer IDs are additional metadata. This excludes the baseline compiler and other corrections required for a complete model. Storage ratios must not be described as an intact MLP replacement.

**Audit.** All selected normal-equation residuals<1e-7. Reordering and refitting all4608 factors reproduces H323 coefficients with relative error1.62e-12. Reference and full-readout hashes verified. Saved selection order, contribution scores, all fitted subset models, ridge curves and per-document errors. Native weights unchanged.

**Next discriminator.** Use the curve to choose between sparse native factors and distributed mixtures. Promote only frozen candidates to fresh cross-source documents; audit semantic and behavioral roles separately. Broad research goal remains active.
