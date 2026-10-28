

## H283 — Frozen feature families have modest complementary information

H282 compared three separately fit64-feature adapters. H283 freezes all their coefficients and feature maps, then selects a convex mixture of their output corrections. The grid has286 triples in.1 increments with nonnegative entries summing<=1; unused mass gives the zero-correction base. Mixtures are formed before final native RMS/head/softcap, not by averaging output probabilities or individual KL values. The existing256 validation documents select weights; no training or fresh-test labels select them. This panel has already selected the underlying adapters, so fresh replication matters. A separate shrunken-single control searches the same grid restricted to at most one nonzero weight, separating complementarity from simple correction shrinkage.

Validation selects linear/quadratic/outside weights(.5,.1,.4), KL.05022582. The best shrunken single is(.9,0,0), KL.05191811; the best unscaled single is(1,0,0), KL.05203628. All grid results are retained. Original adapters and native weights remain unchanged. The three mixture coefficients add little fitting freedom, but inference still requires all included maps; this is not a free compute/storage improvement. The quadratic component prevents folding the entire mixture into a linear readout.

Fresh128 documents exclude2752 earlier unique complete-document hashes. Independent document/window/file hash and exact window-content audits pass. Not a model-training holdout. Selection audit verifies the286-point grid, global minimum within that grid, shrunken-single minimum, and agreement of the pure-linear validation score with H280 within1e-7. Fresh pure-single metrics match the original linear prediction exactly. These are numerical consistency checks, not a proof of optimizer optimality or semantic interpretation.

| Model | Fresh KL | KL recovery vs original compiler | CE increase | Native top1 agreement |
|---|---:|---:|---:|---:|
|constant|0.054076|44.22%|0.06055|83.98%|
|single|0.050044|48.38%|0.05020|83.98%|
|shrink|0.049900|48.53%|0.05070|83.98%|
|mixture|0.048239|50.24%|0.04879|84.77%|

- mixture_gain: 3.61% relative KL reduction, paired document-bootstrap95% [1.47,5.88]%.
- complementarity: 3.33% relative KL reduction, paired document-bootstrap95% [1.55,5.17]%.

Both positive intervals support a small improvement conditional on these fits. Both effect sizes miss the preset5% thresholds, so mixture_gain and complementarity gates FAIL. Repair also FAILS:50.24% recovery is below80%, CE+.04879 exceeds.02, and84.77% top1 agreement is below98%. No threshold relaxed. Intervals condition on fixed selected models and do not include training/model-selection variability; no simultaneous-family confidence claim.

An oracle selects the lowest-native-KL single family separately for each fresh document and achieves meanKL.04570108. This uses fresh native labels, is non-deployable, and is only the best documentwise selector among these three fixed single models. It is not an upper bound on arbitrary mixtures, nonlinear readouts, or model classes. The gap suggests some document-specific variation in family usefulness; it does not establish that a label-free router can predict that usefulness.

**Hypothesis update.** Individually weaker maps can contribute complementary information: the fixed mixture beats the shrinkage control by3.33% on fresh data. Thus H282 did not establish redundancy of outside directions or quadratic features. However, the gain is small and leaves about half the original compiler KL error. This is evidence about overlapping approximation errors, not independent semantic units. The best explanation remains a combination of incomplete feature coverage and estimation/metric dependence, with no uniquely identified computational ontology.

**Next discriminator.** Recheck intervention behavior of this frozen mixture. On fresh documents, compare native source-removal responses against base, single, shrink and mixture corrections, for all nine source heads plus matched-angle payload controls. Measure response fidelity and conditional downstream KL with a common native intact baseline, and separately distinguish intact replacement from conditional response repair. Lock thresholds before outcomes. A natural-state gain with worsened responses would support distribution-specific fitting; gains on both would support broader computational coverage. Keep per-head results and paired document uncertainty. Do not promote the mixture solely from intact KL.

All selection/evaluation jobs terminal. Runtime arrays and full grid retained under bilin18_l17_feature_mixture*. Broad research goal remains active.
