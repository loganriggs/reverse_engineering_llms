## H994 — Bounded coordinate atlas identifies strong candidate readouts and weak common-site targets

Previous turn was progress: H993 tested task/coactivation groupings and found no advantage over simpler faithful supports. This turn measures finite downstream coordinate effects rather than treating activation-energy signatures as causal profiles. It stays bounded across six tasks, without detailed single-task tracing.

For each original training family, choose the lowest-index H988 discovery context passing native competence, retain allfour source pairs (no individual filtering), and freeze that panel before interventions.48directed edits/24pairs. AtMLP12 and the frozen H991 PCA128 basis, replace one coordinate with its donor value. Scan all128coordinates. Each coordinate edit has an independent seeded Gaussian direction control with identical signed coefficient and double-precision edit norm. Include identity, full-write donor and densePCA128 references. Total48*(256+3)=12,432forwards. Random-direction control is not a semantic nuisance intervention; donor coefficient magnitude still varies by coordinate/task.

Freeze each family's top8 coordinates by mean signed donor-directed answer-margin effect, with low-index tie break. This is discovery-only selection for the next validation experiment. It intentionally favors changes toward the donor answer, not necessarily maximum fidelity to a full-write intervention; when a full write moves against that answer the distinction matters.

| Family | Full-write mean margin effect | Top coordinate (zero-indexed) | Top-coordinate mean | Norm-matched random RMS | Positive fraction (8edits) |
|---|---:|---:|---:|---:|---:|
|agreement|1.616|8|0.705|0.050|1.000|
|capital_recall|0.377|20|0.144|0.168|0.625|
|entity_binding|-0.050|42|0.015|0.008|0.875|
|sentiment|0.835|4|0.811|0.095|1.000|
|negation|1.110|4|1.088|0.196|1.000|
|pronoun_gender|0.493|16|0.143|0.021|1.000|

Coordinate4 has large effects for both sentiment and negation, while agreement and pronoun gender have different leading coordinates. This is compatible with a shared valence-related output channel. Both tasks use the same positive/negative answer groups, so it is not evidence that coordinate4 implements negation or is shared upstream computation. Alldirections are PCA-derived in a fixed gauge, not proven natural units.

Capital recall's leading coordinate20 mean0.144 is smaller than its random-direction RMS0.168 and has only5/8 positive directed effects. Entity binding's entireMLP12 donor swap has mean-0.050; its best positive-coordinate subset is not a faithful substitute for that full write by construction. These tasks should remain weak-site diagnostics, not be promoted as successful causal anchors simply because the native model answers correctly. Earlier localization already identified other sites for binding; do not infer model inability from this common-layer assay.

Single-atom effects are not generally additive: agreement's sum over128means1.695 versus densePCA128 mean1.616, and pronoun0.512 versus0.493. Sentiment/negation are closer in this panel but that does not license joint-effect claims. Top8 single-effect sums are reported only as descriptive quantities; actual joint-subspace interventions must be measured next, including negative/complement effects where fidelity is assessed.

Conditional discovery ranking stability between source0–1 and2–3 (Spearman across all128 coordinate means): agreement0.273, capital0.131, binding0.211, sentiment0.471, negation0.370, pronoun0.644. Onlytwo pairs per half and many tiny effects; these are descriptive, not a validated stability threshold or independent replication. Frozen top8sets overlap across several tasks, but such overlap alone does not establish reuse.

Audits:12,432/12,432forwards, identity logit error1.82e-5. Double-precision edit-norm mismatch1.39e-15 before model float32 casting; exact float32 perturbation equality is not claimed. Full source/manifest hashes unchanged; all outputs finite. Allsummary means and top8selections independently recomputed. No validation task outcomes used for selection, no featurizer training and no reserved-test access. Natural-document pool26,112.

Next: apply the frozen top8subspaces to the SAME cross-task validation panel, measure actual joint finite effects and full-reference KL, and compare own-task versus other-task subspaces at identical rank. Test context transfer and response selectivity before using profiles to group atoms. Keep weak native/causal cohorts explicit. This directly tests reusable response structure without another prolonged sentiment case study. Bootstrap unlabeled-task discovery/retraining remains outstanding; H961 context-bridge predictor evaluation remains pending.
